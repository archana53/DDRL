import argparse
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from modules.data.datasets import (
    CelebAHQMaskDataset,
    DatasetWithFeatures,
    DepthDataset,
    KeyPointDataset,
)
from modules.decoders import PixelwiseMLPHead
from modules.feature_loader import FeatureLoader
from modules.ldms import UnconditionalDiffusionModel, UnconditionalDiffusionModelConfig
from modules.trainer import PLModelTrainer
import wandb
from pytorch_lightning.loggers import WandbLogger

TASK_CONFIG = {
    "Depth_Estimation": {
        "dataloader": DepthDataset,
        "head": PixelwiseMLPHead,
        "criterion": torch.nn.MSELoss,
        "out_channels": 1,
        "root_path": "/coc/flash5/schermala3/Datasets/BIWIKinectHeads/preprocessed",
        "feature_store_path": "/coc/flash8/akutumbaka3/DDRL/data/depth_timesteps_0_1000_100_features.h5",
    },
    "Facial_Keypoint_Detection": {
        "dataloader": KeyPointDataset,
        "head": None,
        "criterion": None,
        "out_channels": 19,  # 19 classes
    },
    "Facial_Segmentation": {
        "dataloader": CelebAHQMaskDataset,
        "head": PixelwiseMLPHead,
        "criterion": torch.nn.CrossEntropyLoss,
        "out_channels": 1,
        "root_path": "/coc/flash5/schermala3/Datasets/CelebAMask-HQ/",
        "feature_store_path": "/coc/flash8/akutumbaka3/DDRL/data/celebahqmask_timesteps_0_300_100_features.h5",
    },
}


def setup_diffusion_model(config, args, device):
    if args.conditional:
        raise ValueError("Conditional Diffusion Models not supported yet")
    else:
        model_config = UnconditionalDiffusionModelConfig()
        model = UnconditionalDiffusionModel(model_config)

    model.set_feature_scales_and_direction(config.scales, config.scale_direction)
    return model


def setup_feature_loader(config, args):
    if args.feature_store_path is None:
        raise ValueError("Feature Loader path not specified")

    feature_store_path = TASK_CONFIG[config.name]["feature_store_path"]
    feature_loader = FeatureLoader(
        h5_file=Path(feature_store_path),
        scales=config.scales,
        scale_directions=config.scale_direction,
        timestep=config.timestep,
        resolution=(args.img_res, args.img_res),
    )
    return feature_loader


def setup_dataloaders(dataset_cls, feature_loader=None, config=None):
    root_path = TASK_CONFIG[config.name]["root_path"]
    task_dataset = dataset_cls(root=Path(root_path), mode="val", size=(args.img_res, args.img_res))

    # create dataset splits optionally
    # TODO: check if any dataset has splits already
    task_train_dataset = task_dataset
    task_val_dataset, task_test_dataset = None, None
    if not args.donot_create_dataset_splits:
        task_train_dataset, task_val_dataset, task_test_dataset = data.random_split(
            task_dataset,
            [0.8, 0.2 * 0.33, 0.2 * 0.67],
            generator=torch.Generator().manual_seed(42),
        )
    datasets = [task_train_dataset, task_val_dataset, task_test_dataset]

    # create a DatasetWithFeatures object if using feature loader
    if feature_loader is not None:
        datasets = [
            DatasetWithFeatures(dataset, feature_loader) if dataset is not None else None
            for dataset in datasets
        ]

    # create dataloaders
    dataloaders = [
        data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
        if dataset is not None
        else None
        for dataset in datasets
    ]

    return dataloaders


def train():
    device = "cpu"  # TODO CHANGE BACK TO DEVICE

    # Set up feature extraction method
    model = None
    feature_loader = None
    all_trainable_params = []
    feature_size = None

    # Setup experiment name
    run = wandb.init()
    config = wandb.config
    if config.full_finetuning:
        config.batch_size = 4
        args.use_feature_loader = False
        args.use_diffusion = True
    else:
        args.use_feature_loader = True

    experiment_name = f"{config.name}_t={config.timestep}_scaledir={config.scale_direction}_lr={config.lr}_batchsize={config.batch_size}"
    run.name = experiment_name
    run.save()

    if args.use_diffusion:
        model = setup_diffusion_model(config, args, device)
        feature_size = model.feature_size

        if config.full_finetuning:
            lora_layers = model.add_lora_compatibility(lora_rank=args.lora_rank)
            for param in model.parameters():
                param.requires_grad = False
            for param in lora_layers.parameters():
                param.requires_grad = True
            all_trainable_params = list(lora_layers.parameters())

    else:  # args.use_feature_loader:
        feature_loader = setup_feature_loader(config, args)
        feature_size = feature_loader.feature_size

    # Setup Finetuning Task Dataloader and Head
    task_config = TASK_CONFIG[config.name]
    task_dataset_cls = task_config["dataloader"]
    (
        task_train_dataloader,
        task_val_dataloader,
        task_test_dataloader,
    ) = setup_dataloaders(
        dataset_cls=task_dataset_cls, feature_loader=feature_loader, config=config
    )

    in_features = feature_size[1]
    out_features = task_config["out_channels"]

    task_head = task_config["head"](in_channels=in_features, out_channels=out_features)
    task_criterion = task_config["criterion"]()
    all_trainable_params += list(task_head.parameters())
    optimizer = torch.optim.Adam(all_trainable_params, lr=config.lr)

    # Initialise Logger
    wandb_logger = WandbLogger()

    # Initialize ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", experiment_name),
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        devices=args.gpus,
        max_steps=args.max_steps,
        log_every_n_steps=50,
        val_check_interval=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Initialize the model
    model_trainer = PLModelTrainer(
        model,
        task_head,
        task_criterion,
        optimizer,
        timestep=config.timestep,
        use_precomputed_features=args.use_feature_loader,
    )

    # Train the model
    trainer.fit(
        model_trainer,
        train_dataloaders=task_train_dataloader,
        val_dataloaders=task_val_dataloader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune pre-trained diffusion features on a downstream task"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Depth_Estimation",
        help="Name of the downstream task",
    )
    parser.add_argument(
        "--conditional",
        "-c",
        action="store_true",
        help="Conditional/Unconditional Diffusion",
    )
    parser.add_argument(
        "--use_diffusion",
        "-d",
        action="store_true",
        help="Use Diffusion Backbone to extract features",
    )
    parser.add_argument(
        "--use_feature_loader",
        "-l",
        action="store_true",
        help="Use Feature Loader to load features",
    )
    parser.add_argument(
        "--donot_create_dataset_splits", action="store_true", help="Create dataset splits"
    )
    parser.add_argument("--max_steps", type=int, default=30000, help="Number of steps to train for")
    # General feature extraction parameters
    parser.add_argument("--img_res", type=int, default=256, help="Image resolution")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )

    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA layer")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer to use")
    parser.add_argument(
        "--sweep_id",
        type=str,
    )
    args = parser.parse_args()
    sweep_config = {
        "method": "grid",  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        "metric": {  # This is the metric we are interested in maximizing
            "name": "val_loss",
            "goal": "minimize",
        },
        # Paramters and parameter values we are sweeping across
        "parameters": {
            "name": {"values": [args.name]},
            "lr": {"values": [1e-4, 1e-3, 5e-3]},
            "timestep": {"values": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
            "full_finetuning": {"values": [False, True]},
            "scale_direction": {"values": [["down"], ["up"], ["up", "down"]]},
            "scales": {"values": [[0, 1, 2, 3]]},
            "batch_size": {"values": [16, 32]},
        },
    }
    entity = "notarchana"
    project = "test_sweep_ddrl"

    # Create the sweep
    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    # Run an agent 🕵️ to try out 5 hyperparameter combinations
    else:
        sweep_id = args.sweep_id

    wandb.agent(sweep_id, function=train)
