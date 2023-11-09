import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from modules.ldms import UnconditionalDiffusionModel, UnconditionalDiffusionModelConfig
import torch
from modules.data.datasets import (
    CelebAHQMaskDataset,
    DepthDataset,
    KeyPointDataset,
    DatasetWithFeatures,
)
from modules.feature_loader import FeatureLoader
from modules.decoders import ConvHead, MLPHead, PixelwiseMLPHead
from modules.trainer import PLModelTrainer
from pathlib import Path
import numpy as np
import torch.utils.data as data
import os

TASK_CONFIG = {
    "Depth_Estimation": {
        "dataloader": DepthDataset,
        "head": PixelwiseMLPHead,
        "criterion": torch.nn.MSELoss,
        "out_channels": 1,
    },
    "Facial_Keypoint_Detection": {
        "dataloader": KeyPointDataset,
        "head": None,
        "criterion": None,
    },
    "Facial_Segmentation": {
        "dataloader": CelebAHQMaskDataset,
        "head": None,
        "criterion": None,
        "out_channels": 19,  # 19 classes
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune pre-trained diffusion features on a downstream task"
    )

    # Feature extraction Parameters
    feature_extraction_group = parser.add_argument_group("Feature Extraction Arguments")

    # Diffusion or Feature Loader in a mutually exclusive group
    extraction_method_specifc_group = (
        feature_extraction_group.add_mutually_exclusive_group()
    )
    extraction_method_specifc_group.add_argument(
        "--use_diffusion",
        "-d",
        action="store_true",
        help="Use Diffusion Backbone to extract features",
    )
    extraction_method_specifc_group.add_argument(
        "--use_feature_loader",
        "-l",
        action="store_true",
        help="Use Feature Loader to load features",
    )

    # Diffusion Parameters
    feature_extraction_group.add_argument(
        "--conditional",
        "-c",
        action="store_true",
        help="Conditional/Unconditional Diffusion",
    )
    feature_extraction_group.add_argument(
        "--model_path", type=str, help="Path to model"
    )

    # Feature Loader Parameters
    feature_extraction_group.add_argument(
        "--feature_store_path", type=str, help="Path to h5 file containing features"
    )

    # General feature extraction parameters
    feature_extraction_group.add_argument(
        "--img_res", type=int, default=256, help="Image resolution"
    )
    feature_extraction_group.add_argument(
        "--scale_direction",
        type=str,
        nargs="+",
        default=["up", "down"],
        help="Input one of the following options: up, down, both",
    )
    feature_extraction_group.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Input one of the following options: all, 0, 1, 2,3",
    )
    feature_extraction_group.add_argument(
        "--time_step",
        type=int,
        default=100,
        help="Features extracted from which time step",
    )

    # Downstream Task parameters
    task_group = parser.add_argument_group("task_args")
    task_group.add_argument(
        "--name",
        type=str,
        default="Depth_Estimation",
        help="Name of the downstream task",
    )
    task_group.add_argument(
        "--root_path",
        type=str,
        default="/coc/flash5/schermala3/Datasets/BIWIKinectHeads/preprocessed",
        help="Path to task",
    )

    # Training Parameters
    training_group = parser.add_argument_group("train_args")
    training_group.add_argument(
        "--create_dataset_splits", action="store_true", help="Create dataset splits"
    )
    training_group.add_argument(
        "--full_finetuning",
        "-f",
        action="store_true",
        help="If true, finetune diffusion model as well",
    )
    training_group.add_argument(
        "--lora_rank", type=int, default=4, help="Rank of LoRA layer"
    )
    training_group.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer to use"
    )
    training_group.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    training_group.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    training_group.add_argument(
        "--precision", type=str, default="32-true", help="Precision for training"
    )
    training_group.add_argument(
        "--max_steps", type=int, default=30000, help="Number of steps to train for"
    )
    training_group.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to use"
    )
    training_group.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )

    args = parser.parse_args()
    return args


def setup_diffusion_model(args, device):
    if args.conditional:
        raise ValueError("Conditional Diffusion Models not supported yet")
    else:
        model_config = UnconditionalDiffusionModelConfig()
        model = UnconditionalDiffusionModel(model_config)

    model.set_feature_scales_and_direction(args.scales, args.scale_direction)
    return model


def setup_feature_loader(args):
    if args.feature_store_path is None:
        raise ValueError("Feature Loader path not specified")

    feature_loader = FeatureLoader(
        h5_file=Path(args.feature_store_path),
        scales=args.scales,
        scale_directions=args.scale_direction,
        timestep=args.time_step,
        resolution=(args.img_res, args.img_res),
    )
    return feature_loader


def setup_dataloaders(dataset_cls, args, feature_loader=None):
    task_dataset = dataset_cls(
        root=Path(args.root_path), mode="val", size=(args.img_res, args.img_res)
    )

    # create dataset splits optionally
    # TODO: check if any dataset has splits already
    task_train_dataset = task_dataset
    task_val_dataset, task_test_dataset = None, None
    if args.create_dataset_splits:
        task_train_dataset, task_val_dataset, task_test_dataset = data.random_split(
            task_dataset,
            [0.8, 0.2 * 0.33, 0.2 * 0.67],
            generator=torch.Generator().manual_seed(42),
        )
    datasets = [task_train_dataset, task_val_dataset, task_test_dataset]

    # create a DatasetWithFeatures object if using feature loader
    if feature_loader is not None:
        datasets = [
            DatasetWithFeatures(dataset, feature_loader)
            if dataset is not None
            else None
            for dataset in datasets
        ]

    # create dataloaders
    dataloaders = [
        data.DataLoader(
            dataset,
            batch_size=args.batch_size,
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


if __name__ == "__main__":
    args = parse_args()

    # Setup experiment name
    experiment_name = f"{args.name}_timestep={args.time_step}_scales={args.scales}_scaledir={args.scale_direction}_lr={args.lr}_batchsize={args.batch_size}"

    device = "cpu"  # TODO CHANGE BACK TO DEVICE

    # Set up feature extraction method
    model = None
    feature_loader = None
    all_trainable_params = []
    feature_size = None

    if args.use_diffusion:
        model = setup_diffusion_model(args, device)
        feature_size = model.feature_size

        if args.full_finetuning:
            lora_layers = model.add_lora_compatibility(args.lora_rank)
            all_trainable_params = list(lora_layers.parameters())

    else:  # args.use_feature_loader:
        feature_loader = setup_feature_loader(args)
        feature_size = feature_loader.feature_size

    # Setup Finetuning Task Dataloader and Head
    task_config = TASK_CONFIG[args.name]
    task_dataset_cls = task_config["dataloader"]
    (
        task_train_dataloader,
        task_val_dataloader,
        task_test_dataloader,
    ) = setup_dataloaders(task_dataset_cls, args, feature_loader)

    in_features = feature_size[1]
    out_features = task_config["out_channels"]

    task_head = task_config["head"](
        in_channels=in_features, out_channels=out_features
    )  # .to(device)
    task_criterion = task_config["criterion"]()
    all_trainable_params += list(task_head.parameters())
    optimizer = torch.optim.Adam(all_trainable_params, lr=args.lr)

    # Initialise Logger
    log_dir = f"logs/{experiment_name}"

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
        default_root_dir=log_dir,
        log_every_n_steps=50,
        val_check_interval=1,
        callbacks=[checkpoint_callback],
    )

    # Initialize the model
    model_trainer = PLModelTrainer(
        model,
        task_head,
        task_criterion,
        optimizer,
        timestep=args.time_step,
        use_precomputed_features=args.use_feature_loader,
    )

    # Train the model
    trainer.fit(
        model_trainer,
        train_dataloaders=task_train_dataloader,
        val_dataloaders=task_val_dataloader,
    )
