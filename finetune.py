import argparse
import pytorch_lightning as pl
from modules.ldms import UnconditionalDiffusionModel, UnconditionalDiffusionModelConfig
import torch
from modules.data.datasets import CelebAHQMaskDataset, DepthDataset, KeyPointDataset
from modules.decoders import ConvHead, MLPHead, PixelwiseMLPHead, KeyPointHead
from modules.trainer import PLModelTrainer
from pathlib import Path
import numpy as np


TASK_CONFIG = {
    "Depth_Estimation": {
        "dataloader": DepthDataset,
        "head": PixelwiseMLPHead,
        "criterion": torch.nn.MSELoss,
    },
    "Facial_Keypoint_Detection": {
        "dataloader": KeyPointDataset,
        "head": KeyPointHead,
        "criterion": torch.nn.MSELoss,
    },
    "Facial_Segmentation": {
        "dataloader": CelebAHQMaskDataset,
        "head": None,
        "criterion": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune pre-trained diffusion features on a downstream task"
    )

    # Diffusion Parameters
    diffusion_group = parser.add_argument_group("diff_args")
    diffusion_group.add_argument(
        "--img_res", type=int, default=256, help="Image resolution"
    )
    diffusion_group.add_argument(
        "--conditional",
        "-c",
        action="store_true",
        help="Conditional/Unconditional Diffusion",
    )
    diffusion_group.add_argument("--model_path", type=str, help="Path to model")
    diffusion_group.add_argument(
        "--scale_direction",
        type=str,
        nargs="+",
        default=["up", "down"],
        help="Input one of the following options: up, down, both",
    )
    diffusion_group.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Input one of the following options: all, 0, 1, 2",
    )
    diffusion_group.add_argument(
        "--time_step",
        type=int,
        default=50,
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
    training_group.add_argument("--full_finetuning", "-f", action="store_true")
    training_group.add_argument("--lora_rank", type=int, default=4)
    training_group.add_argument("--optimizer", type=str, default="Adam")
    training_group.add_argument("--lr", type=float, default=1e-3)
    training_group.add_argument("--num_epochs", type=int, default=100)
    training_group.add_argument("--batch_size", type=int, default=16)
    training_group.add_argument("--precision", type=str, default="32-true")
    training_group.add_argument("--max_epochs", type=int, default=100)
    training_group.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    device = "cpu"  # TODO CHANGE BACK TO DEVICE
    # Get the backbone model and setup
    if args.conditional:
        raise ValueError("Conditional Diffusion Models not supported yet")
    else:
        model_config = UnconditionalDiffusionModelConfig()
        model = UnconditionalDiffusionModel(model_config)

    model.set_feature_scales_and_direction(args.scales, args.scale_direction)
    model.to(device)

    # Setup Finetuning Task Dataloader and Head
    task_config = TASK_CONFIG[args.name]
    task_train_dataloader = task_config["dataloader"](
        root=Path(args.root_path),
        mode="train",
        size=(args.img_res, args.img_res),
    )
    task_val_dataloader = task_config["dataloader"](
        root=Path(args.root_path),
        mode="val",
        size=(args.img_res, args.img_res),
    )
    if task_config["head"] == PixelwiseMLPHead:
        in_features = model.feature_size[1]
        out_features = 1

    task_head = task_config["head"](
        in_channels=in_features, out_channels=out_features
    ).to(device)
    task_criterion = task_config["criterion"]()

    if args.full_finetuning:
        lora_layers = model.add_lora_compatibility(args.lora_rank)
        all_trainable_params = list(lora_layers.parameters()) + list(
            task_head.parameters()
        )

        optimizer = torch.optim.Adam(all_trainable_params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(task_head.parameters(), lr=args.lr)

    # Initialize the trainer
    trainer = pl.Trainer(devices=args.gpus, max_epochs=args.max_epochs)

    # Initialize the model
    model_trainer = PLModelTrainer(model, task_head, task_criterion, optimizer)

    # Train the model
    trainer.fit(model_trainer, train_dataloaders=task_train_dataloader)
