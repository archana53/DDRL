## Standard libraries
import os
from copy import deepcopy
import argparse

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from ldms import UnconditionalDiffusionModelConfig, UnconditionalDiffusionModel
from functools import partial
from PIL import Image
from diffusers import LDMPipeline
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)


class ContrastiveHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels1=1024, hidden_channels2=256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2
        self.out_channels = out_channels

        self.fc0 = nn.Linear(self.in_channels, self.hidden_channels1)
        self.bn0 = nn.BatchNorm1d(self.hidden_channels1)
        self.fc1 = nn.Linear(self.hidden_channels1, self.hidden_channels2)
        self.bn1 = nn.BatchNorm1d(self.hidden_channels2)
        self.fc = nn.Linear(self.hidden_channels2, self.out_channels)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc(x)
        return x


class ContrastiveLearning(pl.LightningModule):
    def __init__(
        self,
        backbone,
        head,
        optimizer,
        temperature,
        weight_decay,
        timestep,
        scales,
        scale_direction,
        max_epochs=500,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.timestep = timestep.to(device)
        self.scales = scales
        self.scale_direction = scale_direction
        self.optimizer = optimizer

        assert self.temperature > 0.0, "The temperature must be a positive float!"
        self.backbone = backbone
        self.backbone.set_feature_scales_and_direction(self.scales, self.scale_direction)
        self.head = head

        self.configure_optimizers()

    def forward(self, x):
        noisy_pred, x = self.backbone.get_features(x, self.timestep)
        x = self.head(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        # Encode all images
        feats = self.forward(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],  # First position positive example
                cos_sim.masked_fill(pos_mask, -9e15),
            ],
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        loss = self.info_nce_loss(batch, mode="train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.info_nce_loss(batch, mode="val")
        return {"loss": loss}


class ContrastiveTransformations(object):
    def __init__(self, n_views=2):
        self.base_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=96),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=9),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def train_cl(batch_size, unlabeled_data, train_data_contrast,lora_rank,lr,checkpoint_path, max_epochs=500, **kwargs):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        devices=1,
        max_epochs=max_epochs,
        log_every_n_steps=50,
        val_check_interval=1,
        callbacks=[checkpoint_callback],
    )
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(checkpoint_path, "ContrastiveLearning.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = ContrastiveLearning.load_from_checkpoint(
            pretrained_filename
        )  # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = data.DataLoader(
            unlabeled_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = data.DataLoader(
            train_data_contrast,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        pl.seed_everything(42)  # To be reproducable

        diff_model_config = UnconditionalDiffusionModelConfig()
        diff_model = UnconditionalDiffusionModel(diff_model_config)
        lora_layers = diff_model.add_lora_compatibility(lora_rank)
        head = ContrastiveHead(
            in_channels=4480,
            out_channels=16,
        )
        optimizer = optim.Adam(
            lora_layers.parameters(),
            lr=lr,
        )
        model = ContrastiveLearning(
            backbone=diff_model,
            head=head,
            optimizer=optimizer,
            max_epochs=max_epochs,
            **kwargs,
        )

        for params in model.backbone.parameters():
            params.requires_grad = False
        for params in lora_layers.parameters():
            params.requires_grad = True

        trainer.fit(model, train_loader, val_loader)
    return model


class CelebAHQDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = [image_name for image_name in os.listdir(self.root_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.root_dir + "/" + self.images[idx]
        image = Image.open(image_path)

        if self.transform is not None:
            transformed = self.transform(image)
            image = transformed
        label = 1
        return image, label

def parse_args():
    parser = argparse.ArgumentParser(
        description="Contrastive pretraining of LORA layers in unconditional diffusion model"
    )

    # Diffusion Parameters
    diffusion_group = parser.add_argument_group("diff_args")
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
        default=[0, 1, 2, 3],
        help="Input one of the following options: all, 0, 1, 2, 3",
    )
    diffusion_group.add_argument(
        "--time_step",
        type=int,
        default=50,
        help="Features extracted from which time step",
    )

    diffusion_group.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LORA layers",
    )

    #Training parameters
    training_group = parser.add_argument_group("training_args")
    training_group.add_argument(
        "--data_path",
        type=str,
        default="/Users/omscs/Desktop/celebhq",
        help="dataset path",
    )
    training_group.add_argument(
        "--checkpoint_path",
        type=str,
        default="../saved_models/cont_checkpoint",
        help="checkpoint path",
    )
    training_group.add_argument("--lr", type=float, default=5e-4)
    training_group.add_argument("--temp", type=float, default=0.07)
    training_group.add_argument("--weight_decay", type=float, default=1e-4)
    training_group.add_argument("--batch_size", type=int, default=6)
    training_group.add_argument("--max_epochs", type=int, default=100)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = args.data_path
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = args.checkpoint_path
    # In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
    # workers as possible in a data loader, which corresponds to the number of CPU cores

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    celeb_data = CelebAHQDataset(
        root_dir=DATASET_PATH,
        transform=ContrastiveTransformations(n_views=2),
    )
    clr_model = train_cl(
        batch_size=args.batch_size,
        unlabeled_data=celeb_data,
        train_data_contrast=celeb_data,
        lora_rank=args.lora_rank,
        checkpoint_path=CHECKPOINT_PATH,
        lr=args.lr,
        temperature=args.temp,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        timestep=torch.LongTensor([args.time_step]),
        scales=args.scales,
        scale_direction=args.scale_direction,
    )