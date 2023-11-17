# Description: Contrastive pretraining

## Standard libraries
import os
from copy import deepcopy


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



# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)

os.makedirs(CHECKPOINT_PATH, exist_ok=True)



# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)


class ContrastiveLearning(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, lr, temperature, weight_decay,timestep,scales,scale_direction, max_epochs=500, hidden_channels1=256, hidden_channels2=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2
        self.timestep = timestep
        self.scales = scales
        self.scale_direction = scale_direction

        assert self.temperature > 0.0, 'The temperature must be a positive float!'

        self.model_config = UnconditionalDiffusionModelConfig()
        self.model = UnconditionalDiffusionModel(self.model_config)
        self.model.set_feature_scales_and_direction(self.scales,self.scale_direction)
        self.lora_layers = self.model.add_lora_compatibility(4)
        self.fc0 = nn.Linear(self.in_channels, self.hidden_channels1)
        self.bn0 = nn.BatchNorm1d(self.hidden_channels1)
        self.fc1 = nn.Linear(self.hidden_channels1, self.hidden_channels2)
        self.bn1 = nn.BatchNorm1d(self.hidden_channels2)
        self.fc = nn.Linear(self.hidden_channels2, self.out_channels)

    def forward(self, x):
        noisy_pred, x  = self.model.get_features(x,self.timestep)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.max_epochs,
                                                            eta_min=self.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)


        # Encode all images
        feats = self.forward(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        print("###############################################################################################")
        print(nll)

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')
    
class ContrastiveTransformations(object):

    def __init__(self, n_views=2):
        self.base_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=96),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

def train_cl(batch_size,unlabeled_data,train_data_contrast, max_epochs=500, **kwargs):
    trainer = pl.Trainer(
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs
                        )
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'ContrastiveLearning.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = ContrastiveLearning.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = data.DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader = data.DataLoader(train_data_contrast, batch_size=batch_size, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        pl.seed_everything(42) # To be reproducable
        model = ContrastiveLearning(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        #trainer.save_checkpoint(pretrained_filename)
        #model = model = ContrastiveLearning.load_from_checkpoint(pretrained_filename)#ContrastiveLearning.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
    # cl_model = ContrastiveLearning(max_epochs=max_epochs, **kwargs)
    # train_loader = data.DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True,drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    # trainer = pl.Trainer()
    # trainer.fit(model=cl_model, train_dataloaders=train_loader)


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



if __name__ == '__main__':
    celeb_data = CelebAHQDataset(root_dir="/Users/omscs/Desktop/celebhq",transform=ContrastiveTransformations(n_views=2))
    timestep = 100
    scales = [1,2,3]
    scale_direction = ["up","down"]
    clr_model = train_cl(batch_size=24,
                        unlabeled_data = celeb_data,
                        train_data_contrast = celeb_data,
                        in_channels=3360,
                        out_channels = 16,
                        lr=5e-4,
                        temperature=0.07,
                        weight_decay=1e-4,
                        max_epochs=500,
                        hidden_channels1=1024,
                        hidden_channels2=256,
                        timestep = torch.LongTensor([timestep]),
                        scales = scales,
                        scale_direction = scale_direction)