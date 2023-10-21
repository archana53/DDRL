# Description: Contrastive pretraining

## Standard libraries
import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

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
from torchvision.datasets import STL10,CIFAR10
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial17"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)


class ContrastiveLearning(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, lr, temperature, weight_decay, max_epochs=500, hidden_channels1=256, hidden_channels2=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2
        assert self.temperature > 0.0, 'The temperature must be a positive float!'
        self.fc0 = nn.Linear(self.in_channels, self.hidden_channels1)
        self.bn0 = nn.BatchNorm1d(self.hidden_channels1)
        self.fc1 = nn.Linear(self.hidden_channels1, self.hidden_channels2)
        self.bn1 = nn.BatchNorm1d(self.hidden_channels2)
        self.fc = nn.Linear(self.hidden_channels2, self.out_channels)

    def forward(self, x):
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
        imgs = torch.flatten(imgs, start_dim=1, end_dim=3)
        imgs = imgs[:,:16]
        print(imgs.shape)


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
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'ContrastiveLearning'),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])
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
        model = ContrastiveLearning.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    return model   



if __name__ == '__main__':
    unlabeled_data = CIFAR10(root=DATASET_PATH, train=True, download=True,
                       transform=ContrastiveTransformations(n_views=2))
    train_data_contrast = CIFAR10(root=DATASET_PATH, train=False, download=True,
                                transform=ContrastiveTransformations(n_views=2))

    # pl.seed_everything(42)
    # NUM_IMAGES = 6
    # imgs = torch.stack([img for idx in range(NUM_IMAGES) for img in unlabeled_data[idx][0]], dim=0)
    # img_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9)
    # img_grid = img_grid.permute(1, 2, 0)

    # plt.figure(figsize=(10,5))
    # plt.title('Augmented image examples of the STL10 dataset')
    # plt.imshow(img_grid)
    # plt.axis('off')
    # plt.show()
    # plt.close()
    clr_model = train_cl(batch_size=2,
                        unlabeled_data = unlabeled_data,
                        train_data_contrast = train_data_contrast,
                        in_channels=16,
                        out_channels = 16,
                        lr=5e-4,
                        temperature=0.07,
                        weight_decay=1e-4,
                        max_epochs=500,
                        hidden_channels1=16,
                        hidden_channels2=16)