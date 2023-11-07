from modules.trainer import PLModelTrainer

if __name__ == "__main__":

    class RandomVectorDataset(Dataset):
        def __init__(self, num_samples, vector_length):
            self.num_samples = num_samples
            self.vector_length = vector_length
            self.data = torch.randn(num_samples, vector_length)
            self.labels = torch.randint(2, size=(num_samples, 1)).float()

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            sample = {"image": self.data[idx], "label": self.labels[idx]}
            return sample

    # Define a LightningDataModule
    class RandomVectorDataModule(pl.LightningDataModule):
        def __init__(self, num_samples, vector_length, batch_size, num_workers=2):
            super().__init__()
            self.num_samples = num_samples
            self.vector_length = vector_length
            self.batch_size = batch_size
            self.num_workers = num_workers

        def setup(self, stage=None):
            self.dummy_dataset = RandomVectorDataset(
                self.num_samples, self.vector_length
            )

        def train_dataloader(self):
            return DataLoader(
                self.dummy_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        def val_dataloader(self):
            return DataLoader(
                self.dummy_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    num_samples = 100
    vector_length = 10
    batch_size = 16

    dm = RandomVectorDataModule(num_samples, vector_length, batch_size)

    def train() -> None:
        trainer = pl.Trainer(
            max_epochs=100,
            precision="bf16-mixed",
            accelerator="auto",
            log_every_n_steps=5,
        )

        backbone = nn.Sequential(
            nn.Linear(10, 40), nn.ReLU(), nn.Linear(40, 20), nn.ReLU()
        )
        head = nn.Linear(20, 1)
        criterion = nn.BCEWithLogitsLoss()

        model = PLModelTrainer(
            backbone, head, criterion, torch.optim.Adam(head.parameters())
        )
        trainer.fit(model, datamodule=dm)

    train()

    # %load_ext tensorboard
    # %tensorboard --logdir lightning_logs/
