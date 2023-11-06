import pytorch_lightning as pl
import torch


class PLModelTrainer(pl.LightningModule):
    def __init__(
        self, backbone, head, criterion, optimizer, timestep=None, metrics=None
    ):
        super(PLModelTrainer, self).__init__()
        self.backbone = backbone
        self.head = head
        self.criterion = criterion
        self.optimizer = optimizer
        self.time_step = torch.LongTensor([10]) if timestep is None else timestep

        self.configure_optimizers()

    def forward(self, x, t=None):
        if t == None:
            t = self.time_step
        t = t.to(x.device)
        _, features = self.backbone.get_features(
            x, t
        )  # _ , features = self.backbone(x)
        return self.head(features)

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return {"loss": loss}
