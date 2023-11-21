import pytorch_lightning as pl
import torch

import torch.nn.functional as F


class PLModelTrainer(pl.LightningModule):
    def __init__(
        self,
        backbone,
        head,
        criterion,
        optimizer,
        timestep=10,
        metrics=None,
        visualizer=None,
        use_precomputed_features=False,
    ):
        super(PLModelTrainer, self).__init__()
        self.head = head
        self.criterion = criterion
        self.optimizer = optimizer
        self.time_step = torch.LongTensor([timestep])
        self.metrics = metrics
        self.visualizer = visualizer
        self.use_precomputed_features = use_precomputed_features
        self.backbone = backbone if not use_precomputed_features else None
        self.get_features = (
            self.compute_features
            if not use_precomputed_features
            else self.load_features
        )

        self.configure_optimizers()

    def compute_features(self, *, x=None, features=None):
        _, features = self.backbone.get_features(x, self.time_step.to(x.device))
        return features

    def load_features(self, *, x=None, features=None):
        features = [
            F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
            for f in features
        ]
        features = torch.cat(features, dim=1)
        return features

    def forward(self, x, features=None):
        features = self.get_features(x=x, features=features)
        return self.head(features)

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        features = batch.get("features", None)
        y_hat = self.forward(x, features=features)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        features = batch.get("features", None)
        y_hat = self.forward(x, features=features)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        for key, metric in self.metrics.items():
            self.log(f"val_{key}", metric(y_hat, y))
        if self.visualizer is not None:
            self.logger.experiment.add_image(
                "val_predictions", self.visualizer(x, y, y_hat), batch_idx
            )
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        features = batch.get("features", None)
        y_hat = self.forward(x, features=features)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        for key, metric in self.metrics.items():
            self.log(f"test_{key}", metric(y_hat, y))
        return {"loss": loss}
