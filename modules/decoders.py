# Description: Decoders for downstream tasks
# Available decoders: ConvHead, MLPHead, MLPEnsembleHead, KeyPointHead
# Tasks supported: dense classification, dense regression, key point detection

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvHead(nn.Module):
    """Convolutional head for downstream tasks"""

    def __init__(
        self, in_channels, out_channels, hidden_channels=256, kernel_size=3, padding=1
    ):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size, padding=padding
        )
        self.bn0 = nn.BatchNorm2d(hidden_channels)
        self.conv = nn.Conv2d(
            hidden_channels, out_channels, kernel_size, padding=padding
        )

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.conv(x)
        return x


class PixelwiseMLPHead(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels1=256, hidden_channels2=128
    ):
        super(PixelwiseMLPHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels1)
        self.bn0 = nn.BatchNorm1d(hidden_channels1)
        self.fc2 = nn.Linear(hidden_channels1, hidden_channels2)
        self.bn1 = nn.BatchNorm1d(hidden_channels2)
        self.fc3 = nn.Linear(hidden_channels2, out_channels)

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size * height * width, num_channels)
        x = F.relu(self.bn0(self.fc1(x)))
        x = F.relu(self.bn1(self.fc2(x)))
        x = self.fc3(x)
        x = x.reshape(batch_size, height, width, -1)
        x = x.permute(0, 3, 1, 2)
        return x


class MLPHead(nn.Module):
    """MLP head for downstream tasks"""

    def __init__(
        self, in_channels, out_channels, hidden_channels1=256, hidden_channels2=128
    ):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, hidden_channels1)
        self.bn0 = nn.BatchNorm1d(hidden_channels1)
        self.fc1 = nn.Linear(hidden_channels1, hidden_channels2)
        self.bn1 = nn.BatchNorm1d(hidden_channels2)
        self.fc = nn.Linear(hidden_channels2, out_channels)

    def forward(self, x):
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc(x)
        return x


class MLPEnsembleHead(nn.Module):
    """Ensemble of MLPHead modules for downstream tasks.
    Should use majority voting for classification and averaging for regression."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels1=256,
        hidden_channels2=128,
        n_models=3,
    ):
        super().__init__()
        self.n_models = n_models
        self.ensemble = nn.ModuleList(
            [
                MLPHead(in_channels, out_channels, hidden_channels1, hidden_channels2)
                for _ in range(n_models)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [model(x) for model in self.ensemble]
        x = torch.stack(x, dim=1)
        return x
