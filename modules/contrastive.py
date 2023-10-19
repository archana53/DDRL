# Description: Contrastive pretraining

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """MLP head for contrastive learning"""

    def __init__(self, in_channels, out_channels, hidden_channels1=256, hidden_channels2=128):
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