import unittest
import torch
from modules.decoders import ConvHead, MLPHead, MLPEnsembleHead


# Test ConvHead
class TestHeads(unittest.TestCase):
    def test_conv_head(self):
        batch_size = 2
        in_channels = 3
        hidden_channels = 16
        out_channels = 10
        height = 32
        width = 32

        x = torch.randn(batch_size, in_channels, height, width)
        conv_head = ConvHead(in_channels, hidden_channels, out_channels)
        output = conv_head(x)

        assert output.shape == (batch_size, out_channels, height, width)

    # Test MLPHead
    def test_mlp_head(self):
        batch_size = 2
        in_channels = 10
        out_channels = 5

        x = torch.randn(batch_size, in_channels)
        mlp_head = MLPHead(in_channels, out_channels)
        output = mlp_head(x)

        assert output.shape == (batch_size, out_channels)

    # Test MLPEnsembleHead
    def test_mlp_ensemble_head(self):
        batch_size = 2
        in_channels = 10
        out_channels = 5
        n_models = 3

        x = torch.randn(batch_size, in_channels)
        mlp_ensemble_head = MLPEnsembleHead(in_channels, out_channels, n_models=n_models)
        output = mlp_ensemble_head(x)

        assert output.shape == (batch_size, n_models, out_channels)

    # Test KeyPointHead
