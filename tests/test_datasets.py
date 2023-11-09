import unittest
import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from modules.data.datasets import CelebAHQMaskDataset, DepthDataset, DatasetType

CELEBA_PATH = pathlib.Path("/coc/flash5/schermala3/Datasets/CelebAMask-HQ")
BIWI_PATH = pathlib.Path("/coc/flash5/schermala3/Datasets/BIWIKinectHeads/preprocessed")


class TestCelebAHQMaskDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = CelebAHQMaskDataset(root=CELEBA_PATH, size=(256, 256), mode="train")

    def test_len(self):
        self.assertEqual(len(self.dataset), 30000)

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn("image", sample)
        self.assertIn("label", sample)
        self.assertIn("name", sample)
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertIsInstance(sample["label"], torch.Tensor)
        self.assertIsInstance(sample["name"], str)
        self.assertEqual(sample["image"].shape, (3, 256, 256))
        self.assertEqual(sample["label"].shape, (256, 256))



class TestDepthDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = DepthDataset(root=BIWI_PATH, size=(256, 256), mode="train")

    def test_len(self):
        self.assertEqual(len(self.dataset), 15677)

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn("image", sample)
        self.assertIn("label", sample)
        self.assertIn("name", sample)
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertIsInstance(sample["label"], torch.Tensor)
        self.assertIsInstance(sample["name"], str)
        self.assertEqual(sample["image"].shape, (3, 256, 256))
        self.assertEqual(sample["label"].shape, (1, 256, 256))


class TestDatasetType(unittest.TestCase):
    def test_CelebAHQMask(self):
        dataset = DatasetType.CelebAHQMask.value(root=CELEBA_PATH, size=(256, 256), mode="train")
        self.assertIsInstance(dataset, CelebAHQMaskDataset)

    def test_Depth(self):
        dataset = DatasetType.Depth.value(root=BIWI_PATH, size=(256, 256), mode="train")
        self.assertIsInstance(dataset, DepthDataset)


if __name__ == "__main__":
    unittest.main()
