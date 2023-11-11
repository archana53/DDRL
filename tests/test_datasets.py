import unittest
import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from modules.data.datasets import CelebAHQMaskDataset, DepthDataset, DatasetType
from modules.data.datasets import KeyPointDataset
from pathlib import Path

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

class TestALFWDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = KeyPointDataset(root=Path("resources/test_keypoint.GTB"), mode="train")
    def test_len(self):
        self.assertEqual(len(self.dataset), 4)

    def test_getitem(self):
        for data in self.dataset:
            self.assertEqual(data['image'].size(), torch.Size((3, 256, 256)))
            # Not all images have all the keypoints, sometimes they are hidden
            # self.assertEqual(len(data['label']), 19)
            self.assertEqual(len(data['label'][0]), 2)


class TestDatasetType(unittest.TestCase):
    def test_CelebAHQMask(self):
        dataset = DatasetType.CelebAHQMask.value(root=CELEBA_PATH, size=(256, 256), mode="train")
        self.assertIsInstance(dataset, CelebAHQMaskDataset)

    def test_Depth(self):
        dataset = DatasetType.Depth.value(root=BIWI_PATH, size=(256, 256), mode="train")
        self.assertIsInstance(dataset, DepthDataset)

    def test_ALFW(self):
        dataset = DatasetType.KeyPoint.value(root=Path("resources/test_keypoint.GTB"), mode="train")
        self.assertIsInstance(dataset, KeyPointDataset)


if __name__ == "__main__":
    unittest.main()
