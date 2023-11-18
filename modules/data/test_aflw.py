import unittest
import tqdm
from modules.data.datasets import KeyPointDataset
from torch.utils.data import DataLoader

from pathlib import Path


class TestALFWDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = KeyPointDataset(
            root=Path("aflw_dataset/AFLW_lists/front.GTB"), mode="train"
        )
        self.dataloader = DataLoader(self.dataset, batch_size=16, shuffle=False, num_workers=2)

    def test_walkthrough(self):
        try:
            for i, batch in enumerate(self.dataloader):
                num_keypoints = batch['label'].size()[1]
                assert num_keypoints == 19
        except Exception as e:
            print(e)
