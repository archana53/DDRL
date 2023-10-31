import unittest
from modules.data.datasets import KeyPointDataset
from pathlib import Path
import torch
class TestDatasets(unittest.TestCase):

    def test_keypoint(self):
        keypoint_dataset = KeyPointDataset(root=Path("resources/test_keypoint.GTB"))
        self.assertEqual(len(keypoint_dataset), 4)

        for data in keypoint_dataset:
            self.assertEqual(data['image'].size(), torch.Size((3, 256, 256)))
            # Not all images have all the keypoints, sometimes they are hidden
            # self.assertEqual(len(data['label']), 19)
            self.assertEqual(len(data['label'][0]), 2)

if __name__ == '__main__':
    unittest.main()