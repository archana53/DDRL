import unittest
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import torch
from metrics import mIOU, MSE, keypoint_MSE


class TestMetrics(unittest.TestCase):
    def test_mIOU(self):
        test_pred = torch.rand(1, 19, 256, 256)
        test_gt = test_pred.argmax(dim=1)
        iou = mIOU(test_pred, test_gt)
        self.assertAlmostEqual(iou, 1, places=2)

    def test_MSE(self):
        test_pred = torch.rand(1, 1, 256, 256)
        mse = MSE(test_pred, test_pred)
        self.assertAlmostEqual(mse, 0, places=2)

    def test_keypoint_MSE(self):
        test_pred = torch.rand(1, 19, 256, 256)
        keypoint_mse = keypoint_MSE(test_pred, test_pred)
        self.assertAlmostEqual(keypoint_mse, 0, places=2)


if __name__ == "__main__":
    unittest.main()
