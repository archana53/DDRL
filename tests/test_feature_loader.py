import pathlib
import unittest
import sys

import h5py
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from modules.feature_loader import FeatureLoader


class TestFeatureLoader(unittest.TestCase):
    def setUp(self):
        # Set up a test h5 file with some dummy data
        self.h5_file = pathlib.Path("test.h5")
        with h5py.File(self.h5_file, "w") as f:
            f.create_dataset("image1/0/block_xy{1}", data=np.random.rand(4, 25, 25))
            f.create_dataset("image1/0/block_xy{2}", data=np.random.rand(4, 50, 50))
            f.create_dataset("image1/0/block_xy{3}", data=np.random.rand(4, 100, 100))
            f.create_dataset("image1/0/block_xt{1}", data=np.random.rand(4, 25, 25))
            f.create_dataset("image1/0/block_xt{2}", data=np.random.rand(4, 50, 50))
            f.create_dataset("image1/0/block_xt{3}", data=np.random.rand(4, 100, 100))
            
            f.create_dataset("image2/0/block_xy{1}", data=np.random.rand(4, 25, 25))
            f.create_dataset("image2/0/block_xy{2}", data=np.random.rand(4, 50, 50))
            f.create_dataset("image2/0/block_xy{3}", data=np.random.rand(4, 100, 100))
            f.create_dataset("image2/0/block_xt{1}", data=np.random.rand(4, 25, 25))
            f.create_dataset("image2/0/block_xt{2}", data=np.random.rand(4, 50, 50))
            f.create_dataset("image2/0/block_xt{3}", data=np.random.rand(4, 100, 100))

    def tearDown(self):
        # Clean up the test h5 file
        self.h5_file.unlink()

    def test_load_features(self):
        # Test loading features for a single image
        feature_loader = FeatureLoader(
            self.h5_file, scales=[1, 2], scale_directions=["xy", "xt"], timestep=0, resolution=(100, 100)
        )
        features = feature_loader("image1")
        self.assertEqual(features.shape, (1, 16, 100, 100))

    def test_load_features_multiple_images(self):
        # Test loading features for multiple images
        feature_loader = FeatureLoader(
            self.h5_file, scales=[1], scale_directions=["xy", "xt"], timestep=0, resolution=(100, 100)
        )
        features1 = feature_loader("image1")
        features2 = feature_loader("image2")
        self.assertEqual(features1.shape, (1, 8, 100, 100))
        self.assertEqual(features2.shape, (1, 8, 100, 100))

    def test_load_features_missing_scale(self):
        # Test loading features with a missing scale
        with self.assertRaises(KeyError):
            feature_loader = FeatureLoader(
                self.h5_file, scales=[1, 2, 4], scale_directions=["xy", "xt"], timestep=0, resolution=(100, 100)
            )

    def test_load_features_missing_direction(self):
        # Test loading features with a missing scale direction
        with self.assertRaises(KeyError):
            feature_loader = FeatureLoader(
                self.h5_file, scales=[1, 2], scale_directions=["xy", "xz"], timestep=0, resolution=(100, 100)
            )

    def test_load_features_missing_timestep(self):
        # Test loading features with a missing timestep
        with self.assertRaises(KeyError):
            feature_loader = FeatureLoader(
                self.h5_file, scales=[1, 2], scale_directions=["xy", "xt"], timestep=1, resolution=(100, 100)
            )


if __name__ == "__main__":
    unittest.main()