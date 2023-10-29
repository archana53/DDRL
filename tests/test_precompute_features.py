import os
import tempfile
import unittest
from pathlib import Path
import sys

# add modules and data to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "modules"))

import h5py
import torch
import numpy as np

from torch.utils.data import DataLoader
from diffusers import LDMPipeline

from modules.data.datasets_with_enum import DatasetType
from modules.precompute_features import save_features
from modules.unet_extractor import ModelExtractor


CELEBA_PATH = Path("/coc/flash5/schermala3/Datasets/CelebAMask-HQ")
BIWI_PATH = Path("/coc/flash5/schermala3/Datasets/BIWIKinectHeads/preprocessed")


class TestPrecomputeFeatures(unittest.TestCase):
    def setUp(self):
        self.pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        self.unet = self.pipe.unet
        self.dataset_cls = DatasetType["CelebAHQMask"].value
        self.dataset_root = CELEBA_PATH
        self.dataset = self.dataset_cls(root=self.dataset_root, mode="test", size=(256, 256))
        self.batch_size = 1
        self.num_workers = 2
        self.extractor = ModelExtractor(self.pipe, self.unet, ["up"], [0, 1, 2], upsample=False).cuda()
        self.timesteps = torch.arange(0, 1000, 200).cuda()
        self.output_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.output_dir, "celebahq_features.h5")

    def test_save_features(self):
        with h5py.File(self.output_file, "w") as h5_file:
            features = {"up": torch.randn(2, 512, 64, 64)}
            timestep = 0
            image_names = ["image1.jpg", "image2.jpg"]
            save_features(features, timestep, image_names, h5_file)

            self.assertTrue("timestep_0" in h5_file)
            self.assertTrue("feature_up" in h5_file["timestep_0"])
            self.assertTrue("image1.jpg" in h5_file["timestep_0"]["feature_up"])

            self.assertTrue(
                np.allclose(h5_file["timestep_0"]["feature_up"]["image1.jpg"], features["up"][0].cpu().numpy())
            )
            self.assertTrue(
                np.allclose(h5_file["timestep_0"]["feature_up"]["image2.jpg"], features["up"][1].cpu().numpy())
            )

    def test_precompute_features(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # run for 10 batches
        num_iter = 10

        # add save_features
        with h5py.File(self.output_file, "w") as h5_file:
            # run model on dataset
            for i, batch in enumerate(dataloader):
                image = batch["image"].cuda()
                image_name = batch["name"]
                for timestep in self.timesteps:
                    _, features = self.extractor(image, timestep)
                    save_features(features, timestep, image_name, h5_file)

                if i == num_iter - 1:  # break after `num_iter` batches
                    break

        with h5py.File(self.output_file, "r") as h5_file:
            self.assertTrue("timestep_0" in h5_file)
            self.assertTrue("timestep_200" in h5_file)
            self.assertTrue("timestep_800" in h5_file)
            feature_name = list(features.keys())[0]
            self.assertTrue(all(f"feature_{x}" in h5_file["timestep_0"] for x in features.keys()))
            self.assertTrue(all(x in h5_file["timestep_0"][f"feature_{feature_name}"] for x in image_name))
            self.assertEqual(len(h5_file), len(self.timesteps))
            self.assertEqual(len(h5_file["timestep_0"][f"feature_{feature_name}"]), num_iter * self.batch_size)


if __name__ == "__main__":
    unittest.main()
