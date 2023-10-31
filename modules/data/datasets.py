import pathlib
import json
from typing import Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class BaseTaskDataset(Dataset):
    """Base dataset class for downstream tasks. Child classes must implement __getitem__ and __len__.
    Override augmentations in child classes if necessary.
    Returns a dict with keys "image" and "label" for the image and label respectively.
    Args:
        root (pathlib.Path): Path to the root directory of the dataset.
        size (Tuple[int, int], optional): Size of the image as (height, width). Defaults to (256, 256).
        mode (str, optional): Mode of the dataset. Must be one of (train, val, test). Defaults to "train".
    """

    def __init__(
        self,
        root: pathlib.Path,
        size: Tuple[int, int] = (256, 256),
        mode: str = "train",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if mode not in ("train", "val", "test"):
            raise ValueError(
                f"mode must be one of (train, val, test). Given mode: {mode}"
            )
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Root directory {root} is not valid.")

        self.mode = mode
        self.root = root
        self.size = size

        self.transforms = A.Compose(
            [
                # geometric
                A.Flip(),
                A.Rotate(limit=30),
                A.RandomResizedCrop(*size),
                # color
                A.ColorJitter(),
                A.GaussianBlur(),
            ],
            # keypoint_params=A.KeypointParams(format="xy"),
        )

        self.to_tensor = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ]
        )

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CelebAHQMaskDataset(BaseTaskDataset):
    """Dataset class for CelebA-HQ Mask dataset. Modified from https://github.com/zllrunning/face-parsing.PyTorch
    Returns a dict with keys "image" and "label" for the image and segmentation map respectively.

    Args:
        root (pathlib.Path): Path to the root directory of the dataset.
        size (Tuple[int, int], optional): Size of the image. Defaults to (256, 256).
        ignore_label (int, optional): Label to ignore. Defaults to 255.
        mode (str, optional): Mode of the dataset. Must be one of (train, val, test). Defaults to "train".
    """

    def __init__(self, ignore_label: int = 255, *args, **kwargs):
        super(CelebAHQMaskDataset, self).__init__(*args, **kwargs)

        self.ignore_label = ignore_label  # used for loss calculation
        self.image_subfolder = "CelebA-HQ-img"
        self.label_subfolder = "mask"
        self.images = sorted((self.root / self.image_subfolder).glob("*.jpg"))
        self.labels = sorted((self.root / self.label_subfolder).glob("*.png"))

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Number of images and labels do not match. Found {len(self.images)} images and {len(self.labels)} labels."
            )

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)

        label_path = self.labels[idx]
        label = Image.open(
            image_path.parent.parent / self.label_subfolder / label_path
        ).convert("P")

        if self.mode == "train":
            transformed = self.transforms(image=np.array(image), mask=np.array(label))
            image = transformed["image"]
            label = transformed["mask"]

        transformed = self.to_tensor(image=np.array(image), mask=np.array(label))
        image = transformed["image"]
        label = transformed["mask"].long()
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.images)


class DepthDataset(BaseTaskDataset):
    """Dataset class for single image facial depth estimation.
    Returns a dict with keys "image" and "label" for the image and depth map respectively.

    Args:
        root (pathlib.Path): Path to the root directory of the dataset.
        size (Tuple[int, int], optional): Size of the image. Defaults to (256, 256).
        mode (str, optional): Mode of the dataset. Must be one of (train, val, test). Defaults to "train".
    """

    def __init__(self, *args, **kwargs):
        super(DepthDataset, self).__init__(*args, **kwargs)

        self.image_subfolder = "cropped_images"
        self.depth_subfolder = "cropped_depths"
        self.images = sorted((self.root / self.image_subfolder).glob("*.jpg"))
        self.depths = sorted((self.root / self.depth_subfolder).glob("*.png"))

        if len(self.images) != len(self.depths):
            raise ValueError(
                f"Number of images and depths do not match. Found {len(self.images)} images and {len(self.depths)} depths."
            )

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)

        depth_path = self.depths[idx]
        depth = Image.open(
            image_path.parent.parent / self.depth_subfolder / depth_path
        ).convert("L")

        if self.mode == "train":
            transformed = self.transforms(image=np.array(image), mask=np.array(depth))
            image = transformed["image"]
            depth = transformed["mask"]

        transformed = self.to_tensor(image=np.array(image))
        image = transformed["image"]
        depth = np.expand_dims((depth / 127.5 - 1), 2)
        depth = ToTensorV2()(image=np.array(depth))["image"]
        return {"image": image, "label": depth}

    def __len__(self):
        return len(self.images)


class KeyPointDataset(BaseTaskDataset):
    """Dataset class for single image facial keypoint estimation.
    Returns a dict with keys "image" and "label" for the image and keypoint map respectively.
    """

    def __init__(self, *args, **kwargs):
        super(KeyPointDataset, self).__init__(*args, **kwargs)

        self.image_subfolder = "image"
        self.keypoint_subfolder = "keypoint"
        self.images = sorted((self.root / self.image_subfolder).glob("*.jpg"))
        self.keypoints = sorted((self.root / self.keypoint_subfolder).glob("*.json"))

        if len(self.images) != len(self.keypoints):
            raise ValueError(
                f"Number of images and keypoints do not match. Found {len(self.images)} images and {len(self.keypoints)} keypoints."
            )

    def __getitem__(self, idx):
        # TODO: test this
        image_path = self.images[idx]
        image = Image.open(image_path)

        keypoint_path = self.keypoints[idx]
        with open(keypoint_path) as json_file:
            keypoint = json.load(json_file)

        if self.mode == "train":
            transformed = self.transforms(
                image=np.array(image), keypoint=keypoint["keypoints"]
            )
            image = transformed["image"]
            keypoint = transformed["keypoint"]

        transformed = self.to_tensor(image=np.array(image), keypoint=keypoint)
        image = transformed["image"]
        keypoint = transformed["keypoint"]
        return {"image": image, "label": keypoint}

    def __len__(self):
        return len(self.images)
