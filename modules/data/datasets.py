import pathlib
import json
from typing import Tuple
from enum import Enum

import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from enum import Enum
from modules.feature_loader import FeatureLoader


class BaseTaskDataset(Dataset):
    """Base dataset class for downstream tasks. Child classes must implement __getitem__ and __len__.
    Override augmentations in child classes.
    Returns a dict with keys "image", "label", and "name" for the image, label and file name respectively.
    Args:
        root (pathlib.Path): Path to the root directory of the dataset.
        size (Tuple[int, int], optional): Size of the image as (height, width). Defaults to (256, 256).
        mode (str, optional): Mode of the dataset. Must be one of (train, val, test). Defaults to "train".
    """

    def validate_root(self, root):
        return not root.exists() or not root.is_dir()
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
            raise ValueError(f"mode must be one of (train, val, test). Given mode: {mode}")
        if self.validate_root(root):
            raise FileNotFoundError(f"Root directory {root} is not valid.")

        self.mode = mode
        self.root = root
        self.size = size

        self.transforms = None  # override in child classes

        self.to_tensor = A.Compose(
            [
                A.Resize(*size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ],
            is_check_shapes=False,  # CelebAHQMaskDataset has a different shape for image and mask
            keypoint_params=A.KeypointParams(format="xy")
        )

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CelebAHQMaskDataset(BaseTaskDataset):
    """Dataset class for CelebA-HQ Mask dataset. Modified from https://github.com/zllrunning/face-parsing.PyTorch
    Returns a dict with keys "image", "label", and "name" for the image, segmentation map, and file name respectively.

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

        self.transforms = A.Compose(
            [
                # geometric
                A.Flip(),
                A.Rotate(limit=30),
                A.Resize(512, 512),  # resize both image and mask to 512x512 before cropping
                A.RandomResizedCrop(*self.size),
                # color
                A.ColorJitter(),
                A.GaussianBlur(),
            ],
            is_check_shapes=False,  # CelebAHQMaskDataset has a different shape for image and mask
        )

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)

        label_path = self.labels[idx]
        label = Image.open(image_path.parent.parent / self.label_subfolder / label_path).convert("P")

        if self.mode == "train":
            transformed = self.transforms(image=np.array(image), mask=np.array(label))
            image = transformed["image"]
            label = transformed["mask"]

        transformed = self.to_tensor(image=np.array(image), mask=np.array(label))
        image = transformed["image"]
        label = transformed["mask"].unsqueeze(0).long()
        return {"image": image, "label": label, "name": image_path.name}

    def __len__(self):
        return len(self.images)


class DepthDataset(BaseTaskDataset):
    """Dataset class for single image facial depth estimation.
    Returns a dict with keys "image", "label", and "name" for the image, depth map, and file name respectively.

    Args:
        root (pathlib.Path): Path to the root directory of the dataset.
        size (Tuple[int, int], optional): Size of the image. Defaults to (256, 256).
        mode (str, optional): Mode of the dataset. Must be one of (train, val, test). Defaults to "train".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_subfolder = "cropped_images"
        self.depth_subfolder = "cropped_depths"
        self.images = sorted((self.root / self.image_subfolder).glob("*.jpg"))
        self.depths = sorted((self.root / self.depth_subfolder).glob("*.png"))

        if len(self.images) != len(self.depths):
            raise ValueError(
                f"Number of images and depths do not match. Found {len(self.images)} images and {len(self.depths)} depths."
            )

        self.transforms = A.Compose(
            [
                # geometric
                A.Flip(),
                A.Rotate(limit=30),
                A.RandomResizedCrop(*self.size),
                # color
                A.ColorJitter(),
                A.GaussianBlur(),
            ],
        )

        self.depth_to_tensor = A.Compose(
            [
                A.Resize(*self.size),
                ToTensorV2(transpose_mask=True),
            ],
        )

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)

        depth_path = self.depths[idx]
        depth = Image.open(image_path.parent.parent / self.depth_subfolder / depth_path).convert("L")

        if self.mode == "train":
            transformed = self.transforms(image=np.array(image), mask=np.array(depth))
            image = transformed["image"]
            depth = transformed["mask"]

        transformed = self.to_tensor(image=np.array(image))
        image = transformed["image"]
        depth = self.depth_to_tensor(image=np.array(depth))["image"]
        depth = depth / 127.5 - 1
        return {"image": image, "label": depth, "name": image_path.name}

    def __len__(self):
        return len(self.images)


class KeyPointDataset(BaseTaskDataset):
    """Dataset class for single image facial keypoint estimation.
    Returns a dict with keys "image" and "label" for the image and keypoint map respectively.
    """
    def validate_root(self, root):
        return not root.exists() or not root.is_file()

    def get_bounding_box(self, ax, ay, bx, by):
        return ((float(ax), float(ay)), (float(bx), float(by)))
    def get_file_info(self, ground_truth_file):
        with open(ground_truth_file) as file:
            ground_truths = [line.rstrip() for line in file]

        image_paths = []
        pts_paths = []
        bounding_boxes = []
        for ground_truth in ground_truths:
            split_line = ground_truth.split(' ')
            image_paths.append(split_line[0])
            pts_paths.append(split_line[1])
            bounding_box = self.get_bounding_box(split_line[2], split_line[3], split_line[4], split_line[5])
            bounding_boxes.append(bounding_box)

        return image_paths, pts_paths, bounding_boxes
    def get_image_path(self, idx):
        return self.file_info[idx][0]

    def get_keypoints(self, pts_path):
        with open(pts_path) as file:
            lines = [line.rstrip() for line in file]

        coords = []
        for line in lines:
            split_line = line.split(' ')
            coords.append((float(split_line[1]), float(split_line[2])))

        return coords
    def __init__(self, *args, **kwargs):
        super(KeyPointDataset, self).__init__(*args, **kwargs)
        self.image_paths, self.pts_paths, self.bounding_boxes = self.get_file_info(self.root)
        self.transforms = A.Compose(
            [
                # geometric
                A.Flip(),
                A.Rotate(limit=30),
                A.Resize(512, 512),  # resize both image and mask to 512x512 before cropping
                # No random crops to avoid missing out on keypoints
                # color
                A.ColorJitter(),
                A.GaussianBlur(),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )

    def __getitem__(self, idx):
        # TODO: test this
        image_path = self.image_paths[idx]
        keypoint_path = self.pts_paths[idx]

        image = Image.open(image_path)
        keypoints = self.get_keypoints(keypoint_path)

        if self.mode == "train":
            transformed = self.transforms(image=np.array(image), keypoints=keypoints)
            image = transformed["image"]
            keypoints = transformed["keypoints"]

        transformed = self.to_tensor(image=np.array(image), keypoints=keypoints)
        image = transformed["image"]
        keypoints = transformed["keypoints"]
        unfolded_keypoints = []
        for keypoint in keypoints:
            unfolded_keypoints.append(keypoint[0])
            unfolded_keypoints.append(keypoint[1])

        return {"image": image, "label": unfolded_keypoints}

    def __len__(self):
        return len(self.image_paths)


class DatasetType(Enum):
    """Enum for dataset types. Used to instantiate the correct dataset class."""

    CelebAHQMask = CelebAHQMaskDataset
    Depth = DepthDataset
    KeyPoint = KeyPointDataset


class DatasetWithFeatures(Dataset):
    """Dataset class for downstream tasks with precomputed features.
    Takes in a dataset object and a feature loader object.
    Returns keys in the original dataset object and "features" for the features.
    """

    def __init__(self, dataset: Dataset, feature_loader: FeatureLoader):
        super().__init__()
        self.dataset = dataset
        self.feature_loader = feature_loader

    def __getitem__(self, idx):
        data = self.dataset[idx]
        features = self.feature_loader(data["name"])
        data["features"] = features
        return data

    def __len__(self):
        return len(self.dataset)
