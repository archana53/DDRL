import pathlib
from typing import List, Tuple

import h5py
import torch
import torch.nn.functional as F


class FeatureLoader:
    """Callable class to load precomputed features from an h5 file.
    Args:
        h5_file (pathlib.Path): Path to the h5 file.
        scales (List[int]): List of scales to load.
        scale_directions (List[str]): List of scale directions to load.
        timestep (int): Timestep to load.
        resolution (Tuple[int, int]): Resolution to resize the features to.

    Call Args:
        image_name (str): Name of the image to load features for.

    Raises:
        ValueError: If the h5 file is invalid.
        KeyError: If not all required features are available in the h5 file.
    """

    def __init__(
        self,
        h5_file: pathlib.Path,
        scales: List[int],
        scale_directions: List[str],
        timestep: int,
        resolution: Tuple[int, int],
    ):
        if not h5_file.exists() or not h5_file.is_file():
            raise ValueError(f"Invalid h5 file {h5_file}")

        self.h5_file = h5_file
        self.scales = sorted(scales)
        self.scale_directions = sorted(scale_directions)
        self.timestep = timestep
        self.resolution = resolution
        self.feature_names = self._get_required_feature_names()

        # Check if all required features are available
        self.check_features_availablility()

    def check_features_availablility(self):
        """Checks if all required features are available in the h5 file."""
        available_timesteps, available_feature_names = self._get_available_feature_names()
        if self.timestep not in available_timesteps or not self.feature_names.issubset(available_feature_names):
            raise KeyError("Not all required features are available in the h5 file. Please run precompute_features.py.")

    def _get_required_feature_names(self):
        """Returns a set of required feature names."""
        names = set()
        for scale_dir in self.scale_directions:
            if scale_dir == "mid":
                names.add("mid{0}")
                continue
            for scale in self.scales:
                names.add(scale_dir + "{" + str(scale) + "}")
        return names

    def _get_available_feature_names(self):
        """Returns a list of available feature names."""
        with h5py.File(self.h5_file, "r") as f:
            images = list(f.keys())
            timesteps = list(f[images[0]].keys())
            features = list(f[images[0]][timesteps[0]].keys())
            timesteps = [int(timestep) for timestep in timesteps]
            features = [feature.split("_")[1] for feature in features]
        return timesteps, features

    def __call__(self, image_name):
        """Loads features for a given image.
        Args:
            image_name (str): Name of the image to load features for.
        Returns:
            torch.Tensor: A tensor of shape (1, channels, height, width) representing the features.
        """
        features = self._load_features(image_name)
        features = self._filter_features(features)
        features = self._resize_and_concatenate(features)
        return features

    def _load_features(self, image_name):
        """Loads features for a given image from the h5 file.
        Args:
            image_name (str): Name of the image to load features for.
        Returns:
            Dict[str, np.array]: A dictionary containing the features.
        """
        with h5py.File(self.h5_file, "r") as f:
            features = {}
            for feature_name in f[image_name][str(self.timestep)]:
                dest_feature_name = feature_name.split("_")[1]
                features[dest_feature_name] = f[image_name][str(self.timestep)][feature_name][...]
        return features

    def _filter_features(self, features):
        """Filters features based on scale and scale direction.
        Args:
            features (Dict[str, np.array]): A dictionary containing the features.
        Returns:
            List[np.array]: A list of features.
        """
        filtered_features = []
        for feature_name in sorted(features):
            if feature_name in self.feature_names:
                filtered_features.append(features[feature_name])
        return filtered_features

    def _resize_and_concatenate(self, features):
        """Resizes and concatenates features.
        Args:
            features (List[np.array]): A list of features.
        Returns:
            torch.Tensor: A tensor of shape (1, channels, height, width) representing the features.
        """
        resized_features = []
        for feature in features:
            resized_features.append(
                F.interpolate(
                    torch.from_numpy(feature).unsqueeze(0), size=self.resolution, mode="bilinear", align_corners=False
                )
            )
        return torch.cat(resized_features, dim=1)
