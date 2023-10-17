from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_majority_vote(predictions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Performs majority voting on a list of predictions.

    Args:
        predictions (torch.Tensor): A tensor of predictions of shape (batch_size, num_models, num_classes).
        threshold (float): The threshold for considering a prediction as positive.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_classes) representing the majority vote of the predictions.
    """

    # Convert predictions to binary values based on threshold
    binary_predictions = torch.where(predictions >= threshold, torch.ones_like(predictions), torch.zeros_like(predictions))

    # Sum the binary predictions across models
    summed_predictions = binary_predictions.sum(dim=1)

    # get the majority vote for each class
    majority_vote = torch.topk(summed_predictions, k=1, dim=1).values.squeeze()

    return majority_vote


def visualize_keypoints(image: torch.Tensor, keypoints: List[torch.Tensor], color: str = 'red') -> np.ndarray:
    """
    Visualizes keypoints on an image.

    Args:
        image (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the image.
        keypoints (List[torch.Tensor]): A list of tensors of shape (batch_size, 2) representing the keypoints.
        color (str): The color of the keypoints.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, channels, height, width) representing the image with keypoints.
    """
    #TODO: check channels order

    # Convert image to numpy array
    image = image.cpu().numpy()

    # Convert keypoints to numpy array
    keypoints = [keypoint.cpu().numpy() for keypoint in keypoints]

    # Iterate over images
    for i in range(image.shape[0]):

        # Iterate over keypoints
        for keypoint in keypoints:
            # Get x and y coordinates
            x = keypoint[i, 0]
            y = keypoint[i, 1]

            # Draw a circle around the keypoint
            cv2.circle(image[i], (x, y), 5, color, -1)

    return image


def visualize_depth(image: torch.Tensor, depth: torch.Tensor) -> np.ndarray:
    """
    Visualizes depth on an image.

    Args:
        image (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the image.
        depth (torch.Tensor): A tensor of shape (batch_size, 1, height, width) representing the depth.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, channels, height, width) representing the image with depth.
    """

    # Convert image to numpy array
    image = image.cpu().numpy()

    # Convert depth to numpy array
    depth = depth.cpu().numpy()
    depth = (depth - np.min(depth, axis=(2,3), keepdims=True)) / (
        np.max(depth, axis=(2,3), keepdims=True) - np.min(depth, axis=(2,3), keepdims=True)
        )

    return depth


def visualize_segmentation(image: torch.Tensor, segmentation: torch.Tensor) -> np.ndarray:
    """
    Visualizes segmentation on an image.

    Args:
        image (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the image.
        segmentation (torch.Tensor): A tensor of shape (batch_size, 1, height, width) representing the segmentation.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, channels, height, width) representing the image with segmentation.
    """

    # Convert image to numpy array
    image = image.cpu().numpy()

    # Convert segmentation to numpy array
    segmentation = segmentation.cpu().numpy()
    segmentation = np.argmax(segmentation, axis=1)

    # map segmentation to colors
    cmap = plt.get_cmap('tab20')
    segmentation = cmap(segmentation)[:, :, :3]

    return segmentation