from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_majority_vote(
    predictions: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """
    Performs majority voting on a list of predictions.

    Args:
        predictions (torch.Tensor): A tensor of predictions of shape (batch_size, num_models, num_classes).
        threshold (float): The threshold for considering a prediction as positive.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_classes) representing the majority vote of the predictions.
    """

    # Convert predictions to binary values based on threshold
    binary_predictions = torch.where(
        predictions >= threshold,
        torch.ones_like(predictions),
        torch.zeros_like(predictions),
    )

    # Sum the binary predictions across models
    summed_predictions = binary_predictions.sum(dim=1)

    # get the majority vote for each class
    majority_vote = torch.topk(summed_predictions, k=1, dim=1).values.squeeze()

    return majority_vote


def visualize_keypoints(
    image: torch.Tensor, keypoints: List[torch.Tensor], color: str = "red"
) -> np.ndarray:
    """
    Visualizes keypoints on an image.

    Args:
        image (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the image.
        keypoints (List[torch.Tensor]): A list of tensors of shape (batch_size, 2) representing the keypoints.
        color (str): The color of the keypoints.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, channels, height, width) representing the image with keypoints.
    """
    # TODO: check channels order

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


def heatmap_to_keypoints(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Converts a heatmap to a list of keypoints.

    Args:
        heatmap (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the heatmap.

    Returns:
        torch.Tensor: A tensors of shape (batch_size, 2) representing the keypoints.
    """

    # Convert heatmap to numpy array
    heatmap = heatmap.detach().cpu().numpy()

    # Get the x and y coordinates of the keypoints
    argmax_indices = heatmap.reshape(heatmap.shape[0], heatmap.shape[1], -1).argmax(-1)
    keypoints = np.column_stack(np.unravel_index(argmax_indices, heatmap.shape[2:]))

    # Convert keypoints to tensor
    keypoints = torch.tensor(keypoints, dtype=torch.float32)

    return keypoints


def visualize_heatmap(
    x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
) -> np.ndarray:
    """
    Visualizes an RGB image, a heatmap, and a prediction heatmap.
    Args:
        x (torch.Tensor): A tensor of shape (batch_size, 3, height, width) representing the RGB image.
        y (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the heatmap.
        y_hat (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the prediction heatmap.
    Returns:
        np.ndarray: A tensor of shape (3, height, width) representing the image with heatmaps.
    """

    # Convert to numpy array
    x = x[0].cpu().numpy()
    y = y[0].cpu().numpy()
    y_hat = y_hat[0].cpu().numpy()

    # Convert heatmaps to single channel
    y = np.sum(y, axis=0, keepdims=True)
    y_hat = np.sum(y_hat, axis=0, keepdims=True)

    # apply colormap to heatmaps
    cmap = plt.get_cmap("magma")
    y = cmap(y)[:, :, :, :3]
    y_hat = cmap(y_hat)[:, :, :, :3]
    y = np.transpose(y.squeeze(0), (2, 0, 1))
    y_hat = np.transpose(y_hat.squeeze(0), (2, 0, 1))

    # Concatenate images
    image = np.concatenate([x, y, y_hat], axis=2)

    return image


def visualize_depth(
    x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
) -> np.ndarray:
    """
    Visualizes an RGB image, its depth, and a predicted depth.

    Args:
        x (torch.Tensor): A tensor of shape (batch_size, 3, height, width) representing the RGB image.
        y (torch.Tensor): A tensor of shape (batch_size, 1, height, width) representing the depth.
        y_hat (torch.Tensor): A tensor of shape (batch_size, 1, height, width) representing the predicted depth.
    Returns:
        np.ndarray: A tensor of shape (3, height, width) representing the image with depth.
    """

    # Convert to numpy array
    x = x[0].cpu().numpy()
    y = y[0].cpu().numpy()
    y_hat = y_hat[0].cpu().numpy()

    # normalize depths
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y_hat = (y_hat - np.min(y_hat)) / (np.max(y_hat) - np.min(y_hat))

    # apply colormap to depths
    cmap = plt.get_cmap("magma")
    y = cmap(y)[:, :, :, :3]
    y_hat = cmap(y_hat)[:, :, :, :3]
    y = np.transpose(y.squeeze(0), (2, 0, 1))
    y_hat = np.transpose(y_hat.squeeze(0), (2, 0, 1))

    # Concatenate images
    image = np.concatenate([x, y, y_hat], axis=2)

    return image


def visualize_segmentation(
    x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
) -> np.ndarray:
    """
    Visualizes an RGB image, a segmentation, and a predicted segmentation.

    Args:
        x (torch.Tensor): A tensor of shape (batch_size, 3, height, width) representing the RGB image.
        y (torch.Tensor): A tensor of shape (batch_size, height, width) representing the segmentation.
        y_hat (torch.Tensor): A tensor of shape (batch_size, channels, height, width) representing the predicted segmentation.
    Returns:
        np.ndarray: A tensor of shape (3, height, width) representing the image with segmentation.
    """

    # Convert to numpy array
    x = x[0].cpu().numpy()
    y = y[0].cpu().numpy()
    y_hat = y_hat[0].cpu().numpy()

    # Convert predicted segmentation to numpy array
    y_hat = y_hat.argmax(axis=0)

    # apply colormap to segmentation
    cmap = plt.get_cmap("tab20")
    y = cmap(y)[:, :, :3]
    y_hat = cmap(y_hat)[:, :, :3]
    y = np.transpose(y, (2, 0, 1))
    y_hat = np.transpose(y_hat, (2, 0, 1))

    # Concatenate images
    image = np.concatenate([x, y, y_hat], axis=2)

    return image
