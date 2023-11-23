import torch
import torch.nn.functional as F

from utils import heatmap_to_keypoints


def mIOU(y_pred, y_true):
    """
    Computes the mean Intersection over Union (mIOU) metric for semantic segmentation.
    Args:
        y_pred (torch.Tensor): Predicted segmentation mask of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth segmentation mask of shape (batch_size, num_classes, height, width).
    Returns:
        float: Mean Intersection over Union (mIOU) score.
    """
    iou = torch.zeros(y_pred.shape[0], dtype=y_pred.dtype, device=y_pred.device)
    num_classes = y_pred.shape[1]
    y_pred = torch.argmax(y_pred, dim=1)
    for i in range(num_classes):
        intersection = ((y_true == i) & (y_pred == i)).sum(axis=(1, 2))
        union = ((y_true == i) | (y_pred == i)).sum(axis=(1, 2))
        iou[union != 0] += intersection[union != 0] / union[union != 0]
    iou /= num_classes
    return torch.mean(iou)


def MSE(y_pred, y_true):
    """
    Computes the mean squared error (MSE) metric for depth estimation.
    Args:
        y_pred (torch.Tensor): Predicted depth map of shape (batch_size, 1, height, width).
        y_true (torch.Tensor): Ground truth depth map of shape (batch_size, 1, height, width).
    Returns:
        float: Mean squared error (MSE) score.
    """
    return torch.mean((y_true - y_pred) ** 2)


def keypoint_MSE(y_pred, y_true):
    """
    Computes the mean squared error (MSE) metric for keypoint estimation.
    Extracts keypoints from the heatmap and computes the MSE on coordinates.
    Args:
        y_pred (torch.Tensor): Predicted keypoint of shape (batch_size, 2).
        y_true (torch.Tensor): Ground truth keypoint of shape (batch_size, 2).
    Returns:
        float: Mean squared error (MSE) score.
    """
    y_pred = heatmap_to_keypoints(y_pred)
    y_true = heatmap_to_keypoints(y_true)
    return torch.mean((y_true - y_pred) ** 2)


def weighted_heatmap_MSE(y_pred, y_true, threshold=0.2, weight_value=10.0):
    """
    Computes the mean squared error (MSE) metric for keypoint estimation by weighting the class imbalanced keypoints
    loss higher.
    Extracts keypoints from the heatmap and computes the MSE on coordinates.
    Args:
        y_pred (torch.Tensor): Predicted keypoint heat map of shape (batch_size, 1, height, width).
        y_true (torch.Tensor): Ground truth heat map of shape (batch_size, 1, height, width).
    Returns:
        float: Mean squared error (MSE) score after weighting loss of values > threshold.
    """
    # weight map creation
    weight = y_true > threshold
    weight = weight.float() * weight_value

    # weighted loss
    return torch.mean(F.mse_loss(y_pred, y_true, reduction='none') * weight)
