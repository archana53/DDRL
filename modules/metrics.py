import torch

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
