import math
import numpy as np
import torch


def PTSconvert2str(points):
    assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
    assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
    string = ''
    num_pts = points.shape[1]
    for i in range(num_pts):
        ok = False
        if points.shape[0] == 3 and bool(points[2, i]) == True:
            ok = True
        elif points.shape[0] == 2:
            ok = True

        if ok:
            string = string + '{:02d} {:.2f} {:.2f} True\n'.format(i + 1, points[0, i], points[1, i])
    string = string[:-1]
    return string


def PTSconvert2box(points, expand_ratio=None):
    assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
    assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
    if points.shape[0] == 3:
        points = points[:2, points[-1, :].astype('bool')]
    elif points.shape[0] == 2:
        points = points[:2, :]
    else:
        raise Exception('The shape of points is not right : {}'.format(points.shape))
    assert points.shape[1] >= 2, 'To get the box of points, there should be at least 2 vs {}'.format(points.shape)
    box = np.array([points[0, :].min(), points[1, :].min(), points[0, :].max(), points[1, :].max()])
    W = box[2] - box[0]
    H = box[3] - box[1]
    assert W > 0 and H > 0, 'The size of box should be greater than 0 vs {}'.format(box)
    if expand_ratio is not None:
        box[0] = int(math.floor(box[0] - W * expand_ratio))
        box[1] = int(math.floor(box[1] - H * expand_ratio))
        box[2] = int(math.ceil(box[2] + W * expand_ratio))
        box[3] = int(math.ceil(box[3] + H * expand_ratio))
    return box


# A gaussian kernel cache, so we don't have to regenerate them every time.
# This is only a small optimization, generating the kernels is pretty fast.
_gaussians = {}


def generate_gaussian_heatmap(t, x, y, sigma=10):
    """
    Generates a 2D Gaussian point at location x,y in tensor t.

    x should be in range (-1, 1) to match the output of fastai's PointScaler.

    sigma is the standard deviation of the generated 2D Gaussian.
    """
    h, w = t.shape

    # Heatmap pixel per output pixel
    mu_x = int(0.5 * (x + 1.) * w)
    mu_y = int(0.5 * (y + 1.) * h)

    tmp_size = sigma * 3

    # Top-left
    x1, y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

    # Bottom right
    x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
    if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
        return t

    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = _gaussians[sigma] if sigma in _gaussians \
        else torch.Tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))
    _gaussians[sigma] = g

    # Determine the bounds of the source gaussian
    g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
    g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

    # Image range
    img_x_min, img_x_max = max(0, x1), min(x2, w)
    img_y_min, img_y_max = max(0, y1), min(y2, h)

    t[img_y_min:img_y_max, img_x_min:img_x_max] = \
        g[g_y_min:g_y_max, g_x_min:g_x_max]

    return t
