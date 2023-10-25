# Script to preprocess BIWI Kinect dataset
# Step 1: Decode binary depth files to images
# Step 2: Crop face images and decoded depth images using depth masks

from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm


def decode_biwi_depth(filepath: str) -> np.ndarray:
    """Decode BIWI Kinect depth image file from binary format to image
    Modified from https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database/discussion/182970#1205032
    
    Args:
        filepath (str): Path to the BIWI Kinect depth image file.

    Returns:
        np.ndarray: Decoded depth image.
    """
    with open(filepath, "rb") as f:
        width = int.from_bytes(f.read(4), "little")
        height = int.from_bytes(f.read(4), "little")
        depth = np.zeros([width * height], dtype=np.uint16)
        i = 0
        while i < width * height:
            skip = int.from_bytes(f.read(4), "little")
            read = int.from_bytes(f.read(4), "little")
            for j in range(read):
                depth[i + skip + j] = int.from_bytes(f.read(2), "little")
            i += skip + read
    depth = depth.reshape(height, width)

    d_ = np.zeros((depth.shape[0] + 20, depth.shape[1] + 20), dtype=depth.dtype)
    d_[20:, :640] = depth
    depth = cv2.resize(d_, (640, 480))
    return depth


def crop_using_mask(img, depth, mask):
    """Given a binary depth mask, crop the face from the image and depth image.
    Apply a small padding to the cropped images.
    Args:
        img (np.ndarray): Image to crop.
        depth (np.ndarray): Depth image to crop.
        mask (np.ndarray): Binary depth mask.
    
    Returns:
        img (np.ndarray): Cropped image.
        depth (np.ndarray): Cropped depth image.
    """
    # Find the bounding box of the face
    ys, xs = np.where(mask == 255)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

    # Pad the bounding box
    x1 = max(0, x1 - 20)
    y1 = max(0, y1 - 20)
    x2 = min(img.shape[1], x2 + 20)
    y2 = min(img.shape[0], y2 + 20)

    # Crop the face from the image and depth image
    img = img[y1:y2, x1:x2]
    depth = depth[y1:y2, x1:x2]

    return img, depth


def crop_and_save(img_path, depth_path, mask_path, dest_path, dest_name):
    """Crop face images and decoded depth images using depth masks
    Args:
        img_path (str): Path to the BIWI Kinect image file.
        depth_path (str): Path to the BIWI Kinect depth file.
        mask_path (str): Path to the BIWI Kinect depth mask file.
        dest_path (Path): Path to the destination folder to save the cropped images.
        dest_name (str): Name of the cropped image file.
    """
    img = cv2.imread(img_path)
    depth = decode_biwi_depth(depth_path)
    cv2.imwrite(depth_path.replace(".bin", ".png"), depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img.shape[:2] != depth.shape[:2] or img.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image, depth and mask must have the same shape. Found {img.shape}, {depth.shape}, {mask.shape}")

    try:
        img, depth = crop_using_mask(img, depth, mask)
    except ValueError as e:
        print(f"Error cropping {img_path}: {e}")
        return

    # Save the cropped images
    cv2.imwrite(str(dest_path / "cropped_images" / f"{dest_name}.jpg"), img)
    cv2.imwrite(str(dest_path / "cropped_depths" / f"{dest_name}.png"), depth)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="BIWI Kinect Preprocessor. Expected directory structure: "
        "root-dir/faces_0/*/frame_*.png, "
        "root-dir/faces_9/*/frame_*depth.bin, "
        "root-dir/head_pose_masks/*/frame_*_depth_mask.png"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/coc/flash5/schermala3/Datasets/BIWIKinectHeads",
        help="Path to extracted BIWI Kinect dataset",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="/coc/flash5/schermala3/Datasets/BIWIKinectHeads/preprocessed",
        help="Path to destination folder to save the cropped images",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    dest_dir = Path(args.dest_dir)

    # Create destination folders
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "cropped_images").mkdir(parents=True, exist_ok=True)
    (dest_dir / "cropped_depths").mkdir(parents=True, exist_ok=True)

    # Crop images and depth images using depth masks
    for i in tqdm(range(1, 25)):
        for frame in (root_dir / f"faces_0/{i:02}").glob("frame_*rgb.png"):
            frame_name = "_".join(frame.name.split("_")[:2])
            img_path = frame
            depth_path = root_dir / f"faces_0/{i:02}/{frame_name}_depth.bin"
            mask_path = root_dir / f"head_pose_masks/{i:02}/{frame_name}_depth_mask.png"

            if not (mask_path.exists() and depth_path.exists()):
                print(f"Missing mask or depth file for {img_path}. Skipping.")
                continue

            crop_and_save(str(img_path), str(depth_path), str(mask_path), dest_dir, f"{i:02}_{frame_name}")
