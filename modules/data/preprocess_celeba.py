import os
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class CelebAMaskPreprocessor:
    """CelebAMask-HQ Preprocessor class that converts raw masks to a single mask.
    Modified from https://github.com/zllrunning/face-parsing.PyTorch/blob/master/prepropess_data.py
    """

    def __init__(self, image_path, source_mask_path, dest_mask_path):
        self.image_path = image_path
        self.source_mask_path = source_mask_path
        self.dest_mask_path = dest_mask_path
        self.attrs = [
            "skin",
            "l_brow",
            "r_brow",
            "l_eye",
            "r_eye",
            "eye_g",
            "l_ear",
            "r_ear",
            "ear_r",
            "nose",
            "mouth",
            "u_lip",
            "l_lip",
            "neck",
            "neck_l",
            "cloth",
            "hair",
            "hat",
        ]

    def preprocess(self):
        for i in tqdm(range(15)):
            for j in range(i * 2000, (i + 1) * 2000):
                mask = np.zeros((512, 512))

                for l, attr in enumerate(self.attrs, 1):
                    file_name = "".join([str(j).rjust(5, "0"), "_", attr, ".png"])
                    path = os.path.join(self.source_mask_path, str(i), file_name)

                    if os.path.exists(path):
                        sep_mask = np.array(Image.open(path).convert("P"))
                        mask[sep_mask == 225] = l

                cv2.imwrite(f"{self.dest_mask_path}/{j}.png", mask)


if __name__ == "__main__":
    parser = ArgumentParser(description="CelebAMask-HQ Preprocessor")
    parser.add_argument(
        "--image_path",
        type=str,
        default="/coc/flash5/schermala3/Datasets/CelebAMask-HQ/CelebA-HQ-img",
        help="path to image directory",
    )
    parser.add_argument(
        "--source_mask_path",
        type=str,
        default="/coc/flash5/schermala3/Datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
        help="path to raw mask directory",
    )
    parser.add_argument(
        "--dest_mask_path",
        type=str,
        default="/coc/flash5/schermala3/Datasets/CelebAMask-HQ/mask",
        help="desired destination directory for processed mask",
    )

    args = parser.parse_args()

    image_path = args.image_path
    source_mask_path = args.source_mask_path
    dest_mask_path = args.dest_mask_path

    if not os.path.exists(dest_mask_path):
        os.makedirs(dest_mask_path)

    preprocessor = CelebAMaskPreprocessor(image_path, source_mask_path, dest_mask_path)
    preprocessor.preprocess()
