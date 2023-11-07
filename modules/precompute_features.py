import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from diffusers import LDMPipeline
from tqdm import tqdm
from h5py import File

from data.datasets import DatasetType
from unet_extractor import ModelExtractor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["CelebAHQMask", "Depth", "KeyPoint"],
        help="Dataset to use",
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="Root path of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers in the dataloader")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the images (square)")
    parser.add_argument(
        "--scale_directions",
        type=str,
        nargs="+",
        default=["down", "up"],
        choices=["down", "mid", "up"],
        help="UNet block directions to extract features from",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="UNet block indices to extract features from",
    )
    parser.add_argument(
        "--timestep_range",
        type=int,
        nargs="+",
        default=[0, 1000, 100],
        help="Start, stop and step of timesteps",
    )
    parser.add_argument(
        "--no_latent",
        dest="latent",
        action="store_false",
        help="Do not use LDM architecture",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory path to store the features",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output feature store",
    )
    return parser.parse_args()


def _already_exists(feature_names, timestep, image_name, h5_file):
    """Checks if features already exist in h5 file.
    :param feature_names: list of feature names
    :param timestep: timestep of features
    :param image_name: name of image
    :param h5_file: h5 file to check
    :return: True if features already exist, False otherwise
    """
    for i in range(len(image_name)):
        for feature_name in feature_names:
            if f"{image_name[i]}/{timestep}/{feature_name}" not in h5_file:
                # delete all features of image if one feature is missing
                # to avoid h5 overwriting error
                if image_name[i] in h5_file:
                    del h5_file[image_name[i]]
                return False
    return True


def save_features(features, timestep, image_names, h5_file):
    """Saves batch of features to h5 file.
    Each scale is saved as a separate group.
    :param features: batch of features
    :param timestep: timestep of features
    :param image_names: names of images in batch
    :param h5_file: h5 file to save features to
    """
    for feature_name, feature in features.items():
        for i, f in enumerate(feature):
            f = f.detach().cpu().numpy()
            h5_file.create_dataset(f"{image_names[i]}/{timestep}/{feature_name}", data=f)


if __name__ == "__main__":
    args = parse_args()

    pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
    unet = pipe.unet

    # set up dataloaders
    dataset_cls = DatasetType[args.dataset].value
    dataset = dataset_cls(
        root=Path(args.dataset_root),
        mode="test",
        size=(args.resolution, args.resolution),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    extractor = ModelExtractor(pipe, unet, args.scale_directions, args.scales, args.latent, upsample=False).cuda()
    timesteps = torch.arange(*args.timestep_range).cuda()  # save features every `timestep` timesteps

    # set up output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timesteps_suffix = '_'.join([str(x) for x in args.timestep_range])
    output_file = os.path.join(output_dir, f"{args.dataset.lower()}_timesteps_{timesteps_suffix}_features.h5")

    # check if output file exists
    if os.path.exists(output_file) and args.overwrite:
        os.remove(output_file)
        
    # run model on dummy input to get feature names
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution).cuda()
    _ = extractor(dummy_input, timesteps[0])
    feature_names = list(extractor.intermediate_features.keys())

    with File(output_file, "a") as h5_file:
        # run model on dataset
        for i, batch in enumerate(tqdm(dataloader)):
            image = batch["image"].cuda()
            image_name = batch["name"]
            for timestep in timesteps:

                # skip if features already exist
                if _already_exists(feature_names, timestep, image_name, h5_file):
                    continue

                with torch.inference_mode():
                    _ = extractor(image, timestep)
                save_features(extractor.intermediate_features, timestep, image_name, h5_file)
