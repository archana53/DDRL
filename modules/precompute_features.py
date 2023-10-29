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
        "--dataset", type=str, required=True, choices=["CelebAHQMask", "Depth", "KeyPoint"], help="Dataset to use"
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
        "--scales", type=int, nargs="+", default=[0, 1, 2, 3], help="UNet block indices to extract features from"
    )
    parser.add_argument("--timestep_freq", type=int, default=100, help="Frequency of timesteps to save features at")
    parser.add_argument("--no_latent", dest="latent", action="store_false", help="Do not use LDM architecture")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory path to store the features")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output feature store")
    return parser.parse_args()


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
            h5_file.create_dataset(f"timestep_{timestep}/feature_{feature_name}/{image_names[i]}", data=f)


if __name__ == "__main__":
    args = parse_args()

    pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
    unet = pipe.unet

    dataset_cls = DatasetType[args.dataset].value
    dataset = dataset_cls(root=Path(args.dataset_root), mode="test", size=(args.resolution, args.resolution))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    extractor = ModelExtractor(pipe, unet, args.scale_directions, args.scales, args.latent, upsample=False).cuda()
    timesteps = torch.arange(0, 1000, args.timestep_freq).cuda()  # save features every `timestep` timesteps

    # set up output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{args.dataset.lower()}_features.h5")

    # check if output file exists
    if os.path.exists(output_file):
        if args.overwrite:
            os.remove(output_file)
        else:
            raise FileExistsError(f"File {output_file} already exists. Use --overwrite to overwrite it.")

    # add save_features callback to model
    with File(output_file, "w") as h5_file:
        # run model on dataset
        for i, batch in enumerate(tqdm(dataloader)):
            image = batch["image"].cuda()
            image_name = batch["name"]
            for timestep in timesteps:
                _, features = extractor(image, timestep)
                save_features(features, timestep, image_name, h5_file)
