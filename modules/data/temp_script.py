from datasets import KeyPointDataset, DepthDataset
from pathlib import Path
from fastai.torch_core import show_image
import torchvision.transforms as T


keypoint_dataset = KeyPointDataset(root=Path('/home/suyash/Gatech/DDRL/modules/data/aflw_dataset/AFLW_lists/front.GTB'),mode="train")
first_image = keypoint_dataset.__getitem__(0)
T.ToPILImage()(first_image['image']).show()
show_image(first_image['image'])