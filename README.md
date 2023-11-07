# ddrl
Discriminative Diffusion Representation Learning

## Offline Features

### Saving
Sample Command:
```bash
python modules/precompute_features.py --dataset CelebAHQMask --dataset_root /coc/flash5/schermala3/Datasets/CelebAMask-HQ/ --output_dir /coc/flash5/schermala3/Datasets/CelebAMask-HQ/temp_features/
```

### Loading
Sample loading code:
```python
with h5py.File("h5_file_path", "r") as feature_store:
    print("Available images: ", feature_store.keys())
    image = feature_store["image_name"]
    print("Available timesteps: ", image.keys())
    timestep_0 = image["timestep_0"]
    print("Available features: ", timestep_0.keys())
    features = timestep_0["features"]
    print(feature.shape)
```

Loading using the `FeatureLoader` class:
```python
from modules.feature_loader import FeatureLoader

try:
    feature_loader = FeatureLoader("h5_file_path", scales, scale_directions, timestep, resolution)
except KeyError as e:  # features unavailable
    print(e)

# Get features for a single image
features = feature_loader(image_name)
```

## AFLW data setup
1. Download AFLW dataset.zip
2. extract it by `tar xzvf aflw-images-*.tar.gz`.

3. Arrange images and matfile so it looks as follows in directory DDRL/modules/data/aflw_dataset
```
.
├── AFLWinfo_release.mat
└── images
    ├── 0
    ├── 2
    └── 3
```

AFLWinfo_release.mat includes all the annotations as a mat file which
is decoded by the following script for ease of use with a pytorch dataloader.

Run the following script
```
python preprocess_AFLW.py
```
It creates two folders in DDRL/modules/data/aflw_dataset - 

1. AFLW_lists
2. processed

#### AFLW_lists
This will include four files - 
```python
all.GTB
all.GTL
front.GTB
front.GTL
```

all includes info about all the images. (~24k images)

front includes only those images that have been classified
as front facing in our algorithm outlined in the face class in preprocess_AFLW.py (~6k images)

The annotations provided include a Ground Truth Bounding Box and all GTB files use the ground truth bounding box.

These annotations may be wrong and GTL includes bounding boxes that are algorithmically calculated from keypoints. These are the GTL files.

Each GTB/GTL file has the format - 
```python
<image_path> <keypoint_annotation_path> <bounding-box coordinates> <size of image>
```

#### processed
The processed folder has the following file format corresponding to the images -
```
└── annotations
    ├── 0
    ├── 2
    └── 3
```

Each row in each file in the annotations is of the format - 
```python
<keypoint-index> <keypoint-x-coordinate> <keypoint-y-coordinate> <keypoint masked or not>
```