# DDRL
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
