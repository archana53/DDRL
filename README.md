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
    print("Available timesteps: ", feature_store.keys())
    timestep_0 = feature_store["timestep_0"]
    print("Available features: ", timestep_0.keys())
    features = timestep_0["features"]
    print("Images: ", features.keys())
    feature = features["image_name"]
    print(feature.shape)
```
