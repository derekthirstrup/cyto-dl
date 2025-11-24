# Efficient Random Timelapse Sampling with BioIO

## Overview

This guide explains how to efficiently train on timelapse microscopy data using the new `RandomTimelapseDatamodule`. This approach eliminates the need to convert large microscopy files (nd2, czi, ome-zarr) to TIF stacks and enables efficient random sampling of timepoints during training.

## Key Features

✅ **Lazy Loading**: Doesn't enumerate all timepoints at initialization
✅ **Random Sampling**: Uses numpy RNG for reproducible random timepoint sampling
✅ **Multi-Format Support**: Works with any bioio-supported format (nd2, czi, ome-zarr, tif, zarr, etc.)
✅ **Cloud Storage**: Supports S3, GCS, Azure Blob Storage via upath
✅ **Memory Efficient**: Uses Dask for lazy loading and caches metadata to avoid repeated file opens
✅ **Flexible Configuration**: Simple YAML configuration for various use cases

## Why Use This Instead of MultiDimImageDataset?

### The Problem with MultiDimImageDataset

The existing `MultiDimImageDataset` enumerates **all** timepoints at initialization:

```python
# For a timelapse with 1000 frames, this creates 1000 dataset entries!
for timepoint in range(1000):
    dataset_entries.append({...})  # Very inefficient!
```

This approach:
- ❌ Requires opening files multiple times during initialization
- ❌ Creates thousands of dataset entries for large timelapses
- ❌ Wastes memory storing metadata for every single timepoint
- ❌ Makes training from large nd2/czi files impractical
- ❌ Forces users to convert to TIF stacks first

### The Solution: RandomTimelapseDatamodule

The new datamodule:
- ✅ Creates one dataset entry per file
- ✅ Randomly samples timepoints on-the-fly during training
- ✅ Opens files only once to cache metadata
- ✅ Scales to timelapses with thousands of frames
- ✅ Works directly with nd2, czi, ome-zarr files

## Supported File Formats

Thanks to bioio, all these formats are supported without conversion:

| Format | Extensions | Cloud Support |
|--------|-----------|---------------|
| Nikon ND2 | `.nd2` | ✅ |
| Zeiss CZI | `.czi` | ✅ |
| OME-Zarr | `.ome.zarr`, `.zarr` | ✅ |
| Zarr v3 | `.zarr` | ✅ |
| OME-TIFF | `.ome.tif`, `.ome.tiff` | ✅ |
| TIFF Stacks | `.tif`, `.tiff` | ✅ |
| Leica LIF | `.lif` | ✅ |
| And many more... | See bioio docs | ✅ |

## Quick Start

### 1. Prepare Your Metadata CSV

Create a CSV file with your timelapse files:

```csv
path,channel,split,scene,resolution,start,stop
/data/timelapse1.nd2,0,train,0,0,0,500
/data/timelapse2.czi,"0,1,2",train,0,1,10,200
s3://bucket/timelapse3.ome.zarr,1,val,0,0,0,-1
/data/timelapse4.nd2,2,test,1,0,0,100
```

**Column descriptions:**
- `path`: File path (local, S3, GCS, etc.)
- `channel`: Channel index or comma-separated indices
- `split`: train/val/test
- `scene`: Scene index (optional, defaults to 0)
- `resolution`: Resolution level for multi-resolution files (optional, defaults to 0)
- `start`: Starting timepoint (optional, defaults to 0)
- `stop`: Ending timepoint (optional, defaults to all frames, use -1 for all)

### 2. Create Your Config

```yaml
_target_: cyto_dl.datamodules.random_timelapse_datamodule.RandomTimelapseDatamodule

path: /path/to/metadata.csv
split_column: split

# Sample 5 random timepoints per file per batch
num_timepoints: 5
timepoint_sampling: random  # Options: random, sequential, uniform

spatial_dims: 3  # 2 for YX, 3 for ZYX
seed: 42

num_workers: 8
batch_size: 4

transforms:
  train:
    _target_: monai.transforms.Compose
    transforms:
      - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
        path_key: original_path
        out_key: raw
        dask_load: True
        dtype: numpy.float32
      # ... your other transforms ...
```

### 3. Train Your Model

```python
import lightning as L
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("configs/data/im2im/random_timelapse.yaml")

# Instantiate datamodule
datamodule = hydra.utils.instantiate(config)

# Train
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule)
```

## Configuration Options

### Timepoint Sampling Strategies

Control how timepoints are sampled with `timepoint_sampling`:

#### 1. Random Sampling (default)
```yaml
timepoint_sampling: random
num_timepoints: 5
```
Randomly samples 5 timepoints without replacement. Good for diverse training data.

#### 2. Sequential Sampling
```yaml
timepoint_sampling: sequential
num_timepoints: 10
```
Samples 10 consecutive frames starting from a random position. Good for temporal models (e.g., predicting next frame).

#### 3. Uniform Sampling
```yaml
timepoint_sampling: uniform
num_timepoints: 8
```
Samples 8 evenly spaced frames across the timelapse. Good for full coverage of timelapse dynamics.

### Controlling Dataset Size

Use `samples_per_epoch` to oversample or undersample:

```yaml
samples_per_epoch:
  train: 1000  # Sample 1000 batches per epoch (even if CSV has fewer rows)
  val: 100
  test: 50
```

This is useful when:
- You have few files but want more training batches
- You want to limit validation time
- You want reproducible epoch lengths

### Cloud Storage (S3, GCS, Azure)

The datamodule automatically handles cloud paths via `upath`:

```yaml
# Option 1: S3 paths in CSV
path: s3://my-bucket/data/train.csv

# CSV contents:
# path,channel,split
# s3://my-bucket/images/exp1.nd2,0,train
# s3://my-bucket/images/exp2.czi,1,val
```

```yaml
# Option 2: Separate CSV per split
path:
  train: s3://my-bucket/train.csv
  val: s3://my-bucket/val.csv
  test: s3://my-bucket/test.csv
```

**Note**: Ensure you have proper cloud credentials configured (e.g., AWS credentials for S3).

### Inline Metadata (No CSV)

For quick testing, use inline `dict_meta`:

```yaml
path:
  dict_meta:
    path:
      - /data/file1.nd2
      - /data/file2.czi
    channel:
      - "0"
      - "1,2"
    split:
      - train
      - val

split_column: split
```

## Advanced Usage

### Multi-Channel Loading

Load multiple channels by specifying comma-separated indices:

```csv
path,channel,split
/data/file.nd2,"0,2,4",train
```

This loads channels 0, 2, and 4 from the file.

### Multi-Scene Files

For files with multiple scenes (e.g., multi-well plates):

```csv
path,channel,scene,split
/data/plate.nd2,0,0,train
/data/plate.nd2,0,1,train
/data/plate.nd2,0,2,val
```

### Multi-Resolution Files

For pyramidal OME-Zarr or other multi-resolution formats:

```csv
path,channel,resolution,split
/data/pyramid.ome.zarr,0,0,train  # Full resolution
/data/pyramid.ome.zarr,0,1,train  # 2x downsampled
/data/pyramid.ome.zarr,0,2,val    # 4x downsampled
```

### Custom Timepoint Ranges

Specify which frames to sample from:

```csv
path,channel,start,stop,split
/data/long_timelapse.nd2,0,0,100,train    # First 100 frames
/data/long_timelapse.nd2,0,100,200,val    # Frames 100-200
/data/long_timelapse.nd2,0,200,-1,test    # Frames 200 to end
```

### Extra Metadata Columns

Include additional columns in your CSV and they'll be available in the batch:

```yaml
extra_columns:
  - cell_id
  - treatment
  - replicate
```

```csv
path,channel,split,cell_id,treatment,replicate
/data/exp.nd2,0,train,cell_001,drug_A,rep1
```

These will be accessible in your training loop:

```python
for batch in dataloader:
    image = batch["raw"]
    cell_id = batch["cell_id"]
    treatment = batch["treatment"]
    # ...
```

## Performance Tips

### 1. Use Dask Loading
```yaml
- _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
  dask_load: True  # Enables lazy loading
```

### 2. Use Appropriate Data Types
```yaml
dtype: numpy.float16  # Saves memory (vs float32)
```

### 3. Adjust Number of Workers
```yaml
num_workers: 8  # Increase for faster data loading (depends on CPU cores)
```

### 4. Use Persistent Workers
```yaml
persistent_workers: True  # Keeps workers alive between epochs
```

### 5. Pin Memory for GPU Training
```yaml
pin_memory: True  # Faster CPU-to-GPU transfer
```

## Comparison: Old vs New Approach

| Aspect | MultiDimImageDataset (Old) | RandomTimelapseDatamodule (New) |
|--------|---------------------------|--------------------------------|
| Initialization | Opens each file, enumerates all timepoints | Opens each file once to cache metadata |
| Dataset size | # files × # timepoints | # files |
| Memory usage | High (stores all timepoint metadata) | Low (minimal metadata cache) |
| Random sampling | Sample-level (picks from enumerated list) | Timepoint-level (true random sampling) |
| Timelapses with 1000 frames | 1000 dataset entries per file | 1 dataset entry per file |
| Supports nd2/czi directly | Yes (but inefficient) | Yes (efficient!) |
| S3/cloud support | Via upath | Enhanced upath support |
| Configuration complexity | Medium | Simple |

## Example: Training a Denoising Model

Here's a complete example for training a denoising model on timelapse data:

```yaml
# configs/data/timelapse_denoise.yaml
_target_: cyto_dl.datamodules.random_timelapse_datamodule.RandomTimelapseDatamodule

path: /data/microscopy/timelapse_metadata.csv
split_column: split

img_path_column: path
channel_column: channel

# Sample 1 random timepoint per file
num_timepoints: 1
timepoint_sampling: random

spatial_dims: 3
seed: 42

num_workers: 8
batch_size: 16
pin_memory: True

transforms:
  train:
    _target_: monai.transforms.Compose
    transforms:
      # Load raw data
      - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
        path_key: original_path
        out_key: raw
        dask_load: True
        dtype: numpy.float32

      - _target_: monai.transforms.EnsureChannelFirstd
        keys: raw
        channel_dim: "no_channel"

      # Random 3D crop
      - _target_: monai.transforms.RandSpatialCropd
        keys: raw
        roi_size: [64, 128, 128]
        random_size: False

      # Augmentation
      - _target_: monai.transforms.RandFlipd
        keys: raw
        prob: 0.5
        spatial_axis: [0, 1, 2]

      - _target_: monai.transforms.RandRotate90d
        keys: raw
        prob: 0.5
        spatial_axes: [1, 2]

      # Add synthetic noise for denoising task
      - _target_: monai.transforms.RandGaussianNoised
        keys: raw
        prob: 1.0
        mean: 0.0
        std: 0.1

      # Normalize
      - _target_: monai.transforms.NormalizeIntensityd
        keys: raw
        channel_wise: True

      - _target_: monai.transforms.ToTensord
        keys: raw
```

## Troubleshooting

### Issue: "File not found" with S3 paths

**Solution**: Ensure you have:
1. Installed cloud dependencies: `pip install s3fs` (for S3)
2. Configured credentials (AWS credentials for S3)
3. Used correct S3 URI format: `s3://bucket/path/file.nd2`

### Issue: "Scene not found"

**Solution**: Check available scenes:
```python
from bioio import BioImage
img = BioImage("/path/to/file.nd2")
print(img.scenes)  # See available scenes
```

### Issue: High memory usage

**Solutions**:
1. Enable Dask loading: `dask_load: True`
2. Use float16: `dtype: numpy.float16`
3. Reduce batch size
4. Reduce num_workers
5. Use smaller crops in transforms

### Issue: Slow initialization

**Solution**: This is expected for the first epoch as metadata is cached. Subsequent epochs will be much faster.

## Migration Guide: From MultiDimImageDataset

If you're currently using `MultiDimImageDataset`, here's how to migrate:

**Old config:**
```yaml
dataset:
  _target_: cyto_dl.datamodules.multidim_image.MultiDimImageDataset
  csv_path: /data/metadata.csv
  img_path_column: path
  channel_column: channel
  time_start_column: start
  time_stop_column: stop
  spatial_dims: 3
```

**New config:**
```yaml
_target_: cyto_dl.datamodules.random_timelapse_datamodule.RandomTimelapseDatamodule
path: /data/metadata.csv
img_path_column: path
channel_column: channel
time_start_column: start
time_stop_column: stop
spatial_dims: 3
num_timepoints: 1  # NEW: specify how many frames to sample
timepoint_sampling: random  # NEW: sampling strategy
split_column: split  # NEW: specify split column
seed: 42  # NEW: reproducibility
# ... dataloader kwargs ...
```

## References

- [BioIO Documentation](https://bioio-devs.github.io/bioio/)
- [MONAI Transforms](https://docs.monai.io/en/stable/transforms.html)
- [UPath Documentation](https://github.com/fsspec/universal_pathlib)
- [PyTorch Lightning DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)

## Contributing

Found a bug or want to request a feature? Please open an issue on GitHub!
