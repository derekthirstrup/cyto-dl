# Supervised Training with Timelapse Data

## Overview

For **supervised learning** from timelapse microscopy files, you need separate input and target channels. This guide shows how to properly configure your CSV and config files.

---

## ✅ Correct Approach for Supervised Learning

### CSV Format

**Required columns:**
- `path` - File path to timelapse
- `source_channel` - Channel index for INPUT (raw/brightfield)
- `target_channel` - Channel index for TARGET (segmentation/label)
- `split` - train/val/test

**Example CSV:**
```csv
path,source_channel,target_channel,split,start,stop
C:/data/exp1.nd2,0,1,train,0,100
C:/data/exp2.czi,0,2,val,0,50
```

### Config Structure

```yaml
_target_: cyto_dl.datamodules.random_timelapse_datamodule.RandomTimelapseDatamodule

# Column mappings
img_path_column: path
split_column: split

# IMPORTANT: Use extra_columns, NOT channel_column
extra_columns:
  - source_channel  # CSV column for input
  - target_channel  # CSV column for target

# Transforms must load BOTH channels
transforms:
  train:
    transforms:
      # Load INPUT
      - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
        out_key: raw
        kwargs_keys:
          - source_channel  # ← Uses CSV source_channel value

      # Load TARGET
      - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
        out_key: seg
        kwargs_keys:
          - target_channel  # ← Uses CSV target_channel value
```

---

## ❌ Common Mistake: Using Single channel_column

**DON'T do this for supervised learning:**
```yaml
# ❌ WRONG - This only loads ONE channel
channel_column: channel  # Can't specify both input AND target
```

**Why it's wrong:**
- `channel_column` is designed for **unsupervised learning** where you only need one channel
- For **supervised learning**, you need TWO separate channels (input and target)
- Solution: Use `extra_columns` with `source_channel` and `target_channel`

---

## Complete Example

### 1. CSV File (`h2b_supervised.csv`)

```csv
path,source_channel,target_channel,split,start,stop,scene
C:/data/timelapse1.nd2,0,1,train,0,100,0
C:/data/timelapse2.nd2,0,1,train,0,150,0
C:/data/timelapse3.czi,0,2,val,0,80,0
C:/data/timelapse4.ome.zarr,0,1,test,0,120,0
```

**Channel mapping:**
- Channel 0 = Raw H2B signal (INPUT)
- Channel 1 = Segmentation mask (TARGET)
- Channel 2 = Alternative segmentation (TARGET)

### 2. Data Config (`random_timelapse_supervised.yaml`)

```yaml
_target_: cyto_dl.datamodules.random_timelapse_datamodule.RandomTimelapseDatamodule

path: /path/to/h2b_supervised.csv

img_path_column: path
split_column: split

# Make both channel columns available
extra_columns:
  - source_channel
  - target_channel

num_timepoints: 1
timepoint_sampling: random
spatial_dims: 3
seed: 42

transforms:
  train:
    _target_: monai.transforms.Compose
    transforms:
      # Load source channel
      - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
        path_key: original_path
        out_key: raw  # Saved to batch["raw"]
        dask_load: True
        kwargs_keys:
          - dimension_order_out
          - source_channel  # From CSV
          - T

      # Load target channel
      - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
        path_key: original_path
        out_key: seg  # Saved to batch["seg"]
        dask_load: True
        kwargs_keys:
          - dimension_order_out
          - target_channel  # From CSV
          - T

      # Process both together
      - _target_: monai.transforms.EnsureChannelFirstd
        keys: [raw, seg]

      - _target_: monai.transforms.RandSpatialCropd
        keys: [raw, seg]
        roi_size: [32, 64, 64]

      - _target_: monai.transforms.NormalizeIntensityd
        keys: [raw, seg]

      - _target_: monai.transforms.ToTensord
        keys: [raw, seg]
```

### 3. Experiment Config (`h2b_prediction_timelapse.yaml`)

```yaml
# @package _global_

defaults:
  - override /data: im2im/random_timelapse_supervised.yaml
  - override /model: im2im/labelfree.yaml
  - override /trainer: gpu.yaml

source_col: raw  # Model expects batch["raw"]
target_col: seg  # Model expects batch["seg"]

data:
  path: C:/data/h2b_supervised.csv
  batch_size: 4
  num_workers: 4
```

---

## Data Flow

```
CSV Row:
  path: exp.nd2
  source_channel: 0
  target_channel: 1

    ↓ (RandomTimelapseDataset)

Batch dict created:
  original_path: exp.nd2
  source_channel: 0      ← From extra_columns
  target_channel: 1      ← From extra_columns
  T: <random frame>

    ↓ (First BioIOImageLoaderd)

batch["raw"] = exp.nd2[channel=0, T=random]

    ↓ (Second BioIOImageLoaderd)

batch["seg"] = exp.nd2[channel=1, T=random]

    ↓ (Model receives)

input = batch["raw"]
target = batch["seg"]
```

---

## Key Differences: Supervised vs Unsupervised

| Aspect | Unsupervised | Supervised |
|--------|--------------|------------|
| **CSV columns** | `channel` | `source_channel`, `target_channel` |
| **Config param** | `channel_column: channel` | `extra_columns: [source_channel, target_channel]` |
| **Transforms** | Load 1 channel | Load 2 channels (separately) |
| **Batch keys** | `batch["raw"]` | `batch["raw"]`, `batch["seg"]` |
| **Use case** | Self-supervised, MAE, contrastive | Segmentation, prediction, translation |

---

## Files to Use

**Configs:**
- `configs/data/im2im/random_timelapse_supervised.yaml` - Base supervised config
- `configs/experiment/local/h2b_prediction_timelapse.yaml` - H2B experiment example

**CSV Templates:**
- `examples/supervised_timelapse_metadata.csv` - Example with source/target channels

**Documentation:**
- `docs/CSV_CONFIG_MAPPING.md` - Complete mapping reference

---

## Quick Start

1. **Create your CSV:**
```csv
path,source_channel,target_channel,split
C:/data/file1.nd2,0,1,train
C:/data/file2.nd2,0,1,val
```

2. **Run training:**
```bash
python cyto_dl/train.py experiment=local/h2b_prediction_timelapse
```

3. **Verify it's working:**
- Check that transforms load both channels
- Verify batch contains both `"raw"` and `"seg"` keys
- Confirm model receives both input and target

---

## Troubleshooting

### "Only loading one channel"
**Cause:** Using `channel_column` instead of `extra_columns`
**Fix:** Replace `channel_column: channel` with:
```yaml
extra_columns:
  - source_channel
  - target_channel
```

### "Key 'source_channel' not found"
**Cause:** CSV missing required column
**Fix:** Add `source_channel` and `target_channel` columns to CSV

### "Both channels look the same"
**Cause:** Both loaders using same channel
**Fix:** Verify CSV has different values for `source_channel` and `target_channel`

---

## See Also

- [Random Timelapse Sampling Guide](RANDOM_TIMELAPSE_SAMPLING.md)
- [CSV Config Mapping](CSV_CONFIG_MAPPING.md)
