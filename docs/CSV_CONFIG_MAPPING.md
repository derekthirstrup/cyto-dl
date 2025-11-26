# CSV to Config Mapping Guide

This guide explains how CSV column names map to configuration variables in cyto-dl experiment configs.

## 📋 Table of Contents
1. [Approach 1: Separate Input/Output Files](#approach-1-dataframe-with-separate-files)
2. [Approach 2: Single Timelapse Files](#approach-2-timelapse-with-channels)
3. [Variable Reference](#variable-reference)
4. [Common Patterns](#common-patterns)

---

## Approach 1: Dataframe with Separate Files

**Use when:** You have separate files for input (raw) and output (segmentation/target)

### CSV Structure
```csv
raw,seg,split
C:/data/cell001_raw.nd2,C:/data/cell001_seg.nd2,train
C:/data/cell002_raw.nd2,C:/data/cell002_seg.nd2,val
```

### Config: `h2b_prediction_dataframe.yaml`

#### Experiment Config Variables
```yaml
# These define which CSV columns contain image paths
source_col: raw   # Must match CSV column name for INPUT
target_col: seg   # Must match CSV column name for OUTPUT
```

#### Data Config Mapping
```yaml
data:
  path: /path/to/csv
  split_column: split  # CSV column with train/val/test labels

  columns:
    - ${source_col}  # = "raw" → looks for "raw" column in CSV
    - ${target_col}  # = "seg" → looks for "seg" column in CSV
```

#### Transform Mapping
```yaml
transforms:
  train:
    transforms:
      - _target_: monai.transforms.LoadImaged
        keys: ${target_col}  # = "seg" → loads from "seg" column

      - _target_: monai.transforms.LoadImaged
        keys: ${source_col}  # = "raw" → loads from "raw" column

      - _target_: monai.transforms.NormalizeIntensityd
        keys: ${data.columns}  # = ["raw", "seg"] → normalizes both
```

### Complete Flow
```
CSV Column "raw"
  ↓ (referenced by source_col)
  ↓
${source_col} in config
  ↓ (expands to "raw")
  ↓
LoadImaged with keys: raw
  ↓
Creates data dict with key "raw"
  ↓
Model receives batch["raw"]
```

### Example CSV → Config Mapping

| CSV Header | Config Variable | Used In | Purpose |
|------------|----------------|---------|---------|
| `raw` | `source_col: raw` | `columns`, `keys` in transforms | Input image paths |
| `seg` | `target_col: seg` | `columns`, `keys` in transforms | Target/output image paths |
| `split` | `split_column: split` | Datamodule | Train/val/test split |

---

## Approach 2: Timelapse with Channels

**Use when:** Input and output come from the same timelapse file (different channels or timepoints)

### CSV Structure
```csv
path,raw_channel,seg_channel,split,start,stop
C:/data/timelapse1.nd2,0,1,train,0,100
C:/data/timelapse2.czi,0,1,val,0,50
```

### Config: `h2b_prediction_timelapse.yaml`

#### Experiment Config Variables
```yaml
# These define the KEYS in the data dictionary, NOT CSV columns
source_col: raw   # Key for input in batch dict
target_col: seg   # Key for target in batch dict
```

#### Data Config Mapping
```yaml
data:
  path: /path/to/csv
  img_path_column: path     # CSV column with file paths
  split_column: split       # CSV column with train/val/test

  extra_columns:
    - raw_channel           # CSV column to include in batch
    - seg_channel           # CSV column to include in batch
```

#### Transform Mapping
```yaml
transforms:
  train:
    transforms:
      # Load INPUT channel
      - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
        path_key: original_path      # Uses path from CSV "path" column
        out_key: ${source_col}       # = "raw" → saves to batch["raw"]
        kwargs_keys:
          - raw_channel              # Uses CSV "raw_channel" column

      # Load TARGET channel
      - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
        path_key: original_path      # Uses same path
        out_key: ${target_col}       # = "seg" → saves to batch["seg"]
        kwargs_keys:
          - seg_channel              # Uses CSV "seg_channel" column

      # Process both
      - _target_: monai.transforms.NormalizeIntensityd
        keys: [${source_col}, ${target_col}]  # = ["raw", "seg"]
```

### Complete Flow
```
CSV Row:
  path: C:/data/exp.nd2
  raw_channel: 0
  seg_channel: 1
  start: 0
  stop: 100

  ↓ (RandomTimelapseDataset processes)
  ↓
Batch dict created:
  original_path: C:/data/exp.nd2
  raw_channel: 0        # From extra_columns
  seg_channel: 1        # From extra_columns
  T: <random timepoint>
  dimension_order_out: CZYX

  ↓ (First BioIOImageLoaderd)
  ↓
Loads channel 0 → batch["raw"]

  ↓ (Second BioIOImageLoaderd)
  ↓
Loads channel 1 → batch["seg"]

  ↓
Model receives:
  batch["raw"]  = input
  batch["seg"]  = target
```

### Example CSV → Config Mapping

| CSV Column | Config Variable | Used In | Purpose |
|------------|----------------|---------|---------|
| `path` | `img_path_column: path` | RandomTimelapseDataset | File path |
| `raw_channel` | `extra_columns: [raw_channel]` | BioIOImageLoaderd kwargs | Channel index for input |
| `seg_channel` | `extra_columns: [seg_channel]` | BioIOImageLoaderd kwargs | Channel index for target |
| `split` | `split_column: split` | RandomTimelapseDatamodule | Train/val/test |
| `start` | `time_start_column: start` | RandomTimelapseDataset | Timepoint range start |
| `stop` | `time_stop_column: stop` | RandomTimelapseDataset | Timepoint range end |

---

## Variable Reference

### Top-Level Experiment Variables

These are defined at the experiment config level and used throughout:

```yaml
# Experiment config (e.g., h2b_prediction_dataframe.yaml)
source_col: raw   # Name for INPUT data
target_col: seg   # Name for TARGET/OUTPUT data
spatial_dims: 3   # 2D or 3D
```

**Usage:**
- `${source_col}` → expands to `"raw"`
- `${target_col}` → expands to `"seg"`
- `${spatial_dims}` → expands to `3`

### Data Config Variables

#### For DataframeDatamodule (Approach 1):
```yaml
data:
  path: /path/to/csv              # CSV file path
  split_column: <column_name>     # CSV column for splits
  columns:                        # CSV columns with image paths
    - ${source_col}
    - ${target_col}
```

#### For RandomTimelapseDatamodule (Approach 2):
```yaml
data:
  path: /path/to/csv
  img_path_column: <column_name>    # CSV column with file paths
  split_column: <column_name>       # CSV column for splits
  extra_columns:                    # CSV columns to include in batch
    - <column_name_1>
    - <column_name_2>
```

### Transform Variables

#### Common Patterns:

**Load image with specific key:**
```yaml
- _target_: monai.transforms.LoadImaged
  keys: ${source_col}  # Loads into batch["raw"]
```

**Process multiple keys:**
```yaml
- _target_: monai.transforms.NormalizeIntensityd
  keys: [${source_col}, ${target_col}]  # Process both
```

**Load with BioIO:**
```yaml
- _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
  path_key: original_path        # Key in batch with file path
  out_key: ${source_col}         # Save loaded image to this key
  kwargs_keys:                   # CSV columns to use as BioImage kwargs
    - dimension_order_out
    - raw_channel
    - T
```

---

## Common Patterns

### Pattern 1: Changing Input/Output Column Names

**If your CSV uses different column names:**

CSV with `input` and `output`:
```csv
input,output,split
/data/file1_input.nd2,/data/file1_output.nd2,train
```

Update experiment config:
```yaml
source_col: input   # Changed from "raw"
target_col: output  # Changed from "seg"
```

### Pattern 2: Single File, Multiple Channels

**CSV:**
```csv
path,input_ch,output_ch,split
/data/exp.nd2,0,1,train
```

**Config:**
```yaml
source_col: raw
target_col: seg

data:
  extra_columns: [input_ch, output_ch]

  transforms:
    train:
      transforms:
        - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
          out_key: ${source_col}
          kwargs_keys: [input_ch]  # Use input_ch column value

        - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
          out_key: ${target_col}
          kwargs_keys: [output_ch]  # Use output_ch column value
```

### Pattern 3: Same Channel, Different Timepoints

**CSV:**
```csv
path,channel,input_time,output_time,split
/data/exp.nd2,0,0,1,train
```

**Config:**
```yaml
data:
  extra_columns: [channel, input_time, output_time]

  transforms:
    train:
      transforms:
        # Load input timepoint
        - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
          out_key: ${source_col}
          kwargs_keys: [channel, input_time]  # T=input_time

        # Load output timepoint (next frame)
        - _target_: cyto_dl.image.io.bioio_loader.BioIOImageLoaderd
          out_key: ${target_col}
          kwargs_keys: [channel, output_time]  # T=output_time
```

---

## Quick Reference Table

| You Want To... | CSV Column | Config Location | Config Variable |
|----------------|------------|-----------------|-----------------|
| Specify input file path | `raw` | `source_col: raw` + `columns: [${source_col}]` | Top-level + data |
| Specify output file path | `seg` | `target_col: seg` + `columns: [${target_col}]` | Top-level + data |
| Split train/val/test | `split` | `split_column: split` | data |
| Use different channels | `raw_ch`, `seg_ch` | `extra_columns: [raw_ch, seg_ch]` | data |
| Sample timepoint range | `start`, `stop` | `time_start_column: start`, `time_stop_column: stop` | data |
| Include metadata | `treatment`, `cell_id` | `extra_columns: [treatment, cell_id]` | data |
| Specify scene | `scene` | `scene_column: scene` | data |
| Specify resolution | `resolution` | `resolution_column: resolution` | data |

---

## Troubleshooting

### Error: "Key 'raw' not found in batch"
**Problem:** `source_col` doesn't match the key used in transforms
**Solution:** Ensure `out_key: ${source_col}` in loader transform

### Error: "Column 'raw' not found in CSV"
**Problem:** CSV column name doesn't match `source_col`
**Solution:** Either rename CSV column or update `source_col` in experiment config

### Error: "Missing key in data dictionary"
**Problem:** CSV column specified in `extra_columns` doesn't exist
**Solution:** Check CSV has all columns listed in `extra_columns`

---

## Examples Summary

**See these files for complete examples:**
- `configs/experiment/local/h2b_prediction_dataframe.yaml` - Separate files approach
- `configs/experiment/local/h2b_prediction_timelapse.yaml` - Timelapse approach
- `examples/h2b_dataframe_metadata.csv` - CSV for separate files
- `examples/h2b_timelapse_metadata.csv` - CSV for timelapse

**Run with:**
```bash
python cyto_dl/train.py experiment=local/h2b_prediction_dataframe
# or
python cyto_dl/train.py experiment=local/h2b_prediction_timelapse
```
