# Lazy Loading for Large Timelapse Datasets

## Overview

**Problem:** Loading large timelapse zarr datasets can cause memory exhaustion and server resets because each timepoint is loaded fully into memory when accessed, even if you only need a small crop.

**Solution:** Use lazy loading with dask arrays to defer computation until after spatial transforms are applied. This dramatically reduces memory usage by only loading the data you actually need.

## How It Works

### Without Lazy Loading (Default Behavior)
```python
# ❌ Memory intensive - loads entire timepoint
BioIOImageLoaderd(keys=["raw"], dask_load=True)
# → Returns 2048x2048x100 array (8GB) immediately in RAM
```

### With Lazy Loading
```python
# ✅ Memory efficient - delays loading
BioIOImageLoaderd(keys=["raw"], dask_load=True, lazy_load=True)
# → Returns dask array (metadata only, ~KB)

RandSpatialCropd(keys=["raw"], roi_size=[64, 64, 64])
# → Updates dask array metadata (still no data loaded)

ComputeDaskd(keys=["raw"], dtype="float16")
# → NOW loads only 64x64x64 crop (32MB) into RAM
```

**Memory savings:** 8GB → 32MB (250x reduction!)

## Usage

### 1. For BioIO Loaders (Recommended for Timelapse)

```python
from monai.transforms import Compose, RandSpatialCropd, RandFlipd
from cyto_dl.image.io import BioIOImageLoaderd
from cyto_dl.image.transforms import ComputeDaskd

transforms = Compose([
    # Step 1: Load as dask array (no computation yet)
    BioIOImageLoaderd(
        keys=["raw"],
        path_key="original_path",
        dask_load=True,
        lazy_load=True,  # ← Enable lazy loading
        dtype="float16",
    ),

    # Step 2: Apply spatial transforms (dask-aware, no data loaded yet)
    RandSpatialCropd(keys=["raw"], roi_size=[64, 64, 64], random_size=False),

    # Step 3: Compute the cropped region (only loads 64x64x64)
    ComputeDaskd(keys=["raw"], dtype="float16"),

    # Step 4: Apply remaining transforms (now working with numpy arrays)
    RandFlipd(keys=["raw"], prob=0.5),
    # ... more transforms
])
```

### 2. For OME-Zarr Direct Access

```python
from cyto_dl.image.io import OmeZarrReader
from cyto_dl.image.transforms import ComputeDaskd

reader = OmeZarrReader(
    level=0,
    lazy_load=True,  # ← Enable lazy loading
)

transforms = Compose([
    LoadImageD(keys=["raw"], reader=reader),
    RandSpatialCropd(keys=["raw"], roi_size=[128, 128, 128]),
    ComputeDaskd(keys=["raw"], dtype="float16"),
    # ... more transforms
])
```

### 3. Complete Configuration Example

Here's a complete datamodule configuration with lazy loading:

```yaml
# config/data/timelapse.yaml
data:
  _target_: cyto_dl.datamodules.multidim_image.MultiDimImageDataset
  csv_path: /path/to/timelapse_manifest.csv
  spatial_dims: 3
  transform:
    - _target_: cyto_dl.image.io.BioIOImageLoaderd
      keys: [raw]
      path_key: original_path
      dask_load: true
      lazy_load: true  # ← Enable lazy loading
      dtype: float16

    # Apply spatial transforms while still lazy
    - _target_: monai.transforms.RandSpatialCropd
      keys: [raw]
      roi_size: [64, 64, 64]
      random_size: false

    # Compute after cropping
    - _target_: cyto_dl.image.transforms.ComputeDaskd
      keys: [raw]
      dtype: float16

    # Rest of your transforms
    - _target_: monai.transforms.RandFlipd
      keys: [raw]
      prob: 0.5
```

## When to Use Lazy Loading

### ✅ Use Lazy Loading When:
- Working with **large timelapse datasets** (multiple timepoints, each >1GB)
- Loading from **zarr/OME-Zarr** files with chunking
- Using **spatial crops** (only need small region of large image)
- Experiencing **memory issues or server resets** during training
- Working with **multi-resolution** data (loading lower resolution levels)

### ❌ Don't Use Lazy Loading When:
- Images are **small** (already fit in memory)
- You need the **entire image** (no cropping)
- Using **non-dask** data sources (e.g., TIFF files)
- Applying transforms that need **full image statistics** (e.g., normalization based on entire image)

## Performance Comparison

### Example: Training on 100-timepoint timelapse (2GB per timepoint)

| Approach | Memory per Sample | Workers | Total RAM | Result |
|----------|------------------|---------|-----------|--------|
| **No lazy load** | 2GB | 8 | 16GB | ❌ OOM crash |
| **Lazy + crop 64³** | 32MB | 8 | 256MB | ✅ Works! |
| **Lazy + crop 128³** | 256MB | 8 | 2GB | ✅ Works! |

## Important Notes

### 1. Transform Order Matters
```python
# ✅ CORRECT: Crop before compute
BioIOImageLoaderd(lazy_load=True)
RandSpatialCropd(...)  # Dask-aware
ComputeDaskd(...)      # Compute cropped region
RandFlipd(...)         # Numpy-based

# ❌ WRONG: Compute before crop
BioIOImageLoaderd(lazy_load=True)
ComputeDaskd(...)      # Loads entire image!
RandSpatialCropd(...)  # Crops already-loaded data
```

### 2. Dask-Aware Transforms
Most MONAI spatial transforms are dask-aware and will work with lazy arrays:
- ✅ `RandSpatialCropd` - updates chunk metadata
- ✅ `CenterSpatialCropd` - updates chunk metadata
- ✅ `Resized` - works with dask
- ⚠️ `Normalizationd` - may compute if needs full stats

### 3. Dtype Conversion
When using lazy loading, dtype conversion happens in `ComputeDaskd`:
```python
BioIOImageLoaderd(lazy_load=True, dtype="float16")  # ← Ignored when lazy
ComputeDaskd(dtype="float16")  # ← Actual conversion happens here
```

### 4. Debugging Lazy Loading

Check if data is still lazy:
```python
def check_lazy(data):
    for key, value in data.items():
        if hasattr(value, "compute"):
            print(f"{key}: Still lazy (dask array)")
        else:
            print(f"{key}: Computed (numpy array)")
```

## Troubleshooting

### Issue: "Transforms fail with dask arrays"
**Solution:** Make sure `ComputeDaskd` is placed before transforms that require numpy arrays.

### Issue: "Still running out of memory"
**Solutions:**
1. Reduce crop size further
2. Reduce batch size
3. Reduce number of workers
4. Use lower resolution level: `OmeZarrReader(level=1, lazy_load=True)`
5. Install `zarrs-python` for faster I/O: `pip install zarrs-python`

### Issue: "Loading is slow"
**Solutions:**
1. Check zarr chunk size matches crop size
2. Install `zarrs-python` for rust-based acceleration
3. Use local storage instead of network storage for training
4. Increase dask worker threads: `dask.config.set(num_workers=8)`

### Issue: "Data corruption or server resets persist"
**Solutions:**
1. Ensure you're using `zarr>=3.0.9` (versions 3.0.0-3.0.8 have data corruption bug)
2. Update `anndata>=0.12.0` for zarr 3.x compatibility
3. Check zarr version: `python -c "import zarr; print(zarr.__version__)"`

## Advanced: Custom Compute Strategy

For advanced users, you can create custom compute strategies:

```python
class PartialComputeDaskd(Transform):
    """Compute only specific chunks based on ROI."""

    def __init__(self, keys, roi_key="roi"):
        self.keys = keys
        self.roi_key = roi_key

    def __call__(self, data):
        roi = data[self.roi_key]
        for key in self.keys:
            if hasattr(data[key], "compute"):
                # Slice dask array to ROI, then compute
                sliced = data[key][roi]
                data[key] = sliced.compute()
        return data
```

## Related Documentation
- [ZARR3_COMPATIBILITY.md](ZARR3_COMPATIBILITY.md) - Zarr 3.x compatibility information
- [BioIO Documentation](https://bioio-devs.github.io/bioio/OVERVIEW.html)
- [Dask Arrays](https://docs.dask.org/en/latest/array.html)

## Summary

**Key takeaway:** For large timelapse datasets, enable `lazy_load=True` in your loader and add `ComputeDaskd` after spatial transforms. This prevents loading entire timepoints into memory and dramatically reduces RAM usage.

```python
# The magic three-step pattern:
BioIOImageLoaderd(lazy_load=True)     # 1. Load metadata only
RandSpatialCropd(roi_size=[64,64,64]) # 2. Define what you need
ComputeDaskd()                         # 3. Load only what you need
```
