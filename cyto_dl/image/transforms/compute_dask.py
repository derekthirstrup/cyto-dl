"""Transform to compute dask arrays at a controlled point in the pipeline."""

import numpy as np
from monai.data import MetaTensor
from monai.transforms import Transform

from cyto_dl.utils.arg_checking import get_dtype


class ComputeDaskd(Transform):
    """Dictionary-based transform to compute dask arrays into numpy arrays.

    This transform should be used in conjunction with lazy_load=True in BioIOImageLoaderd
    or OmeZarrReader. It computes dask arrays at a controlled point in the transform
    pipeline, allowing you to apply spatial transforms (crops, etc.) to the dask array
    metadata before actually loading data into memory.

    Key benefits:
    - Reduced memory usage for large timelapse datasets
    - Only loads data that will actually be used after cropping/transforms
    - Better performance with chunked zarr data

    Example usage:
        transforms = Compose([
            BioIOImageLoaderd(keys=["raw"], lazy_load=True),  # Returns dask array
            RandSpatialCropd(keys=["raw"], roi_size=[64, 64, 64]),  # Dask-aware crop
            ComputeDaskd(keys=["raw"], dtype="float16"),  # Compute here
            # ... rest of transforms that need numpy arrays
        ])
    """

    def __init__(
        self,
        keys,
        dtype: np.dtype = np.float16,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys : list of str
            Keys in the data dictionary to compute
        dtype : np.dtype, default=np.float16
            Data type to cast the computed array to
        allow_missing_keys : bool, default=False
            Whether to allow missing keys in the data dictionary
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.dtype = get_dtype(dtype)
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
        """Compute dask arrays for specified keys."""
        data = data.copy()

        for key in self.keys:
            if key not in data:
                if not self.allow_missing_keys:
                    raise KeyError(f"Key {key} not found in data dictionary")
                continue

            value = data[key]
            meta_key = f"{key}_meta"

            # Check if it's a dask array
            if hasattr(value, "compute"):
                # It's a dask array, compute it
                computed = value.compute()

                # Convert to target dtype
                computed = computed.astype(self.dtype)

                # If original was MetaTensor, preserve metadata
                if isinstance(value, MetaTensor):
                    data[key] = MetaTensor(computed, meta=value.meta)
                # If metadata was stored separately (from lazy loading), use it
                elif meta_key in data:
                    meta = data[meta_key]
                    data[key] = MetaTensor(computed, meta=meta)
                    # Clean up the metadata key
                    del data[meta_key]
                else:
                    # No metadata available, wrap in MetaTensor with empty meta
                    data[key] = MetaTensor(computed, meta={})

            elif isinstance(value, MetaTensor):
                # It's already computed MetaTensor, just ensure dtype
                if value.dtype != self.dtype:
                    data[key] = MetaTensor(value.astype(self.dtype), meta=value.meta)
            else:
                # It's a regular numpy array
                # Check if metadata was stored separately
                if meta_key in data:
                    meta = data[meta_key]
                    if value.dtype != self.dtype:
                        value = value.astype(self.dtype)
                    data[key] = MetaTensor(value, meta=meta)
                    del data[meta_key]
                elif value.dtype != self.dtype:
                    data[key] = value.astype(self.dtype)

        return data
