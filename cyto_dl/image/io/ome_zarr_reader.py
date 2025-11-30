from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _stack_images
from monai.utils import ensure_tuple, require_pkg
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from upath import UPath as Path


@require_pkg(pkg_name="upath")
@require_pkg(pkg_name="ome_zarr")
class OmeZarrReader(ImageReader):
    def __init__(self, level=0, image_name="default", channels=None, lazy_load=False):
        """
        Parameters
        ----------
        level : int
            Resolution level to read from multi-resolution zarr
        image_name : str
            Name of the image within the zarr store
        channels : list, optional
            List of channel indices or names to extract
        lazy_load : bool
            If True, returns dask arrays without computing (saves memory).
            If False, computes arrays immediately (default, original behavior).
            When True, you must add ComputeDaskd transform later in your pipeline.
        """
        super().__init__()
        self.level = level
        self.image_name = image_name
        self.channels = channels
        self.lazy_load = lazy_load

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for path in filenames:
            if self.image_name:
                path = str(Path(path) / self.image_name)

            reader = Reader(parse_url(path))
            node = next(iter(reader()))
            img_.append(node)

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []

        for img_obj in ensure_tuple(img):
            # Get dask array from zarr store
            data = img_obj.data[self.level]

            # Remove time dimension (first dimension) - select first timepoint
            # Note: For timelapse data, time selection should be done via BioImage, not here
            if data.ndim > 0:
                data = data[0]

            if self.channels:
                _metadata_channels = img_obj.metadata["name"]
                _channels = [
                    ch if isinstance(ch, int) else _metadata_channels.index(ch)
                    for ch in self.channels
                ]
                data = data[_channels]

            # Only compute if not lazy loading
            if not self.lazy_load:
                data = data.compute()

            img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        for fname in ensure_tuple(filename):
            if not str(fname).endswith("zarr"):
                return False
        return True
