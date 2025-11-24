from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from monai.data import Dataset
from omegaconf import OmegaConf
from upath import UPath


class RandomSamplingTimelapseDataset(Dataset):
    """Dataset for randomly sampling timepoints from timelapse microscopy files.

    This dataset efficiently handles timelapse data by:
    1. NOT enumerating all timepoints at initialization (lazy loading)
    2. Randomly sampling N timepoints per file during training
    3. Supporting all bioio formats (nd2, czi, ome-zarr, zarr, tif stacks, etc.)
    4. Properly handling S3/cloud paths through upath
    5. Using numpy RNG for reproducible random sampling

    Unlike MultiDimImageDataset which creates a separate dataset entry for each timepoint
    (inefficient for large timelapses), this dataset creates one entry per file and
    randomly samples timepoints on-the-fly.
    """

    def __init__(
        self,
        csv_path: Optional[Union[Path, str]] = None,
        img_path_column: str = "path",
        channel_column: str = "channel",
        spatial_dims: int = 3,
        scene_column: str = "scene",
        resolution_column: str = "resolution",
        num_timepoints: int = 1,
        timepoint_sampling: str = "random",
        time_start_column: str = "start",
        time_stop_column: str = "stop",
        extra_columns: Sequence[str] = [],
        dict_meta: Optional[Dict] = None,
        transform: Optional[Union[Callable, Sequence[Callable]]] = None,
        seed: Optional[int] = None,
        samples_per_epoch: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        csv_path: Union[Path, str], optional
            Path to CSV file with metadata. Supports local and S3/cloud paths via upath.
        img_path_column: str
            Column containing path to microscopy file. Supports local, S3, and other cloud paths.
        channel_column: str
            Column containing channel index/indices to extract. Can be comma-separated for multiple channels.
        spatial_dims: int
            Spatial dimension of output image. Must be 2 for YX or 3 for ZYX.
        scene_column: str
            Column containing scene index/indices. If not present or -1, uses first scene.
        resolution_column: str
            Column containing resolution level. If not present, uses resolution 0.
        num_timepoints: int
            Number of timepoints to sample per file. Default: 1
        timepoint_sampling: str
            How to sample timepoints: 'random' (default), 'sequential', 'uniform'.
            - 'random': Randomly sample num_timepoints with replacement
            - 'sequential': Sample num_timepoints consecutive frames starting from random position
            - 'uniform': Sample num_timepoints evenly spaced across the timelapse
        time_start_column: str
            Column specifying starting timepoint (inclusive). If not present, starts at 0.
        time_stop_column: str
            Column specifying ending timepoint (exclusive). If not present, uses all timepoints.
        extra_columns: Sequence[str]
            Additional columns to include in output dictionary.
        dict_meta: Optional[Dict]
            Dictionary version of CSV file. If provided, csv_path is ignored.
        transform: Optional[Callable]
            MONAI-style transforms to apply. Should include BioIOImageLoaderd.
        seed: Optional[int]
            Random seed for reproducibility. If None, uses random state.
        samples_per_epoch: Optional[int]
            Number of samples per epoch. If None, uses len(dataframe).
            Useful for oversampling or undersampling during training.
        """
        super().__init__([])  # Initialize with empty data, we'll override __getitem__

        # Read metadata
        if csv_path is not None:
            csv_path = UPath(csv_path)  # Support S3/cloud paths
            self.df = pd.read_csv(csv_path)
        elif dict_meta is not None:
            self.df = pd.DataFrame(OmegaConf.to_container(dict_meta))
        else:
            raise ValueError("Must provide either csv_path or dict_meta")

        # Store configuration
        self.img_path_column = img_path_column
        self.channel_column = channel_column
        self.scene_column = scene_column
        self.resolution_column = resolution_column
        self.time_start_column = time_start_column
        self.time_stop_column = time_stop_column
        self.extra_columns = list(extra_columns)
        self.num_timepoints = num_timepoints
        self.timepoint_sampling = timepoint_sampling
        self.samples_per_epoch = samples_per_epoch or len(self.df)

        if spatial_dims not in (2, 3):
            raise ValueError(f"`spatial_dims` must be 2 or 3, got {spatial_dims}")
        self.spatial_dims = spatial_dims

        # Initialize RNG for reproducible random sampling
        self.rng = np.random.default_rng(seed)

        # Store transform
        self.transform = transform

        # Cache for file metadata (scenes, timepoints) to avoid reopening files
        self._file_metadata_cache = {}

    def _get_file_metadata(self, file_path: str, row_dict: Dict):
        """Get metadata for a file (number of timepoints, scenes) without loading full image.

        Uses bioio to open file and cache metadata. This is much more efficient than
        loading the full image.
        """
        if file_path in self._file_metadata_cache:
            return self._file_metadata_cache[file_path]

        # Import here to avoid loading bioio unless needed
        from bioio import BioImage

        # UPath handles S3, GCS, Azure, local paths transparently
        img = BioImage(UPath(file_path))

        # Get scene
        scene = row_dict.get(self.scene_column, -1)
        if scene == -1:
            scene = img.scenes[0] if img.scenes else 0
        else:
            # Handle comma-separated scenes (take first one)
            if isinstance(scene, str) and "," in scene:
                scene = scene.split(",")[0].strip()
            if scene not in img.scenes:
                raise ValueError(
                    f"Scene {scene} not found in {file_path}. Available: {img.scenes}"
                )

        # Set scene
        img.set_scene(scene)

        # Get timepoint range
        start = row_dict.get(self.time_start_column, 0)
        stop = row_dict.get(self.time_stop_column, -1)

        # If stop not specified, use all timepoints
        if stop == -1:
            stop = img.dims.T

        metadata = {
            "num_timepoints": img.dims.T,
            "num_scenes": len(img.scenes),
            "scenes": img.scenes,
            "scene": scene,
            "time_range": (start, stop),
        }

        # Cache it
        self._file_metadata_cache[file_path] = metadata
        return metadata

    def _sample_timepoints(self, start: int, stop: int, num_samples: int):
        """Sample timepoints based on sampling strategy.

        Parameters
        ----------
        start: int
            Starting timepoint (inclusive)
        stop: int
            Ending timepoint (exclusive)
        num_samples: int
            Number of timepoints to sample

        Returns
        -------
        List[int]
            List of sampled timepoint indices
        """
        available_timepoints = stop - start

        if num_samples > available_timepoints:
            # If requesting more samples than available, sample with replacement
            timepoints = self.rng.integers(start, stop, size=num_samples).tolist()
        else:
            if self.timepoint_sampling == "random":
                # Random sampling without replacement
                timepoints = self.rng.choice(
                    range(start, stop), size=num_samples, replace=False
                ).tolist()
            elif self.timepoint_sampling == "sequential":
                # Sample consecutive frames starting from random position
                max_start = stop - num_samples
                seq_start = self.rng.integers(start, max_start + 1)
                timepoints = list(range(seq_start, seq_start + num_samples))
            elif self.timepoint_sampling == "uniform":
                # Evenly spaced samples across the range
                indices = np.linspace(start, stop - 1, num_samples, dtype=int)
                timepoints = indices.tolist()
            else:
                raise ValueError(
                    f"Unknown timepoint_sampling: {self.timepoint_sampling}. "
                    f"Must be 'random', 'sequential', or 'uniform'."
                )

        return sorted(timepoints)

    def __len__(self):
        """Return number of samples per epoch (may differ from dataframe length)."""
        return self.samples_per_epoch

    def __getitem__(self, index):
        """Get a sample with randomly sampled timepoint(s).

        Returns a list of dictionaries (one per timepoint) if num_timepoints > 1,
        or a single dictionary if num_timepoints == 1.
        """
        # Map index to dataframe row (supports oversampling/undersampling)
        df_index = index % len(self.df)
        row = self.df.iloc[df_index].to_dict()

        # Get file path
        file_path = row[self.img_path_column]

        # Get file metadata
        metadata = self._get_file_metadata(file_path, row)
        start, stop = metadata["time_range"]

        # Sample timepoints
        timepoints = self._sample_timepoints(start, stop, self.num_timepoints)

        # Create sample dictionaries for each timepoint
        samples = []
        for timepoint in timepoints:
            sample = {
                "dimension_order_out": "C" + "ZYX"[-self.spatial_dims :],
                "C": row[self.channel_column],
                "scene": metadata["scene"],
                "T": timepoint,
                "original_path": file_path,
                "resolution": row.get(self.resolution_column, 0),
            }

            # Add extra columns
            for col in self.extra_columns:
                sample[col] = row.get(col)

            # Apply transforms if provided
            if self.transform is not None:
                sample = self.transform(sample)

            samples.append(sample)

        # If single timepoint, return single dict instead of list
        if self.num_timepoints == 1:
            return samples[0]

        return samples
