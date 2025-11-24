from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
from lightning import LightningDataModule
from monai.data import DataLoader
from upath import UPath

from .random_timelapse import RandomSamplingTimelapseDataset


class RandomTimelapseDatamodule(LightningDataModule):
    """A PyTorch Lightning datamodule for efficiently training on timelapse microscopy data.

    This datamodule enables efficient random sampling of timepoints from large timelapse
    microscopy files (nd2, czi, ome-zarr, etc.) without requiring conversion to TIF stacks.

    Key features:
    - Lazy loading: doesn't enumerate all timepoints at initialization
    - Random timepoint sampling: uses RNG for reproducible random sampling
    - Multi-format support: works with any bioio-supported format (nd2, czi, ome-zarr, tif, etc.)
    - Cloud storage: supports S3, GCS, Azure blob storage via upath
    - Efficient caching: caches file metadata to avoid repeated file opens

    Unlike MultiDimImageDataset which creates a dataset entry for each timepoint,
    this datamodule creates one entry per file and samples timepoints on-the-fly.
    """

    def __init__(
        self,
        path: Union[UPath, Path, str, Dict],
        transforms: Dict,
        img_path_column: str = "path",
        channel_column: str = "channel",
        spatial_dims: int = 3,
        scene_column: str = "scene",
        resolution_column: str = "resolution",
        num_timepoints: int = 1,
        timepoint_sampling: str = "random",
        time_start_column: str = "start",
        time_stop_column: str = "stop",
        split_column: Optional[str] = None,
        split_map: Optional[Dict] = None,
        extra_columns: Sequence[str] = [],
        samples_per_epoch: Optional[Dict[str, int]] = None,
        seed: int = 42,
        **dataloader_kwargs,
    ):
        """
        Parameters
        ----------
        path: Union[Path, str, Dict]
            Path to CSV file, or dictionary of paths per split {"train": "train.csv", "val": "val.csv"},
            or inline dict_meta for single dataset. Supports local and S3/cloud paths.

        transforms: Dict
            Transforms for each split. Should include BioIOImageLoaderd.
            Format: {"train": [...], "val": [...], "test": [...]}

        img_path_column: str
            Column containing microscopy file paths. Supports bioio formats and cloud paths.

        channel_column: str
            Column containing channel indices (can be comma-separated).

        spatial_dims: int
            Output spatial dimensions (2 for YX, 3 for ZYX).

        scene_column: str
            Column containing scene index. If not present, uses first scene.

        resolution_column: str
            Column containing resolution level. If not present, uses level 0.

        num_timepoints: int
            Number of timepoints to sample per file per epoch.

        timepoint_sampling: str
            Sampling strategy: 'random' (default), 'sequential', or 'uniform'.
            - 'random': Sample randomly with replacement
            - 'sequential': Sample consecutive frames from random start
            - 'uniform': Sample evenly spaced frames

        time_start_column: str
            Column specifying timepoint range start. If not present, starts at 0.

        time_stop_column: str
            Column specifying timepoint range end. If not present, uses all frames.

        split_column: Optional[str]
            Column name for train/val/test split. Required if path is a single CSV.

        split_map: Optional[Dict]
            Map split values to canonical names: {"training": "train", "validation": "val"}

        extra_columns: Sequence[str]
            Additional columns to include in sample dictionaries.

        samples_per_epoch: Optional[Dict[str, int]]
            Number of samples per epoch per split. If None, uses dataframe length.
            Example: {"train": 1000, "val": 100} for oversampling/undersampling.

        seed: int
            Random seed for reproducibility.

        dataloader_kwargs:
            Additional arguments for DataLoader (num_workers, batch_size, etc.)
        """
        super().__init__()
        torch.manual_seed(seed)
        self.seed = seed

        # Store config
        self.path = path
        self.img_path_column = img_path_column
        self.channel_column = channel_column
        self.spatial_dims = spatial_dims
        self.scene_column = scene_column
        self.resolution_column = resolution_column
        self.num_timepoints = num_timepoints
        self.timepoint_sampling = timepoint_sampling
        self.time_start_column = time_start_column
        self.time_stop_column = time_stop_column
        self.split_column = split_column
        self.split_map = split_map or {}
        self.extra_columns = extra_columns
        self.samples_per_epoch = samples_per_epoch or {}
        self.dataloader_kwargs = dataloader_kwargs

        # Parse transforms
        self.transforms = self._parse_transforms(transforms)

        # Create datasets for each split
        self.datasets = self._create_datasets()

        # Initialize RNG
        self.rng = np.random.default_rng(seed=seed)

        # Store dataloaders
        self.dataloaders = {}

    def _parse_transforms(self, transforms):
        """Parse transform configuration to ensure all splits are covered."""
        if not isinstance(transforms, dict):
            # If single transform provided, use for all splits
            return {
                "train": transforms,
                "val": transforms,
                "test": transforms,
                "predict": transforms,
            }

        # Fill in missing splits with train transforms
        parsed = {}
        train_transform = transforms.get("train", transforms.get("val", None))

        for split in ["train", "val", "test", "predict"]:
            # Try canonical name, then alias
            if split in transforms:
                parsed[split] = transforms[split]
            elif split == "val" and "valid" in transforms:
                parsed[split] = transforms["valid"]
            else:
                # Default to train transform
                parsed[split] = train_transform

        return parsed

    def _create_datasets(self):
        """Create datasets for each split."""
        datasets = {}

        if isinstance(self.path, dict):
            # Multiple CSV files, one per split
            for split, csv_path in self.path.items():
                # Normalize split name
                canonical_split = self._get_canonical_split_name(split)
                datasets[canonical_split] = self._create_dataset(
                    csv_path=csv_path,
                    transform=self.transforms[canonical_split],
                    split=canonical_split,
                )
        else:
            # Single CSV with split column
            if self.split_column is None:
                raise ValueError(
                    "When using a single CSV, split_column must be specified."
                )

            # Load CSV once and filter by split
            import pandas as pd

            csv_path = UPath(self.path)
            df = pd.read_csv(csv_path)

            for split in ["train", "val", "test"]:
                # Get rows for this split
                split_values = self._get_split_values(split)
                split_df = df[df[self.split_column].isin(split_values)]

                if len(split_df) > 0:
                    datasets[split] = self._create_dataset(
                        dict_meta=split_df.to_dict("list"),
                        transform=self.transforms[split],
                        split=split,
                    )

            # Predict uses all data
            datasets["predict"] = self._create_dataset(
                dict_meta=df.to_dict("list"),
                transform=self.transforms["predict"],
                split="predict",
            )

        return datasets

    def _create_dataset(
        self,
        csv_path: Optional[str] = None,
        dict_meta: Optional[Dict] = None,
        transform: Optional[Callable] = None,
        split: str = "train",
    ):
        """Create a single RandomSamplingTimelapseDataset."""
        return RandomSamplingTimelapseDataset(
            csv_path=csv_path,
            dict_meta=dict_meta,
            img_path_column=self.img_path_column,
            channel_column=self.channel_column,
            spatial_dims=self.spatial_dims,
            scene_column=self.scene_column,
            resolution_column=self.resolution_column,
            num_timepoints=self.num_timepoints,
            timepoint_sampling=self.timepoint_sampling,
            time_start_column=self.time_start_column,
            time_stop_column=self.time_stop_column,
            extra_columns=self.extra_columns,
            transform=transform,
            seed=self.seed,
            samples_per_epoch=self.samples_per_epoch.get(split),
        )

    def _get_canonical_split_name(self, split: str) -> str:
        """Convert split name to canonical form (train/val/test/predict)."""
        # Apply user mapping first
        split = self.split_map.get(split, split)

        # Normalize common aliases
        aliases = {
            "training": "train",
            "validation": "val",
            "valid": "val",
            "testing": "test",
        }
        return aliases.get(split.lower(), split.lower())

    def _get_split_values(self, split: str):
        """Get all values in split column that map to this split."""
        values = [split]

        # Add aliases
        if split == "train":
            values.extend(["training", "Train", "TRAIN"])
        elif split == "val":
            values.extend(["valid", "validation", "Val", "VALID", "VALIDATION"])
        elif split == "test":
            values.extend(["testing", "Test", "TESTING"])

        # Apply inverse split_map (find original values that map to this split)
        for orig, mapped in self.split_map.items():
            if mapped == split:
                values.append(orig)

        return values

    def _make_dataloader(self, split: str):
        """Create dataloader for a split."""
        kwargs = {**self.dataloader_kwargs}
        kwargs["shuffle"] = kwargs.get("shuffle", True) and split == "train"

        dataset = self.datasets.get(split)
        if dataset is None:
            raise ValueError(f"No dataset found for split: {split}")

        return DataLoader(dataset=dataset, **kwargs)

    def train_dataloader(self):
        """Return training dataloader."""
        if "train" not in self.datasets:
            raise ValueError("No train split found in datasets")
        if "train" not in self.dataloaders:
            self.dataloaders["train"] = self._make_dataloader("train")
        return self.dataloaders["train"]

    def val_dataloader(self):
        """Return validation dataloader."""
        if "val" not in self.datasets:
            raise ValueError("No val split found in datasets")
        if "val" not in self.dataloaders:
            self.dataloaders["val"] = self._make_dataloader("val")
        return self.dataloaders["val"]

    def test_dataloader(self):
        """Return test dataloader."""
        if "test" not in self.datasets:
            raise ValueError("No test split found in datasets")
        if "test" not in self.dataloaders:
            self.dataloaders["test"] = self._make_dataloader("test")
        return self.dataloaders["test"]

    def predict_dataloader(self):
        """Return prediction dataloader."""
        if "predict" not in self.datasets:
            raise ValueError("No predict split found in datasets")
        if "predict" not in self.dataloaders:
            self.dataloaders["predict"] = self._make_dataloader("predict")
        return self.dataloaders["predict"]
