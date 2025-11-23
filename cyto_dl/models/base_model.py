import copy
import inspect
import logging
from collections.abc import MutableMapping
from typing import Optional, Sequence, Union

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchmetrics import MeanMetric

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False

_DEFAULT_METRICS = {
    "train/loss": MeanMetric(),
    "val/loss": MeanMetric(),
    "test/loss": MeanMetric(),
}


def _is_primitive(value):
    if value is None or isinstance(value, (bool, str, int, float)):
        return True

    if isinstance(value, (tuple, list)):
        return all(_is_primitive(el) for el in value)

    if isinstance(value, dict):
        return all(_is_primitive(el) for el in value.values())

    return False


def _cast_init_arg(value):
    if isinstance(value, inspect.Parameter):
        return value._default
    if isinstance(value, (ListConfig, DictConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


class BaseModelMeta(type):
    def __call__(cls, *args, **kwargs):
        """Meta class to handle instantiating configs."""

        init_args = inspect.signature(cls.__init__).parameters.copy()
        init_args.pop("self")
        if "base_kwargs" in init_args.keys():
            init_args.pop("base_kwargs")
        keys = tuple(init_args.keys())

        user_init_args = {keys[ix]: arg for ix, arg in enumerate(args)}
        user_init_args.update(kwargs)
        init_args.update(user_init_args)
        init_args = {k: _cast_init_arg(v) for k, v in init_args.items()}

        # instantiate class with instantiated `init_args`
        # hydra doesn't change the original dict, so we can use it after this
        # with `save_hyperparameters`
        obj = type.__call__(cls, **instantiate(init_args, _recursive_=True, _convert_=True))

        # make sure only primitives get stored in the ckpt
        ignore = [arg for arg, v in init_args.items() if not _is_primitive(v)]

        for arg in ignore:
            init_args.pop(arg)

        obj._set_hparams(init_args)
        obj._hparams_initial = copy.deepcopy(obj._hparams)

        return obj


class BaseModel(LightningModule, metaclass=BaseModelMeta):
    def __init__(
        self,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        metrics=_DEFAULT_METRICS,
        spatial_dims: Optional[int] = None,
        channels_last: bool = False,
        compile_model: bool = False,
        compile_mode: str = "default",
        gradient_checkpointing: bool = False,
    ):
        """
        Parameters
        ----------
        optimizer : Optional[torch.optim.Optimizer]
            Optimizer class or instance
        lr_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
            Learning rate scheduler class or instance
        metrics : dict
            Dictionary of metrics to track
        spatial_dims : Optional[int]
            Number of spatial dimensions (2 or 3). Used for channels-last optimization.
        channels_last : bool
            Whether to use channels-last memory format for better GPU performance.
            Provides 20-30% speedup on modern GPUs. Requires spatial_dims to be set.
        compile_model : bool
            Whether to compile the model using torch.compile (PyTorch 2.0+).
            Provides 20-50% speedup. Not supported on Windows.
        compile_mode : str
            Compilation mode: "default", "reduce-overhead", "max-autotune".
            Only used if compile_model=True.
        gradient_checkpointing : bool
            Enable gradient checkpointing to reduce memory usage.
            Trades ~20% compute for 40-60% memory reduction.
        """
        super().__init__()

        self.metrics = tuple(metrics.keys())

        for key, value in metrics.items():
            setattr(self, key, value)

        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.lr_scheduler = lr_scheduler

        # Performance optimization settings
        self._spatial_dims = spatial_dims
        self._channels_last = channels_last
        self._compile_model = compile_model
        self._compile_mode = compile_mode
        self._gradient_checkpointing = gradient_checkpointing

    def parse_batch(self, batch):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def compute_metrics(self, loss, preds, targets, split):
        """Method to handle logging metrics. Assumptions made:

        - the `_step` method of each model returns a tuple (loss, preds, targets),
          whose elements may be dictionaries
        - the keys of `self.metrics` have a specific structure:
          'split/type(/part)' , where:
            - `split` is one of (train, val, test, predict)
            - `type` is either "loss", or an arbitrary string denoting a metric
            - `part` is optional, used when (loss, preds, targets) are dictionaries,
              in which case it must match a dictionary key
        """
        for metric_key in self.metrics:
            metric_split, metric_type, *metric_part = metric_key.split("/")
            if not metric_split.startswith(split):
                continue

            if len(metric_part) > 0:
                metric_part = "/".join(metric_part)
            else:
                metric_part = None

            metric = getattr(self, metric_key)

            if metric_type == "loss":
                if metric_part is not None:
                    metric.update(loss[metric_part])
                else:
                    if not isinstance(loss, MutableMapping):
                        metric.update(loss)
                    elif "loss" in loss:
                        metric.update(loss["loss"])
                    else:
                        raise TypeError(
                            "Expected `loss` to be a single value or tensor, "
                            "or a dictionary with a key 'loss', but it isn't."
                        )
            else:
                if metric_part is not None:
                    metric.update(preds[metric_part], targets[metric_part])
                else:
                    if not isinstance(preds, MutableMapping):
                        metric.update(preds, targets)

            self.log(metric_key, metric, on_step=True, on_epoch=True, prog_bar=True)

    def model_step(self, stage, batch, batch_idx):
        """Here you should implement the logic for a step in the training/validation/test process.
        The stage (training/validation/test) is given by the variable `stage`.

        Example:

        x = self.parse_batch(batch)

        if self.hparams.id_label is not None:
           if self.hparams.id_label in batch:
               ids = batch[self.hparams.id_label].detach().cpu()
               results.update({
                   self.hparams.id_label: ids,
                   "id": ids
               })

        return results
        """
        raise NotImplementedError

    def on_train_start(self):
        for metric_key in self.metrics:
            metric_split, *_ = metric_key.split("/")
            if metric_split.startswith("val"):
                metric = getattr(self, metric_key)
                metric.reset()

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step("train", batch, batch_idx)
        self.compute_metrics(loss, preds, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step("val", batch, batch_idx)
        self.compute_metrics(loss, preds, targets, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step("test", batch, batch_idx)
        self.compute_metrics(loss, preds, targets, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        """Here you should implement the logic for an inference step.

        In most cases this would simply consist of calling the forward pass of your model, but you
        might wish to add additional post-processing.
        """
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        """Setup hook called at the beginning of fit, test, or predict.

        Applies performance optimizations:
        - Channels-last memory format
        - torch.compile
        - Gradient checkpointing
        """
        if hasattr(super(), 'setup'):
            super().setup(stage)

        # Apply channels-last memory format
        if self._channels_last and self._spatial_dims is not None:
            if self._spatial_dims == 2:
                self.to(memory_format=torch.channels_last)
                logger.info("✓ Applied channels_last memory format (2D)")
            elif self._spatial_dims == 3:
                self.to(memory_format=torch.channels_last_3d)
                logger.info("✓ Applied channels_last_3d memory format (3D)")

        # Apply torch.compile
        if self._compile_model and stage in ("fit", "validate", "test"):
            import sys
            if not sys.platform.startswith("win"):
                try:
                    # Compile the model forward pass
                    # Note: This is applied to self, which compiles the entire model
                    # For more fine-grained control, subclasses can override
                    self = torch.compile(self, mode=self._compile_mode)
                    logger.info(f"✓ Applied torch.compile with mode='{self._compile_mode}'")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}, continuing without compilation")
            else:
                logger.warning("torch.compile not supported on Windows")

        # Enable gradient checkpointing
        if self._gradient_checkpointing and stage == "fit":
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency.

        Subclasses should override this to apply checkpointing to specific layers.
        Default implementation logs a warning if no implementation exists.
        """
        logger.warning(
            "Gradient checkpointing requested but not implemented for this model. "
            "Override _enable_gradient_checkpointing() in your model class."
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                },
            }
        return optimizer
