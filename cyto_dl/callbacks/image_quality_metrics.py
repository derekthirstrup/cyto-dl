"""Callback to log image quality metrics (SSIM, PSNR, Pearson) during training.

Works in conjunction with MetricTrackingLoss to extract computed metrics and log them via
Lightning's logging system.
"""

from lightning.pytorch.callbacks import Callback


class ImageQualityMetrics(Callback):
    """Reads metrics from MetricTrackingLoss on task heads and logs them.

    Parameters
    ----------
    stages : list
        Which stages to log metrics for. Default: ["val"]
    """

    def __init__(self, stages=None):
        self.stages = stages or ["val"]
        # Running averages per epoch
        self._epoch_metrics = {}

    def _reset_epoch_metrics(self):
        self._epoch_metrics = {}

    def _collect_metrics(self, pl_module, stage):
        """Read metrics from MetricTrackingLoss on each task head."""
        if not hasattr(pl_module, "task_heads"):
            return

        for head_name, head in pl_module.task_heads.items():
            loss_fn = getattr(head, "loss", None)
            if loss_fn is None:
                continue

            # Handle nested wrappers
            if hasattr(loss_fn, "base_loss"):
                tracker = loss_fn
            else:
                continue

            if tracker.last_ssim is None:
                continue

            key_prefix = f"{stage}/{head_name}"
            if key_prefix not in self._epoch_metrics:
                self._epoch_metrics[key_prefix] = {
                    "ssim": [],
                    "psnr": [],
                    "pearson": [],
                }

            self._epoch_metrics[key_prefix]["ssim"].append(tracker.last_ssim)
            self._epoch_metrics[key_prefix]["psnr"].append(tracker.last_psnr)
            self._epoch_metrics[key_prefix]["pearson"].append(tracker.last_pearson)

            # Log per-step values
            pl_module.log(f"{stage}/ssim/{head_name}", tracker.last_ssim, prog_bar=False)
            pl_module.log(f"{stage}/psnr/{head_name}", tracker.last_psnr, prog_bar=False)
            pl_module.log(f"{stage}/pearson/{head_name}", tracker.last_pearson, prog_bar=False)

    def _log_epoch_averages(self, pl_module, stage):
        """Log epoch-averaged metrics."""
        cross_head_avgs = {"ssim": [], "psnr": [], "pearson": []}
        for key_prefix, metrics in self._epoch_metrics.items():
            if not key_prefix.startswith(stage):
                continue
            for metric_name, values in metrics.items():
                if not values:
                    continue
                avg = sum(values) / len(values)
                head_name = key_prefix.split("/", 1)[1]
                pl_module.log(
                    f"{stage}/{metric_name}_epoch/{head_name}",
                    avg,
                    prog_bar=(metric_name == "ssim"),
                )
                cross_head_avgs.setdefault(metric_name, []).append(avg)

        # Cross-head average — useful for single-task convenience and avoids
        # the per-head value being overwritten when multiple heads contribute.
        for metric_name, avgs in cross_head_avgs.items():
            if avgs:
                pl_module.log(f"{stage}/{metric_name}_epoch", sum(avgs) / len(avgs))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "train" in self.stages:
            self._collect_metrics(pl_module, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if "val" in self.stages:
            self._collect_metrics(pl_module, "val")

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val" in self.stages:
            self._log_epoch_averages(pl_module, "val")
            # Reset for next epoch
            self._epoch_metrics = {
                k: v for k, v in self._epoch_metrics.items() if not k.startswith("val")
            }

    def on_train_epoch_end(self, trainer, pl_module):
        if "train" in self.stages:
            self._log_epoch_averages(pl_module, "train")
            self._epoch_metrics = {
                k: v for k, v in self._epoch_metrics.items() if not k.startswith("train")
            }

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if "test" in self.stages:
            self._collect_metrics(pl_module, "test")

    def on_test_epoch_end(self, trainer, pl_module):
        if "test" in self.stages:
            self._log_epoch_averages(pl_module, "test")
