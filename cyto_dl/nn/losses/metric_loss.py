"""Loss wrapper that computes image quality metrics alongside the primary loss.

Used by the autoresearch system to track SSIM, PSNR, and Pearson correlation
during training/validation without modifying the model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ssim_3d(pred, target, window_size=7, data_range=None):
    """Compute SSIM for 3D volumes (or 2D images).

    Simplified SSIM that works on arbitrary spatial dims by computing
    statistics over all spatial dimensions.
    """
    if data_range is None:
        data_range = max(target.max() - target.min(), 1e-8)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_pred = pred.mean()
    mu_target = target.mean()

    sigma_pred_sq = ((pred - mu_pred) ** 2).mean()
    sigma_target_sq = ((target - mu_target) ** 2).mean()
    sigma_cross = ((pred - mu_pred) * (target - mu_target)).mean()

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / (
        (mu_pred**2 + mu_target**2 + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )
    return ssim.clamp(-1, 1)


def _pearson_corr(pred, target):
    """Compute Pearson correlation coefficient."""
    pred_flat = pred.flatten().float()
    target_flat = target.flatten().float()

    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()

    num = (pred_centered * target_centered).sum()
    denom = torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum() + 1e-8)
    return (num / denom).clamp(-1, 1)


def _psnr(mse_val, data_range=None, pred=None, target=None):
    """Compute PSNR from MSE value."""
    if data_range is None and pred is not None and target is not None:
        data_range = max(target.max().item() - target.min().item(), 1e-8)
    elif data_range is None:
        data_range = 1.0

    if mse_val <= 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(torch.tensor(data_range**2 / max(mse_val, 1e-10)))


class MetricTrackingLoss(nn.Module):
    """Wraps a base loss function and computes SSIM, PSNR, Pearson as side effects.

    Metrics are stored as attributes and read by the ImageQualityMetrics callback.

    Parameters
    ----------
    base_loss : nn.Module
        The primary loss function (e.g., MSELoss, L1Loss)
    """

    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.last_ssim = None
        self.last_psnr = None
        self.last_pearson = None
        self.last_mse = None

    def forward(self, y_hat, y):
        loss = self.base_loss(y_hat, y)

        with torch.no_grad():
            mse_val = F.mse_loss(y_hat, y).item()
            self.last_mse = mse_val
            self.last_ssim = _ssim_3d(y_hat, y).item()
            self.last_pearson = _pearson_corr(y_hat, y).item()
            self.last_psnr = _psnr(mse_val, pred=y_hat, target=y).item()

        return loss
