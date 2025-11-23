"""Accuracy validation tools for optimized models.

Ensures that optimizations (quantization, TensorRT, etc.) maintain model accuracy
within acceptable thresholds.

Usage:
    from cyto_dl.utils.accuracy_validation import AccuracyValidator

    # Compare baseline vs optimized model
    validator = AccuracyValidator(
        baseline_model=model_baseline,
        optimized_model=model_optimized,
        validation_loader=val_loader
    )

    results = validator.validate()
    validator.print_report()
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for validation.

    Attributes
    ----------
    mae : float
        Mean Absolute Error
    mse : float
        Mean Squared Error
    rmse : float
        Root Mean Squared Error
    psnr : float
        Peak Signal-to-Noise Ratio (dB)
    ssim : float
        Structural Similarity Index (if images)
    cosine_similarity : float
        Cosine similarity between outputs
    max_absolute_diff : float
        Maximum absolute difference
    relative_error : float
        Relative error (%)
    """

    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    cosine_similarity: float = 0.0
    max_absolute_diff: float = 0.0
    relative_error: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class AccuracyValidator:
    """Validate accuracy of optimized models vs baseline.

    Parameters
    ----------
    baseline_model : nn.Module
        Baseline model
    optimized_model : nn.Module
        Optimized model to validate
    validation_loader : DataLoader
        Validation dataloader
    device : str
        Device to use
    tolerance : float
        Acceptable accuracy degradation (default 1%)

    Examples
    --------
    >>> validator = AccuracyValidator(
    ...     baseline_model=model_fp32,
    ...     optimized_model=model_quantized,
    ...     validation_loader=val_loader,
    ...     tolerance=0.01
    ... )
    >>> results = validator.validate()
    >>> if results['passed']:
    ...     print("✓ Accuracy validation passed!")
    """

    def __init__(
        self,
        baseline_model: nn.Module,
        optimized_model: nn.Module,
        validation_loader: DataLoader,
        device: str = "cuda",
        tolerance: float = 0.01,
    ):
        self.baseline_model = baseline_model.to(device).eval()
        self.optimized_model = optimized_model.to(device).eval()
        self.validation_loader = validation_loader
        self.device = device
        self.tolerance = tolerance
        self.metrics: Optional[AccuracyMetrics] = None

    def validate(
        self, custom_metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, any]:
        """Run validation.

        Parameters
        ----------
        custom_metrics : Dict[str, Callable], optional
            Custom metric functions (name -> function(pred, target))

        Returns
        -------
        Dict[str, any]
            Validation results
        """
        logger.info("Running accuracy validation...")

        # Collect outputs
        baseline_outputs = []
        optimized_outputs = []
        targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.validation_loader):
                if isinstance(batch, (list, tuple)):
                    inputs, target = batch
                    targets.append(target.cpu())
                else:
                    inputs = batch
                    target = None

                inputs = inputs.to(self.device)

                # Baseline prediction
                baseline_out = self.baseline_model(inputs)
                baseline_outputs.append(baseline_out.cpu())

                # Optimized prediction
                optimized_out = self.optimized_model(inputs)
                optimized_outputs.append(optimized_out.cpu())

                if batch_idx % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1} batches...")

        # Concatenate all outputs
        baseline_outputs = torch.cat(baseline_outputs, dim=0)
        optimized_outputs = torch.cat(optimized_outputs, dim=0)

        if targets:
            targets = torch.cat(targets, dim=0)
        else:
            targets = None

        # Compute metrics
        self.metrics = self._compute_metrics(
            baseline_outputs, optimized_outputs, targets, custom_metrics
        )

        # Check if validation passed
        passed = self._check_tolerance()

        results = {
            "passed": passed,
            "metrics": self.metrics,
            "tolerance": self.tolerance,
        }

        return results

    def _compute_metrics(
        self,
        baseline: torch.Tensor,
        optimized: torch.Tensor,
        targets: Optional[torch.Tensor],
        custom_metrics: Optional[Dict[str, Callable]],
    ) -> AccuracyMetrics:
        """Compute accuracy metrics."""
        # Absolute difference
        diff = (baseline - optimized).abs()

        # Mean Absolute Error
        mae = diff.mean().item()

        # Mean Squared Error
        mse = (diff ** 2).mean().item()

        # RMSE
        rmse = np.sqrt(mse)

        # Maximum absolute difference
        max_abs_diff = diff.max().item()

        # Relative error
        baseline_norm = baseline.abs().mean().item()
        relative_error = (mae / baseline_norm * 100) if baseline_norm > 0 else 0.0

        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            max_val = max(baseline.max().item(), optimized.max().item())
            psnr = 20 * np.log10(max_val / np.sqrt(mse)) if max_val > 0 else 0.0
        else:
            psnr = float('inf')

        # Cosine similarity
        baseline_flat = baseline.flatten()
        optimized_flat = optimized.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            baseline_flat.unsqueeze(0), optimized_flat.unsqueeze(0)
        ).item()

        # SSIM (if images)
        ssim = 0.0
        if baseline.ndim >= 4:  # Batch of images
            try:
                ssim = self._compute_ssim(baseline, optimized)
            except:
                pass

        # Custom metrics
        custom_results = {}
        if custom_metrics and targets is not None:
            for name, metric_fn in custom_metrics.items():
                try:
                    baseline_score = metric_fn(baseline, targets)
                    optimized_score = metric_fn(optimized, targets)
                    custom_results[name] = {
                        "baseline": baseline_score,
                        "optimized": optimized_score,
                        "diff": optimized_score - baseline_score,
                    }
                except Exception as e:
                    logger.warning(f"Custom metric {name} failed: {e}")

        metrics = AccuracyMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            psnr=psnr,
            ssim=ssim,
            cosine_similarity=cosine_sim,
            max_absolute_diff=max_abs_diff,
            relative_error=relative_error,
            custom_metrics=custom_results,
        )

        return metrics

    def _compute_ssim(
        self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
    ) -> float:
        """Compute SSIM between two image batches.

        Simple SSIM implementation. For production, use pytorch-msssim.
        """
        # Normalize to [0, 1]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
        img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)

        # Compute means
        mu1 = img1.mean()
        mu2 = img2.mean()

        # Compute variances
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()

        # Compute covariance
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim = (numerator / denominator).item()

        return ssim

    def _check_tolerance(self) -> bool:
        """Check if accuracy is within tolerance."""
        if self.metrics is None:
            return False

        # Check relative error
        if self.metrics.relative_error > (self.tolerance * 100):
            logger.warning(
                f"Relative error {self.metrics.relative_error:.2f}% exceeds "
                f"tolerance {self.tolerance * 100:.2f}%"
            )
            return False

        # Check cosine similarity (should be close to 1)
        if self.metrics.cosine_similarity < (1 - self.tolerance):
            logger.warning(
                f"Cosine similarity {self.metrics.cosine_similarity:.4f} below "
                f"threshold {1 - self.tolerance:.4f}"
            )
            return False

        return True

    def print_report(self):
        """Print validation report."""
        if self.metrics is None:
            logger.warning("No metrics available. Run validation first.")
            return

        print("\n" + "=" * 70)
        print("ACCURACY VALIDATION REPORT")
        print("=" * 70)
        print(f"Tolerance:              {self.tolerance * 100:.2f}%")
        print("-" * 70)
        print("Metrics:")
        print(f"  Mean Absolute Error:  {self.metrics.mae:.6f}")
        print(f"  Mean Squared Error:   {self.metrics.mse:.6f}")
        print(f"  RMSE:                 {self.metrics.rmse:.6f}")
        print(f"  Max Absolute Diff:    {self.metrics.max_absolute_diff:.6f}")
        print(f"  Relative Error:       {self.metrics.relative_error:.2f}%")
        print(f"  PSNR:                 {self.metrics.psnr:.2f} dB")
        print(f"  Cosine Similarity:    {self.metrics.cosine_similarity:.6f}")

        if self.metrics.ssim > 0:
            print(f"  SSIM:                 {self.metrics.ssim:.6f}")

        if self.metrics.custom_metrics:
            print("\nCustom Metrics:")
            for name, result in self.metrics.custom_metrics.items():
                print(f"  {name}:")
                print(f"    Baseline:  {result['baseline']:.6f}")
                print(f"    Optimized: {result['optimized']:.6f}")
                print(f"    Diff:      {result['diff']:+.6f}")

        # Verdict
        print("-" * 70)
        passed = self._check_tolerance()
        if passed:
            print("✓ VALIDATION PASSED - Accuracy within tolerance")
        else:
            print("✗ VALIDATION FAILED - Accuracy degraded beyond tolerance")

        print("=" * 70 + "\n")


class OutputComparator:
    """Compare outputs of two models on the same inputs.

    Useful for debugging optimization issues.

    Examples
    --------
    >>> comparator = OutputComparator(model_baseline, model_optimized)
    >>> comparison = comparator.compare(input_tensor)
    >>> comparator.visualize_diff(comparison)
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        device: str = "cuda",
    ):
        self.model_a = model_a.to(device).eval()
        self.model_b = model_b.to(device).eval()
        self.device = device

    def compare(
        self, input_tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compare outputs.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input to both models

        Returns
        -------
        Dict[str, torch.Tensor]
            Comparison results
        """
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output_a = self.model_a(input_tensor)
            output_b = self.model_b(input_tensor)

        diff = (output_a - output_b).abs()
        relative_diff = diff / (output_a.abs() + 1e-8)

        return {
            "output_a": output_a.cpu(),
            "output_b": output_b.cpu(),
            "absolute_diff": diff.cpu(),
            "relative_diff": relative_diff.cpu(),
            "max_diff": diff.max().item(),
            "mean_diff": diff.mean().item(),
        }

    def visualize_diff(self, comparison: Dict[str, torch.Tensor]):
        """Visualize differences (console output).

        Parameters
        ----------
        comparison : Dict[str, torch.Tensor]
            Comparison results from compare()
        """
        print("\n" + "=" * 70)
        print("OUTPUT COMPARISON")
        print("=" * 70)
        print(f"Max Absolute Diff:  {comparison['max_diff']:.6f}")
        print(f"Mean Absolute Diff: {comparison['mean_diff']:.6f}")
        print("\nOutput Statistics:")
        print(f"  Model A - Mean: {comparison['output_a'].mean():.6f}, "
              f"Std: {comparison['output_a'].std():.6f}")
        print(f"  Model B - Mean: {comparison['output_b'].mean():.6f}, "
              f"Std: {comparison['output_b'].std():.6f}")
        print("=" * 70 + "\n")


def validate_optimization(
    baseline_model: nn.Module,
    optimized_model: nn.Module,
    validation_loader: DataLoader,
    tolerance: float = 0.01,
    device: str = "cuda",
) -> bool:
    """Quick validation helper.

    Parameters
    ----------
    baseline_model : nn.Module
        Baseline model
    optimized_model : nn.Module
        Optimized model
    validation_loader : DataLoader
        Validation data
    tolerance : float
        Acceptable degradation
    device : str
        Device to use

    Returns
    -------
    bool
        True if validation passed

    Examples
    --------
    >>> passed = validate_optimization(
    ...     baseline_model=model_fp32,
    ...     optimized_model=model_quantized,
    ...     validation_loader=val_loader,
    ...     tolerance=0.01
    ... )
    >>> if passed:
    ...     print("✓ Safe to deploy optimized model")
    """
    validator = AccuracyValidator(
        baseline_model=baseline_model,
        optimized_model=optimized_model,
        validation_loader=validation_loader,
        device=device,
        tolerance=tolerance,
    )

    results = validator.validate()
    validator.print_report()

    return results["passed"]
