#!/usr/bin/env python3
"""Immutable evaluator for label-free fluorescence prediction quality.

DO NOT MODIFY THIS FILE DURING AUTORESEARCH EXPERIMENTS.
This evaluator is the ground truth for measuring model quality.

Reads CSV logger output from a cyto-dl training run and computes a composite
score from SSIM, PSNR, and Pearson correlation.

Usage:
    python labelfree_evaluator.py <log_dir>

    # Returns a single float score to stdout (higher is better)
    # Exit code 0 = success, 1 = failure/crash
"""

import argparse
import csv
import math
import sys
from contextlib import suppress
from pathlib import Path


def find_csv_log(log_dir):
    """Find the CSV metrics file from a training run."""
    log_dir = Path(log_dir)

    # Lightning CSV logger writes to: <log_dir>/csv/version_X/metrics.csv
    # or directly to: <log_dir>/metrics.csv
    candidates = [
        log_dir / "csv" / "version_0" / "metrics.csv",
        log_dir / "csv" / "metrics.csv",
        log_dir / "metrics.csv",
    ]

    # Also search for any metrics.csv recursively
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Glob fallback
    found = list(log_dir.rglob("metrics.csv"))
    if found:
        return sorted(found)[0]

    return None


def parse_metrics(csv_path):
    """Parse the CSV log file and extract final epoch metrics.

    Returns dict with metric name -> value for the last epoch that has data.
    """
    metrics = {}
    last_epoch_rows = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return metrics

    # Collect all non-empty values across rows (Lightning logs different
    # metrics on different rows within the same step)
    for row in rows:
        for key, value in row.items():
            if value and value.strip():
                with suppress(ValueError, TypeError):
                    val = float(value)
                    if not math.isnan(val) and not math.isinf(val):
                        metrics[key] = val

    return metrics


def compute_composite_score(metrics):
    """Compute the composite quality score.

    Score = 0.5 * SSIM + 0.3 * (PSNR/50) + 0.2 * Pearson

    All components normalized to roughly [0, 1] range. Higher is better.

    Returns (score, details_dict)
    """
    # Try multiple key patterns for each metric
    ssim_keys = [
        "val/ssim_epoch",
        "val/ssim_epoch/signal",
        "val/ssim/signal",
    ]
    psnr_keys = [
        "val/psnr_epoch",
        "val/psnr_epoch/signal",
        "val/psnr/signal",
    ]
    pearson_keys = [
        "val/pearson_epoch",
        "val/pearson_epoch/signal",
        "val/pearson/signal",
    ]
    loss_keys = [
        "val/loss_epoch",
        "val/loss",
    ]

    def _get(keys, default=None):
        for k in keys:
            if k in metrics:
                return metrics[k]
        return default

    ssim = _get(ssim_keys)
    psnr = _get(psnr_keys)
    pearson = _get(pearson_keys)
    val_loss = _get(loss_keys)

    details = {
        "ssim": ssim,
        "psnr": psnr,
        "pearson": pearson,
        "val_loss": val_loss,
    }

    # Compute composite score
    score = 0.0
    components = 0

    if ssim is not None:
        # SSIM is already in [-1, 1], typically [0, 1] for reasonable predictions
        ssim_contrib = max(0, ssim) * 0.5
        score += ssim_contrib
        components += 1
        details["ssim_contribution"] = ssim_contrib

    if psnr is not None:
        # PSNR typically 10-50 dB for images, normalize to [0, 1]
        psnr_norm = max(0, min(psnr, 50)) / 50.0
        psnr_contrib = psnr_norm * 0.3
        score += psnr_contrib
        components += 1
        details["psnr_contribution"] = psnr_contrib

    if pearson is not None:
        # Pearson in [-1, 1], we want [0, 1]
        pearson_contrib = max(0, pearson) * 0.2
        score += pearson_contrib
        components += 1
        details["pearson_contribution"] = pearson_contrib

    if components == 0:
        # Fallback: use negative val_loss (lower loss = higher score)
        if val_loss is not None:
            # Invert loss: score = 1 / (1 + loss) so lower loss -> higher score
            score = 1.0 / (1.0 + val_loss)
            details["fallback"] = True
        else:
            score = 0.0
            details["error"] = "no metrics found"

    details["composite_score"] = score
    details["num_components"] = components
    return score, details


def evaluate(log_dir):
    """Run evaluation on a training run's output.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing the training run's output (with CSV logs)

    Returns
    -------
    tuple of (float, dict)
        (composite_score, details)
    """
    csv_path = find_csv_log(log_dir)
    if csv_path is None:
        return 0.0, {"error": f"No metrics.csv found in {log_dir}"}

    metrics = parse_metrics(csv_path)
    if not metrics:
        return 0.0, {"error": f"No valid metrics in {csv_path}"}

    return compute_composite_score(metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate label-free fluorescence prediction quality"
    )
    parser.add_argument("log_dir", help="Path to training run output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed metrics")
    args = parser.parse_args()

    try:
        score, details = evaluate(args.log_dir)
    except Exception as e:
        print(f"CRASH: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("--- Evaluation Details ---", file=sys.stderr)
        for k, v in details.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}", file=sys.stderr)
            else:
                print(f"  {k}: {v}", file=sys.stderr)
        print("--------------------------", file=sys.stderr)

    # Print ONLY the score to stdout (autoresearch reads this)
    print(f"{score:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
