#!/usr/bin/env python3
"""Parse metrics from cyto-dl CSV logger output.

Utility for extracting training metrics from Lightning's CSV logger files.
Used by the experiment runner and evaluator.
"""

import csv
import math
import sys
from pathlib import Path


def find_latest_metrics_csv(base_dir):
    """Find the most recent metrics.csv in a training output directory."""
    base_dir = Path(base_dir)
    candidates = list(base_dir.rglob("metrics.csv"))
    if not candidates:
        return None
    # Return most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_csv_metrics(csv_path):
    """Parse a Lightning CSV logger metrics file.

    Lightning writes different metrics on different rows (same step),
    so we need to aggregate across rows to get a complete picture.

    Returns
    -------
    dict
        Final epoch metrics (key -> value)
    """
    all_metrics = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value and value.strip():
                    try:
                        val = float(value)
                        if not math.isnan(val) and not math.isinf(val):
                            all_metrics[key] = val
                    except (ValueError, TypeError):
                        pass

    return all_metrics


def get_metric_history(csv_path, metric_name):
    """Get the history of a specific metric across epochs.

    Returns list of (epoch/step, value) tuples.
    """
    history = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric_name in row and row[metric_name].strip():
                try:
                    val = float(row[metric_name])
                    step = int(row.get("step", row.get("epoch", 0)))
                    if not math.isnan(val):
                        history.append((step, val))
                except (ValueError, TypeError):
                    pass

    return history


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_metrics.py <log_dir> [metric_name]")
        sys.exit(1)

    log_dir = sys.argv[1]
    metric_name = sys.argv[2] if len(sys.argv) > 2 else None

    csv_path = find_latest_metrics_csv(log_dir)
    if csv_path is None:
        print(f"No metrics.csv found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    if metric_name:
        history = get_metric_history(csv_path, metric_name)
        for step, val in history:
            print(f"{step}\t{val:.6f}")
    else:
        metrics = parse_csv_metrics(csv_path)
        for key, val in sorted(metrics.items()):
            print(f"{key}\t{val:.6f}")


if __name__ == "__main__":
    main()
