#!/usr/bin/env python
"""
Benchmark script for CytoDL training performance across different branches.

Usage:
    python scripts/benchmark_train.py --config benchmarks/train_tasks.csv --output results_train.json

CSV Format:
    branch,experiment,model,data_config,trainer,max_epochs,batch_size
    main,im2im/segmentation,model/im2im/segmentation,data/im2im/segmentation,gpu,5,8
    phase/optimization-1,im2im/segmentation,model/im2im/segmentation,data/im2im/segmentation,gpu,5,8
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import tempfile

import torch
import numpy as np


@contextmanager
def timer(name: str, metrics: Dict[str, Any]):
    """Context manager for timing code blocks."""
    start_time = time.perf_counter()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    yield

    elapsed = time.perf_counter() - start_time
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    metrics[name] = {
        "elapsed_seconds": elapsed,
        "memory_delta_mb": (end_memory - start_memory) / (1024 ** 2) if torch.cuda.is_available() else None,
        "peak_memory_mb": peak_memory / (1024 ** 2) if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def checkout_branch(branch: str, original_branch: str) -> bool:
    """Checkout a git branch."""
    try:
        # Check if branch exists locally
        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            # Try to fetch from remote
            print(f"Branch {branch} not found locally, attempting to fetch from remote...")
            subprocess.run(
                ["git", "fetch", "origin", f"{branch}:{branch}"],
                check=True,
                capture_output=True
            )

        # Checkout the branch
        subprocess.run(["git", "checkout", branch], check=True, capture_output=True)
        print(f"✓ Checked out branch: {branch}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to checkout branch {branch}: {e}")
        return False


def get_git_commit_info() -> Dict[str, str]:
    """Get current git commit information."""
    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        commit_date = subprocess.run(
            ["git", "show", "-s", "--format=%ci", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        return {
            "branch": branch,
            "commit_hash": commit_hash,
            "commit_date": commit_date,
        }
    except subprocess.CalledProcessError:
        return {"branch": "unknown", "commit_hash": "unknown", "commit_date": "unknown"}


def instrument_training(experiment: str, overrides: List[str], metrics: Dict[str, Any]):
    """Run instrumented training with detailed timing."""
    from cyto_dl.api import CytoDLModel
    from cyto_dl import utils
    from omegaconf import DictConfig
    import hydra
    from lightning import Trainer, LightningModule

    with timer("total_training", metrics):
        # Initialize model
        with timer("model_initialization", metrics):
            model = CytoDLModel()

        # Load experiment configuration
        with timer("config_loading", metrics):
            model.load_default_experiment(
                experiment,
                output_dir=tempfile.mkdtemp(),
                overrides=overrides
            )

        # Get config to inspect data setup
        cfg = model.cfg

        # Setup data
        with timer("data_setup", metrics):
            if hasattr(model, 'datamodule'):
                datamodule = model.datamodule
            else:
                datamodule = utils.create_dataloader(cfg.data, None)

        # Setup model
        with timer("model_setup", metrics):
            lightning_model: LightningModule = hydra.utils.instantiate(
                cfg.model, _recursive_=False
            )

        # Setup trainer
        with timer("trainer_setup", metrics):
            callbacks = utils.instantiate_callbacks(cfg.get("callbacks"))
            logger = utils.instantiate_loggers(cfg.get("logger"))
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer, callbacks=callbacks, logger=logger
            )

        # Run first epoch with detailed timing
        epoch_metrics = {}

        with timer("first_epoch", epoch_metrics):
            # Data loading timing
            with timer("data_loading_first_batch", epoch_metrics):
                if hasattr(datamodule, 'train_dataloader'):
                    train_loader = datamodule.train_dataloader()
                    first_batch = next(iter(train_loader))

            # Model forward pass timing
            with timer("model_forward_first_batch", epoch_metrics):
                if torch.cuda.is_available():
                    for key in first_batch:
                        if isinstance(first_batch[key], torch.Tensor):
                            first_batch[key] = first_batch[key].cuda()

                lightning_model.eval()
                if torch.cuda.is_available():
                    lightning_model = lightning_model.cuda()

                with torch.no_grad():
                    output = lightning_model(first_batch)

        metrics["epoch_breakdown"] = epoch_metrics

        # Full training run
        with timer("full_training", metrics):
            if hasattr(datamodule, 'train_dataloader'):
                trainer.fit(model=lightning_model, datamodule=datamodule)

        # Extract key metrics
        metrics["final_metrics"] = {
            k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
            for k, v in trainer.callback_metrics.items()
        }

        # Training statistics
        if hasattr(datamodule, 'train_dataloader'):
            train_loader = datamodule.train_dataloader()
            metrics["dataset_info"] = {
                "num_train_batches": len(train_loader),
                "batch_size": cfg.data.get("batch_size", "unknown"),
            }


def run_benchmark(task_config: Dict[str, str], original_branch: str) -> Dict[str, Any]:
    """Run a single benchmark task."""
    branch = task_config["branch"]
    experiment = task_config["experiment"]

    print(f"\n{'='*80}")
    print(f"Benchmarking: {branch} | {experiment}")
    print(f"{'='*80}\n")

    result = {
        "branch": branch,
        "experiment": experiment,
        "task_config": task_config,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None,
        "metrics": {},
    }

    # Checkout branch
    if not checkout_branch(branch, original_branch):
        result["error"] = f"Failed to checkout branch {branch}"
        return result

    # Get git info
    result["git_info"] = get_git_commit_info()

    # Build overrides
    overrides = [
        f"trainer={task_config.get('trainer', 'gpu')}",
        f"trainer.max_epochs={task_config.get('max_epochs', '5')}",
        f"datamodule.batch_size={task_config.get('batch_size', '8')}",
        "trainer.limit_train_batches=10",  # Limit for benchmarking
        "trainer.limit_val_batches=5",
    ]

    # Add any custom overrides
    if task_config.get("overrides"):
        overrides.extend(task_config["overrides"].split(","))

    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Run instrumented training
        instrument_training(experiment, overrides, result["metrics"])

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        print(f"✗ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

    return result


def load_tasks_from_csv(csv_path: str) -> List[Dict[str, str]]:
    """Load benchmark tasks from CSV file."""
    tasks = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty or comment lines
            if not row.get('branch') or row['branch'].startswith('#'):
                continue
            tasks.append(row)
    return tasks


def print_summary(results: List[Dict[str, Any]]):
    """Print benchmark summary."""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    for result in results:
        status = "✓" if result["success"] else "✗"
        branch = result["branch"]
        experiment = result["experiment"]

        print(f"{status} {branch:30s} | {experiment:30s}")

        if result["success"]:
            metrics = result["metrics"]
            total_time = metrics.get("total_training", {}).get("elapsed_seconds", 0)
            first_epoch = metrics.get("epoch_breakdown", {}).get("first_epoch", {}).get("elapsed_seconds", 0)

            print(f"   Total time: {total_time:.2f}s | First epoch: {first_epoch:.2f}s")

            # Show memory if available
            if "peak_memory_mb" in metrics.get("total_training", {}):
                peak_mem = metrics["total_training"]["peak_memory_mb"]
                if peak_mem:
                    print(f"   Peak memory: {peak_mem:.1f} MB")
        else:
            print(f"   Error: {result['error']}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark CytoDL training performance")
    parser.add_argument("--config", required=True, help="Path to CSV file with benchmark tasks")
    parser.add_argument("--output", required=True, help="Path to output JSON file with results")
    parser.add_argument("--skip-failed", action="store_true", help="Continue on failures")

    args = parser.parse_args()

    # Get original branch to restore later
    original_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()

    print(f"Original branch: {original_branch}")

    # Load tasks
    tasks = load_tasks_from_csv(args.config)
    print(f"Loaded {len(tasks)} benchmark tasks from {args.config}")

    # Run benchmarks
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] ", end="")
        result = run_benchmark(task, original_branch)
        results.append(result)

        if not result["success"] and not args.skip_failed:
            print("Stopping due to failure. Use --skip-failed to continue.")
            break

    # Restore original branch
    print(f"\nRestoring original branch: {original_branch}")
    checkout_branch(original_branch, original_branch)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
