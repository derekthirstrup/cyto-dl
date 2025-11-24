#!/usr/bin/env python
"""
Benchmark script for CytoDL inference/prediction performance across different branches.

Usage:
    python scripts/benchmark_inference.py --config benchmarks/inference_tasks.csv --output results_inference.json

CSV Format:
    branch,experiment,checkpoint_path,data_path,trainer,batch_size
    main,im2im/segmentation,path/to/checkpoint.ckpt,path/to/data,gpu,8
    phase/optimization-1,im2im/segmentation,path/to/checkpoint.ckpt,path/to/data,gpu,8
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


def instrument_inference(experiment: str, checkpoint_path: str, overrides: List[str], metrics: Dict[str, Any]):
    """Run instrumented inference with detailed timing."""
    from cyto_dl.api import CytoDLModel
    from cyto_dl import utils
    from omegaconf import DictConfig
    import hydra
    from lightning import Trainer, LightningModule

    with timer("total_inference", metrics):
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

        # Get config
        cfg = model.cfg

        # Setup data
        with timer("data_setup", metrics):
            datamodule = utils.create_dataloader(cfg.data, None)

        # Setup model
        with timer("model_setup", metrics):
            lightning_model: LightningModule = hydra.utils.instantiate(
                cfg.model, _recursive_=False
            )

        # Load checkpoint
        with timer("checkpoint_loading", metrics):
            lightning_model, load_params = utils.load_checkpoint(
                lightning_model,
                {"ckpt_path": checkpoint_path}
            )

        # Setup trainer
        with timer("trainer_setup", metrics):
            callbacks = utils.instantiate_callbacks(cfg.get("callbacks"))
            logger = utils.instantiate_loggers(cfg.get("logger"))
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer, callbacks=callbacks, logger=logger
            )

        # Get data loader
        if hasattr(datamodule, 'predict_dataloader'):
            predict_loader = datamodule.predict_dataloader()
        elif hasattr(datamodule, 'test_dataloader'):
            predict_loader = datamodule.test_dataloader()
        else:
            predict_loader = datamodule

        # Detailed timing for first batch
        batch_metrics = {}

        with timer("first_batch_processing", batch_metrics):
            # Get first batch
            with timer("first_batch_loading", batch_metrics):
                first_batch = next(iter(predict_loader))

            # Move to GPU if available
            with timer("first_batch_to_device", batch_metrics):
                if torch.cuda.is_available():
                    for key in first_batch:
                        if isinstance(first_batch[key], torch.Tensor):
                            first_batch[key] = first_batch[key].cuda()

                    lightning_model = lightning_model.cuda()

            # Model inference
            with timer("first_batch_forward", batch_metrics):
                lightning_model.eval()
                with torch.no_grad():
                    output = lightning_model(first_batch)

            # Post-processing (if any)
            with timer("first_batch_postprocess", batch_metrics):
                # This would include any post-processing steps
                pass

        metrics["batch_breakdown"] = batch_metrics

        # Run full prediction
        with timer("full_prediction", metrics):
            outputs = trainer.predict(
                model=lightning_model,
                dataloaders=predict_loader,
                ckpt_path=None  # Already loaded
            )

        # Inference statistics
        metrics["inference_info"] = {
            "num_batches": len(predict_loader) if hasattr(predict_loader, '__len__') else "unknown",
            "batch_size": cfg.data.get("batch_size", "unknown"),
            "num_outputs": len(outputs) if outputs else 0,
        }

        # Per-batch statistics
        if len(predict_loader) > 0:
            total_batches = len(predict_loader) if hasattr(predict_loader, '__len__') else len(outputs)
            if total_batches > 0:
                avg_time_per_batch = metrics["full_prediction"]["elapsed_seconds"] / total_batches
                metrics["inference_info"]["avg_time_per_batch"] = avg_time_per_batch

                # Estimate throughput
                batch_size = cfg.data.get("batch_size", 1)
                if isinstance(batch_size, int):
                    throughput = batch_size / avg_time_per_batch
                    metrics["inference_info"]["throughput_samples_per_sec"] = throughput


def run_benchmark(task_config: Dict[str, str], original_branch: str) -> Dict[str, Any]:
    """Run a single benchmark task."""
    branch = task_config["branch"]
    experiment = task_config["experiment"]
    checkpoint_path = task_config.get("checkpoint_path", "")

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
        f"datamodule.batch_size={task_config.get('batch_size', '8')}",
        "trainer.limit_predict_batches=10",  # Limit for benchmarking
    ]

    # Add data path if provided
    if task_config.get("data_path"):
        overrides.append(f"data.path={task_config['data_path']}")

    # Add any custom overrides
    if task_config.get("overrides"):
        overrides.extend(task_config["overrides"].split(","))

    try:
        # Validate checkpoint exists
        if checkpoint_path and not Path(checkpoint_path).exists():
            result["error"] = f"Checkpoint not found: {checkpoint_path}"
            result["warning"] = "Checkpoint path is required for inference benchmarking"
            return result

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Run instrumented inference
        instrument_inference(experiment, checkpoint_path, overrides, result["metrics"])

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
            total_time = metrics.get("total_inference", {}).get("elapsed_seconds", 0)
            first_batch = metrics.get("batch_breakdown", {}).get("first_batch_forward", {}).get("elapsed_seconds", 0)

            print(f"   Total time: {total_time:.2f}s | First batch forward: {first_batch:.4f}s")

            # Show throughput if available
            inference_info = metrics.get("inference_info", {})
            if "throughput_samples_per_sec" in inference_info:
                throughput = inference_info["throughput_samples_per_sec"]
                print(f"   Throughput: {throughput:.2f} samples/sec")

            # Show memory if available
            if "peak_memory_mb" in metrics.get("total_inference", {}):
                peak_mem = metrics["total_inference"]["peak_memory_mb"]
                if peak_mem:
                    print(f"   Peak memory: {peak_mem:.1f} MB")
        else:
            print(f"   Error: {result['error']}")
            if result.get("warning"):
                print(f"   Warning: {result['warning']}")

        print()


def compare_results(results: List[Dict[str, Any]]):
    """Compare results across branches."""
    if len(results) < 2:
        return

    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}\n")

    # Group by experiment
    by_experiment = {}
    for result in results:
        if not result["success"]:
            continue
        exp = result["experiment"]
        if exp not in by_experiment:
            by_experiment[exp] = []
        by_experiment[exp].append(result)

    for experiment, exp_results in by_experiment.items():
        print(f"\nExperiment: {experiment}")
        print("-" * 80)

        # Sort by total time
        exp_results.sort(
            key=lambda r: r["metrics"].get("total_inference", {}).get("elapsed_seconds", float('inf'))
        )

        baseline = exp_results[0]
        baseline_time = baseline["metrics"]["total_inference"]["elapsed_seconds"]

        for result in exp_results:
            branch = result["branch"]
            total_time = result["metrics"]["total_inference"]["elapsed_seconds"]
            speedup = baseline_time / total_time if total_time > 0 else 0

            print(f"  {branch:30s}: {total_time:8.2f}s  (speedup: {speedup:.2f}x)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CytoDL inference performance")
    parser.add_argument("--config", required=True, help="Path to CSV file with benchmark tasks")
    parser.add_argument("--output", required=True, help="Path to output JSON file with results")
    parser.add_argument("--skip-failed", action="store_true", help="Continue on failures")
    parser.add_argument("--compare", action="store_true", help="Show performance comparison")

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

    # Compare results if requested
    if args.compare:
        compare_results(results)


if __name__ == "__main__":
    main()
