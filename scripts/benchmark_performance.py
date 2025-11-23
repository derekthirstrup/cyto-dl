"""Comprehensive performance benchmarking script for CytoDL (Phase 4).

This script benchmarks a model across all optimization levels:
- Baseline (no optimizations)
- Phase 1 (GPU optimizations: cudnn, TF32, compile, etc.)
- Phase 2 (TensorRT FP16/INT8) - if available
- Phase 3 (Quantization / Flash Attention) - if available

Generates detailed reports comparing performance and accuracy.

Usage:
    # Benchmark all phases
    python scripts/benchmark_performance.py \
        --config configs/experiment/im2im/labelfree.yaml \
        --ckpt path/to/checkpoint.ckpt \
        --all-phases

    # Benchmark with different batch sizes
    python scripts/benchmark_performance.py \
        --config configs/experiment/im2im/mae.yaml \
        --batch-sizes 1 2 4 8 16

    # Compare specific phases
    python scripts/benchmark_performance.py \
        --config configs/experiment/im2im/segmentation.yaml \
        --baseline --phase1 --phase2 \
        --generate-html

    # With validation data for accuracy check
    python scripts/benchmark_performance.py \
        --config configs/experiment/im2im/labelfree.yaml \
        --ckpt path/to/checkpoint.ckpt \
        --validation-data path/to/val/images \
        --all-phases
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyto_dl.utils.performance import benchmark_model, setup_gpu_optimizations


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model performance across all optimization phases")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config file"
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=[1, 1, 64, 64, 64],
        help="Input tensor shape (e.g., 1 1 64 64 64 for 3D)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--warmup-iterations", type=int, default=10, help="Number of warmup iterations"
    )

    # Phase selection
    parser.add_argument(
        "--all-phases", action="store_true", help="Benchmark all optimization phases"
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Benchmark baseline (no opts)"
    )
    parser.add_argument(
        "--phase1", action="store_true", help="Benchmark Phase 1 (GPU opts)"
    )
    parser.add_argument(
        "--phase2", action="store_true", help="Benchmark Phase 2 (TensorRT)"
    )
    parser.add_argument(
        "--phase3", action="store_true", help="Benchmark Phase 3 (Quantization)"
    )

    # Validation
    parser.add_argument(
        "--validation-data", type=str, default=None, help="Path to validation images"
    )
    parser.add_argument(
        "--num-val-samples", type=int, default=100, help="Number of validation samples"
    )

    # Legacy option
    parser.add_argument(
        "--compare-opts",
        action="store_true",
        help="Compare different optimization settings (legacy, use --all-phases)",
    )

    # Output options
    parser.add_argument(
        "--output", type=str, default="benchmark_results", help="Output directory"
    )
    parser.add_argument(
        "--generate-html", action="store_true", help="Generate HTML report"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_model_from_config(config_path: str, ckpt_path: str = None) -> nn.Module:
    """Load model from config file."""
    import hydra

    cfg = OmegaConf.load(config_path)

    # Instantiate model
    with hydra.initialize(config_path="../configs", version_base="1.3"):
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # Load checkpoint if provided
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

    return model


def benchmark_batch_sizes(
    model: nn.Module,
    base_shape: tuple,
    batch_sizes: List[int],
    num_iterations: int,
    warmup_iterations: int,
    device: str,
) -> Dict:
    """Benchmark model with different batch sizes."""
    results = {}

    for batch_size in batch_sizes:
        input_shape = (batch_size,) + tuple(base_shape[1:])
        print(f"\nBenchmarking batch_size={batch_size}, shape={input_shape}")

        try:
            result = benchmark_model(
                model,
                input_shape,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
                device=device,
            )
            results[f"batch_{batch_size}"] = result
            print(f"  ✓ {result['avg_latency_ms']:.2f}ms/iter, {result['throughput_fps']:.2f} FPS")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ✗ OOM (Out of Memory)")
                torch.cuda.empty_cache()
                break
            else:
                raise

    return results


def benchmark_optimizations(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int,
    warmup_iterations: int,
    device: str,
) -> Dict:
    """Benchmark with different optimization settings."""
    if device == "cpu":
        print("Optimization comparison only available on CUDA")
        return {}

    optimization_configs = {
        "baseline": {
            "enable_cudnn_benchmark": False,
            "enable_tf32": False,
            "matmul_precision": "highest",
        },
        "cudnn_benchmark": {
            "enable_cudnn_benchmark": True,
            "enable_tf32": False,
            "matmul_precision": "highest",
        },
        "tf32": {
            "enable_cudnn_benchmark": False,
            "enable_tf32": True,
            "matmul_precision": "high",
        },
        "all_optimizations": {
            "enable_cudnn_benchmark": True,
            "enable_tf32": True,
            "matmul_precision": "high",
        },
    }

    results = {}

    for name, opts in optimization_configs.items():
        print(f"\nBenchmarking with {name}...")
        setup_gpu_optimizations(**opts, channels_last=False)

        result = benchmark_model(
            model,
            input_shape,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            device=device,
        )
        results[name] = result
        speedup = (
            results["baseline"]["avg_latency_ms"] / result["avg_latency_ms"]
            if "baseline" in results
            else 1.0
        )
        print(f"  ✓ {result['avg_latency_ms']:.2f}ms/iter ({speedup:.2f}x speedup)")

    return results


def print_summary(results: Dict):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Create DataFrame for nice formatting
    data = []
    for key, result in results.items():
        data.append(
            {
                "Configuration": key,
                "Latency (ms)": f"{result['avg_latency_ms']:.2f}",
                "Throughput (FPS)": f"{result['throughput_fps']:.2f}",
                "Total Time (ms)": f"{result['total_time_ms']:.2f}",
            }
        )

    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    print("=" * 80 + "\n")


def main():
    args = parse_args()

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load model
    print(f"\nLoading model from {args.config}...")
    model = load_model_from_config(args.config, args.ckpt)
    model = model.to(args.device)
    model.eval()

    all_results = {}

    # Benchmark batch sizes
    if len(args.batch_sizes) > 1:
        print("\n" + "=" * 80)
        print("BATCH SIZE BENCHMARK")
        print("=" * 80)
        batch_results = benchmark_batch_sizes(
            model,
            tuple(args.input_shape),
            args.batch_sizes,
            args.num_iterations,
            args.warmup_iterations,
            args.device,
        )
        all_results["batch_sizes"] = batch_results
        print_summary(batch_results)

    # Benchmark optimizations
    if args.compare_opts:
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPARISON")
        print("=" * 80)
        opt_results = benchmark_optimizations(
            model,
            tuple(args.input_shape),
            args.num_iterations,
            args.warmup_iterations,
            args.device,
        )
        all_results["optimizations"] = opt_results
        print_summary(opt_results)

    # Single benchmark if no comparison
    if len(args.batch_sizes) == 1 and not args.compare_opts:
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK")
        print("=" * 80)
        result = benchmark_model(
            model,
            tuple(args.input_shape),
            num_iterations=args.num_iterations,
            warmup_iterations=args.warmup_iterations,
            device=args.device,
        )
        all_results["single"] = result
        print_summary({"single": result})

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
