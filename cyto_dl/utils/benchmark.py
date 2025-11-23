"""Comprehensive benchmarking framework for CytoDL.

Provides tools for:
- End-to-end performance benchmarking
- Comparing optimization levels (baseline vs optimized)
- Accuracy validation after optimizations
- Performance regression testing
- Generating detailed benchmark reports

Usage:
    from cyto_dl.utils.benchmark import BenchmarkSuite, ModelBenchmark

    # Benchmark a single model
    benchmark = ModelBenchmark(model, sample_input)
    results = benchmark.run_all()
    benchmark.print_report()

    # Compare multiple optimization levels
    suite = BenchmarkSuite()
    suite.add_model("baseline", model_baseline)
    suite.add_model("optimized", model_optimized)
    comparison = suite.run_comparison()
    suite.generate_report("benchmark_report.html")
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes
    ----------
    name : str
        Benchmark name
    latency_ms : float
        Average latency in milliseconds
    throughput_fps : float
        Throughput in frames/samples per second
    memory_allocated_gb : float
        GPU memory allocated in GB
    memory_reserved_gb : float
        GPU memory reserved in GB
    accuracy : Optional[float]
        Model accuracy (if validation data provided)
    model_size_mb : float
        Model size in MB
    device : str
        Device used (cuda/cpu)
    batch_size : int
        Batch size used
    iterations : int
        Number of iterations run
    metadata : Dict[str, Any]
        Additional metadata
    """

    name: str
    latency_ms: float
    throughput_fps: float
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    accuracy: Optional[float] = None
    model_size_mb: float = 0.0
    device: str = "cuda"
    batch_size: int = 1
    iterations: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "latency_ms": self.latency_ms,
            "throughput_fps": self.throughput_fps,
            "memory_allocated_gb": self.memory_allocated_gb,
            "memory_reserved_gb": self.memory_reserved_gb,
            "accuracy": self.accuracy,
            "model_size_mb": self.model_size_mb,
            "device": self.device,
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "metadata": self.metadata,
        }


class ModelBenchmark:
    """Benchmark a single model.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark
    sample_input : torch.Tensor
        Sample input tensor
    device : str
        Device to use
    name : str
        Benchmark name

    Examples
    --------
    >>> model = MyModel().cuda()
    >>> sample_input = torch.randn(1, 1, 256, 256).cuda()
    >>> benchmark = ModelBenchmark(model, sample_input, name="baseline")
    >>> results = benchmark.run_all()
    >>> benchmark.print_report()
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        device: str = "cuda",
        name: str = "model",
    ):
        self.model = model
        self.sample_input = sample_input
        self.device = device
        self.name = name
        self.results: Optional[BenchmarkResult] = None

        self.model = self.model.to(device)
        self.model.eval()

    def measure_latency(
        self,
        batch_size: int = 1,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Tuple[float, float]:
        """Measure inference latency and throughput.

        Parameters
        ----------
        batch_size : int
            Batch size
        num_iterations : int
            Number of iterations
        warmup_iterations : int
            Warmup iterations

        Returns
        -------
        Tuple[float, float]
            (latency_ms, throughput_fps)
        """
        # Create batched input
        batch_input = self.sample_input.repeat(
            batch_size, *([1] * (self.sample_input.ndim - 1))
        ).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(batch_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Measure
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(batch_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time

        latency_ms = (elapsed / num_iterations) * 1000
        throughput_fps = (batch_size * num_iterations) / elapsed

        return latency_ms, throughput_fps

    def measure_memory(self, batch_size: int = 1) -> Tuple[float, float]:
        """Measure GPU memory usage.

        Parameters
        ----------
        batch_size : int
            Batch size

        Returns
        -------
        Tuple[float, float]
            (allocated_gb, reserved_gb)
        """
        if self.device != "cuda":
            return 0.0, 0.0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        batch_input = self.sample_input.repeat(
            batch_size, *([1] * (self.sample_input.ndim - 1))
        ).to(self.device)

        with torch.no_grad():
            _ = self.model(batch_input)

        allocated_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9

        return allocated_gb, reserved_gb

    def measure_model_size(self) -> float:
        """Measure model size in MB.

        Returns
        -------
        float
            Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def validate_accuracy(
        self, dataloader: DataLoader, metric_fn: callable
    ) -> float:
        """Validate model accuracy.

        Parameters
        ----------
        dataloader : DataLoader
            Validation dataloader
        metric_fn : callable
            Metric function (output, target) -> score

        Returns
        -------
        float
            Accuracy score
        """
        total_score = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None

                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)

                outputs = self.model(inputs)

                if targets is not None:
                    score = metric_fn(outputs, targets)
                    total_score += score * inputs.size(0)
                    total_samples += inputs.size(0)

        accuracy = total_score / total_samples if total_samples > 0 else 0.0
        return accuracy

    def run_all(
        self,
        batch_size: int = 1,
        num_iterations: int = 100,
        validation_loader: Optional[DataLoader] = None,
        metric_fn: Optional[callable] = None,
    ) -> BenchmarkResult:
        """Run complete benchmark.

        Parameters
        ----------
        batch_size : int
            Batch size
        num_iterations : int
            Number of iterations
        validation_loader : DataLoader, optional
            Validation dataloader
        metric_fn : callable, optional
            Metric function

        Returns
        -------
        BenchmarkResult
            Benchmark results
        """
        logger.info(f"Running benchmark: {self.name}")

        # Measure latency
        logger.info("  Measuring latency...")
        latency_ms, throughput_fps = self.measure_latency(
            batch_size, num_iterations
        )

        # Measure memory
        logger.info("  Measuring memory...")
        memory_allocated_gb, memory_reserved_gb = self.measure_memory(batch_size)

        # Measure model size
        logger.info("  Measuring model size...")
        model_size_mb = self.measure_model_size()

        # Validate accuracy
        accuracy = None
        if validation_loader is not None and metric_fn is not None:
            logger.info("  Validating accuracy...")
            accuracy = self.validate_accuracy(validation_loader, metric_fn)

        self.results = BenchmarkResult(
            name=self.name,
            latency_ms=latency_ms,
            throughput_fps=throughput_fps,
            memory_allocated_gb=memory_allocated_gb,
            memory_reserved_gb=memory_reserved_gb,
            accuracy=accuracy,
            model_size_mb=model_size_mb,
            device=self.device,
            batch_size=batch_size,
            iterations=num_iterations,
        )

        return self.results

    def print_report(self):
        """Print benchmark report."""
        if self.results is None:
            logger.warning("No results available. Run benchmark first.")
            return

        print("\n" + "=" * 70)
        print(f"BENCHMARK REPORT: {self.results.name}")
        print("=" * 70)
        print(f"Device:          {self.results.device}")
        print(f"Batch Size:      {self.results.batch_size}")
        print(f"Iterations:      {self.results.iterations}")
        print("-" * 70)
        print(f"Latency:         {self.results.latency_ms:.2f} ms")
        print(f"Throughput:      {self.results.throughput_fps:.1f} FPS")
        print(f"Model Size:      {self.results.model_size_mb:.2f} MB")

        if self.device == "cuda":
            print(f"Memory (Alloc):  {self.results.memory_allocated_gb:.2f} GB")
            print(f"Memory (Resv):   {self.results.memory_reserved_gb:.2f} GB")

        if self.results.accuracy is not None:
            print(f"Accuracy:        {self.results.accuracy:.4f}")

        print("=" * 70 + "\n")

    def save_results(self, output_path: str):
        """Save results to JSON file.

        Parameters
        ----------
        output_path : str
            Output file path
        """
        if self.results is None:
            logger.warning("No results to save")
            return

        with open(output_path, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)

        logger.info(f"✓ Results saved to {output_path}")


class BenchmarkSuite:
    """Benchmark suite for comparing multiple models/configurations.

    Examples
    --------
    >>> suite = BenchmarkSuite()
    >>> suite.add_model("baseline", model_baseline, input_baseline)
    >>> suite.add_model("optimized", model_optimized, input_optimized)
    >>> comparison = suite.run_comparison(batch_size=4)
    >>> suite.print_comparison()
    >>> suite.generate_report("comparison.html")
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.benchmarks: Dict[str, ModelBenchmark] = {}
        self.results: Dict[str, BenchmarkResult] = {}

    def add_model(
        self, name: str, model: nn.Module, sample_input: torch.Tensor
    ):
        """Add model to benchmark suite.

        Parameters
        ----------
        name : str
            Model name
        model : nn.Module
            Model to benchmark
        sample_input : torch.Tensor
            Sample input
        """
        self.benchmarks[name] = ModelBenchmark(
            model, sample_input, device=self.device, name=name
        )
        logger.info(f"✓ Added model: {name}")

    def run_comparison(
        self,
        batch_size: int = 1,
        num_iterations: int = 100,
        validation_loader: Optional[DataLoader] = None,
        metric_fn: Optional[callable] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmarks on all models.

        Parameters
        ----------
        batch_size : int
            Batch size
        num_iterations : int
            Number of iterations
        validation_loader : DataLoader, optional
            Validation dataloader
        metric_fn : callable, optional
            Metric function

        Returns
        -------
        Dict[str, BenchmarkResult]
            Results for each model
        """
        logger.info("=" * 70)
        logger.info("RUNNING BENCHMARK COMPARISON")
        logger.info("=" * 70)

        for name, benchmark in self.benchmarks.items():
            result = benchmark.run_all(
                batch_size=batch_size,
                num_iterations=num_iterations,
                validation_loader=validation_loader,
                metric_fn=metric_fn,
            )
            self.results[name] = result

        return self.results

    def print_comparison(self):
        """Print comparison table."""
        if not self.results:
            logger.warning("No results available. Run comparison first.")
            return

        print("\n" + "=" * 110)
        print("BENCHMARK COMPARISON")
        print("=" * 110)

        # Header
        print(
            f"{'Model':<20} {'Latency (ms)':<15} {'Throughput (FPS)':<20} "
            f"{'Memory (GB)':<15} {'Size (MB)':<15} {'Accuracy':<10}"
        )
        print("-" * 110)

        # Baseline for speedup calculation
        baseline_name = list(self.results.keys())[0]
        baseline_latency = self.results[baseline_name].latency_ms

        # Rows
        for name, result in self.results.items():
            speedup = baseline_latency / result.latency_ms
            speedup_str = f"{speedup:.2f}x" if name != baseline_name else "baseline"

            accuracy_str = (
                f"{result.accuracy:.4f}" if result.accuracy is not None else "N/A"
            )

            print(
                f"{name:<20} {result.latency_ms:<7.2f} ({speedup_str:<6}) "
                f"{result.throughput_fps:<20.1f} {result.memory_allocated_gb:<15.2f} "
                f"{result.model_size_mb:<15.2f} {accuracy_str:<10}"
            )

        print("=" * 110 + "\n")

        # Print recommendations
        self._print_recommendations()

    def _print_recommendations(self):
        """Print optimization recommendations based on results."""
        if len(self.results) < 2:
            return

        print("RECOMMENDATIONS:")
        print("-" * 70)

        baseline_name = list(self.results.keys())[0]
        baseline = self.results[baseline_name]

        for name, result in list(self.results.items())[1:]:
            speedup = baseline.latency_ms / result.latency_ms
            memory_savings = (
                baseline.memory_allocated_gb - result.memory_allocated_gb
            )
            size_reduction = baseline.model_size_mb / result.model_size_mb

            print(f"\n{name}:")
            print(f"  ✓ {speedup:.2f}x faster than baseline")

            if memory_savings > 0:
                print(f"  ✓ {memory_savings:.2f} GB less memory")

            if size_reduction > 1.1:
                print(f"  ✓ {size_reduction:.2f}x smaller model")

            if result.accuracy is not None and baseline.accuracy is not None:
                acc_diff = result.accuracy - baseline.accuracy
                if abs(acc_diff) < 0.01:
                    print(f"  ✓ Accuracy maintained ({acc_diff:+.4f})")
                elif acc_diff < 0:
                    print(f"  ⚠️  Accuracy loss: {acc_diff:.4f}")

        print("-" * 70)

    def generate_report(self, output_path: str):
        """Generate HTML benchmark report.

        Parameters
        ----------
        output_path : str
            Output HTML file path
        """
        if not self.results:
            logger.warning("No results available")
            return

        html = self._generate_html_report()

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"✓ HTML report saved to {output_path}")

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        # Calculate baseline for comparisons
        baseline_name = list(self.results.keys())[0]
        baseline = self.results[baseline_name]

        # Build comparison table
        rows = []
        for name, result in self.results.items():
            speedup = baseline.latency_ms / result.latency_ms
            speedup_str = (
                f"{speedup:.2f}x" if name != baseline_name else "baseline"
            )

            accuracy_str = (
                f"{result.accuracy:.4f}" if result.accuracy is not None else "N/A"
            )

            row = f"""
            <tr>
                <td>{name}</td>
                <td>{result.latency_ms:.2f} ms ({speedup_str})</td>
                <td>{result.throughput_fps:.1f} FPS</td>
                <td>{result.memory_allocated_gb:.2f} GB</td>
                <td>{result.model_size_mb:.2f} MB</td>
                <td>{accuracy_str}</td>
            </tr>
            """
            rows.append(row)

        table_rows = "\n".join(rows)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CytoDL Benchmark Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metadata {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <h1>CytoDL Benchmark Report</h1>

            <div class="metadata">
                <p><strong>Device:</strong> {baseline.device}</p>
                <p><strong>Batch Size:</strong> {baseline.batch_size}</p>
                <p><strong>Iterations:</strong> {baseline.iterations}</p>
            </div>

            <h2>Performance Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Latency</th>
                        <th>Throughput</th>
                        <th>Memory</th>
                        <th>Size</th>
                        <th>Accuracy</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </body>
        </html>
        """

        return html

    def save_results(self, output_path: str):
        """Save all results to JSON.

        Parameters
        ----------
        output_path : str
            Output JSON file path
        """
        if not self.results:
            logger.warning("No results to save")
            return

        results_dict = {
            name: result.to_dict() for name, result in self.results.items()
        }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"✓ Results saved to {output_path}")


def quick_benchmark(
    model: nn.Module,
    sample_input: torch.Tensor,
    batch_size: int = 1,
    num_iterations: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Quick benchmark helper function.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark
    sample_input : torch.Tensor
        Sample input
    batch_size : int
        Batch size
    num_iterations : int
        Number of iterations
    device : str
        Device to use

    Returns
    -------
    Dict[str, float]
        Benchmark metrics

    Examples
    --------
    >>> model = MyModel().cuda()
    >>> input_tensor = torch.randn(1, 1, 256, 256).cuda()
    >>> results = quick_benchmark(model, input_tensor, batch_size=4)
    >>> print(f"Latency: {results['latency_ms']:.2f} ms")
    """
    benchmark = ModelBenchmark(model, sample_input, device=device)
    result = benchmark.run_all(batch_size=batch_size, num_iterations=num_iterations)

    return {
        "latency_ms": result.latency_ms,
        "throughput_fps": result.throughput_fps,
        "memory_gb": result.memory_allocated_gb,
        "model_size_mb": result.model_size_mb,
    }
