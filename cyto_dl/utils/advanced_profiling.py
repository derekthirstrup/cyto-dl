"""Advanced profiling and monitoring tools for performance optimization.

Provides detailed performance metrics including:
- Memory profiling with snapshots
- Operation-level timing
- Bottleneck detection
- GPU utilization monitoring
- Recommendations for optimization

Usage:
    from cyto_dl.utils.advanced_profiling import ProfilerContext, MemoryProfiler

    # Profile training loop
    with ProfilerContext("training") as profiler:
        train_one_epoch(model, dataloader)

    profiler.print_summary()
    profiler.save_report("profile_report.html")
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Advanced memory profiler with snapshots and leak detection.

    Tracks GPU memory usage over time and identifies memory leaks.

    Examples
    --------
    >>> profiler = MemoryProfiler()
    >>> profiler.start()
    >>>
    >>> # Training code
    >>> for batch in dataloader:
    >>>     loss = model(batch)
    >>>     profiler.snapshot("after_forward")
    >>>     loss.backward()
    >>>     profiler.snapshot("after_backward")
    >>>
    >>> profiler.print_summary()
    >>> profiler.detect_leaks()
    """

    def __init__(self):
        self.snapshots = []
        self.start_time = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def start(self):
        """Start profiling."""
        self.start_time = time.time()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.snapshot("start")
        logger.info("✓ Memory profiling started")

    def snapshot(self, name: str):
        """Take a memory snapshot.

        Parameters
        ----------
        name : str
            Snapshot name
        """
        if self.device == "cuda":
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
        else:
            allocated = reserved = peak = 0.0

        self.snapshots.append({
            "name": name,
            "time": time.time() - self.start_time,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "peak_gb": peak,
        })

    def print_summary(self):
        """Print memory usage summary."""
        if not self.snapshots:
            logger.warning("No snapshots taken")
            return

        print("\n" + "="*80)
        print("MEMORY PROFILING SUMMARY")
        print("="*80)
        print(f"{'Snapshot':<30} {'Time (s)':<12} {'Allocated (GB)':<15} {'Reserved (GB)':<15} {'Peak (GB)':<15}")
        print("-"*80)

        for snap in self.snapshots:
            print(f"{snap['name']:<30} {snap['time']:<12.2f} {snap['allocated_gb']:<15.2f} "
                  f"{snap['reserved_gb']:<15.2f} {snap['peak_gb']:<15.2f}")

        print("="*80)

        # Show memory growth
        if len(self.snapshots) > 1:
            growth = self.snapshots[-1]['allocated_gb'] - self.snapshots[0]['allocated_gb']
            print(f"\nMemory Growth: {growth:.2f} GB")

            if growth > 1.0:
                print("⚠️  Warning: Significant memory growth detected!")
                print("   Consider gradient checkpointing or reducing batch size")

    def detect_leaks(self, threshold_gb: float = 0.1) -> List[str]:
        """Detect potential memory leaks.

        Parameters
        ----------
        threshold_gb : float
            Growth threshold to flag as leak (GB)

        Returns
        -------
        List[str]
            List of suspected leak locations
        """
        leaks = []

        for i in range(1, len(self.snapshots)):
            growth = self.snapshots[i]['allocated_gb'] - self.snapshots[i-1]['allocated_gb']

            if growth > threshold_gb:
                leaks.append(
                    f"{self.snapshots[i-1]['name']} → {self.snapshots[i]['name']}: +{growth:.2f} GB"
                )

        if leaks:
            print("\n⚠️  Potential Memory Leaks Detected:")
            for leak in leaks:
                print(f"   {leak}")
        else:
            print("\n✓ No significant memory leaks detected")

        return leaks

    def save_report(self, output_path: str):
        """Save memory profiling report.

        Parameters
        ----------
        output_path : str
            Output file path (.json or .csv)
        """
        import json
        output_path = Path(output_path)

        if output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(self.snapshots, f, indent=2)
        elif output_path.suffix == ".csv":
            import csv
            with open(output_path, "w", newline="") as f:
                if self.snapshots:
                    writer = csv.DictWriter(f, fieldnames=self.snapshots[0].keys())
                    writer.writeheader()
                    writer.writerows(self.snapshots)

        logger.info(f"✓ Memory report saved to {output_path}")


class ProfilerContext:
    """Context manager for profiling code blocks.

    Provides detailed performance metrics including:
    - Execution time
    - Memory usage
    - CUDA kernel activity
    - CPU/GPU utilization

    Examples
    --------
    >>> with ProfilerContext("training") as prof:
    >>>     train_model(model, dataloader)
    >>>
    >>> prof.print_summary()
    >>> prof.export_chrome_trace("trace.json")
    """

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        use_cuda: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        """Initialize profiler context.

        Parameters
        ----------
        name : str
            Profiler name
        enabled : bool
            Enable profiling
        use_cuda : bool
            Profile CUDA operations
        profile_memory : bool
            Profile memory usage
        with_stack : bool
            Record stack traces (slower but more detailed)
        """
        self.name = name
        self.enabled = enabled
        self.profiler = None
        self.start_time = None
        self.end_time = None

        if enabled:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if use_cuda and torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            self.profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=True,
            )

    def __enter__(self):
        if self.enabled:
            self.start_time = time.time()
            self.profiler.__enter__()
            logger.info(f"✓ Profiling '{self.name}' started")
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.profiler.__exit__(*args)
            self.end_time = time.time()
            logger.info(f"✓ Profiling '{self.name}' complete ({self.end_time - self.start_time:.2f}s)")

    def print_summary(self, sort_by: str = "cuda_time_total", top_k: int = 20):
        """Print profiling summary.

        Parameters
        ----------
        sort_by : str
            Sort key: 'cpu_time_total', 'cuda_time_total', 'cpu_memory_usage', etc.
        top_k : int
            Number of top operations to show
        """
        if not self.enabled or self.profiler is None:
            logger.warning("Profiler not enabled")
            return

        print("\n" + "="*80)
        print(f"PROFILING SUMMARY: {self.name}")
        print("="*80)
        print(f"Total time: {self.end_time - self.start_time:.2f}s\n")

        print(f"Top {top_k} operations by {sort_by}:")
        print(self.profiler.key_averages().table(sort_by=sort_by, row_limit=top_k))
        print("="*80)

    def export_chrome_trace(self, output_path: str):
        """Export Chrome trace for visualization.

        Parameters
        ----------
        output_path : str
            Output path for trace file
        """
        if not self.enabled or self.profiler is None:
            logger.warning("Profiler not enabled")
            return

        self.profiler.export_chrome_trace(output_path)
        logger.info(f"✓ Chrome trace exported to {output_path}")
        logger.info(f"   View at: chrome://tracing")

    def export_stacks(self, output_path: str):
        """Export stack traces.

        Parameters
        ----------
        output_path : str
            Output file path
        """
        if not self.enabled or self.profiler is None:
            logger.warning("Profiler not enabled")
            return

        self.profiler.export_stacks(output_path, "self_cuda_time_total")
        logger.info(f"✓ Stack traces exported to {output_path}")


class BottleneckDetector:
    """Detect performance bottlenecks in training/inference.

    Identifies:
    - Data loading bottlenecks
    - GPU underutilization
    - Memory bottlenecks
    - Inefficient operations

    Examples
    --------
    >>> detector = BottleneckDetector()
    >>> detector.start()
    >>>
    >>> for batch in dataloader:
    >>>     detector.mark("data_loading")
    >>>     output = model(batch)
    >>>     detector.mark("forward")
    >>>     loss.backward()
    >>>     detector.mark("backward")
    >>>
    >>> bottlenecks = detector.analyze()
    >>> detector.print_recommendations()
    """

    def __init__(self):
        self.timestamps = []
        self.marks = []
        self.gpu_utils = []

    def start(self):
        """Start bottleneck detection."""
        self.start_time = time.time()
        self.mark("start")

    def mark(self, name: str):
        """Mark a point in execution.

        Parameters
        ----------
        name : str
            Mark name
        """
        current_time = time.time() - self.start_time

        # Record GPU utilization if available
        gpu_util = 0.0
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except:
                pass

        self.marks.append({
            "name": name,
            "time": current_time,
            "gpu_util": gpu_util,
        })

    def analyze(self) -> Dict[str, float]:
        """Analyze bottlenecks.

        Returns
        -------
        Dict[str, float]
            Time spent in each phase
        """
        if len(self.marks) < 2:
            logger.warning("Need at least 2 marks to analyze")
            return {}

        phases = {}
        for i in range(1, len(self.marks)):
            phase_name = f"{self.marks[i-1]['name']} → {self.marks[i]['name']}"
            duration = self.marks[i]['time'] - self.marks[i-1]['time']
            phases[phase_name] = duration

        return phases

    def print_recommendations(self):
        """Print optimization recommendations based on detected bottlenecks."""
        phases = self.analyze()

        if not phases:
            return

        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS & RECOMMENDATIONS")
        print("="*80)

        # Calculate total time
        total_time = sum(phases.values())

        # Print phase breakdown
        print("\nPhase Breakdown:")
        for phase, duration in sorted(phases.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100
            print(f"  {phase:<40} {duration:>8.3f}s ({percentage:>5.1f}%)")

        # Detect bottlenecks and recommend
        print("\nRecommendations:")

        data_loading_time = sum(d for p, d in phases.items() if "data_loading" in p)
        compute_time = sum(d for p, d in phases.items() if "forward" in p or "backward" in p)

        if data_loading_time > compute_time * 0.2:
            print("  ⚠️  Data loading bottleneck detected!")
            print("     → Increase num_workers in DataLoader")
            print("     → Enable pin_memory=True")
            print("     → Use persistent_workers=True")
            print("     → Consider data caching/prefetching")

        # Check GPU utilization
        avg_gpu_util = sum(m['gpu_util'] for m in self.marks) / len(self.marks)
        if avg_gpu_util < 70:
            print(f"  ⚠️  Low GPU utilization ({avg_gpu_util:.1f}%)")
            print("     → Increase batch size")
            print("     → Reduce data loading time")
            print("     → Enable mixed precision training")
            print("     → Use torch.compile")

        print("="*80)


@contextmanager
def memory_snapshot(name: str):
    """Context manager for quick memory snapshots.

    Examples
    --------
    >>> with memory_snapshot("forward_pass"):
    >>>     output = model(input)
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1e9

    start_time = time.time()

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated() / 1e9
        mem_diff = end_mem - start_mem

        print(f"\n{name}:")
        print(f"  Time: {time.time() - start_time:.3f}s")
        print(f"  Memory: {mem_diff:+.3f} GB (now {end_mem:.3f} GB)")
    else:
        print(f"\n{name}:")
        print(f"  Time: {time.time() - start_time:.3f}s")
