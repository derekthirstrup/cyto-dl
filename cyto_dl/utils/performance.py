"""Performance optimization utilities for GPU acceleration."""

import logging
import os
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def setup_gpu_optimizations(
    enable_cudnn_benchmark: bool = True,
    enable_tf32: bool = True,
    matmul_precision: str = "high",
    channels_last: bool = True,
) -> dict:
    """Setup global GPU optimizations for PyTorch.

    Parameters
    ----------
    enable_cudnn_benchmark : bool
        Enable cudnn autotuner for optimal convolution algorithms.
        Best for fixed input sizes. Default True.
    enable_tf32 : bool
        Enable TF32 tensor cores on Ampere+ GPUs (A100, 4090, 5080, etc).
        Provides speedup with minimal accuracy impact. Default True.
    matmul_precision : str
        Matrix multiplication precision: "highest", "high", or "medium".
        "high" enables TF32, "medium" uses TF32+other optimizations. Default "high".
    channels_last : bool
        Whether to recommend channels-last memory format.
        Returns recommendation, actual conversion happens per-model.

    Returns
    -------
    dict
        Dictionary with applied settings for logging/debugging
    """
    applied_settings = {}

    if torch.cuda.is_available():
        # Enable cudnn benchmarking for optimal conv algorithms
        if enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            applied_settings["cudnn_benchmark"] = True
            logger.info("✓ Enabled cudnn.benchmark for optimal convolution algorithms")

        # Enable TF32 for matrix multiplications (Ampere+ GPUs)
        if enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            applied_settings["tf32_enabled"] = True
            logger.info("✓ Enabled TF32 tensor cores for matmul and cudnn")

        # Set matmul precision
        torch.set_float32_matmul_precision(matmul_precision)
        applied_settings["matmul_precision"] = matmul_precision
        logger.info(f"✓ Set float32 matmul precision to '{matmul_precision}'")

        # Log channels-last recommendation
        if channels_last:
            applied_settings["channels_last_recommended"] = True
            logger.info(
                "✓ Channels-last memory format recommended (apply per-model with model.to(memory_format=torch.channels_last))"
            )

        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        applied_settings["gpu"] = gpu_name
        logger.info(f"✓ GPU detected: {gpu_name}")

    else:
        logger.warning("No CUDA GPU detected, GPU optimizations skipped")
        applied_settings["gpu"] = "CPU"

    return applied_settings


def convert_to_channels_last(model: torch.nn.Module, spatial_dims: int = 3) -> torch.nn.Module:
    """Convert model to channels-last memory format for better GPU performance.

    Channels-last format provides 20-30% speedup on modern GPUs for convolutions.
    Optimal for Ampere/Ada Lovelace architecture (A100, 4090, 5080).

    Parameters
    ----------
    model : torch.nn.Module
        Model to convert
    spatial_dims : int
        Number of spatial dimensions (2 for 2D, 3 for 3D)

    Returns
    -------
    torch.nn.Module
        Model with channels-last memory format
    """
    if spatial_dims == 2:
        model = model.to(memory_format=torch.channels_last)
        logger.info("✓ Converted model to channels_last (NHWC) memory format for 2D")
    elif spatial_dims == 3:
        model = model.to(memory_format=torch.channels_last_3d)
        logger.info("✓ Converted model to channels_last_3d (NDHWC) memory format for 3D")
    return model


def get_optimal_num_workers() -> int:
    """Calculate optimal number of dataloader workers based on CPU count.

    Returns
    -------
    int
        Recommended number of workers
    """
    cpu_count = os.cpu_count() or 4
    # Use half of CPU cores, capped at 8 to avoid memory issues
    num_workers = min(cpu_count // 2, 8)
    # Ensure at least 2 workers for parallelism
    num_workers = max(num_workers, 2)
    return num_workers


def enable_compile_if_available(
    model: torch.nn.Module,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> torch.nn.Module:
    """Apply torch.compile if available and supported.

    Parameters
    ----------
    model : torch.nn.Module
        Model to compile
    mode : str
        Compilation mode: "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        - "default": Balanced speed and compilation time
        - "reduce-overhead": Best for inference (CUDA graphs)
        - "max-autotune": Longest compile, best performance
    fullgraph : bool
        Whether to compile the entire graph (may fail for dynamic models)
    dynamic : bool
        Whether to support dynamic shapes (slower but more flexible)

    Returns
    -------
    torch.nn.Module
        Compiled model or original model if compilation unavailable
    """
    import sys

    if sys.platform.startswith("win"):
        logger.warning("torch.compile not supported on Windows, skipping")
        return model

    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(
                model, mode=mode, fullgraph=fullgraph, dynamic=dynamic
            )
            logger.info(f"✓ Applied torch.compile with mode='{mode}'")
            return compiled_model
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}, using uncompiled model")
            return model
    else:
        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
        return model


class CUDAGraphWrapper:
    """Wrapper to capture and replay CUDA graphs for faster inference.

    CUDA graphs reduce CPU overhead by capturing GPU operations once
    and replaying them without CPU involvement.

    Best for:
    - Fixed input shapes
    - Inference (not training)
    - Repetitive operations

    Speedup: 5-15% for inference
    """

    def __init__(self, model: torch.nn.Module, sample_input: torch.Tensor):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to wrap
        sample_input : torch.Tensor
            Sample input tensor for capturing graph (must match inference shape)
        """
        self.model = model
        self.model.eval()

        # Create static copies for graph capture
        self.static_input = torch.zeros_like(sample_input)
        self.static_output = None

        # Warmup runs
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(3):
                self.model(self.static_input)

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

        logger.info("✓ CUDA graph captured for model inference")

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference using captured CUDA graph.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor (must match captured shape)

        Returns
        -------
        torch.Tensor
            Model output
        """
        # Copy input to static buffer
        self.static_input.copy_(input_tensor)

        # Replay graph (fast!)
        self.graph.replay()

        # Return output
        return self.static_output.detach().clone()


def benchmark_model(
    model: torch.nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> dict:
    """Benchmark model inference speed.

    Parameters
    ----------
    model : torch.nn.Module
        Model to benchmark
    input_shape : tuple
        Input tensor shape (e.g., (1, 1, 64, 64, 64) for 3D)
    num_iterations : int
        Number of iterations for benchmarking
    warmup_iterations : int
        Number of warmup iterations
    device : str
        Device to run on

    Returns
    -------
    dict
        Benchmark results with timing statistics
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
    else:
        import time

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end = time.perf_counter()
        elapsed_time_ms = (end - start) * 1000

    avg_time_ms = elapsed_time_ms / num_iterations
    throughput = 1000 / avg_time_ms  # images per second

    results = {
        "avg_latency_ms": round(avg_time_ms, 2),
        "throughput_fps": round(throughput, 2),
        "total_time_ms": round(elapsed_time_ms, 2),
        "num_iterations": num_iterations,
        "input_shape": input_shape,
        "device": device,
    }

    logger.info(f"Benchmark results: {avg_time_ms:.2f}ms per iteration, {throughput:.2f} FPS")

    return results
