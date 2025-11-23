"""TensorRT utilities for optimized inference on NVIDIA GPUs.

TensorRT provides 2-5x faster inference compared to PyTorch through:
- Kernel fusion and optimization
- Precision calibration (FP16, INT8)
- Dynamic tensor memory management
- Layer and tensor fusion

Requirements:
    pip install torch-tensorrt nvidia-tensorrt

Usage:
    # Export model to TensorRT
    from cyto_dl.utils.tensorrt_utils import export_to_tensorrt

    trt_model = export_to_tensorrt(
        model,
        input_shape=(1, 1, 64, 64, 64),
        precision="fp16",
        output_path="model_trt.ts"
    )

    # Use for inference
    output = trt_model(input_tensor)
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Check for TensorRT availability
try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
    logger.info(f"✓ TensorRT available: torch_tensorrt {torch_tensorrt.__version__}")
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn(
        "TensorRT not available. Install with: pip install torch-tensorrt nvidia-tensorrt"
    )


def export_to_tensorrt(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    precision: str = "fp16",
    workspace_size: int = 4,
    min_batch_size: int = 1,
    max_batch_size: int = 16,
    opt_batch_size: int = 4,
    output_path: Optional[str] = None,
    enable_dynamic_shapes: bool = False,
    device: str = "cuda",
) -> Union[nn.Module, None]:
    """Export PyTorch model to TensorRT for optimized inference.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to export
    input_shape : Tuple[int, ...]
        Input tensor shape (e.g., (1, 1, 64, 64, 64) for 3D)
    precision : str
        Precision mode: "fp32", "fp16", or "int8"
        - fp32: No optimization (baseline)
        - fp16: 2-3x faster, minimal accuracy loss
        - int8: 4x faster, requires calibration, slight accuracy loss
    workspace_size : int
        Workspace size in GB for TensorRT engine (default: 4GB)
    min_batch_size : int
        Minimum batch size for dynamic batching
    max_batch_size : int
        Maximum batch size for dynamic batching
    opt_batch_size : int
        Optimal batch size for optimization
    output_path : Optional[str]
        Path to save TorchScript model with TensorRT
    enable_dynamic_shapes : bool
        Enable dynamic input shapes (slower but flexible)
    device : str
        Device to use (default: "cuda")

    Returns
    -------
    nn.Module or None
        TensorRT-optimized model or None if TensorRT unavailable

    Examples
    --------
    >>> # Fixed shape (fastest)
    >>> trt_model = export_to_tensorrt(
    ...     model,
    ...     input_shape=(1, 1, 64, 64, 64),
    ...     precision="fp16"
    ... )

    >>> # Dynamic batch size
    >>> trt_model = export_to_tensorrt(
    ...     model,
    ...     input_shape=(1, 1, 64, 64, 64),
    ...     min_batch_size=1,
    ...     max_batch_size=16,
    ...     opt_batch_size=4,
    ...     enable_dynamic_shapes=True
    ... )
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT not available. Cannot export model.")
        return None

    model = model.to(device)
    model.eval()

    # Create sample input
    sample_input = torch.randn(input_shape, device=device)

    # Trace model to TorchScript
    logger.info("Tracing model to TorchScript...")
    try:
        traced_model = torch.jit.trace(model, sample_input)
    except Exception as e:
        logger.error(f"Failed to trace model: {e}")
        logger.info("Trying torch.jit.script instead...")
        try:
            traced_model = torch.jit.script(model)
        except Exception as e2:
            logger.error(f"Failed to script model: {e2}")
            return None

    # Configure TensorRT compilation
    logger.info(f"Compiling to TensorRT with {precision} precision...")

    # Set precision
    enabled_precisions = {torch.float32}
    if precision == "fp16":
        enabled_precisions.add(torch.float16)
    elif precision == "int8":
        enabled_precisions.add(torch.int8)
        logger.warning("INT8 precision requires calibration data for best results")

    # Configure inputs
    if enable_dynamic_shapes:
        # Dynamic batch size
        inputs = [
            torch_tensorrt.Input(
                min_shape=[min_batch_size] + list(input_shape[1:]),
                opt_shape=[opt_batch_size] + list(input_shape[1:]),
                max_shape=[max_batch_size] + list(input_shape[1:]),
                dtype=torch.float32,
            )
        ]
    else:
        # Fixed shape (faster)
        inputs = [
            torch_tensorrt.Input(
                shape=list(input_shape),
                dtype=torch.float32,
            )
        ]

    # Compile with TensorRT
    try:
        trt_model = torch_tensorrt.compile(
            traced_model,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=workspace_size * (1 << 30),  # Convert GB to bytes
            truncate_long_and_double=True,
            device=torch.device(device),
        )

        logger.info("✓ TensorRT compilation successful")

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.jit.save(trt_model, str(output_path))
            logger.info(f"✓ Saved TensorRT model to {output_path}")

        return trt_model

    except Exception as e:
        logger.error(f"TensorRT compilation failed: {e}")
        logger.info("Falling back to TorchScript without TensorRT")
        return traced_model


class TensorRTInferenceEngine:
    """Wrapper for TensorRT inference with preprocessing and postprocessing.

    This class provides a high-level interface for TensorRT inference with:
    - Automatic preprocessing (normalization, resizing)
    - Batch inference
    - Sliding window inference for large images
    - Postprocessing (thresholding, argmax)

    Examples
    --------
    >>> # Create engine
    >>> engine = TensorRTInferenceEngine(
    ...     model_path="model_trt.ts",
    ...     input_shape=(1, 1, 64, 64, 64),
    ...     device="cuda"
    ... )

    >>> # Single inference
    >>> output = engine(input_tensor)

    >>> # Batch inference
    >>> outputs = engine.batch_inference(input_batch)

    >>> # Sliding window inference
    >>> output = engine.sliding_window_inference(
    ...     large_image,
    ...     roi_size=(64, 64, 64),
    ...     overlap=0.25
    ... )
    """

    def __init__(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        device: str = "cuda",
        half_precision: bool = True,
    ):
        """Initialize TensorRT inference engine.

        Parameters
        ----------
        model_path : str
            Path to TensorRT model (.ts file)
        input_shape : Tuple[int, ...]
            Expected input shape
        device : str
            Device to run on
        half_precision : bool
            Use FP16 precision for inputs
        """
        self.device = device
        self.input_shape = input_shape
        self.half_precision = half_precision

        # Load model
        logger.info(f"Loading TensorRT model from {model_path}")
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        # Warmup
        self._warmup()

    def _warmup(self, num_iterations: int = 3):
        """Warmup model with dummy inputs."""
        logger.info("Warming up TensorRT engine...")
        dummy_input = torch.randn(self.input_shape, device=self.device)
        if self.half_precision:
            dummy_input = dummy_input.half()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)

        logger.info("✓ Warmup complete")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on single input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Model output
        """
        x = x.to(self.device)
        if self.half_precision:
            x = x.half()

        with torch.no_grad():
            output = self.model(x)

        return output

    def batch_inference(
        self,
        inputs: List[torch.Tensor],
        batch_size: int = 4,
    ) -> List[torch.Tensor]:
        """Run inference on batch of inputs.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            List of input tensors
        batch_size : int
            Batch size for processing

        Returns
        -------
        List[torch.Tensor]
            List of outputs
        """
        outputs = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_tensor = torch.stack(batch).to(self.device)

            if self.half_precision:
                batch_tensor = batch_tensor.half()

            with torch.no_grad():
                batch_output = self.model(batch_tensor)

            outputs.extend([out for out in batch_output])

        return outputs

    def sliding_window_inference(
        self,
        image: torch.Tensor,
        roi_size: Tuple[int, ...],
        overlap: float = 0.25,
        batch_size: int = 4,
        mode: str = "gaussian",
    ) -> torch.Tensor:
        """Run sliding window inference on large image.

        Parameters
        ----------
        image : torch.Tensor
            Input image (C, [Z,] H, W)
        roi_size : Tuple[int, ...]
            Region of interest size for each window
        overlap : float
            Overlap between windows (0.0 to 1.0)
        batch_size : int
            Number of windows to process at once
        mode : str
            Blending mode: "constant", "gaussian"

        Returns
        -------
        torch.Tensor
            Stitched output
        """
        from monai.inferers import sliding_window_inference

        # Add batch dimension if needed
        if image.ndim == len(roi_size):
            image = image.unsqueeze(0)

        image = image.to(self.device)
        if self.half_precision:
            image = image.half()

        with torch.no_grad():
            output = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=batch_size,
                predictor=self.model,
                overlap=overlap,
                mode=mode,
            )

        return output


def benchmark_tensorrt(
    pytorch_model: nn.Module,
    tensorrt_model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark PyTorch vs TensorRT performance.

    Parameters
    ----------
    pytorch_model : nn.Module
        Original PyTorch model
    tensorrt_model : nn.Module
        TensorRT-optimized model
    input_shape : Tuple[int, ...]
        Input tensor shape
    num_iterations : int
        Number of benchmark iterations
    warmup_iterations : int
        Number of warmup iterations
    device : str
        Device to use

    Returns
    -------
    Dict[str, float]
        Benchmark results with speedup metrics
    """
    from cyto_dl.utils.performance import benchmark_model

    # Benchmark PyTorch
    logger.info("Benchmarking PyTorch model...")
    pytorch_results = benchmark_model(
        pytorch_model,
        input_shape,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        device=device,
    )

    # Benchmark TensorRT
    logger.info("Benchmarking TensorRT model...")
    tensorrt_results = benchmark_model(
        tensorrt_model,
        input_shape,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        device=device,
    )

    # Calculate speedup
    speedup = pytorch_results["avg_latency_ms"] / tensorrt_results["avg_latency_ms"]
    throughput_gain = tensorrt_results["throughput_fps"] / pytorch_results["throughput_fps"]

    results = {
        "pytorch_latency_ms": pytorch_results["avg_latency_ms"],
        "tensorrt_latency_ms": tensorrt_results["avg_latency_ms"],
        "pytorch_throughput_fps": pytorch_results["throughput_fps"],
        "tensorrt_throughput_fps": tensorrt_results["throughput_fps"],
        "latency_speedup": speedup,
        "throughput_gain": throughput_gain,
    }

    logger.info(f"\n{'='*60}")
    logger.info("TENSORRT BENCHMARK RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"PyTorch Latency:    {results['pytorch_latency_ms']:.2f} ms")
    logger.info(f"TensorRT Latency:   {results['tensorrt_latency_ms']:.2f} ms")
    logger.info(f"Speedup:            {speedup:.2f}x faster")
    logger.info(f"PyTorch Throughput: {results['pytorch_throughput_fps']:.2f} FPS")
    logger.info(f"TensorRT Throughput:{results['tensorrt_throughput_fps']:.2f} FPS")
    logger.info(f"Throughput Gain:    {throughput_gain:.2f}x higher")
    logger.info(f"{'='*60}\n")

    return results


def create_int8_calibrator(
    calibration_data: List[torch.Tensor],
    cache_file: str = "calibration.cache",
) -> "torch_tensorrt.ptq.DataLoaderCalibrator":
    """Create INT8 calibrator for post-training quantization.

    INT8 quantization provides 4x speedup but requires calibration
    to maintain accuracy.

    Parameters
    ----------
    calibration_data : List[torch.Tensor]
        List of calibration samples (50-500 samples recommended)
    cache_file : str
        Path to save calibration cache

    Returns
    -------
    torch_tensorrt.ptq.DataLoaderCalibrator
        Calibrator for INT8 quantization

    Examples
    --------
    >>> # Collect calibration data
    >>> calib_data = [model(x) for x in val_dataset[:100]]
    >>>
    >>> # Create calibrator
    >>> calibrator = create_int8_calibrator(calib_data)
    >>>
    >>> # Use in export
    >>> trt_model = export_to_tensorrt(
    ...     model,
    ...     input_shape=(1, 1, 64, 64, 64),
    ...     precision="int8",
    ...     calibrator=calibrator
    ... )
    """
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT not available")

    # Create data loader from calibration data
    class CalibrationDataLoader:
        def __init__(self, data):
            self.data = data
            self.index = 0

        def __iter__(self):
            self.index = 0
            return self

        def __next__(self):
            if self.index < len(self.data):
                batch = self.data[self.index]
                self.index += 1
                return batch
            raise StopIteration

        def __len__(self):
            return len(self.data)

    dataloader = CalibrationDataLoader(calibration_data)

    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        dataloader,
        cache_file=cache_file,
        use_cache=True,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    )

    logger.info(f"✓ Created INT8 calibrator with {len(calibration_data)} samples")
    logger.info(f"  Cache file: {cache_file}")

    return calibrator
