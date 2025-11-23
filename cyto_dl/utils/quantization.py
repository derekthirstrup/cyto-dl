"""PyTorch native quantization utilities for model optimization.

This module provides INT8 quantization for PyTorch models without requiring TensorRT.
Useful for CPU deployment, edge devices, or when TensorRT is not available.

Quantization provides:
- 4x smaller model size
- 2-4x faster inference (CPU and GPU)
- Minimal accuracy loss (<1-2% with proper calibration)

Types of Quantization:
1. Dynamic Quantization - Easiest, good for RNN/LSTM/Transformer
2. Static Quantization - Best accuracy, requires calibration
3. Quantization Aware Training (QAT) - Best results, requires retraining

Usage:
    # Dynamic quantization (easiest)
    from cyto_dl.utils.quantization import quantize_dynamic
    quantized_model = quantize_dynamic(model)

    # Static quantization (best accuracy)
    from cyto_dl.utils.quantization import quantize_static
    quantized_model = quantize_static(model, calibration_loader)
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

import torch
import torch.nn as nn
from torch.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    quantize_dynamic,
    fuse_modules,
)

logger = logging.getLogger(__name__)


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    output_path: Optional[str] = None,
) -> nn.Module:
    """Apply dynamic quantization to model.

    Dynamic quantization quantizes weights statically and activations dynamically.
    Best for models with:
    - Linear/LSTM/GRU layers
    - Varying input sizes
    - CPU inference

    Parameters
    ----------
    model : nn.Module
        Model to quantize
    dtype : torch.dtype
        Quantization dtype (qint8 or float16)
    output_path : Optional[str]
        Path to save quantized model

    Returns
    -------
    nn.Module
        Dynamically quantized model

    Examples
    --------
    >>> quantized = quantize_model_dynamic(model)
    >>> # Model is now 4x smaller
    >>> output = quantized(input)  # 2-4x faster on CPU
    """
    logger.info("Applying dynamic quantization...")

    # Apply dynamic quantization to Linear and LSTM layers
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell},
        dtype=dtype,
    )

    # Calculate size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    reduction = (1 - quantized_size / original_size) * 100

    logger.info(f"✓ Dynamic quantization complete")
    logger.info(f"  Original size: {original_size / 1e6:.2f} MB")
    logger.info(f"  Quantized size: {quantized_size / 1e6:.2f} MB")
    logger.info(f"  Reduction: {reduction:.1f}%")

    if output_path:
        torch.save(quantized_model.state_dict(), output_path)
        logger.info(f"✓ Saved to {output_path}")

    return quantized_model


def fuse_model_layers(model: nn.Module, fuse_list: Optional[List[List[str]]] = None) -> nn.Module:
    """Fuse consecutive operations for better quantization.

    Fusing combines operations like Conv+BN+ReLU into a single operation,
    which improves quantization accuracy and performance.

    Parameters
    ----------
    model : nn.Module
        Model to fuse
    fuse_list : Optional[List[List[str]]]
        List of layer names to fuse. If None, uses common patterns.

    Returns
    -------
    nn.Module
        Model with fused layers

    Examples
    --------
    >>> # Auto-detect and fuse
    >>> fused_model = fuse_model_layers(model)
    >>>
    >>> # Manual fusion
    >>> fused_model = fuse_model_layers(model, [['conv1', 'bn1', 'relu1']])
    """
    logger.info("Fusing model layers...")

    if fuse_list is None:
        # Common fusion patterns
        fuse_list = []

        # Find Conv+BN+ReLU patterns
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                layers = list(module.children())
                if len(layers) >= 2:
                    # Conv + BN
                    if isinstance(layers[0], (nn.Conv2d, nn.Conv3d)) and isinstance(layers[1], (nn.BatchNorm2d, nn.BatchNorm3d)):
                        pattern = [f"{name}.0", f"{name}.1"]
                        if len(layers) >= 3 and isinstance(layers[2], nn.ReLU):
                            pattern.append(f"{name}.2")
                        fuse_list.append(pattern)

    if fuse_list:
        try:
            model = fuse_modules(model, fuse_list)
            logger.info(f"✓ Fused {len(fuse_list)} layer groups")
        except Exception as e:
            logger.warning(f"Layer fusion failed: {e}")
    else:
        logger.info("No fusible layers found")

    return model


def quantize_model_static(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    backend: str = "fbgemm",
    output_path: Optional[str] = None,
) -> nn.Module:
    """Apply static quantization with calibration.

    Static quantization achieves better accuracy than dynamic by calibrating
    activation ranges using representative data.

    Parameters
    ----------
    model : nn.Module
        Model to quantize
    calibration_loader : DataLoader
        DataLoader with calibration data (50-500 samples)
    backend : str
        Quantization backend: 'fbgemm' (x86 CPU) or 'qnnpack' (ARM)
    output_path : Optional[str]
        Path to save quantized model

    Returns
    -------
    nn.Module
        Statically quantized model

    Examples
    --------
    >>> # Prepare calibration data
    >>> calib_loader = DataLoader(calib_dataset, batch_size=8)
    >>>
    >>> # Quantize
    >>> quantized = quantize_model_static(model, calib_loader)
    """
    logger.info("Applying static quantization...")

    # Set backend
    torch.backends.quantized.engine = backend
    logger.info(f"Using backend: {backend}")

    # Set model to eval mode
    model.eval()

    # Fuse layers
    model = fuse_model_layers(model)

    # Attach qconfig
    model.qconfig = get_default_qconfig(backend)

    # Prepare for calibration
    model_prepared = prepare(model, inplace=False)

    # Calibrate with representative data
    logger.info(f"Calibrating with {len(calibration_loader)} batches...")
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get('input', batch.get('raw', batch.get('x', None)))
            else:
                inputs = batch

            if inputs is None:
                logger.warning(f"Batch {i}: Could not find input tensor")
                continue

            try:
                _ = model_prepared(inputs)
            except Exception as e:
                logger.warning(f"Batch {i} failed: {e}")

    logger.info("✓ Calibration complete")

    # Convert to quantized model
    quantized_model = convert(model_prepared, inplace=False)

    # Calculate size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    reduction = (1 - quantized_size / original_size) * 100

    logger.info(f"✓ Static quantization complete")
    logger.info(f"  Original size: {original_size / 1e6:.2f} MB")
    logger.info(f"  Quantized size: {quantized_size / 1e6:.2f} MB")
    logger.info(f"  Reduction: {reduction:.1f}%")

    if output_path:
        torch.save(quantized_model.state_dict(), output_path)
        logger.info(f"✓ Saved to {output_path}")

    return quantized_model


class QuantizationAwareTraining:
    """Quantization Aware Training (QAT) for best quantization accuracy.

    QAT simulates quantization during training, allowing the model to learn
    quantization-friendly weights. Provides best accuracy but requires retraining.

    Examples
    --------
    >>> qat = QuantizationAwareTraining(model, backend='fbgemm')
    >>> qat_model = qat.prepare()
    >>>
    >>> # Train for a few epochs
    >>> for epoch in range(3):
    >>>     train_one_epoch(qat_model, train_loader, optimizer)
    >>>
    >>> # Convert to quantized
    >>> quantized_model = qat.convert()
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = "fbgemm",
    ):
        """Initialize QAT.

        Parameters
        ----------
        model : nn.Module
            Model to quantize
        backend : str
            Quantization backend
        """
        self.model = model
        self.backend = backend
        self.prepared_model = None

        torch.backends.quantized.engine = backend

    def prepare(self) -> nn.Module:
        """Prepare model for QAT.

        Returns
        -------
        nn.Module
            Model ready for QAT training
        """
        logger.info("Preparing model for Quantization Aware Training...")

        # Fuse layers
        self.model = fuse_model_layers(self.model)

        # Set training mode
        self.model.train()

        # Attach qconfig
        from torch.quantization import get_default_qat_qconfig
        self.model.qconfig = get_default_qat_qconfig(self.backend)

        # Prepare for QAT
        from torch.quantization import prepare_qat
        self.prepared_model = prepare_qat(self.model, inplace=False)

        logger.info("✓ Model prepared for QAT")
        logger.info("  Train for 3-5 epochs with normal training loop")

        return self.prepared_model

    def convert(self) -> nn.Module:
        """Convert QAT model to fully quantized model.

        Returns
        -------
        nn.Module
            Quantized model
        """
        if self.prepared_model is None:
            raise RuntimeError("Must call prepare() before convert()")

        logger.info("Converting QAT model to quantized model...")

        self.prepared_model.eval()
        quantized_model = convert(self.prepared_model, inplace=False)

        logger.info("✓ QAT conversion complete")

        return quantized_model


def benchmark_quantized_model(
    original_model: nn.Module,
    quantized_model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
    device: str = "cpu",
) -> dict:
    """Benchmark original vs quantized model.

    Parameters
    ----------
    original_model : nn.Module
        Original FP32 model
    quantized_model : nn.Module
        Quantized model
    input_shape : tuple
        Input tensor shape
    num_iterations : int
        Number of benchmark iterations
    device : str
        Device to benchmark on ('cpu' recommended for quantization)

    Returns
    -------
    dict
        Benchmark results
    """
    import time

    logger.info(f"Benchmarking on {device}...")

    original_model = original_model.to(device)
    quantized_model = quantized_model.to(device)

    original_model.eval()
    quantized_model.eval()

    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original_model(dummy_input)
            _ = quantized_model(dummy_input)

    # Benchmark original
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = original_model(dummy_input)
    original_time = time.time() - start

    # Benchmark quantized
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = quantized_model(dummy_input)
    quantized_time = time.time() - start

    speedup = original_time / quantized_time

    results = {
        "original_time_ms": (original_time / num_iterations) * 1000,
        "quantized_time_ms": (quantized_time / num_iterations) * 1000,
        "speedup": speedup,
        "device": device,
    }

    logger.info("="*60)
    logger.info("QUANTIZATION BENCHMARK")
    logger.info("="*60)
    logger.info(f"Original:  {results['original_time_ms']:.2f} ms")
    logger.info(f"Quantized: {results['quantized_time_ms']:.2f} ms")
    logger.info(f"Speedup:   {speedup:.2f}x faster")
    logger.info("="*60)

    return results


def export_quantized_model(
    model: nn.Module,
    output_path: str,
    example_input: torch.Tensor,
    export_format: str = "torchscript",
) -> None:
    """Export quantized model for deployment.

    Parameters
    ----------
    model : nn.Module
        Quantized model
    output_path : str
        Output file path
    example_input : torch.Tensor
        Example input for tracing
    export_format : str
        Export format: 'torchscript', 'onnx', or 'state_dict'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if export_format == "torchscript":
        traced = torch.jit.trace(model, example_input)
        torch.jit.save(traced, str(output_path))
        logger.info(f"✓ Exported to TorchScript: {output_path}")

    elif export_format == "onnx":
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
        )
        logger.info(f"✓ Exported to ONNX: {output_path}")

    elif export_format == "state_dict":
        torch.save(model.state_dict(), str(output_path))
        logger.info(f"✓ Exported state dict: {output_path}")

    else:
        raise ValueError(f"Unknown format: {export_format}")
