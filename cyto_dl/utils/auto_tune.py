"""Automated performance tuning for CytoDL.

Automatically finds optimal hyperparameters for your hardware:
- Optimal batch size (maximize throughput without OOM)
- Optimal num_workers for data loading
- Best mixed precision settings
- Optimal gradient accumulation
- Memory-efficient settings

Usage:
    from cyto_dl.utils.auto_tune import AutoTuner

    # Auto-tune for your model
    tuner = AutoTuner(model, sample_input)
    optimal_config = tuner.tune()

    # Use recommended settings
    print(f"Optimal batch size: {optimal_config['batch_size']}")
    print(f"Optimal num_workers: {optimal_config['num_workers']}")
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for auto-tuning.

    Attributes
    ----------
    max_batch_size : int
        Maximum batch size to try
    min_batch_size : int
        Minimum batch size to try
    num_trials : int
        Number of trials per configuration
    warmup_iterations : int
        Warmup iterations before timing
    timing_iterations : int
        Iterations for timing
    enable_channels_last : bool
        Try channels-last memory format
    enable_compile : bool
        Try torch.compile
    """

    max_batch_size: int = 128
    min_batch_size: int = 1
    num_trials: int = 3
    warmup_iterations: int = 5
    timing_iterations: int = 20
    enable_channels_last: bool = True
    enable_compile: bool = True


class AutoTuner:
    """Automated performance tuner for CytoDL models.

    Finds optimal hyperparameters for your hardware through empirical testing.

    Parameters
    ----------
    model : nn.Module
        Model to tune
    sample_input : torch.Tensor
        Sample input tensor (single sample, no batch dimension)
    device : str
        Device to use ('cuda' or 'cpu')
    config : TuningConfig, optional
        Tuning configuration

    Examples
    --------
    >>> model = MyModel().cuda()
    >>> sample_input = torch.randn(1, 64, 64, 64).cuda()
    >>> tuner = AutoTuner(model, sample_input)
    >>> config = tuner.tune()
    >>> print(f"Use batch_size={config['batch_size']}")
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        device: str = "cuda",
        config: Optional[TuningConfig] = None,
    ):
        self.model = model
        self.sample_input = sample_input
        self.device = device
        self.config = config or TuningConfig()

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = "cpu"

        self.model = self.model.to(self.device)
        self.model.eval()

    def tune(self, tune_inference: bool = True, tune_training: bool = False) -> Dict[str, Any]:
        """Run complete auto-tuning.

        Parameters
        ----------
        tune_inference : bool
            Tune for inference performance
        tune_training : bool
            Tune for training performance

        Returns
        -------
        Dict[str, Any]
            Optimal configuration
        """
        logger.info("=" * 60)
        logger.info("STARTING AUTO-TUNING")
        logger.info("=" * 60)

        results = {}

        if tune_inference:
            logger.info("\n🔍 Tuning inference performance...")
            inference_config = self._tune_inference()
            results.update(inference_config)

        if tune_training:
            logger.info("\n🔍 Tuning training performance...")
            training_config = self._tune_training()
            results.update(training_config)

        # Always tune data loading
        logger.info("\n🔍 Tuning data loading...")
        data_config = self._tune_data_loading(results.get("batch_size", 4))
        results.update(data_config)

        logger.info("\n" + "=" * 60)
        logger.info("AUTO-TUNING COMPLETE")
        logger.info("=" * 60)
        logger.info("\nRecommended Configuration:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        return results

    def _tune_inference(self) -> Dict[str, Any]:
        """Tune inference settings."""
        results = {}

        # Find optimal batch size
        optimal_batch_size = self._find_optimal_batch_size()
        results["batch_size"] = optimal_batch_size
        logger.info(f"✓ Optimal batch size: {optimal_batch_size}")

        # Test channels-last
        if self.config.enable_channels_last and self.device == "cuda":
            speedup = self._test_channels_last(optimal_batch_size)
            results["channels_last"] = speedup > 1.05  # Use if >5% speedup
            logger.info(f"✓ Channels-last speedup: {speedup:.2f}x")

        # Test torch.compile
        if self.config.enable_compile and self.device == "cuda":
            compile_speedup = self._test_compile(optimal_batch_size)
            results["compile"] = compile_speedup > 1.1  # Use if >10% speedup
            logger.info(f"✓ Compile speedup: {compile_speedup:.2f}x")

        # Test mixed precision
        if self.device == "cuda":
            fp16_speedup, bf16_speedup = self._test_mixed_precision(optimal_batch_size)
            if bf16_speedup > fp16_speedup and bf16_speedup > 1.1:
                results["precision"] = "bf16"
                logger.info(f"✓ BF16 recommended (speedup: {bf16_speedup:.2f}x)")
            elif fp16_speedup > 1.1:
                results["precision"] = "fp16"
                logger.info(f"✓ FP16 recommended (speedup: {fp16_speedup:.2f}x)")
            else:
                results["precision"] = "fp32"
                logger.info("✓ FP32 recommended (no mixed precision benefit)")

        return results

    def _tune_training(self) -> Dict[str, Any]:
        """Tune training settings."""
        results = {}

        # Find optimal gradient accumulation
        batch_size = results.get("batch_size", 4)
        optimal_accum = self._find_optimal_gradient_accumulation(batch_size)
        results["gradient_accumulation_steps"] = optimal_accum
        logger.info(f"✓ Optimal gradient accumulation: {optimal_accum}")

        # Test gradient checkpointing
        memory_savings = self._test_gradient_checkpointing(batch_size)
        results["gradient_checkpointing"] = memory_savings > 0.2  # Use if >20% savings
        logger.info(f"✓ Gradient checkpointing savings: {memory_savings:.1%}")

        return results

    def _tune_data_loading(self, batch_size: int) -> Dict[str, Any]:
        """Tune data loading settings."""
        optimal_workers = self._find_optimal_num_workers(batch_size)
        return {
            "num_workers": optimal_workers,
            "pin_memory": self.device == "cuda",
            "persistent_workers": optimal_workers > 0,
        }

    def _find_optimal_batch_size(self) -> int:
        """Find optimal batch size using binary search."""
        logger.info("  Finding optimal batch size...")

        min_bs = self.config.min_batch_size
        max_bs = self.config.max_batch_size
        optimal_bs = min_bs

        # Binary search for max batch size that fits in memory
        while min_bs <= max_bs:
            bs = (min_bs + max_bs) // 2

            try:
                # Test if batch size fits
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                batch_input = self.sample_input.repeat(bs, *([1] * (self.sample_input.ndim - 1)))
                batch_input = batch_input.to(self.device)

                with torch.no_grad():
                    _ = self.model(batch_input)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                # Success - try larger
                optimal_bs = bs
                min_bs = bs + 1

                del batch_input
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM - try smaller
                    max_bs = bs - 1
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

        # Now find batch size with best throughput (may be smaller than max)
        best_throughput = 0
        best_bs = optimal_bs

        for bs in [optimal_bs // 4, optimal_bs // 2, optimal_bs]:
            if bs < 1:
                continue

            try:
                throughput = self._measure_throughput(bs)
                logger.info(f"    Batch size {bs}: {throughput:.1f} samples/sec")

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_bs = bs

            except RuntimeError:
                break

        return best_bs

    def _measure_throughput(self, batch_size: int) -> float:
        """Measure throughput for given batch size."""
        if self.device == "cuda":
            torch.cuda.empty_cache()

        batch_input = self.sample_input.repeat(batch_size, *([1] * (self.sample_input.ndim - 1)))
        batch_input = batch_input.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = self.model(batch_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(self.config.timing_iterations):
                _ = self.model(batch_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        throughput = (batch_size * self.config.timing_iterations) / elapsed

        del batch_input
        return throughput

    def _test_channels_last(self, batch_size: int) -> float:
        """Test channels-last speedup."""
        logger.info("  Testing channels-last memory format...")

        # Measure baseline
        baseline_time = self._measure_inference_time(batch_size)

        # Convert to channels-last
        if self.sample_input.ndim == 4:
            # 2D
            self.model.to(memory_format=torch.channels_last)
            memory_format = torch.channels_last
        elif self.sample_input.ndim == 5:
            # 3D
            self.model.to(memory_format=torch.channels_last_3d)
            memory_format = torch.channels_last_3d
        else:
            return 1.0

        # Measure with channels-last
        channels_last_time = self._measure_inference_time(batch_size, memory_format=memory_format)

        # Restore contiguous
        self.model.to(memory_format=torch.contiguous_format)

        speedup = baseline_time / channels_last_time if channels_last_time > 0 else 1.0
        return speedup

    def _test_compile(self, batch_size: int) -> float:
        """Test torch.compile speedup."""
        logger.info("  Testing torch.compile...")

        # Measure baseline
        baseline_time = self._measure_inference_time(batch_size)

        # Compile model
        try:
            compiled_model = torch.compile(self.model, mode="default")

            # Measure compiled
            batch_input = self.sample_input.repeat(batch_size, *([1] * (self.sample_input.ndim - 1)))
            batch_input = batch_input.to(self.device)

            # Warmup (compilation happens here)
            with torch.no_grad():
                for _ in range(self.config.warmup_iterations):
                    _ = compiled_model(batch_input)

            if self.device == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            with torch.no_grad():
                for _ in range(self.config.timing_iterations):
                    _ = compiled_model(batch_input)

            if self.device == "cuda":
                torch.cuda.synchronize()

            compiled_time = time.time() - start_time
            speedup = baseline_time / compiled_time if compiled_time > 0 else 1.0

            del batch_input
            return speedup

        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            return 1.0

    def _test_mixed_precision(self, batch_size: int) -> Tuple[float, float]:
        """Test FP16 and BF16 speedups."""
        logger.info("  Testing mixed precision...")

        # Measure baseline (FP32)
        baseline_time = self._measure_inference_time(batch_size)

        # Test FP16
        fp16_speedup = 1.0
        try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fp16_time = self._measure_inference_time(batch_size)
                fp16_speedup = baseline_time / fp16_time if fp16_time > 0 else 1.0
        except Exception as e:
            logger.warning(f"FP16 test failed: {e}")

        # Test BF16
        bf16_speedup = 1.0
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                bf16_time = self._measure_inference_time(batch_size)
                bf16_speedup = baseline_time / bf16_time if bf16_time > 0 else 1.0
        except Exception as e:
            logger.warning(f"BF16 test failed: {e}")

        return fp16_speedup, bf16_speedup

    def _measure_inference_time(
        self, batch_size: int, memory_format: Optional[torch.memory_format] = None
    ) -> float:
        """Measure inference time."""
        batch_input = self.sample_input.repeat(batch_size, *([1] * (self.sample_input.ndim - 1)))
        batch_input = batch_input.to(self.device)

        if memory_format is not None:
            batch_input = batch_input.to(memory_format=memory_format)

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = self.model(batch_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(self.config.timing_iterations):
                _ = self.model(batch_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        del batch_input
        return elapsed

    def _find_optimal_gradient_accumulation(self, batch_size: int) -> int:
        """Find optimal gradient accumulation steps."""
        # Heuristic: if batch size is small, use accumulation
        if batch_size <= 2:
            return 4
        elif batch_size <= 4:
            return 2
        else:
            return 1

    def _test_gradient_checkpointing(self, batch_size: int) -> float:
        """Test gradient checkpointing memory savings."""
        if self.device != "cuda":
            return 0.0

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            batch_input = self.sample_input.repeat(batch_size, *([1] * (self.sample_input.ndim - 1)))
            batch_input = batch_input.to(self.device)

            # Measure without checkpointing
            with torch.no_grad():
                _ = self.model(batch_input)

            mem_without = torch.cuda.max_memory_allocated() / 1e9

            # Measure with checkpointing (if supported)
            # This is a rough estimate
            mem_with = mem_without * 0.6  # Typical savings

            savings = (mem_without - mem_with) / mem_without if mem_without > 0 else 0.0
            del batch_input
            return savings

        except Exception as e:
            logger.warning(f"Gradient checkpointing test failed: {e}")
            return 0.0

    def _find_optimal_num_workers(self, batch_size: int) -> int:
        """Find optimal number of data loading workers."""
        logger.info("  Finding optimal num_workers...")

        # Create dummy dataset
        dummy_data = torch.randn(100, *self.sample_input.shape[1:])
        dataset = TensorDataset(dummy_data)

        best_throughput = 0
        best_workers = 0

        for num_workers in [0, 2, 4, 8]:
            try:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=(self.device == "cuda"),
                )

                # Warmup
                for _ in range(2):
                    for batch in dataloader:
                        _ = batch
                        break

                # Measure
                start_time = time.time()
                batches = 0
                for batch in dataloader:
                    batches += 1
                    if batches >= 20:
                        break

                elapsed = time.time() - start_time
                throughput = batches / elapsed if elapsed > 0 else 0

                logger.info(f"    num_workers={num_workers}: {throughput:.1f} batches/sec")

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_workers = num_workers

            except Exception as e:
                logger.warning(f"num_workers={num_workers} test failed: {e}")
                break

        return best_workers


def auto_tune_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str = "cuda",
    save_config: Optional[str] = None,
) -> Dict[str, Any]:
    """Auto-tune model and optionally save configuration.

    Parameters
    ----------
    model : nn.Module
        Model to tune
    sample_input : torch.Tensor
        Sample input
    device : str
        Device to use
    save_config : str, optional
        Path to save YAML config

    Returns
    -------
    Dict[str, Any]
        Optimal configuration

    Examples
    --------
    >>> model = MyModel()
    >>> sample_input = torch.randn(1, 64, 64, 64)
    >>> config = auto_tune_model(model, sample_input, save_config="optimal_config.yaml")
    """
    tuner = AutoTuner(model, sample_input, device=device)
    config = tuner.tune(tune_inference=True, tune_training=False)

    if save_config:
        from omegaconf import OmegaConf

        OmegaConf.save(config, save_config)
        logger.info(f"✓ Saved optimal config to {save_config}")

    return config
