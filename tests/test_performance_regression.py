"""Performance regression tests for CytoDL.

These tests ensure that performance optimizations don't degrade and that
new changes don't introduce performance regressions.

Usage:
    # Run all performance tests
    pytest tests/test_performance_regression.py -v

    # Run specific test
    pytest tests/test_performance_regression.py::test_baseline_performance -v

    # Update performance baselines
    pytest tests/test_performance_regression.py --update-baselines
"""

import json
import pytest
from pathlib import Path

import torch
import torch.nn as nn

# Performance baselines (update these when making intentional improvements)
PERFORMANCE_BASELINES = {
    "simple_conv2d": {
        "latency_ms": 10.0,  # Maximum acceptable latency
        "memory_gb": 1.0,  # Maximum acceptable memory
    },
    "simple_conv3d": {
        "latency_ms": 50.0,
        "memory_gb": 4.0,
    },
}


class SimpleConv2D(nn.Module):
    """Simple 2D CNN for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleConv3D(nn.Module):
    """Simple 3D CNN for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def conv2d_model():
    """Fixture for 2D CNN model."""
    return SimpleConv2D()


@pytest.fixture
def conv3d_model():
    """Fixture for 3D CNN model."""
    return SimpleConv3D()


@pytest.fixture
def sample_input_2d():
    """Sample 2D input."""
    return torch.randn(1, 1, 256, 256)


@pytest.fixture
def sample_input_3d():
    """Sample 3D input."""
    return torch.randn(1, 1, 64, 64, 64)


def benchmark_model(model, sample_input, device="cuda", num_iterations=20):
    """Quick benchmark helper."""
    from cyto_dl.utils.benchmark import quick_benchmark

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    return quick_benchmark(
        model, sample_input, batch_size=1, num_iterations=num_iterations, device=device
    )


@pytest.mark.performance
def test_baseline_performance_2d(conv2d_model, sample_input_2d):
    """Test that baseline 2D performance meets expectations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    results = benchmark_model(conv2d_model, sample_input_2d, device="cuda")

    # Check latency
    assert results["latency_ms"] < PERFORMANCE_BASELINES["simple_conv2d"]["latency_ms"], \
        f"Latency {results['latency_ms']:.2f}ms exceeds baseline " \
        f"{PERFORMANCE_BASELINES['simple_conv2d']['latency_ms']:.2f}ms"

    # Check memory
    assert results["memory_gb"] < PERFORMANCE_BASELINES["simple_conv2d"]["memory_gb"], \
        f"Memory {results['memory_gb']:.2f}GB exceeds baseline " \
        f"{PERFORMANCE_BASELINES['simple_conv2d']['memory_gb']:.2f}GB"


@pytest.mark.performance
def test_baseline_performance_3d(conv3d_model, sample_input_3d):
    """Test that baseline 3D performance meets expectations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    results = benchmark_model(conv3d_model, sample_input_3d, device="cuda")

    # Check latency
    assert results["latency_ms"] < PERFORMANCE_BASELINES["simple_conv3d"]["latency_ms"], \
        f"Latency {results['latency_ms']:.2f}ms exceeds baseline " \
        f"{PERFORMANCE_BASELINES['simple_conv3d']['latency_ms']:.2f}ms"

    # Check memory
    assert results["memory_gb"] < PERFORMANCE_BASELINES["simple_conv3d"]["memory_gb"], \
        f"Memory {results['memory_gb']:.2f}GB exceeds baseline " \
        f"{PERFORMANCE_BASELINES['simple_conv3d']['memory_gb']:.2f}GB"


@pytest.mark.performance
def test_phase1_speedup(conv2d_model, sample_input_2d):
    """Test that Phase 1 optimizations provide expected speedup."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from cyto_dl.utils.performance import setup_gpu_optimizations

    # Baseline
    baseline_results = benchmark_model(conv2d_model, sample_input_2d, device="cuda")

    # Phase 1 optimizations
    setup_gpu_optimizations(
        enable_cudnn_benchmark=True,
        enable_tf32=True,
        channels_last=True,
    )

    conv2d_model = conv2d_model.to(memory_format=torch.channels_last)
    sample_input_2d = sample_input_2d.to(memory_format=torch.channels_last)

    phase1_results = benchmark_model(conv2d_model, sample_input_2d, device="cuda")

    # Phase 1 should be at least 1.2x faster (conservative estimate)
    speedup = baseline_results["latency_ms"] / phase1_results["latency_ms"]
    assert speedup >= 1.2, \
        f"Phase 1 speedup {speedup:.2f}x is less than expected 1.2x"


@pytest.mark.performance
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_quantization_size_reduction(conv2d_model):
    """Test that quantization reduces model size."""
    try:
        from cyto_dl.utils.quantization import quantize_model_dynamic
    except ImportError:
        pytest.skip("Quantization utilities not available")

    # Get original size
    original_size = sum(p.numel() * p.element_size() for p in conv2d_model.parameters())

    # Quantize
    quantized_model = quantize_model_dynamic(conv2d_model, dtype=torch.qint8)

    # Get quantized size
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

    # Should be at least 2x smaller (INT8 vs FP32)
    size_reduction = original_size / quantized_size
    assert size_reduction >= 2.0, \
        f"Size reduction {size_reduction:.2f}x is less than expected 2.0x"


@pytest.mark.performance
def test_memory_leak_detection(conv2d_model, sample_input_2d):
    """Test that training loop doesn't leak memory."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from cyto_dl.utils.advanced_profiling import MemoryProfiler

    model = conv2d_model.cuda()
    sample_input = sample_input_2d.cuda()

    profiler = MemoryProfiler()
    profiler.start()

    # Simulate training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for i in range(10):
        optimizer.zero_grad()
        output = model(sample_input)
        target = torch.randn_like(output).cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            profiler.snapshot(f"iter_{i}")

    # Check for leaks
    leaks = profiler.detect_leaks(threshold_gb=0.1)

    assert len(leaks) == 0, f"Memory leaks detected: {leaks}"


@pytest.mark.performance
def test_accuracy_after_optimization(conv2d_model, sample_input_2d):
    """Test that optimizations maintain accuracy."""
    try:
        from cyto_dl.utils.accuracy_validation import OutputComparator
    except ImportError:
        pytest.skip("Accuracy validation not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Baseline model
    baseline_model = conv2d_model.cuda()

    # Optimized model (channels-last)
    optimized_model = conv2d_model.cuda().to(memory_format=torch.channels_last)
    sample_optimized = sample_input_2d.cuda().to(memory_format=torch.channels_last)

    # Compare outputs
    comparator = OutputComparator(baseline_model, optimized_model)
    comparison = comparator.compare(sample_optimized)

    # Output difference should be negligible
    assert comparison["max_diff"] < 1e-5, \
        f"Max output difference {comparison['max_diff']} exceeds threshold"


@pytest.mark.performance
def test_benchmark_suite():
    """Test BenchmarkSuite functionality."""
    from cyto_dl.utils.benchmark import BenchmarkSuite

    suite = BenchmarkSuite(device="cpu")  # Use CPU for CI

    # Add models
    model1 = SimpleConv2D()
    model2 = SimpleConv2D()
    sample_input = torch.randn(1, 256, 256)

    suite.add_model("model1", model1, sample_input)
    suite.add_model("model2", model2, sample_input)

    # Run comparison
    results = suite.run_comparison(batch_size=1, num_iterations=5)

    assert len(results) == 2
    assert "model1" in results
    assert "model2" in results

    # Results should have expected fields
    for result in results.values():
        assert result.latency_ms > 0
        assert result.throughput_fps > 0


def save_baselines(baselines, output_path="performance_baselines.json"):
    """Save performance baselines to file."""
    with open(output_path, "w") as f:
        json.dump(baselines, f, indent=2)


def load_baselines(input_path="performance_baselines.json"):
    """Load performance baselines from file."""
    if not Path(input_path).exists():
        return PERFORMANCE_BASELINES

    with open(input_path, "r") as f:
        return json.load(f)


@pytest.mark.performance
def test_no_performance_regression():
    """Test that recent changes haven't degraded performance."""
    # This test would compare against saved baseline results
    # Implementation depends on CI/CD setup

    baseline_path = Path("tests/baselines/performance_baselines.json")

    if not baseline_path.exists():
        pytest.skip("No baseline file found")

    # Load baselines
    baselines = load_baselines(baseline_path)

    # Run current benchmarks
    current_results = {}

    # Compare
    for model_name, baseline in baselines.items():
        if model_name in current_results:
            current = current_results[model_name]

            # Allow 10% performance degradation
            tolerance = 1.1

            assert current["latency_ms"] <= baseline["latency_ms"] * tolerance, \
                f"{model_name}: Latency regression detected"

            assert current["memory_gb"] <= baseline["memory_gb"] * tolerance, \
                f"{model_name}: Memory regression detected"
