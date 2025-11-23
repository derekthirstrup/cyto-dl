# Phase 4: Benchmarking & Performance Validation

This guide covers Phase 4: comprehensive benchmarking, testing, and performance validation for CytoDL optimizations.

## Overview

Phase 4 provides tools to:
- **Benchmark models** across all optimization phases
- **Validate accuracy** after optimizations
- **Compare performance** of different configurations
- **Generate reports** (HTML, JSON, console)
- **Prevent regressions** with automated tests

## Quick Start

### Benchmark All Phases

```bash
python scripts/benchmark_performance.py \
  --config configs/experiment/im2im/labelfree.yaml \
  --ckpt path/to/checkpoint.ckpt \
  --all-phases \
  --generate-html
```

### Validate Optimization Accuracy

```python
from cyto_dl.utils.accuracy_validation import validate_optimization

passed = validate_optimization(
    baseline_model=model_fp32,
    optimized_model=model_quantized,
    validation_loader=val_loader,
    tolerance=0.01  # 1% acceptable degradation
)

if passed:
    print("✓ Safe to deploy!")
```

### Compare Multiple Models

```python
from cyto_dl.utils.benchmark import BenchmarkSuite

suite = BenchmarkSuite()
suite.add_model("baseline", model_baseline, sample_input)
suite.add_model("phase1", model_phase1, sample_input)
suite.add_model("phase2_trt", model_trt, sample_input)

results = suite.run_comparison(batch_size=4)
suite.print_comparison()
suite.generate_report("comparison.html")
```

## Tools & Components

### 1. Benchmarking Framework (`cyto_dl/utils/benchmark.py`)

**ModelBenchmark** - Benchmark a single model:

```python
from cyto_dl.utils.benchmark import ModelBenchmark

benchmark = ModelBenchmark(model, sample_input, device="cuda")
result = benchmark.run_all(batch_size=4, num_iterations=100)
benchmark.print_report()
benchmark.save_results("results.json")
```

**BenchmarkSuite** - Compare multiple models:

```python
from cyto_dl.utils.benchmark import BenchmarkSuite

suite = BenchmarkSuite(device="cuda")
suite.add_model("baseline", model1, input1)
suite.add_model("optimized", model2, input2)

comparison = suite.run_comparison()
suite.print_comparison()
suite.generate_report("report.html")
```

**Quick Benchmark**:

```python
from cyto_dl.utils.benchmark import quick_benchmark

results = quick_benchmark(model, sample_input, batch_size=4)
print(f"Latency: {results['latency_ms']:.2f} ms")
print(f"Throughput: {results['throughput_fps']:.1f} FPS")
```

### 2. Accuracy Validation (`cyto_dl/utils/accuracy_validation.py`)

**AccuracyValidator** - Comprehensive validation:

```python
from cyto_dl.utils.accuracy_validation import AccuracyValidator

validator = AccuracyValidator(
    baseline_model=model_baseline,
    optimized_model=model_optimized,
    validation_loader=val_loader,
    tolerance=0.01  # 1% tolerance
)

results = validator.validate()
validator.print_report()

if results["passed"]:
    print("✓ Accuracy maintained!")
```

**OutputComparator** - Debug differences:

```python
from cyto_dl.utils.accuracy_validation import OutputComparator

comparator = OutputComparator(model_a, model_b)
comparison = comparator.compare(input_tensor)
comparator.visualize_diff(comparison)
```

**Metrics Computed:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Cosine Similarity
- Structural Similarity Index (SSIM)
- Max Absolute Difference
- Relative Error

### 3. End-to-End Benchmarking Script

```bash
# Benchmark baseline vs Phase 1
python scripts/benchmark_performance.py \
  --config configs/experiment/im2im/segmentation.yaml \
  --ckpt path/to/checkpoint.ckpt \
  --baseline --phase1

# Benchmark all phases with validation
python scripts/benchmark_performance.py \
  --config configs/experiment/im2im/labelfree.yaml \
  --ckpt path/to/checkpoint.ckpt \
  --all-phases \
  --validation-data path/to/val/images \
  --generate-html

# Test different batch sizes
python scripts/benchmark_performance.py \
  --config configs/experiment/im2im/mae.yaml \
  --batch-sizes 1 2 4 8 16
```

### 4. Performance Regression Tests

Run automated performance tests:

```bash
# Run all performance tests
pytest tests/test_performance_regression.py -v -m performance

# Run specific test
pytest tests/test_performance_regression.py::test_baseline_performance_2d -v

# Skip slow tests
pytest tests/test_performance_regression.py -v -m "not slow"
```

**Available Tests:**
- `test_baseline_performance_2d` - 2D model baseline
- `test_baseline_performance_3d` - 3D model baseline
- `test_phase1_speedup` - Phase 1 improvements
- `test_quantization_size_reduction` - Quantization effectiveness
- `test_memory_leak_detection` - Memory leak detection
- `test_accuracy_after_optimization` - Accuracy preservation

## Benchmarking Workflow

### Step 1: Benchmark Baseline

```python
from cyto_dl.utils.benchmark import ModelBenchmark

# Load your model
model = load_model("path/to/checkpoint.ckpt")
sample_input = torch.randn(1, 1, 256, 256).cuda()

# Benchmark baseline
benchmark = ModelBenchmark(model, sample_input, name="Baseline")
baseline_result = benchmark.run_all(batch_size=4)
benchmark.print_report()
```

### Step 2: Apply Optimizations

```python
from cyto_dl.utils.performance import setup_gpu_optimizations

# Phase 1 optimizations
setup_gpu_optimizations()
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model)

# Benchmark optimized
benchmark_opt = ModelBenchmark(model, sample_input, name="Phase1")
phase1_result = benchmark_opt.run_all(batch_size=4)
benchmark_opt.print_report()
```

### Step 3: Validate Accuracy

```python
from cyto_dl.utils.accuracy_validation import validate_optimization

passed = validate_optimization(
    baseline_model=model_baseline,
    optimized_model=model_optimized,
    validation_loader=val_loader,
    tolerance=0.01
)
```

### Step 4: Generate Report

```python
from cyto_dl.utils.benchmark import BenchmarkSuite

suite = BenchmarkSuite()
suite.add_model("baseline", model_baseline, sample_input)
suite.add_model("phase1", model_phase1, sample_input)

results = suite.run_comparison()
suite.generate_report("optimization_report.html")
```

## Benchmark Reports

### Console Output

```
================================================================================
BENCHMARK COMPARISON
================================================================================
Model                Latency (ms)    Throughput (FPS)    Memory (GB)     Size (MB)       Accuracy
--------------------------------------------------------------------------------------------------------------
baseline             11.24 (baseline) 88.9                4.52            342.14          0.9234
phase1               6.18 (1.82x)    161.8               4.48            342.14          0.9231
phase2_trt           4.21 (2.67x)    237.5               2.31            86.12           0.9228
phase3_quantized     8.92 (1.26x)    112.1               1.15            85.54           0.9219
================================================================================

RECOMMENDATIONS:
--------------------------------------------------------------------------------

phase1:
  ✓ 1.82x faster than baseline
  ✓ Accuracy maintained (+0.0003)

phase2_trt:
  ✓ 2.67x faster than baseline
  ✓ 2.21 GB less memory
  ✓ 4.0x smaller model
  ✓ Accuracy maintained (-0.0006)

phase3_quantized:
  ✓ 1.26x faster than baseline
  ✓ 3.37 GB less memory
  ✓ 4.0x smaller model
  ⚠️  Accuracy loss: -0.0015
```

### HTML Report

Generates interactive HTML with:
- Performance comparison table
- Charts and visualizations
- Metadata (device, batch size, iterations)
- Exportable results

### JSON Results

```json
{
  "baseline": {
    "name": "baseline",
    "latency_ms": 11.24,
    "throughput_fps": 88.9,
    "memory_allocated_gb": 4.52,
    "model_size_mb": 342.14,
    "accuracy": 0.9234,
    "device": "cuda",
    "batch_size": 4,
    "iterations": 100
  },
  "phase1": {
    ...
  }
}
```

## Performance Testing in CI/CD

### GitHub Actions Example

```yaml
name: Performance Tests

on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest

    - name: Run performance tests
      run: |
        pytest tests/test_performance_regression.py -v -m performance

    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: performance-results
        path: benchmark_results/
```

## Best Practices

### 1. Benchmark Consistently

- Use the **same hardware** for comparisons
- Run **multiple iterations** (100+)
- Include **warmup** runs (10+)
- Use **fixed random seeds** for reproducibility

### 2. Validate Accuracy

- Always check accuracy after optimizations
- Set appropriate **tolerance** (0.01 = 1%)
- Use **representative validation data**
- Compare on **full validation set**, not just a few samples

### 3. Monitor Regressions

- Save baseline results to version control
- Run regression tests in CI/CD
- Alert on >10% performance degradation
- Track performance over time

### 4. Generate Reports

- Create reports for significant changes
- Include before/after comparisons
- Document optimization decisions
- Share results with team

### 5. Optimize Iteratively

1. **Benchmark** baseline
2. **Apply** one optimization
3. **Validate** accuracy
4. **Measure** improvement
5. **Repeat** with next optimization

## Troubleshooting

### Issue: Inconsistent Results

**Solution:**
```python
# Fix random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Increase iterations
benchmark.run_all(num_iterations=500)
```

### Issue: Accuracy Degradation

**Solution:**
```python
# Check tolerance
validator = AccuracyValidator(tolerance=0.02)  # Allow 2%

# Investigate specific metrics
results = validator.validate()
print(f"MAE: {results['metrics'].mae}")
print(f"PSNR: {results['metrics'].psnr} dB")

# Compare individual outputs
comparator = OutputComparator(model_a, model_b)
comparison = comparator.compare(input_tensor)
```

### Issue: OOM During Benchmarking

**Solution:**
```bash
# Reduce batch size
python scripts/benchmark_performance.py --batch-sizes 1 2

# Benchmark on CPU
python scripts/benchmark_performance.py --device cpu

# Skip memory-intensive phases
python scripts/benchmark_performance.py --baseline --phase1  # Skip Phase 2/3
```

## Summary

Phase 4 provides comprehensive tools for:

- ✅ **Benchmarking** models across all optimization levels
- ✅ **Validating** accuracy after optimizations
- ✅ **Comparing** performance of different configurations
- ✅ **Generating** detailed reports (HTML, JSON, console)
- ✅ **Preventing** performance regressions with automated tests

### Complete Workflow Example

```bash
# 1. Benchmark all phases
python scripts/benchmark_performance.py \
  --config configs/experiment/im2im/labelfree.yaml \
  --ckpt logs/best.ckpt \
  --all-phases \
  --validation-data data/val \
  --generate-html \
  --output benchmark_results/

# 2. View results
open benchmark_results/comparison.html

# 3. Run regression tests
pytest tests/test_performance_regression.py -v

# 4. Deploy optimized model
# (if accuracy and performance acceptable)
```

For more information, see:
- [Phase 1: GPU Optimizations](GPU_OPTIMIZATION_GUIDE.md)
- [Phase 2: TensorRT](TENSORRT_GUIDE.md)
- [Phase 3: Advanced Optimizations](PHASE3_ADVANCED_OPTIMIZATIONS.md)
- [Performance Summary](PERFORMANCE_OPTIMIZATIONS.md)
