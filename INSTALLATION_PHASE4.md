# Phase 4: Installation Guide

Phase 4 benchmarking and testing tools require **NO new dependencies!**

## Quick Start

Phase 4 works immediately with your existing CytoDL installation:

```bash
# Verify Phase 4 is ready
python -c "from cyto_dl.utils.benchmark import BenchmarkSuite; print('✓ Phase 4 ready!')"
python -c "from cyto_dl.utils.accuracy_validation import AccuracyValidator; print('✓ Accuracy validation ready!')"
```

**Expected output:**
```
✓ Phase 4 ready!
✓ Accuracy validation ready!
```

## What's Included

Phase 4 includes (all built into existing dependencies):

| Tool | File | Dependencies |
|------|------|--------------|
| **Benchmarking Framework** | `cyto_dl/utils/benchmark.py` | PyTorch (already installed) |
| **Accuracy Validation** | `cyto_dl/utils/accuracy_validation.py` | PyTorch (already installed) |
| **Benchmark Script** | `scripts/benchmark_performance.py` | PyTorch (already installed) |
| **Regression Tests** | `tests/test_performance_regression.py` | pytest (dev dependency) |

## Installation

### Option 1: Already Installed!

If you have CytoDL installed from Phase 1-3, Phase 4 is ready to use:

```bash
# No installation needed!
python scripts/benchmark_performance.py --help
```

### Option 2: Fresh Installation

```bash
# Clone repository
git clone https://github.com/derekthirstrup/cyto-dl
cd cyto-dl

# Checkout Phase 4 branch
git checkout claude/benchmarking-phase4-014viXtwt7gNsiG4xedaMKNA

# Install in development mode
pip install -e .

# Verify
python -c "from cyto_dl.utils.benchmark import BenchmarkSuite; print('✓ Ready!')"
```

### Option 3: For Testing/Development

Install development dependencies for running tests:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or install pytest manually
pip install pytest

# Run performance tests
pytest tests/test_performance_regression.py -v
```

## Verification

### Verify All Tools

```python
# verify_phase4.py
import torch

print("Verifying Phase 4 Installation...")
print("=" * 60)

# 1. Benchmarking Framework
try:
    from cyto_dl.utils.benchmark import BenchmarkSuite, ModelBenchmark, quick_benchmark
    print("✓ Benchmarking Framework: Available")
except ImportError as e:
    print(f"✗ Benchmarking Framework: Failed - {e}")

# 2. Accuracy Validation
try:
    from cyto_dl.utils.accuracy_validation import AccuracyValidator, OutputComparator
    print("✓ Accuracy Validation: Available")
except ImportError as e:
    print(f"✗ Accuracy Validation: Failed - {e}")

# 3. Benchmark Script
try:
    from pathlib import Path
    script_path = Path("scripts/benchmark_performance.py")
    if script_path.exists():
        print("✓ Benchmark Script: Available")
    else:
        print("✗ Benchmark Script: Not found")
except Exception as e:
    print(f"✗ Benchmark Script: Failed - {e}")

# 4. Test basic functionality
try:
    import torch.nn as nn

    # Create dummy model
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
    sample_input = torch.randn(1, 10)

    # Quick benchmark
    results = quick_benchmark(model, sample_input, device="cpu", num_iterations=5)

    print(f"✓ Benchmark Test: Passed (latency={results['latency_ms']:.2f}ms)")
except Exception as e:
    print(f"✗ Benchmark Test: Failed - {e}")

print("=" * 60)
print("\nPhase 4 Verification Complete!")
```

Run verification:
```bash
python verify_phase4.py
```

**Expected output:**
```
Verifying Phase 4 Installation...
============================================================
✓ Benchmarking Framework: Available
✓ Accuracy Validation: Available
✓ Benchmark Script: Available
✓ Benchmark Test: Passed (latency=0.23ms)
============================================================

Phase 4 Verification Complete!
```

## Quick Test

Test the benchmarking framework:

```bash
# Create a simple test script
cat > test_phase4.py << 'EOF'
import torch
import torch.nn as nn
from cyto_dl.utils.benchmark import quick_benchmark

# Simple model
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10)
)

# Sample input
sample_input = torch.randn(1, 28, 28)

# Benchmark
device = "cuda" if torch.cuda.is_available() else "cpu"
results = quick_benchmark(model, sample_input, batch_size=4, device=device)

print(f"Latency:    {results['latency_ms']:.2f} ms")
print(f"Throughput: {results['throughput_fps']:.1f} FPS")
print(f"Memory:     {results['memory_gb']:.2f} GB")

print("\n✓ Phase 4 is working!")
EOF

# Run test
python test_phase4.py
```

## Troubleshooting

### Issue: Import Errors

**Error:**
```
ImportError: cannot import name 'BenchmarkSuite'
```

**Solution:**
```bash
# Ensure you're on the correct branch
git checkout claude/benchmarking-phase4-014viXtwt7gNsiG4xedaMKNA

# Reinstall in development mode
pip install -e .

# Verify
python -c "from cyto_dl.utils.benchmark import BenchmarkSuite"
```

### Issue: Pytest Not Found

**Error:**
```
bash: pytest: command not found
```

**Solution:**
```bash
# Install pytest
pip install pytest

# Or install dev dependencies
pip install -e ".[dev]"

# Verify
pytest --version
```

### Issue: Performance Tests Fail

**Error:**
```
test_baseline_performance_2d FAILED - CUDA not available
```

**Solution:**
```bash
# Skip CUDA-dependent tests
pytest tests/test_performance_regression.py -v -k "not cuda"

# Or run on CPU only
pytest tests/test_performance_regression.py -v --device cpu
```

## Platform Notes

### Linux (Ubuntu/Debian)

Phase 4 works out-of-the-box:

```bash
# No special setup needed
python -c "from cyto_dl.utils.benchmark import BenchmarkSuite; print('✓')"
```

### Windows

Phase 4 works on Windows:

```powershell
# Verify
python -c "from cyto_dl.utils.benchmark import BenchmarkSuite; print('✓')"

# Run tests
pytest tests\test_performance_regression.py -v
```

### macOS

Phase 4 works on macOS (CPU mode):

```bash
# Verify
python -c "from cyto_dl.utils.benchmark import BenchmarkSuite; print('✓')"

# Benchmark on CPU
python scripts/benchmark_performance.py --device cpu
```

## Dependencies Summary

Phase 4 uses only existing dependencies:

```
# Required (already in base CytoDL)
torch >= 2.0
numpy

# Optional (for testing)
pytest  # Only for running tests
```

No new packages to install! 🎉

## Next Steps

1. **Run verification**: `python verify_phase4.py`
2. **Try benchmarking**: `python scripts/benchmark_performance.py --help`
3. **Run tests**: `pytest tests/test_performance_regression.py -v`
4. **Read docs**: See `docs/PHASE4_BENCHMARKING.md`

---

## Summary

✅ **No installation required** - Phase 4 uses existing PyTorch

✅ **Works immediately** - All tools ready to use

✅ **Cross-platform** - Linux, Windows, macOS supported

Enjoy benchmarking! 🚀
