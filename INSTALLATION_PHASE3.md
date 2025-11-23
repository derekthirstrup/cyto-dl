# Phase 3: Installation Guide

This guide covers installation for Phase 3 advanced optimizations.

## Quick Start

**Most Phase 3 features work out-of-the-box with existing CytoDL installation!**

The only optional dependency is **Flash Attention** for Vision Transformer models.

## What's Included

Phase 3 includes:

| Feature | Dependencies | Installation Required |
|---------|--------------|----------------------|
| **PyTorch Quantization** | Built into PyTorch | ❌ No |
| **Advanced Profiling** | Built into PyTorch | ❌ No |
| **Multi-GPU DDP** | Built into PyTorch | ❌ No |
| **Auto-Tuning** | Built into PyTorch | ❌ No |
| **Flash Attention** | `flash-attn` package | ✅ Yes (optional) |

---

## Installation Options

### Option 1: Basic Phase 3 (No New Dependencies)

Most Phase 3 features work immediately:

```bash
# No installation needed!
# Test that Phase 3 works:
python -c "from cyto_dl.utils.quantization import quantize_model_dynamic; print('✓ Quantization ready')"
python -c "from cyto_dl.utils.advanced_profiling import MemoryProfiler; print('✓ Profiling ready')"
python -c "from cyto_dl.utils.distributed import setup_ddp_optimizations; print('✓ DDP ready')"
python -c "from cyto_dl.utils.auto_tune import AutoTuner; print('✓ Auto-tuning ready')"
```

**Expected output:**
```
✓ Quantization ready
✓ Profiling ready
✓ DDP ready
✓ Auto-tuning ready
```

### Option 2: With Flash Attention (Recommended for ViT Models)

Install Flash Attention for 2-4x faster Vision Transformer models:

#### Requirements

- **NVIDIA GPU** with compute capability >= 8.0 (Ampere or newer):
  - ✅ RTX 3000 series (3060, 3070, 3080, 3090)
  - ✅ RTX 4000 series (4060, 4070, 4080, 4090)
  - ✅ RTX 5000 series (5070, 5080, 5090)
  - ✅ A100, H100 (data center GPUs)
  - ⚠️ RTX 2000 series: Limited support
- **CUDA** >= 11.6
- **PyTorch** >= 2.0

#### Installation

```bash
# Option A: Via pip (recommended)
pip install flash-attn --no-build-isolation

# Option B: Via conda
conda install -c conda-forge flash-attn

# Verify installation
python -c "import flash_attn; print(f'✓ Flash Attention {flash_attn.__version__} installed')"
```

**Expected output:**
```
✓ Flash Attention 2.x.x installed
```

---

## Verification

### Verify All Phase 3 Features

Run this script to verify everything is working:

```python
# verify_phase3.py
import torch

print("Verifying Phase 3 Installation...")
print("=" * 60)

# 1. PyTorch Quantization
try:
    from cyto_dl.utils.quantization import quantize_model_dynamic
    print("✓ PyTorch Quantization: Available")
except ImportError as e:
    print(f"✗ PyTorch Quantization: Failed - {e}")

# 2. Flash Attention (optional)
try:
    from cyto_dl.nn.vits.flash_attention import FlashAttentionBlock
    import flash_attn
    print(f"✓ Flash Attention: Available (v{flash_attn.__version__})")
except ImportError:
    print("⚠ Flash Attention: Not installed (optional)")

# 3. Advanced Profiling
try:
    from cyto_dl.utils.advanced_profiling import MemoryProfiler
    print("✓ Advanced Profiling: Available")
except ImportError as e:
    print(f"✗ Advanced Profiling: Failed - {e}")

# 4. Multi-GPU DDP
try:
    from cyto_dl.utils.distributed import setup_ddp_optimizations
    print("✓ Multi-GPU DDP: Available")
except ImportError as e:
    print(f"✗ Multi-GPU DDP: Failed - {e}")

# 5. Auto-Tuning
try:
    from cyto_dl.utils.auto_tune import AutoTuner
    print("✓ Auto-Tuning: Available")
except ImportError as e:
    print(f"✗ Auto-Tuning: Failed - {e}")

print("=" * 60)
print("\nSystem Information:")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

print("=" * 60)
print("\nPhase 3 Verification Complete!")
```

Run verification:
```bash
python verify_phase3.py
```

---

## Troubleshooting

### Issue: Flash Attention Installation Fails

**Error:**
```
ERROR: Failed building wheel for flash-attn
```

**Solution 1: Pre-built wheels**
```bash
# Check for pre-built wheels
pip install flash-attn --no-build-isolation
```

**Solution 2: Check CUDA version**
```bash
# Verify CUDA version
python -c "import torch; print(torch.version.cuda)"

# Ensure CUDA >= 11.6
# If CUDA is too old, update:
# https://developer.nvidia.com/cuda-downloads
```

**Solution 3: Install build dependencies**
```bash
# Install ninja for faster builds
pip install ninja

# Install with verbose output
pip install flash-attn --no-build-isolation -v
```

**Solution 4: Use pre-compiled binaries**
```bash
# Download pre-compiled wheel from:
# https://github.com/Dao-AILab/flash-attention/releases

# Install wheel
pip install flash_attn-*.whl
```

### Issue: GPU Not Supported

**Error:**
```
RuntimeError: FlashAttention only supports Ampere GPUs or newer
```

**Solution:**
Flash Attention requires compute capability >= 8.0. If you have an older GPU:

```python
# Fall back to PyTorch's optimized SDPA
from cyto_dl.nn.vits.flash_attention import FlashAttentionBlock

# This will automatically use SDPA instead
block = FlashAttentionBlock(dim=768, num_heads=12, use_flash=True)
# Still faster than naive attention, just not as fast as Flash Attention
```

### Issue: Out of Memory with Flash Attention

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Flash Attention uses less memory, but if you still get OOM:

# 1. Reduce batch size
batch_size = 4  # Instead of 8

# 2. Use gradient checkpointing
from cyto_dl.nn.vits.flash_attention import OptimizedTransformerBlock

block = OptimizedTransformerBlock(...)
# Enable gradient checkpointing in training loop

# 3. Use mixed precision
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
```

### Issue: Import Errors

**Error:**
```
ImportError: cannot import name 'quantize_model_dynamic'
```

**Solution:**
```bash
# Ensure you're on the correct branch
git checkout claude/advanced-optimizations-phase3-014viXtwt7gNsiG4xedaMKNA

# Reinstall in development mode
pip install -e .

# Verify
python -c "from cyto_dl.utils.quantization import quantize_model_dynamic"
```

---

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

Flash Attention works best on Linux:

```bash
# Install build dependencies
sudo apt update
sudo apt install -y build-essential ninja-build

# Install Flash Attention
pip install flash-attn --no-build-isolation
```

### Windows

Flash Attention on Windows requires:

1. **Visual Studio 2019 or newer** with C++ build tools
2. **CUDA Toolkit** installed

```powershell
# Install Flash Attention
pip install flash-attn --no-build-isolation

# If fails, try WSL2:
wsl --install
# Then install in WSL2 environment
```

### macOS

⚠️ **Flash Attention requires NVIDIA GPU** - not supported on macOS.

However, other Phase 3 features work:

```bash
# All features except Flash Attention work on macOS
python -c "from cyto_dl.utils.quantization import quantize_model_dynamic; print('✓')"
python -c "from cyto_dl.utils.auto_tune import AutoTuner; print('✓')"
```

---

## Testing Installation

### Quick Test: Quantization

```bash
python -c "
from cyto_dl.utils.quantization import quantize_model_dynamic
import torch.nn as nn
import torch

model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
quantized = quantize_model_dynamic(model, dtype=torch.qint8)
print('✓ Quantization works!')
"
```

### Quick Test: Flash Attention

```bash
python -c "
from cyto_dl.nn.vits.flash_attention import FlashAttentionBlock
import torch

block = FlashAttentionBlock(dim=768, num_heads=12)
if torch.cuda.is_available():
    block = block.cuda()
    x = torch.randn(2, 64, 768).cuda()
    output = block(x)
    print(f'✓ Flash Attention works! Output shape: {output.shape}')
else:
    print('⚠ CUDA not available, Flash Attention requires GPU')
"
```

### Quick Test: Profiling

```bash
python -c "
from cyto_dl.utils.advanced_profiling import MemoryProfiler

profiler = MemoryProfiler()
profiler.start()
profiler.snapshot('test')
profiler.print_summary()
print('✓ Profiling works!')
"
```

### Quick Test: Auto-Tuning

```bash
python -c "
from cyto_dl.utils.auto_tune import AutoTuner
import torch
import torch.nn as nn

if torch.cuda.is_available():
    model = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU()).cuda()
    sample = torch.randn(1, 64, 64).cuda()
    tuner = AutoTuner(model, sample, device='cuda')
    print('✓ Auto-tuning ready!')
else:
    print('⚠ CUDA not available, auto-tuning requires GPU')
"
```

---

## Complete Installation Script

Run this script to install everything:

```bash
#!/bin/bash
# install_phase3.sh

echo "Installing Phase 3 Advanced Optimizations..."

# Step 1: Verify PyTorch
echo "Step 1: Verifying PyTorch installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"

# Step 2: Verify CUDA (if available)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "Step 2: CUDA detected"
    python -c "import torch; print(f'✓ CUDA {torch.version.cuda}')"
    python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"

    # Step 3: Install Flash Attention
    echo "Step 3: Installing Flash Attention..."
    pip install flash-attn --no-build-isolation

    if python -c "import flash_attn" 2>/dev/null; then
        echo "✓ Flash Attention installed successfully"
    else
        echo "⚠ Flash Attention installation failed (optional)"
    fi
else
    echo "Step 2: No CUDA detected - skipping Flash Attention"
fi

# Step 4: Verify Phase 3
echo "Step 4: Verifying Phase 3 features..."
python verify_phase3.py

echo "Installation complete!"
```

Make executable and run:
```bash
chmod +x install_phase3.sh
./install_phase3.sh
```

---

## Summary

### ✅ No Installation Required

Most Phase 3 features work immediately:
- PyTorch Quantization
- Advanced Profiling
- Multi-GPU DDP
- Auto-Tuning

### ✅ Optional: Flash Attention

For Vision Transformer models:
```bash
pip install flash-attn --no-build-isolation
```

### Next Steps

1. **Test Phase 3**: Run verification script
2. **Try examples**: `python examples/phase3_workflow.py`
3. **Read docs**: See `docs/PHASE3_ADVANCED_OPTIMIZATIONS.md`
4. **Optimize your models**: Apply Phase 3 to your workflows

---

## Support

If you encounter issues:

1. **Check system requirements**: CUDA >= 11.6, PyTorch >= 2.0
2. **Run verification script**: `python verify_phase3.py`
3. **Check GPU compatibility**: Compute capability >= 8.0 for Flash Attention
4. **See troubleshooting**: Common issues above
5. **File an issue**: [GitHub Issues](https://github.com/derekthirstrup/cyto-dl/issues)

---

## Version Compatibility

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| CUDA | 11.6 | 12.0+ |
| flash-attn | 2.0 | 2.5+ |

Enjoy Phase 3 optimizations! 🚀
