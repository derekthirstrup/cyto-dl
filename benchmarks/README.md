# CytoDL Performance Benchmarking

This directory contains benchmark scripts and configurations for measuring performance improvements across optimization phases.

## Optimization Phases

All 4 optimization phases have been completed:

1. **Phase 1**: GPU Performance Optimizations (`claude/optimize-gpu-performance-014viXtwt7gNsiG4xedaMKNA`)
   - CUDA optimizations
   - Memory management improvements
   - Batch processing enhancements

2. **Phase 2**: TensorRT Integration (`claude/tensorrt-integration-phase2-014viXtwt7gNsiG4xedaMKNA`)
   - TensorRT model optimization
   - FP16 precision support
   - Dynamic shape optimization

3. **Phase 3**: Advanced Optimizations (`claude/advanced-optimizations-phase3-014viXtwt7gNsiG4xedaMKNA`)
   - Advanced kernel fusion
   - Data pipeline optimizations
   - Multi-GPU improvements

4. **Phase 4**: Benchmarking Enhancements (`claude/benchmarking-phase4-014viXtwt7gNsiG4xedaMKNA`)
   - Profiling instrumentation
   - Performance monitoring
   - Metrics collection

**Combined Branch**: `claude/all-optimizations-combined-014viXtwt7gNsiG4xedaMKNA` contains all optimizations merged together.

## Benchmark Scripts

### Training Benchmark

Measures training performance with detailed timing metrics.

```bash
python scripts/benchmark_train.py \
    --config benchmarks/train_tasks.csv \
    --output results/train_benchmark.json \
    --skip-failed
```

**Metrics collected:**
- Total training time
- Model initialization time
- Data setup time
- First epoch time breakdown
- Data loading time (first batch)
- Model forward pass time (first batch)
- Full training time
- Peak GPU memory usage
- Final training metrics

### Inference Benchmark

Measures inference/prediction performance with detailed timing metrics.

```bash
python scripts/benchmark_inference.py \
    --config benchmarks/inference_tasks.csv \
    --output results/inference_benchmark.json \
    --skip-failed \
    --compare
```

**Metrics collected:**
- Total inference time
- Model initialization time
- Checkpoint loading time
- First batch processing breakdown:
  - Batch loading time
  - Data transfer to device time
  - Model forward pass time
  - Post-processing time
- Full prediction time
- Throughput (samples/sec)
- Average time per batch
- Peak GPU memory usage

## CSV Configuration Format

### Training Tasks CSV

```csv
branch,experiment,trainer,max_epochs,batch_size,overrides
main,im2im/segmentation,gpu,5,8,
phase-branch-name,im2im/segmentation,gpu,5,8,custom_override=value
```

**Columns:**
- `branch`: Git branch to benchmark
- `experiment`: Experiment config from `configs/experiment/`
- `trainer`: Trainer config (gpu, cpu, ddp)
- `max_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `overrides`: Optional comma-separated config overrides

### Inference Tasks CSV

```csv
branch,experiment,checkpoint_path,data_path,trainer,batch_size,overrides
main,im2im/segmentation,path/to/checkpoint.ckpt,path/to/data,gpu,16,
phase-branch-name,im2im/segmentation,path/to/checkpoint.ckpt,path/to/data,gpu,16,
```

**Columns:**
- `branch`: Git branch to benchmark
- `experiment`: Experiment config from `configs/experiment/`
- `checkpoint_path`: Path to trained model checkpoint
- `data_path`: Path to test data
- `trainer`: Trainer config (gpu, cpu)
- `batch_size`: Inference batch size
- `overrides`: Optional comma-separated config overrides

## Usage Examples

### 1. Benchmark Training Performance

```bash
# Run training benchmarks for all phases
python scripts/benchmark_train.py \
    --config benchmarks/train_tasks.csv \
    --output results/train_benchmark.json

# Skip failures and continue
python scripts/benchmark_train.py \
    --config benchmarks/train_tasks.csv \
    --output results/train_benchmark.json \
    --skip-failed
```

### 2. Benchmark Inference Performance

```bash
# First, ensure you have a trained checkpoint
# Update checkpoint_path and data_path in inference_tasks.csv

# Run inference benchmarks
python scripts/benchmark_inference.py \
    --config benchmarks/inference_tasks.csv \
    --output results/inference_benchmark.json \
    --compare

# The --compare flag will show speedup comparisons
```

### 3. Custom Benchmark Configuration

Create a custom CSV with specific tasks:

**custom_benchmark.csv:**
```csv
branch,experiment,trainer,max_epochs,batch_size,overrides
main,im2im/segmentation,gpu,10,16,
claude/all-optimizations-combined-014viXtwt7gNsiG4xedaMKNA,im2im/segmentation,gpu,10,16,
```

Run:
```bash
python scripts/benchmark_train.py \
    --config custom_benchmark.csv \
    --output results/custom_results.json
```

### 4. Multiple Experiments

Benchmark different experiments across phases:

```csv
branch,experiment,trainer,max_epochs,batch_size,overrides
main,im2im/segmentation,gpu,5,8,
main,im2im/label_free,gpu,5,8,
claude/all-optimizations-combined-014viXtwt7gNsiG4xedaMKNA,im2im/segmentation,gpu,5,8,
claude/all-optimizations-combined-014viXtwt7gNsiG4xedaMKNA,im2im/label_free,gpu,5,8,
```

## Output Format

Results are saved in JSON format with the following structure:

```json
[
  {
    "branch": "main",
    "experiment": "im2im/segmentation",
    "timestamp": "2025-11-24T12:00:00",
    "success": true,
    "git_info": {
      "branch": "main",
      "commit_hash": "abc123...",
      "commit_date": "2025-11-24"
    },
    "metrics": {
      "total_training": {
        "elapsed_seconds": 120.5,
        "peak_memory_mb": 4096.2
      },
      "epoch_breakdown": {
        "first_epoch": {"elapsed_seconds": 25.3},
        "data_loading_first_batch": {"elapsed_seconds": 0.5},
        "model_forward_first_batch": {"elapsed_seconds": 0.1}
      },
      "dataset_info": {
        "num_train_batches": 100,
        "batch_size": 8
      }
    }
  }
]
```

## Interpreting Results

### Training Metrics

1. **Total Training Time**: Overall time for the entire training run
2. **First Epoch Time**: Time for first training epoch (includes JIT compilation overhead)
3. **Data Loading Time**: Time to load and preprocess first batch
4. **Model Forward Time**: Time for model inference on first batch
5. **Peak Memory**: Maximum GPU memory used during training

### Inference Metrics

1. **Total Inference Time**: Overall time for all predictions
2. **First Batch Forward Time**: Model inference time (critical metric)
3. **Throughput**: Samples processed per second
4. **Average Time Per Batch**: Mean processing time across all batches

### Expected Improvements

Based on optimization phases:

- **Phase 1 (GPU)**: 10-30% speedup in model forward pass
- **Phase 2 (TensorRT)**: 20-50% speedup in inference
- **Phase 3 (Advanced)**: 15-40% reduction in data loading time
- **Phase 4 (Benchmarking)**: No performance change, adds instrumentation
- **Combined**: 40-80% overall speedup (cumulative effect)

## Tips

1. **Warm-up runs**: First runs may be slower due to JIT compilation
2. **GPU memory**: Monitor peak memory to avoid OOM errors
3. **Batch size**: Larger batch sizes generally improve throughput
4. **Data pipeline**: Ensure data is on fast storage (SSD/NVMe)
5. **Reproducibility**: Results may vary by ±5% between runs

## Requirements

- PyTorch with CUDA support
- Lightning framework
- All CytoDL dependencies installed
- Git repository with all phase branches

## Troubleshooting

### Branch Checkout Failures

```bash
# Manually fetch all branches
git fetch origin

# List available branches
git branch -r
```

### Checkpoint Not Found

Update the `checkpoint_path` in `inference_tasks.csv` to point to a valid checkpoint.

### CUDA Out of Memory

Reduce `batch_size` in the CSV configuration or use smaller models.

### Import Errors

Ensure you're in the correct conda/virtual environment with all dependencies:

```bash
conda activate cyto-dl
pip install -e .
```

## Contributing

To add new benchmark configurations:

1. Create a new CSV file in `benchmarks/`
2. Follow the format specified above
3. Run benchmarks and save results
4. Document any custom configurations

## Contact

For questions about benchmarking or optimization phases, please open an issue on the GitHub repository.
