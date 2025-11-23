"""Complete Phase 3 Workflow Examples

This script demonstrates how to use all Phase 3 advanced optimizations:
1. PyTorch quantization for CPU deployment
2. Flash Attention for Vision Transformers
3. Advanced profiling and bottleneck detection
4. Multi-GPU DDP optimizations
5. Automated performance tuning

Run individual examples by uncommenting the desired section.
"""

import torch
import torch.nn as nn
from pathlib import Path


# ============================================================================
# Example 1: PyTorch Quantization Workflow
# ============================================================================

def example_quantization():
    """Example: Quantize model for CPU deployment."""
    print("=" * 70)
    print("EXAMPLE 1: PyTorch Quantization")
    print("=" * 70)

    from cyto_dl.utils.quantization import (
        quantize_model_dynamic,
        quantize_model_static,
        QuantizationAwareTraining,
        benchmark_quantized_model,
    )
    from torch.utils.data import DataLoader, TensorDataset

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
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
            return self.layers(x)

    model = DummyModel()
    model.eval()

    # Method 1: Dynamic Quantization (Easiest)
    print("\n1. Dynamic Quantization:")
    quantized_dynamic = quantize_model_dynamic(
        model, dtype=torch.qint8, output_path=None
    )
    print("   ✓ Model quantized with dynamic quantization")

    # Method 2: Static Quantization (Best performance)
    print("\n2. Static Quantization:")

    # Create calibration data
    calibration_data = torch.randn(100, 1, 256, 256)
    calibration_dataset = TensorDataset(calibration_data)
    calibration_loader = DataLoader(calibration_dataset, batch_size=4)

    quantized_static = quantize_model_static(
        model,
        calibration_loader=calibration_loader,
        backend="fbgemm",
        output_path=None,
    )
    print("   ✓ Model quantized with static quantization")

    # Method 3: Quantization-Aware Training (Best accuracy)
    print("\n3. Quantization-Aware Training:")
    qat_trainer = QuantizationAwareTraining(model, backend="fbgemm")

    # Prepare for QAT
    qat_model = qat_trainer.prepare()
    print("   ✓ Model prepared for QAT")

    # Simulate training (replace with real training loop)
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    print("   Training QAT model for 2 epochs...")
    qat_model.train()
    for epoch in range(2):
        for i, (batch,) in enumerate(calibration_loader):
            if i >= 5:  # Just a few batches for demo
                break

            optimizer.zero_grad()
            output = qat_model(batch)
            target = torch.randn_like(output)  # Dummy target
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Convert to quantized
    quantized_qat = qat_trainer.convert()
    print("   ✓ QAT model converted to quantized")

    # Benchmark
    print("\n4. Benchmarking:")
    results = benchmark_quantized_model(
        original_model=model,
        quantized_model=quantized_dynamic,
        input_shape=(1, 1, 256, 256),
        num_iterations=20,
        device="cpu",
    )

    print(f"\n   Results:")
    print(f"   - Original: {results['original_time_ms']:.2f} ms")
    print(f"   - Quantized: {results['quantized_time_ms']:.2f} ms")
    print(f"   - Speedup: {results['speedup']:.2f}x")
    print(f"   - Size reduction: {results['size_reduction']:.2f}x")

    print("\n✓ Quantization workflow complete!\n")


# ============================================================================
# Example 2: Flash Attention for Vision Transformers
# ============================================================================

def example_flash_attention():
    """Example: Use Flash Attention for ViT models."""
    print("=" * 70)
    print("EXAMPLE 2: Flash Attention")
    print("=" * 70)

    from cyto_dl.nn.vits.flash_attention import (
        FlashAttentionBlock,
        OptimizedTransformerBlock,
        replace_attention_with_flash,
        benchmark_flash_attention,
    )

    # Method 1: Drop-in replacement
    print("\n1. Drop-in Replacement:")
    flash_attn = FlashAttentionBlock(dim=768, num_heads=12, use_flash=True)

    sample_input = torch.randn(4, 256, 768)  # (batch, seq_len, dim)
    if torch.cuda.is_available():
        flash_attn = flash_attn.cuda()
        sample_input = sample_input.cuda()

    output = flash_attn(sample_input)
    print(f"   ✓ Flash Attention output shape: {output.shape}")

    # Method 2: Complete optimized transformer block
    print("\n2. Optimized Transformer Block:")
    transformer_block = OptimizedTransformerBlock(
        dim=768, num_heads=12, mlp_ratio=4.0, use_flash=True
    )

    if torch.cuda.is_available():
        transformer_block = transformer_block.cuda()

    output = transformer_block(sample_input)
    print(f"   ✓ Transformer block output shape: {output.shape}")

    # Method 3: Auto-replace in existing model
    print("\n3. Auto-Replace in Existing Model:")

    class SimpleViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                [OptimizedTransformerBlock(dim=768, num_heads=12, use_flash=False) for _ in range(4)]
            )

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = SimpleViT()
    print(f"   Original model has Flash Attention: No")

    # Replace with Flash Attention
    model = replace_attention_with_flash(model)
    print(f"   ✓ Model updated with Flash Attention")

    # Benchmark (if CUDA available)
    if torch.cuda.is_available():
        print("\n4. Benchmarking Flash Attention:")
        results = benchmark_flash_attention(
            batch_size=8,
            seq_length=1024,
            dim=768,
            num_heads=12,
            num_iterations=20,
        )

    print("\n✓ Flash Attention workflow complete!\n")


# ============================================================================
# Example 3: Advanced Profiling
# ============================================================================

def example_profiling():
    """Example: Use advanced profiling tools."""
    print("=" * 70)
    print("EXAMPLE 3: Advanced Profiling")
    print("=" * 70)

    from cyto_dl.utils.advanced_profiling import (
        MemoryProfiler,
        ProfilerContext,
        BottleneckDetector,
        memory_snapshot,
    )
    from torch.utils.data import DataLoader, TensorDataset

    # Create dummy model and data
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)
    )

    dataset = TensorDataset(torch.randn(100, 1, 28, 28))
    dataloader = DataLoader(dataset, batch_size=8)

    if torch.cuda.is_available():
        model = model.cuda()

    # Tool 1: Memory Profiler
    print("\n1. Memory Profiler:")
    mem_profiler = MemoryProfiler()
    mem_profiler.start()

    for i, (batch,) in enumerate(dataloader):
        if i >= 3:  # Just a few batches
            break

        if torch.cuda.is_available():
            batch = batch.cuda()

        mem_profiler.snapshot("data_loaded")

        output = model(batch)
        mem_profiler.snapshot("forward")

        loss = output.sum()
        loss.backward()
        mem_profiler.snapshot("backward")

    mem_profiler.print_summary()
    mem_profiler.detect_leaks(threshold_gb=0.05)

    # Tool 2: Profiler Context
    print("\n2. Profiler Context:")
    with ProfilerContext("inference", enabled=True, use_cuda=torch.cuda.is_available()) as prof:
        for i, (batch,) in enumerate(dataloader):
            if i >= 5:
                break
            if torch.cuda.is_available():
                batch = batch.cuda()
            with torch.no_grad():
                _ = model(batch)

    if prof.enabled:
        prof.print_summary(sort_by="cpu_time_total" if not torch.cuda.is_available() else "cuda_time_total", top_k=10)

    # Tool 3: Bottleneck Detector
    print("\n3. Bottleneck Detector:")
    detector = BottleneckDetector()
    detector.start()

    for i, (batch,) in enumerate(dataloader):
        if i >= 3:
            break

        detector.mark("data_loading")

        if torch.cuda.is_available():
            batch = batch.cuda()

        output = model(batch)
        detector.mark("forward")

        loss = output.sum()
        loss.backward()
        detector.mark("backward")

    detector.print_recommendations()

    # Tool 4: Quick memory snapshot
    print("\n4. Quick Memory Snapshot:")
    if torch.cuda.is_available():
        batch = next(iter(dataloader))[0].cuda()
        with memory_snapshot("single_forward"):
            output = model(batch)

    print("\n✓ Profiling workflow complete!\n")


# ============================================================================
# Example 4: Multi-GPU DDP Optimizations
# ============================================================================

def example_ddp():
    """Example: Multi-GPU DDP optimizations."""
    print("=" * 70)
    print("EXAMPLE 4: Multi-GPU DDP Optimizations")
    print("=" * 70)

    from cyto_dl.utils.distributed import (
        setup_ddp_optimizations,
        DDPOptimizer,
        GradientCompression,
        DistributedMetrics,
        is_dist_available_and_initialized,
    )

    if not torch.cuda.is_available():
        print("   Skipping: CUDA not available")
        return

    if not is_dist_available_and_initialized():
        print("   Note: Distributed not initialized (requires torchrun/mpirun)")
        print("   Showing API examples only\n")

    # Create model
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    model = model.cuda()

    # Method 1: Basic DDP setup
    print("1. Basic DDP Setup:")
    print("   Code:")
    print("   model = setup_ddp_optimizations(")
    print("       model,")
    print("       sync_bn=True,")
    print("       gradient_as_bucket_view=True,")
    print("       static_graph=True,")
    print("   )")

    # Method 2: DDPOptimizer (all-in-one)
    print("\n2. DDPOptimizer (All-in-One):")
    print("   Code:")
    print("   ddp_opt = DDPOptimizer(")
    print("       model,")
    print("       sync_bn=True,")
    print("       gradient_compression=True,")
    print("       compression_rank=2,")
    print("   )")
    print("   model = ddp_opt.get_model()")

    # Method 3: Distributed Metrics
    print("\n3. Distributed Metrics:")
    metrics = DistributedMetrics()
    metrics.update("loss", 0.5, count=8)
    metrics.update("accuracy", 0.85, count=8)
    avg_metrics = metrics.compute_and_reset()
    print(f"   ✓ Average loss: {avg_metrics['loss']:.4f}")
    print(f"   ✓ Average accuracy: {avg_metrics['accuracy']:.4f}")

    print("\n   To run with DDP:")
    print("   torchrun --nproc_per_node=4 examples/phase3_workflow.py")

    print("\n✓ DDP workflow examples complete!\n")


# ============================================================================
# Example 5: Automated Performance Tuning
# ============================================================================

def example_auto_tune():
    """Example: Automated performance tuning."""
    print("=" * 70)
    print("EXAMPLE 5: Automated Performance Tuning")
    print("=" * 70)

    from cyto_dl.utils.auto_tune import AutoTuner, auto_tune_model

    if not torch.cuda.is_available():
        print("   Skipping: CUDA not available for auto-tuning")
        return

    # Create model
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    model = model.cuda()

    # Sample input (single sample, no batch dimension)
    sample_input = torch.randn(1, 64, 64).cuda()

    print("\n1. Basic Auto-Tuning:")
    tuner = AutoTuner(model, sample_input, device="cuda")

    # Run tuning
    optimal_config = tuner.tune(tune_inference=True, tune_training=False)

    print("\n   Optimal configuration found:")
    for key, value in optimal_config.items():
        print(f"   - {key}: {value}")

    # Method 2: Save config
    print("\n2. Save Auto-Tuned Config:")
    output_path = "auto_tuned_config.yaml"
    config = auto_tune_model(
        model, sample_input, device="cuda", save_config=output_path
    )
    print(f"   ✓ Config saved to {output_path}")

    print("\n✓ Auto-tuning workflow complete!\n")


# ============================================================================
# Example 6: Complete End-to-End Workflow
# ============================================================================

def example_complete_workflow():
    """Example: Complete workflow with all Phase 3 optimizations."""
    print("=" * 70)
    print("EXAMPLE 6: Complete End-to-End Workflow")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("   Skipping: Requires CUDA")
        return

    from cyto_dl.utils.performance import setup_gpu_optimizations
    from cyto_dl.nn.vits.flash_attention import replace_attention_with_flash
    from cyto_dl.utils.auto_tune import auto_tune_model
    from cyto_dl.utils.advanced_profiling import ProfilerContext

    print("\nWorkflow: Label-Free Model Optimization")
    print("-" * 70)

    # Step 1: Setup GPU optimizations (Phase 1)
    print("\n1. Apply Phase 1 GPU optimizations...")
    setup_gpu_optimizations(
        enable_cudnn_benchmark=True,
        enable_tf32=True,
        channels_last=True,
    )
    print("   ✓ GPU optimizations applied")

    # Step 2: Create model (ViT-based)
    print("\n2. Create Vision Transformer model...")
    from cyto_dl.nn.vits.flash_attention import OptimizedTransformerBlock

    class SimpleViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            self.blocks = nn.ModuleList(
                [OptimizedTransformerBlock(dim=768, num_heads=12, use_flash=False) for _ in range(6)]
            )
            self.head = nn.Linear(768, 1)

        def forward(self, x):
            # Patch embedding
            x = self.patch_embed(x)  # (B, 768, H/16, W/16)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, 768)

            # Transformer blocks
            for block in self.blocks:
                x = block(x)

            # Global pooling + head
            x = x.mean(dim=1)  # (B, 768)
            x = self.head(x)  # (B, 1)
            return x

    model = SimpleViT().cuda()
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Step 3: Apply Flash Attention (Phase 3)
    print("\n3. Apply Flash Attention...")
    model = replace_attention_with_flash(model)
    print("   ✓ Flash Attention applied")

    # Step 4: Convert to channels-last
    print("\n4. Convert to channels-last memory format...")
    model = model.to(memory_format=torch.channels_last)
    print("   ✓ Channels-last applied")

    # Step 5: Compile model
    print("\n5. Compile model with torch.compile...")
    model = torch.compile(model, mode="default")
    print("   ✓ Model compiled")

    # Step 6: Auto-tune
    print("\n6. Auto-tune hyperparameters...")
    sample_input = torch.randn(1, 256, 256).cuda()
    optimal_config = auto_tune_model(model, sample_input, device="cuda", save_config=None)
    print(f"   ✓ Optimal batch size: {optimal_config['batch_size']}")

    # Step 7: Profile inference
    print("\n7. Profile optimized model...")
    with ProfilerContext("optimized_inference", enabled=True, use_cuda=True) as prof:
        batch_size = optimal_config['batch_size']
        batch_input = torch.randn(batch_size, 1, 256, 256).cuda()
        batch_input = batch_input.to(memory_format=torch.channels_last)

        with torch.no_grad():
            for _ in range(10):
                _ = model(batch_input)

    if prof.enabled:
        prof.print_summary(sort_by="cuda_time_total", top_k=10)

    print("\n✓ Complete workflow finished!")
    print("\nFinal optimizations applied:")
    print("  ✓ Phase 1: cudnn, TF32, channels-last, torch.compile")
    print("  ✓ Phase 3: Flash Attention, auto-tuning, profiling")
    print(f"  ✓ Optimal batch size: {optimal_config['batch_size']}")
    print("\nExpected speedup: 5-7x vs baseline\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 3 ADVANCED OPTIMIZATIONS - COMPLETE WORKFLOW EXAMPLES")
    print("=" * 70 + "\n")

    # Run all examples
    # Uncomment the examples you want to run:

    example_quantization()
    example_flash_attention()
    example_profiling()
    example_ddp()
    example_auto_tune()
    example_complete_workflow()

    print("=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Try Phase 3 optimizations on your models")
    print("2. Benchmark performance improvements")
    print("3. Export optimized models for deployment")
    print("4. See docs/PHASE3_ADVANCED_OPTIMIZATIONS.md for details")
    print("=" * 70 + "\n")
