"""Complete TensorRT workflow example for CytoDL.

This example demonstrates:
1. Training a label-free model
2. Exporting to TensorRT
3. Running optimized inference
4. Comparing PyTorch vs TensorRT performance

Usage:
    python examples/tensorrt_workflow.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def train_model():
    """Train a simple label-free model."""
    from cyto_dl.api import CytoDLModel

    print("="*60)
    print("STEP 1: Training Model")
    print("="*60)

    model = CytoDLModel()

    # Download example data
    print("\nDownloading example data...")
    model.download_example_data()

    # Load experiment with optimizations
    print("\nLoading experiment configuration...")
    model.load_default_experiment(
        "segmentation",  # Use segmentation as example
        output_dir="./tensorrt_example",
        overrides=[
            "trainer=gpu_optimized",
            "performance=gpu_optimized",
            "trainer.max_epochs=2",  # Quick training for demo
            "data.batch_size=4",
        ]
    )

    # Train
    print("\nTraining model...")
    model.train()

    # Get checkpoint path
    checkpoint_dir = Path("tensorrt_example/checkpoints")
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if ckpt_files:
        ckpt_path = str(ckpt_files[0])
        print(f"\n✓ Training complete! Checkpoint: {ckpt_path}")
        return ckpt_path
    else:
        print("\n✗ No checkpoint found!")
        return None


def export_to_tensorrt(ckpt_path):
    """Export trained model to TensorRT."""
    print("\n" + "="*60)
    print("STEP 2: Exporting to TensorRT")
    print("="*60)

    from cyto_dl.utils.tensorrt_utils import TENSORRT_AVAILABLE, export_to_tensorrt
    import hydra
    from omegaconf import OmegaConf

    if not TENSORRT_AVAILABLE:
        print("\n⚠️  TensorRT not available!")
        print("Install with: pip install torch-tensorrt nvidia-tensorrt")
        return None

    # Load model from checkpoint
    print("\nLoading model...")
    cfg = OmegaConf.load("configs/experiment/im2im/segmentation.yaml")

    with hydra.initialize(config_path="../configs", version_base="1.3"):
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda()
    model.eval()

    # Export to TensorRT FP16
    print("\nExporting to TensorRT (FP16)...")
    input_shape = (1, 1, 64, 64, 64)  # 3D patch

    trt_model = export_to_tensorrt(
        model,
        input_shape=input_shape,
        precision="fp16",
        workspace_size=4,
        output_path="tensorrt_example/model_trt_fp16.ts",
        device="cuda",
    )

    if trt_model:
        print("\n✓ TensorRT export successful!")
        return "tensorrt_example/model_trt_fp16.ts", model
    else:
        print("\n✗ TensorRT export failed!")
        return None, model


def benchmark_inference(trt_model_path, pytorch_model):
    """Benchmark TensorRT vs PyTorch inference."""
    print("\n" + "="*60)
    print("STEP 3: Benchmarking Inference")
    print("="*60)

    from cyto_dl.utils.tensorrt_utils import TensorRTInferenceEngine
    from cyto_dl.utils.performance import benchmark_model

    input_shape = (1, 1, 64, 64, 64)

    # Benchmark PyTorch
    print("\nBenchmarking PyTorch model...")
    pytorch_results = benchmark_model(
        pytorch_model,
        input_shape,
        num_iterations=50,
        warmup_iterations=5,
        device="cuda",
    )

    # Benchmark TensorRT
    print("\nBenchmarking TensorRT model...")
    trt_model = torch.jit.load(trt_model_path)
    trt_results = benchmark_model(
        trt_model,
        input_shape,
        num_iterations=50,
        warmup_iterations=5,
        device="cuda",
    )

    # Calculate speedup
    speedup = pytorch_results["avg_latency_ms"] / trt_results["avg_latency_ms"]

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"PyTorch Latency:     {pytorch_results['avg_latency_ms']:.2f} ms")
    print(f"TensorRT Latency:    {trt_results['avg_latency_ms']:.2f} ms")
    print(f"Speedup:             {speedup:.2f}x faster")
    print("")
    print(f"PyTorch Throughput:  {pytorch_results['throughput_fps']:.2f} FPS")
    print(f"TensorRT Throughput: {trt_results['throughput_fps']:.2f} FPS")
    print("="*60)

    return speedup


def run_inference_example(trt_model_path):
    """Demonstrate inference with TensorRT engine."""
    print("\n" + "="*60)
    print("STEP 4: Running Inference Example")
    print("="*60)

    from cyto_dl.utils.tensorrt_utils import TensorRTInferenceEngine

    # Create inference engine
    print("\nLoading TensorRT inference engine...")
    engine = TensorRTInferenceEngine(
        model_path=trt_model_path,
        input_shape=(1, 1, 64, 64, 64),
        device="cuda",
        half_precision=True,
    )

    # Create dummy input
    print("\nRunning inference on dummy data...")
    input_tensor = torch.randn(1, 1, 64, 64, 64).cuda()

    # Single inference
    output = engine(input_tensor)
    print(f"✓ Single inference complete! Output shape: {output.shape}")

    # Batch inference
    print("\nRunning batch inference...")
    batch = [torch.randn(1, 1, 64, 64, 64) for _ in range(4)]
    outputs = engine.batch_inference(batch, batch_size=4)
    print(f"✓ Batch inference complete! Processed {len(outputs)} images")

    print("\n" + "="*60)
    print("Inference examples complete!")
    print("="*60)


def main():
    """Run complete TensorRT workflow."""
    print("\n" + "="*80)
    print("CytoDL TensorRT Workflow Example")
    print("="*80)

    try:
        # Step 1: Train model
        ckpt_path = train_model()
        if not ckpt_path:
            print("\n✗ Training failed! Exiting...")
            return

        # Step 2: Export to TensorRT
        result = export_to_tensorrt(ckpt_path)
        if result[0] is None:
            print("\n✗ TensorRT export failed! Exiting...")
            print("You can still use the PyTorch model with Phase 1 optimizations.")
            return

        trt_model_path, pytorch_model = result

        # Step 3: Benchmark
        speedup = benchmark_inference(trt_model_path, pytorch_model)

        # Step 4: Run inference examples
        run_inference_example(trt_model_path)

        # Summary
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE!")
        print("="*80)
        print(f"✓ Model trained and saved to: {ckpt_path}")
        print(f"✓ TensorRT model exported to: {trt_model_path}")
        print(f"✓ TensorRT is {speedup:.2f}x faster than PyTorch")
        print("")
        print("Next steps:")
        print("1. Use the TensorRT model for production inference")
        print("2. Try INT8 quantization for 4x speedup")
        print("3. Integrate into your workflow application")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nWorkflow failed! Check the error message above.")


if __name__ == "__main__":
    main()
