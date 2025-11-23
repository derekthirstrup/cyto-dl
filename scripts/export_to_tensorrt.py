"""Export trained CytoDL models to TensorRT for optimized inference.

This script converts PyTorch models to TensorRT, providing 2-5x faster inference
on NVIDIA GPUs.

Requirements:
    pip install torch-tensorrt nvidia-tensorrt

Usage:
    # Basic export (FP16)
    python scripts/export_to_tensorrt.py \
        --config configs/experiment/im2im/segmentation.yaml \
        --ckpt path/to/checkpoint.ckpt \
        --output model_trt.ts \
        --input-shape 1 1 64 64 64

    # INT8 quantization (4x faster, requires calibration)
    python scripts/export_to_tensorrt.py \
        --config configs/experiment/im2im/labelfree.yaml \
        --ckpt path/to/checkpoint.ckpt \
        --output model_trt_int8.ts \
        --input-shape 1 1 256 256 \
        --precision int8 \
        --calibration-data path/to/calibration/images

    # Dynamic batch size
    python scripts/export_to_tensorrt.py \
        --config configs/experiment/im2im/mae.yaml \
        --ckpt path/to/checkpoint.ckpt \
        --output model_trt_dynamic.ts \
        --input-shape 1 1 64 64 64 \
        --dynamic-shapes \
        --min-batch 1 \
        --max-batch 16 \
        --opt-batch 4
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Export CytoDL model to TensorRT")

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for TensorRT model (.ts file)",
    )

    # Input configuration
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        required=True,
        help="Input tensor shape (e.g., 1 1 64 64 64 for 3D)",
    )

    # TensorRT configuration
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Precision mode (fp32, fp16, int8)",
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        default=4,
        help="Workspace size in GB (default: 4)",
    )

    # Dynamic shapes
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Enable dynamic batch size",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Minimum batch size for dynamic shapes",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=16,
        help="Maximum batch size for dynamic shapes",
    )
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=4,
        help="Optimal batch size for dynamic shapes",
    )

    # INT8 calibration
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to directory with calibration images (for INT8)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples to use (default: 100)",
    )

    # Benchmarking
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark PyTorch vs TensorRT after export",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def load_model_from_checkpoint(config_path: str, ckpt_path: str):
    """Load model from config and checkpoint."""
    import hydra
    from lightning import LightningModule

    # Load config
    cfg = OmegaConf.load(config_path)

    # Instantiate model
    print(f"Loading model from config: {config_path}")
    with hydra.initialize(config_path="../configs", version_base="1.3"):
        model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def load_calibration_data(
    data_path: str,
    num_samples: int,
    input_shape: tuple,
    device: str,
):
    """Load calibration data for INT8 quantization."""
    from PIL import Image
    import numpy as np
    from torchvision import transforms

    data_path = Path(data_path)
    calibration_data = []

    # Find image files
    image_files = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg"))
    image_files = image_files[:num_samples]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_path}")

    print(f"Loading {len(image_files)} calibration samples from {data_path}")

    # Create transform
    transform = transforms.Compose([
        transforms.Resize(input_shape[-2:]),  # Resize to input shape
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    for img_file in image_files:
        img = Image.open(img_file).convert('L')  # Convert to grayscale
        img_tensor = transform(img).unsqueeze(0)  # Add batch dim

        # Adjust shape if needed (for 3D models, just use 2D slices)
        while img_tensor.ndim < len(input_shape):
            img_tensor = img_tensor.unsqueeze(2)

        # Crop/pad to match input shape
        img_tensor = img_tensor[..., :input_shape[-2], :input_shape[-1]]

        calibration_data.append(img_tensor.to(device))

    return calibration_data


def main():
    args = parse_args()

    # Check TensorRT availability
    try:
        from cyto_dl.utils.tensorrt_utils import (
            TENSORRT_AVAILABLE,
            export_to_tensorrt,
            benchmark_tensorrt,
            create_int8_calibrator,
        )
    except ImportError:
        print("Error: TensorRT utilities not found")
        print("Make sure you've installed the package correctly")
        sys.exit(1)

    if not TENSORRT_AVAILABLE:
        print("Error: TensorRT not available")
        print("Install with: pip install torch-tensorrt nvidia-tensorrt")
        sys.exit(1)

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model = load_model_from_checkpoint(args.config, args.ckpt)
    model = model.to(args.device)

    # Prepare calibration for INT8
    calibrator = None
    if args.precision == "int8":
        if args.calibration_data is None:
            print("Warning: INT8 precision requested but no calibration data provided")
            print("Accuracy may be reduced. Provide --calibration-data for best results")
        else:
            print(f"\nPreparing INT8 calibration...")
            calibration_data = load_calibration_data(
                args.calibration_data,
                args.num_calibration_samples,
                tuple(args.input_shape),
                args.device,
            )
            calibrator = create_int8_calibrator(
                calibration_data,
                cache_file=str(Path(args.output).with_suffix(".cache")),
            )

    # Export to TensorRT
    print(f"\nExporting to TensorRT...")
    print(f"  Input shape: {args.input_shape}")
    print(f"  Precision: {args.precision}")
    print(f"  Workspace size: {args.workspace_size} GB")
    print(f"  Dynamic shapes: {args.dynamic_shapes}")

    trt_model = export_to_tensorrt(
        model,
        input_shape=tuple(args.input_shape),
        precision=args.precision,
        workspace_size=args.workspace_size,
        min_batch_size=args.min_batch if args.dynamic_shapes else args.input_shape[0],
        max_batch_size=args.max_batch if args.dynamic_shapes else args.input_shape[0],
        opt_batch_size=args.opt_batch if args.dynamic_shapes else args.input_shape[0],
        output_path=args.output,
        enable_dynamic_shapes=args.dynamic_shapes,
        device=args.device,
    )

    if trt_model is None:
        print("\nError: TensorRT export failed")
        sys.exit(1)

    print(f"\n✓ Successfully exported to {args.output}")

    # Benchmark if requested
    if args.benchmark:
        print(f"\nRunning benchmark comparison...")
        results = benchmark_tensorrt(
            pytorch_model=model,
            tensorrt_model=trt_model,
            input_shape=tuple(args.input_shape),
            num_iterations=args.benchmark_iterations,
            warmup_iterations=10,
            device=args.device,
        )

        # Save benchmark results
        import json
        results_file = Path(args.output).with_suffix(".benchmark.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Benchmark results saved to {results_file}")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"1. Test the TensorRT model:")
    print(f"   >>> import torch")
    print(f"   >>> model = torch.jit.load('{args.output}')")
    print(f"   >>> output = model(torch.randn({args.input_shape}).cuda())")
    print(f"")
    print(f"2. Use with TensorRTInferenceEngine:")
    print(f"   >>> from cyto_dl.utils.tensorrt_utils import TensorRTInferenceEngine")
    print(f"   >>> engine = TensorRTInferenceEngine('{args.output}', {args.input_shape})")
    print(f"   >>> output = engine(input_tensor)")
    print(f"")
    print(f"3. Benchmark your model:")
    print(f"   python scripts/benchmark_performance.py \\")
    print(f"     --tensorrt-model {args.output} \\")
    print(f"     --input-shape {' '.join(map(str, args.input_shape))}")
    print("="*60)


if __name__ == "__main__":
    main()
