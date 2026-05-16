"""CytoDL Training GUI - Streamlit application for configuring and launching im2im model training.

Replaces manual YAML editing with an interactive GUI for brightfield-to-nuclei/cell prediction
model training. Optimized for CUDA 13 / Blackwell GPU workstations.
"""

import datetime
import logging
import os
import re
import signal
import subprocess  # nosec: B404
import threading
from contextlib import suppress
from pathlib import Path

import streamlit as st
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

EXPERIMENT_TYPES = {
    "Segmentation (BF → Nuclei/Cell)": "segmentation",
    "Label-Free (BF → Fluorescence)": "labelfree",
}

BACKBONE_ARCHITECTURES = {
    "DynUNet (MONAI)": {
        "target": "monai.networks.nets.DynUNet",
        "description": "Dynamic UNet – proven performer for 2D/3D biomedical segmentation. "
        "Auto-sizes skip connections. Best overall for BF→nuclei/cell tasks.",
    },
    "SwinUNETR (MONAI)": {
        "target": "monai.networks.nets.SwinUNETR",
        "description": "Swin Transformer UNet – shifted-window self-attention encoder with "
        "UNet decoder. State-of-the-art on many 3D medical benchmarks. "
        "Higher VRAM usage but often better accuracy than DynUNet.",
    },
    "UNETR (MONAI)": {
        "target": "monai.networks.nets.UNETR",
        "description": "ViT encoder + CNN decoder. Good for capturing long-range spatial "
        "dependencies. Benefits from Flash Attention on Blackwell GPUs.",
    },
    "AttentionUNet (MONAI)": {
        "target": "monai.networks.nets.AttentionUnet",
        "description": "UNet with attention gates – focuses on relevant features at each "
        "skip connection. Lightweight attention overhead, good for cell boundary tasks.",
    },
}

LOSS_FUNCTIONS = {
    "DiceCE Loss (segmentation)": {
        "target": "monai.losses.DiceCELoss",
        "kwargs": {"sigmoid": True},
    },
    "Dice Loss": {"target": "monai.losses.DiceLoss", "kwargs": {"sigmoid": True}},
    "MSE Loss (label-free)": {"target": "torch.nn.MSELoss", "kwargs": {}},
    "L1 Loss": {"target": "torch.nn.L1Loss", "kwargs": {}},
    "Focal Loss": {"target": "monai.losses.FocalLoss", "kwargs": {}},
    "Tversky Loss": {"target": "monai.losses.TverskyLoss", "kwargs": {"sigmoid": True}},
}

HEAD_TYPES = {
    "BaseHead (standard)": {
        "target": "cyto_dl.nn.BaseHead",
        "description": "Standard task head — identity forward pass, just applies loss and postprocessing.",
        "extra_params": False,
    },
    "ResBlocksHead (conv + optional upsample)": {
        "target": "cyto_dl.nn.ResBlocksHead",
        "description": "Task head with extra convolutional layers and optional upsampling. "
        "Use for super-resolution or when the backbone output needs refinement.",
        "extra_params": True,
    },
}

POSTPROCESS_ACTIVATIONS = {
    "Sigmoid": {"_target_": "torch.nn.Sigmoid"},
    "Softmax": {"_target_": "torch.nn.Softmax", "dim": 1},
    "ReLU": {"_target_": "torch.nn.ReLU"},
    "None": None,
}

OPTIMIZERS = {
    "AdamW (fused, recommended)": {"target": "torch.optim.AdamW", "fused": True},
    "Adam (fused)": {"target": "torch.optim.Adam", "fused": True},
    "AdamW": {"target": "torch.optim.AdamW", "fused": False},
    "Adam": {"target": "torch.optim.Adam", "fused": False},
    "SGD": {"target": "torch.optim.SGD", "fused": False},
}

LR_SCHEDULERS = {
    "ExponentialLR": {
        "target": "torch.optim.lr_scheduler.ExponentialLR",
        "params": {"gamma": 0.995},
    },
    "CosineAnnealingLR": {
        "target": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "params": {"T_max": 100},
    },
    "ReduceLROnPlateau": {
        "target": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "params": {"factor": 0.5, "patience": 10},
    },
    "StepLR": {
        "target": "torch.optim.lr_scheduler.StepLR",
        "params": {"step_size": 30, "gamma": 0.1},
    },
    "None": None,
}

LOGGERS = {
    "CSV (local)": "csv",
    "TensorBoard": "tensorboard",
    "Weights & Biases": "wandb",
    "MLflow": "mlflow",
    "None": None,
}

PRECISION_OPTIONS = {
    "BF16 Mixed (Blackwell optimal)": "bf16-mixed",
    "FP16 Mixed": "16-mixed",
    "FP32 (full)": "32-true",
    "FP64 (double)": "64-true",
}

TRT_PRECISION_OPTIONS = ["fp16", "fp32", "int8"]

# ---------------------------------------------------------------------------
# Model Zoo — comprehensive architecture catalog with recommendations
# ---------------------------------------------------------------------------
# Task suitability ratings: best / good / fair / poor
MODEL_ZOO = {
    # ---- MONAI Architectures ----
    "DynUNet": {
        "target": "monai.networks.nets.DynUNet",
        "source": "MONAI",
        "category": "UNet Variants",
        "description": (
            "Dynamic UNet with auto-configured skip connections. Kernel sizes, strides, "
            "and upsampling are parameterized per-level, making it very flexible for "
            "different spatial resolutions. The default backbone in CytoDL."
        ),
        "strengths": "Flexible depth, residual blocks, proven on biomedical data, fast training",
        "weaknesses": "Less capable on very large receptive fields than transformers",
        "params": "~2-30M depending on filter config",
        "tasks": {
            "BF → Nuclei Segmentation": "best",
            "BF → Cell Segmentation": "best",
            "BF → Fluorescence (label-free)": "good",
            "Instance Segmentation": "best",
            "Super-Resolution": "fair",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "strides": [1, 2, 2, 2],
            "kernel_size": [3, 3, 3, 3],
            "upsample_kernel_size": [2, 2, 2],
            "filters": [32, 64, 128, 256],
            "dropout": 0.0,
            "res_block": True,
        },
        "vram_estimate": "~2-6 GB",
    },
    "SwinUNETR": {
        "target": "monai.networks.nets.SwinUNETR",
        "source": "MONAI",
        "category": "Transformer Hybrids",
        "description": (
            "Swin Transformer encoder with UNet-style decoder. Uses shifted-window "
            "self-attention to capture both local and global context efficiently. "
            "State-of-the-art on BTCV, MSD, and many 3D medical imaging benchmarks."
        ),
        "strengths": "Long-range context, SOTA accuracy on 3D benchmarks, good scaling",
        "weaknesses": "Higher VRAM, slower training than CNN-only, needs larger patches",
        "params": "~25-60M",
        "tasks": {
            "BF → Nuclei Segmentation": "best",
            "BF → Cell Segmentation": "best",
            "BF → Fluorescence (label-free)": "best",
            "Instance Segmentation": "good",
            "Super-Resolution": "good",
        },
        "recommended_config": {
            "img_size": [96, 96, 96],
            "in_channels": 1,
            "out_channels": 1,
            "spatial_dims": 3,
            "feature_size": 48,
            "drop_rate": 0.0,
        },
        "vram_estimate": "~8-16 GB",
    },
    "UNETR": {
        "target": "monai.networks.nets.UNETR",
        "source": "MONAI",
        "category": "Transformer Hybrids",
        "description": (
            "Vision Transformer encoder paired with CNN decoder. Patches the input into "
            "tokens, processes through standard ViT blocks, then uses skip connections "
            "to a convolutional decoder. Benefits from Flash Attention on Blackwell GPUs."
        ),
        "strengths": "Global context from ViT, flexible patch sizes, Flash Attention compatible",
        "weaknesses": "Higher VRAM than UNets, can overfit on small datasets",
        "params": "~90M (ViT-B)",
        "tasks": {
            "BF → Nuclei Segmentation": "good",
            "BF → Cell Segmentation": "good",
            "BF → Fluorescence (label-free)": "good",
            "Instance Segmentation": "fair",
            "Super-Resolution": "fair",
        },
        "recommended_config": {
            "img_size": [96, 96, 96],
            "in_channels": 1,
            "out_channels": 1,
            "spatial_dims": 3,
            "hidden_size": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "dropout_rate": 0.0,
        },
        "vram_estimate": "~12-24 GB",
    },
    "AttentionUnet": {
        "target": "monai.networks.nets.AttentionUnet",
        "source": "MONAI",
        "category": "UNet Variants",
        "description": (
            "Standard UNet with attention gates at skip connections. The gates learn to "
            "suppress irrelevant background features and highlight salient regions. "
            "Lightweight attention overhead compared to full transformer approaches."
        ),
        "strengths": "Focused skip connections, low overhead, good for boundary detection",
        "weaknesses": "Less powerful than full transformer attention, fixed architecture",
        "params": "~2-15M",
        "tasks": {
            "BF → Nuclei Segmentation": "good",
            "BF → Cell Segmentation": "best",
            "BF → Fluorescence (label-free)": "fair",
            "Instance Segmentation": "good",
            "Super-Resolution": "fair",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [32, 64, 128, 256],
            "strides": [2, 2, 2],
            "kernel_size": [3, 3, 3, 3],
            "dropout": 0.0,
        },
        "vram_estimate": "~2-6 GB",
    },
    "SegResNet": {
        "target": "monai.networks.nets.SegResNet",
        "source": "MONAI",
        "category": "UNet Variants",
        "description": (
            "Encoder-decoder with residual blocks and variational autoencoder (VAE) "
            "regularization option. Winner of BraTS 2018 challenge. Uses group "
            "normalization and has a clean, efficient architecture. Very good "
            "out-of-the-box performance with minimal tuning."
        ),
        "strengths": "Excellent defaults, group norm stability, VAE regularization option",
        "weaknesses": "Fixed encoder structure less flexible than DynUNet",
        "params": "~5-20M",
        "tasks": {
            "BF → Nuclei Segmentation": "best",
            "BF → Cell Segmentation": "good",
            "BF → Fluorescence (label-free)": "good",
            "Instance Segmentation": "good",
            "Super-Resolution": "fair",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "init_filters": 32,
            "blocks_down": [1, 2, 2, 4],
            "blocks_up": [1, 1, 1],
            "dropout_prob": 0.0,
        },
        "vram_estimate": "~3-8 GB",
    },
    "BasicUNet": {
        "target": "monai.networks.nets.BasicUNet",
        "source": "MONAI",
        "category": "UNet Variants",
        "description": (
            "Simple, clean UNet implementation with configurable encoder/decoder depth. "
            "Good starting point and baseline model. Fast to train, easy to debug. "
            "Ideal for prototyping and establishing baselines before trying heavier models."
        ),
        "strengths": "Simple, fast, low VRAM, easy to understand and debug",
        "weaknesses": "No residual connections, less expressive than DynUNet",
        "params": "~1-10M",
        "tasks": {
            "BF → Nuclei Segmentation": "good",
            "BF → Cell Segmentation": "good",
            "BF → Fluorescence (label-free)": "good",
            "Instance Segmentation": "fair",
            "Super-Resolution": "poor",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "features": [32, 32, 64, 128, 256, 32],
            "dropout": 0.0,
        },
        "vram_estimate": "~1-4 GB",
    },
    "VNet": {
        "target": "monai.networks.nets.VNet",
        "source": "MONAI",
        "category": "UNet Variants",
        "description": (
            "V-Net architecture designed specifically for 3D volumetric segmentation. "
            "Uses volumetric convolutions with residual connections throughout. "
            "Introduced the Dice loss for training. Efficient 3D processing."
        ),
        "strengths": "Purpose-built for 3D, efficient volumetric convolutions, residual learning",
        "weaknesses": "3D only, fixed architecture depth, older design",
        "params": "~10-45M",
        "tasks": {
            "BF → Nuclei Segmentation": "good",
            "BF → Cell Segmentation": "good",
            "BF → Fluorescence (label-free)": "fair",
            "Instance Segmentation": "fair",
            "Super-Resolution": "poor",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "dropout_prob": 0.0,
            "dropout_dim": 3,
        },
        "vram_estimate": "~4-12 GB",
    },
    "HighResNet": {
        "target": "monai.networks.nets.HighResNet",
        "source": "MONAI",
        "category": "Specialized",
        "description": (
            "High-resolution network that maintains high-res representations throughout "
            "the network. No downsampling/upsampling — processes at full resolution. "
            "Good for tasks where fine spatial detail matters (cell boundaries, small structures)."
        ),
        "strengths": "Preserves spatial detail, no information loss from downsampling",
        "weaknesses": "Very VRAM intensive at high resolution, limited receptive field",
        "params": "~1-5M",
        "tasks": {
            "BF → Nuclei Segmentation": "fair",
            "BF → Cell Segmentation": "good",
            "BF → Fluorescence (label-free)": "fair",
            "Instance Segmentation": "good",
            "Super-Resolution": "good",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
        },
        "vram_estimate": "~4-16 GB (resolution dependent)",
    },
    "FlexibleUNet": {
        "target": "monai.networks.nets.FlexibleUNet",
        "source": "MONAI",
        "category": "UNet Variants",
        "description": (
            "UNet with a swappable encoder backbone from any torchvision or timm model. "
            "Drop in EfficientNet, ResNet, ConvNeXt, etc. as the encoder while keeping "
            "the UNet decoder. Great for transfer learning from ImageNet pretrained weights."
        ),
        "strengths": "Use any pretrained encoder, transfer learning, flexible",
        "weaknesses": "2D-focused encoders need adaptation for 3D, decoder is basic",
        "params": "Varies by encoder (5-100M+)",
        "tasks": {
            "BF → Nuclei Segmentation": "good",
            "BF → Cell Segmentation": "good",
            "BF → Fluorescence (label-free)": "good",
            "Instance Segmentation": "good",
            "Super-Resolution": "fair",
        },
        "recommended_config": {
            "in_channels": 1,
            "out_channels": 1,
            "backbone": "efficientnet-b0",
            "pretrained": False,
        },
        "vram_estimate": "~2-12 GB (encoder dependent)",
    },
    "DiNTS": {
        "target": "monai.networks.nets.DiNTS",
        "source": "MONAI",
        "category": "Neural Architecture Search",
        "description": (
            "Differentiable Neural Network Topology Search — automatically discovers "
            "optimal network architecture for your specific dataset. Uses a supernet "
            "with searchable topology and cell structures. Can find architectures that "
            "outperform hand-designed models on your specific data distribution."
        ),
        "strengths": "Auto-discovers optimal architecture, can outperform hand-designed nets",
        "weaknesses": "Expensive search phase, complex setup, needs large dataset",
        "params": "Variable (discovered)",
        "tasks": {
            "BF → Nuclei Segmentation": "good",
            "BF → Cell Segmentation": "good",
            "BF → Fluorescence (label-free)": "fair",
            "Instance Segmentation": "fair",
            "Super-Resolution": "fair",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "in_channels": 1,
            "num_classes": 1,
        },
        "vram_estimate": "~8-24 GB (search phase)",
    },
    # ---- CytoDL Custom Architectures ----
    "MAE (CytoDL)": {
        "target": "cyto_dl.nn.vits.MAE",
        "source": "CytoDL",
        "category": "Self-Supervised",
        "description": (
            "Masked Autoencoder — masks random patches of the input and learns to "
            "reconstruct them. Excellent for self-supervised pretraining when you have "
            "lots of unlabeled data. Pretrain on unlabeled BF images, then fine-tune "
            "for segmentation."
        ),
        "strengths": "Self-supervised pretraining, works with unlabeled data, transferable",
        "weaknesses": "Requires two-stage training (pretrain then fine-tune), ViT overhead",
        "params": "~10-80M",
        "tasks": {
            "BF → Nuclei Segmentation": "good (after pretrain)",
            "BF → Cell Segmentation": "good (after pretrain)",
            "BF → Fluorescence (label-free)": "good",
            "Instance Segmentation": "fair",
            "Super-Resolution": "fair",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "patch_size": [4, 8, 8],
            "emb_dim": 192,
            "encoder_layer": 6,
            "encoder_head": 6,
            "decoder_layer": 4,
            "decoder_head": 6,
        },
        "vram_estimate": "~6-16 GB",
    },
    "HieraMAE (CytoDL)": {
        "target": "cyto_dl.nn.vits.HieraMAE",
        "source": "CytoDL",
        "category": "Self-Supervised",
        "description": (
            "Hierarchical Masked Autoencoder — extends MAE with a hierarchical "
            "architecture that progressively increases resolution. Better at "
            "capturing multi-scale features than standard MAE. Facebook/Meta research."
        ),
        "strengths": "Multi-scale features, better than vanilla MAE, hierarchical attention",
        "weaknesses": "More complex, higher VRAM, requires careful hyperparameter tuning",
        "params": "~15-100M",
        "tasks": {
            "BF → Nuclei Segmentation": "good (after pretrain)",
            "BF → Cell Segmentation": "good (after pretrain)",
            "BF → Fluorescence (label-free)": "good",
            "Instance Segmentation": "fair",
            "Super-Resolution": "good",
        },
        "recommended_config": {
            "spatial_dims": 3,
            "patch_size": [4, 8, 8],
            "emb_dim": 192,
            "encoder_layer": 6,
            "encoder_head": 6,
        },
        "vram_estimate": "~8-20 GB",
    },
}

# Task recommendation order (best first)
TASK_RECOMMENDATIONS = {
    "BF → Nuclei Segmentation": [
        ("DynUNet", "Best all-around: fast training, proven accuracy, low VRAM. Start here."),
        ("SwinUNETR", "Higher accuracy potential but needs more VRAM and larger patches."),
        ("SegResNet", "Great defaults, stable training with group norm. BraTS winner."),
        ("AttentionUnet", "Good for difficult boundaries with attention-gated skip connections."),
    ],
    "BF → Cell Segmentation": [
        ("DynUNet", "Reliable baseline — good at multi-scale cell morphology."),
        ("AttentionUnet", "Attention gates help with cell boundary detection."),
        ("SwinUNETR", "Best accuracy when VRAM allows — captures long-range cell context."),
        ("HighResNet", "Preserves fine spatial detail for thin cell boundaries."),
    ],
    "BF → Fluorescence (label-free)": [
        ("SwinUNETR", "Best for label-free — transformer captures complex BF→fluor mapping."),
        ("DynUNet", "Good baseline with fast iteration. Start here to establish baseline."),
        ("UNETR", "ViT encoder captures global context well for intensity prediction."),
        ("BasicUNet", "Fast prototyping to test if the task is learnable."),
    ],
    "Instance Segmentation": [
        ("DynUNet", "Best for multi-output instance seg (semantic + boundary + centroid)."),
        ("AttentionUnet", "Attention helps separate touching instances."),
        ("SegResNet", "Stable training for the challenging multi-task instance loss."),
    ],
    "Super-Resolution": [
        ("SwinUNETR", "Transformer captures global context for upsampling."),
        ("HighResNet", "Maintains spatial detail — natural fit for super-resolution."),
        ("FlexibleUNet", "Use pretrained encoder for transfer learning to SR task."),
    ],
}


def _scan_installed_models() -> dict:
    """Scan for available model architectures from MONAI and timm at runtime."""
    results = {"monai_models": [], "timm_models": [], "monai_version": None, "timm_version": None}

    with suppress(ImportError):
        import monai

        results["monai_version"] = monai.__version__
        import monai.networks.nets as nets

        all_nets = [x for x in dir(nets) if not x.startswith("_") and x[0].isupper()]
        results["monai_models"] = sorted(all_nets)

    with suppress(ImportError):
        import timm

        results["timm_version"] = timm.__version__
        # Get segmentation-relevant models
        all_models = timm.list_models()
        results["timm_models"] = sorted(all_models)

    return results


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "training_process": None,
        "training_log": [],
        "training_running": False,
        "generated_config": None,
        "log_thread": None,
        "trt_process": None,
        "trt_log": [],
        "trt_running": False,
        "trt_log_thread": None,
        "task_heads_config": [],
        "eval_process": None,
        "eval_log": [],
        "eval_running": False,
        "autokernel_process": None,
        "autokernel_log": [],
        "autokernel_running": False,
        "autokernel_log_thread": None,
        "autokernel_results": None,
        "autokernel_setup_process": None,
        "autokernel_setup_log": [],
        "autokernel_setup_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Data preview helpers
# ---------------------------------------------------------------------------
def _scan_data_directory(data_path: str) -> dict:
    """Scan a data directory and return info about its contents."""
    info = {
        "exists": False,
        "is_file": False,
        "is_dir": False,
        "manifest_files": [],
        "image_files": [],
        "total_files": 0,
        "subdirs": [],
        "manifest_df": None,
        "manifest_preview": None,
    }

    p = Path(data_path)
    if not p.exists():
        return info

    info["exists"] = True
    info["is_file"] = p.is_file()
    info["is_dir"] = p.is_dir()

    if p.is_file() and p.suffix in (".csv", ".parquet", ".tsv"):
        info["manifest_files"] = [str(p)]
    elif p.is_dir():
        info["subdirs"] = sorted([d.name for d in p.iterdir() if d.is_dir()])
        for ext in ("*.csv", "*.parquet", "*.tsv"):
            info["manifest_files"].extend(sorted(str(f) for f in p.glob(ext)))
        for ext in (
            "*.tif",
            "*.tiff",
            "*.czi",
            "*.ome.tiff",
            "*.ome.zarr",
            "*.png",
            "*.jpg",
            "*.zarr",
        ):
            found = list(p.rglob(ext))
            info["image_files"].extend(str(f) for f in found[:500])
        info["total_files"] = sum(1 for _ in p.rglob("*") if _.is_file())

    return info


def _load_manifest(filepath: str):
    """Load a CSV/Parquet manifest and return a pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        return None

    p = Path(filepath)
    try:
        if p.suffix == ".parquet":
            return pd.read_parquet(filepath)
        elif p.suffix == ".tsv":
            return pd.read_csv(filepath, sep="\t")
        else:
            return pd.read_csv(filepath)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.debug("Failed to load manifest %s: %s", filepath, exc)
        return None


def _try_load_image_preview(filepath: str):
    """Try to load an image file and return a numpy array for display."""
    try:
        import numpy as np
    except ImportError:
        return None, "numpy not installed"

    p = Path(filepath)
    ext = p.suffix.lower()

    # Try tifffile for .tif/.tiff/.ome.tiff
    if ext in (".tif", ".tiff"):
        try:
            import tifffile

            img = tifffile.imread(filepath)
            # For multi-dimensional, take a middle slice
            while img.ndim > 2:
                mid = img.shape[0] // 2
                img = img[mid]
            return img, None
        except Exception as exc:
            return None, str(exc)

    # Try PIL for standard image formats
    if ext in (".png", ".jpg", ".jpeg", ".bmp"):
        try:
            from PIL import Image

            with Image.open(filepath) as img:
                return np.array(img), None
        except (OSError, ValueError) as exc:
            return None, str(exc)

    # CZI files
    if ext == ".czi":
        try:
            from bioio import BioImage

            reader = BioImage(filepath)
            img = reader.get_image_data(
                "YX", C=0, Z=reader.dims.Z // 2 if reader.dims.Z > 1 else 0, T=0
            )
            return img, None
        except Exception as exc:
            return None, str(exc)

    return None, f"Unsupported format: {ext}"


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def _build_backbone_config(arch_key, spatial_dims, in_channels, filters, dropout, res_block):
    arch = BACKBONE_ARCHITECTURES[arch_key]
    target = arch["target"]

    if target == "monai.networks.nets.DynUNet":
        n_layers = len(filters)
        strides = [1] + [2] * (n_layers - 1)
        kernel_size = [3] * n_layers
        upsample_kernel_size = [2] * (n_layers - 1)
        return {
            "_target_": target,
            "spatial_dims": spatial_dims,
            "in_channels": in_channels,
            "out_channels": 1,
            "strides": strides,
            "kernel_size": kernel_size,
            "upsample_kernel_size": upsample_kernel_size,
            "filters": filters,
            "dropout": dropout,
            "res_block": res_block,
        }
    elif target == "monai.networks.nets.SwinUNETR":
        img_size = [96, 96, 96] if spatial_dims == 3 else [96, 96]
        return {
            "_target_": target,
            "img_size": img_size,
            "in_channels": in_channels,
            "out_channels": 1,
            "spatial_dims": spatial_dims,
            "feature_size": filters[0] if filters else 24,
            "drop_rate": dropout,
        }
    elif target == "monai.networks.nets.UNETR":
        img_size = [96, 96, 96] if spatial_dims == 3 else [96, 96]
        return {
            "_target_": target,
            "img_size": img_size,
            "in_channels": in_channels,
            "out_channels": 1,
            "spatial_dims": spatial_dims,
            "hidden_size": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "dropout_rate": dropout,
        }
    elif target == "monai.networks.nets.AttentionUnet":
        strides = [2] * (len(filters) - 1)
        kernel_size = [3] * len(filters)
        return {
            "_target_": target,
            "spatial_dims": spatial_dims,
            "in_channels": in_channels,
            "out_channels": 1,
            "channels": filters,
            "strides": strides,
            "kernel_size": kernel_size,
            "dropout": dropout,
        }
    return {}


def _build_task_head_config(head_cfg: dict, exp_type: str) -> dict:
    """Build a single task head config dict from GUI parameters."""
    head_info = HEAD_TYPES[head_cfg["head_type"]]

    # Loss
    loss_info = LOSS_FUNCTIONS[head_cfg["loss_function"]]
    loss_cfg = {"_target_": loss_info["target"]}
    loss_cfg.update(loss_info["kwargs"])

    # Postprocess
    act_key = head_cfg.get(
        "prediction_activation", "Sigmoid" if exp_type == "segmentation" else "None"
    )
    activation_cfg = POSTPROCESS_ACTIVATIONS.get(act_key)

    pred_postprocess = {
        "_target_": "cyto_dl.models.im2im.utils.postprocessing.ActThreshLabel",
        "rescale_dtype": head_cfg.get("rescale_dtype", "numpy.uint8"),
    }
    if activation_cfg is not None:
        pred_postprocess["activation"] = activation_cfg

    postprocess = {
        "input": {
            "_target_": "cyto_dl.models.im2im.utils.postprocessing.ActThreshLabel",
            "rescale_dtype": head_cfg.get("rescale_dtype", "numpy.uint8"),
        },
        "prediction": pred_postprocess,
    }

    result = {
        "_target_": head_info["target"],
        "loss": loss_cfg,
        "postprocess": postprocess,
    }

    # ResBlocksHead extra params
    if head_info["extra_params"]:
        result["in_channels"] = head_cfg.get("in_channels", 64)
        result["out_channels"] = head_cfg.get("out_channels", 1)
        result["spatial_dims"] = head_cfg.get("spatial_dims", 3)
        result["n_convs"] = head_cfg.get("n_convs", 1)
        result["dropout"] = head_cfg.get("head_dropout", 0.0)
        result["dense"] = head_cfg.get("dense", False)
        if head_cfg.get("resolution", "lr") == "hr":
            result["resolution"] = "hr"
            result["upsample_method"] = head_cfg.get("upsample_method", "pixelshuffle")
            ratio = head_cfg.get("upsample_ratio", 2)
            result["upsample_ratio"] = [ratio] * head_cfg.get("spatial_dims", 3)
        final_act = head_cfg.get("final_activation", "None")
        if final_act == "Sigmoid":
            result["final_act"] = {"_target_": "torch.nn.Sigmoid"}
        elif final_act == "ReLU":
            result["final_act"] = {"_target_": "torch.nn.ReLU"}

    return result


def build_full_config(params: dict) -> dict:
    """Construct a complete Hydra-compatible config dict from GUI parameters."""
    exp_type = params["experiment_type"]

    # ---- backbone ----
    backbone = _build_backbone_config(
        params["architecture"],
        params["spatial_dims"],
        params["in_channels"],
        params["filters"],
        params["dropout"],
        params.get("res_block", True),
    )

    # ---- optimizer ----
    opt_info = OPTIMIZERS[params["optimizer"]]
    opt_cfg = {
        "_partial_": True,
        "_target_": opt_info["target"],
        "lr": params["learning_rate"],
        "weight_decay": params["weight_decay"],
    }
    if opt_info["fused"]:
        opt_cfg["fused"] = True

    # ---- lr scheduler ----
    lr_sched_cfg = {}
    sched_info = LR_SCHEDULERS[params["lr_scheduler"]]
    if sched_info is not None:
        lr_sched_cfg = {"_partial_": True, "_target_": sched_info["target"]}
        lr_sched_cfg.update(sched_info["params"])
        if params["lr_scheduler"] == "CosineAnnealingLR":
            lr_sched_cfg["T_max"] = params["max_epochs"]

    # ---- task heads (multi-task) ----
    task_heads_list = params.get("task_heads", [])
    if not task_heads_list:
        # Fallback: single head from legacy params
        loss_info = LOSS_FUNCTIONS[params["loss_function"]]
        loss_cfg = {"_target_": loss_info["target"]}
        loss_cfg.update(loss_info["kwargs"])
        if exp_type == "segmentation":
            postprocess = {
                "input": {
                    "_target_": "cyto_dl.models.im2im.utils.postprocessing.ActThreshLabel",
                    "rescale_dtype": "numpy.uint8",
                },
                "prediction": {
                    "_target_": "cyto_dl.models.im2im.utils.postprocessing.ActThreshLabel",
                    "activation": {"_target_": "torch.nn.Sigmoid"},
                    "rescale_dtype": "numpy.uint8",
                },
            }
        else:
            postprocess = {
                "input": {
                    "_target_": "cyto_dl.models.im2im.utils.postprocessing.ActThreshLabel",
                    "rescale_dtype": "numpy.uint8",
                },
                "prediction": {
                    "_target_": "cyto_dl.models.im2im.utils.postprocessing.ActThreshLabel",
                    "rescale_dtype": "numpy.uint8",
                },
            }
        task_heads = {
            params["target_col"]: {
                "_target_": "cyto_dl.nn.BaseHead",
                "loss": loss_cfg,
                "postprocess": postprocess,
            },
        }
    else:
        task_heads = {}
        for head in task_heads_list:
            task_heads[head["task_name"]] = _build_task_head_config(head, exp_type)

    # ---- model ----
    model_cfg = {
        "_target_": "cyto_dl.models.im2im.MultiTaskIm2Im",
        "save_images_every_n_epochs": params["save_images_every_n"],
        "save_dir": "${paths.output_dir}",
        "x_key": params["source_col"],
        "backbone": backbone,
        "task_heads": task_heads,
        "optimizer": {"generator": opt_cfg},
        "inference_args": {
            "sw_batch_size": 1,
            "roi_size": params["patch_shape"],
            "overlap": params.get("inference_overlap", 0),
            "mode": "gaussian",
            "progress": True,
        },
    }
    if lr_sched_cfg:
        model_cfg["lr_scheduler"] = {"generator": lr_sched_cfg}

    # ---- trainer ----
    trainer_cfg = {
        "_target_": "lightning.Trainer",
        "default_root_dir": "${paths.output_dir}",
        "min_epochs": 1,
        "max_epochs": params["max_epochs"],
        "accelerator": "gpu",
        "devices": params["num_gpus"],
        "precision": params["precision"],
        "check_val_every_n_epoch": params["val_every_n"],
        "deterministic": False,
        "detect_anomaly": False,
        "gradient_clip_val": params["gradient_clip"] if params["gradient_clip"] > 0 else None,
        "gradient_clip_algorithm": "norm",
    }
    if params["num_gpus"] > 1:
        trainer_cfg["strategy"] = "ddp"

    # ---- callbacks ----
    callbacks_cfg = {
        "model_checkpoint": {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "dirpath": "${paths.output_dir}/checkpoints",
            "filename": "epoch_{epoch:03d}",
            "monitor": "val/loss",
            "mode": "min",
            "save_last": True,
            "auto_insert_metric_name": False,
        },
        "model_summary": {
            "_target_": "lightning.pytorch.callbacks.RichModelSummary",
            "max_depth": -1,
        },
        "rich_progress_bar": {
            "_target_": "lightning.pytorch.callbacks.RichProgressBar",
        },
    }
    if params["early_stopping"]:
        callbacks_cfg["early_stopping"] = {
            "_target_": "lightning.pytorch.callbacks.EarlyStopping",
            "monitor": "val/loss",
            "patience": params["early_stopping_patience"],
            "mode": "min",
        }
    if params["lr_monitor"]:
        callbacks_cfg["learning_rate_monitor"] = {
            "_target_": "lightning.pytorch.callbacks.LearningRateMonitor",
            "logging_interval": "epoch",
        }

    callbacks_cfg["saving"] = {
        "_target_": "cyto_dl.callbacks.ImageSaver",
        "save_dir": "${paths.output_dir}",
        "save_every_n_epochs": params["save_images_every_n"],
        "stages": ["train", "test", "val"],
        "save_input": True,
    }

    # ---- logger ----
    logger_name = LOGGERS[params["logger"]]
    logger_cfg = None
    if logger_name == "csv":
        logger_cfg = {
            "csv": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": "${paths.output_dir}",
                "name": params["experiment_name"],
            }
        }
    elif logger_name == "tensorboard":
        logger_cfg = {
            "tensorboard": {
                "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
                "save_dir": "${paths.output_dir}",
                "name": params["experiment_name"],
            }
        }
    elif logger_name == "wandb":
        logger_cfg = {
            "wandb": {
                "_target_": "lightning.pytorch.loggers.WandbLogger",
                "project": params["experiment_name"],
                "name": params["run_name"],
                "save_dir": "${paths.output_dir}",
            }
        }
    elif logger_name == "mlflow":
        logger_cfg = {
            "mlflow": {
                "_target_": "lightning.pytorch.loggers.MLFlowLogger",
                "experiment_name": params["experiment_name"],
                "run_name": params["run_name"],
                "save_dir": "${paths.output_dir}",
            }
        }

    # ---- performance / GPU optimizations ----
    perf_cfg = None
    if params["gpu_optimizations"]:
        perf_cfg = {
            "enable_cudnn_benchmark": True,
            "enable_tf32": True,
            "matmul_precision": params["matmul_precision"],
            "channels_last": params["channels_last"],
            "compile": {
                "enabled": params["torch_compile"],
                "mode": params["compile_mode"],
                "fullgraph": False,
                "dynamic": False,
            },
            "cuda_graphs": {"enabled": False, "warmup_iterations": 3},
            "gradient_checkpointing": {
                "enabled": params["gradient_checkpointing"],
            },
        }

    # ---- data ----
    # Collect all column names from task heads
    all_target_cols = []
    if task_heads_list:
        all_target_cols = [h["task_name"] for h in task_heads_list]
    else:
        all_target_cols = [params["target_col"]]
    all_columns = [params["source_col"]] + all_target_cols

    data_cfg = {
        "_target_": "cyto_dl.datamodules.dataframe.DataframeDatamodule",
        "path": params["data_path"],
        "cache_dir": params.get("cache_dir") or None,
        "num_workers": params["num_workers"],
        "batch_size": params["batch_size"],
        "pin_memory": True,
        "persistent_workers": True,
        "split_column": None,
        "columns": all_columns,
    }

    # ---- paths ----
    paths_cfg = {
        "root_dir": str(PROJECT_ROOT),
        "data_dir": (
            str(Path(params["data_path"]).parent)
            if params["data_path"]
            else str(PROJECT_ROOT / "data")
        ),
        "log_dir": params.get("log_dir") or str(PROJECT_ROOT / "logs"),
        "output_dir": params.get("output_dir")
        or str(PROJECT_ROOT / "logs" / params["experiment_name"] / params["run_name"]),
    }

    # ---- assemble top-level ----
    config = {
        "experiment_name": params["experiment_name"],
        "run_name": params["run_name"],
        "task_name": "train",
        "tags": ["streamlit-gui", exp_type],
        "train": True,
        "test": params["run_test"],
        "seed": params["seed"] if params["seed"] > 0 else None,
        "source_col": params["source_col"],
        "target_col": all_target_cols[0] if all_target_cols else params["target_col"],
        "spatial_dims": params["spatial_dims"],
        "raw_im_channels": params["in_channels"],
        "checkpoint": {
            "ckpt_path": params.get("resume_checkpoint") or None,
            "weights_only": None,
            "strict": True,
        },
        "paths": paths_cfg,
        "data": data_cfg,
        "model": model_cfg,
        "trainer": trainer_cfg,
        "callbacks": callbacks_cfg,
    }
    if logger_cfg:
        config["logger"] = logger_cfg
    if perf_cfg:
        config["performance"] = perf_cfg

    return config


# ---------------------------------------------------------------------------
# Process launcher helpers
# ---------------------------------------------------------------------------
def _stream_output(process, log_list):
    """Read merged stdout+stderr line-by-line into the session log list."""
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                log_list.append(line.rstrip("\n"))
    except (ValueError, OSError):
        pass
    finally:
        with suppress(OSError):
            process.stdout.close()


def launch_training(config: dict, config_path: Path):
    """Write config YAML and launch training subprocess."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["CYTODL_CONFIG_PATH"] = str(config_path.parent)

    cmd = [
        "python",
        "-u",
        str(PROJECT_ROOT / "cyto_dl" / "train.py"),
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
    ]

    process = subprocess.Popen(  # nosec: B603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=env,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    st.session_state.training_process = process
    st.session_state.training_log = []
    st.session_state.training_running = True

    t = threading.Thread(
        target=_stream_output, args=(process, st.session_state.training_log), daemon=True
    )
    t.start()
    st.session_state.log_thread = t


def _kill_process_group(proc):
    """Safely terminate a process group with SIGKILL escalation."""
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid) if os.name != "nt" else proc.pid
        if os.name != "nt":
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
            pass
    except OSError:
        pass


def stop_training():
    _kill_process_group(st.session_state.training_process)
    st.session_state.training_running = False


def _parse_loss_from_logs(log_lines: list) -> dict:
    """Parse training/validation loss values from Lightning log lines.

    Looks for patterns like:
      - 'train/loss': 0.1234
      - val/loss: 0.0567
      - epoch 5, train_loss=0.1234
      - Epoch 5/100 ... loss=0.1234
    Returns dict with keys 'train_loss', 'val_loss', each a list of (epoch, value).
    """
    train_losses = []
    val_losses = []
    current_epoch = 0

    for line in log_lines:
        # Try to extract epoch number
        epoch_match = re.search(r"[Ee]poch[\s:=]*(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        # Pattern: 'train/loss': <value> or train/loss=<value> or train/loss: <value>
        train_match = re.search(r'train[/_]loss[\'"\s:=]+([0-9.eE+-]+)', line)
        if train_match:
            with suppress(ValueError):
                val = float(train_match.group(1))
                train_losses.append((current_epoch, val))

        # Pattern: 'val/loss': <value> or val/loss=<value> or val/loss: <value>
        val_match = re.search(r'val[/_]loss[\'"\s:=]+([0-9.eE+-]+)', line)
        if val_match:
            with suppress(ValueError):
                val = float(val_match.group(1))
                val_losses.append((current_epoch, val))

    return {"train_loss": train_losses, "val_loss": val_losses}


def _load_csv_metrics(output_dir: str):
    """Load metrics from Lightning CSVLogger output.

    CSVLogger writes to <save_dir>/<name>/version_X/metrics.csv
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    output_path = Path(output_dir)
    # Search for metrics.csv in common locations
    candidates = list(output_path.rglob("metrics.csv"))
    if not candidates:
        return None

    # Use the most recently modified one
    metrics_file = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        df = pd.read_csv(metrics_file)
        return df
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.debug("Failed to load CSV metrics %s: %s", metrics_file, exc)
        return None


def _find_best_checkpoint(output_dir: str):
    """Find the best checkpoint in the output directory."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.exists():
        # Try searching more broadly
        ckpt_dir = Path(output_dir)

    candidates = list(ckpt_dir.rglob("*.ckpt"))
    if not candidates:
        return None

    # Prefer 'last.ckpt' for mid-training eval, otherwise most recent
    for c in candidates:
        if c.name == "last.ckpt":
            return str(c)

    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def launch_evaluation(config_path: str, checkpoint_path: str, output_dir: str):
    """Launch a prediction/evaluation subprocess using the current best checkpoint."""
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)

    eval_output = Path(output_dir) / "eval_preview"
    eval_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-u",
        str(PROJECT_ROOT / "cyto_dl" / "eval.py"),
        "--config-path",
        str(Path(config_path).parent),
        "--config-name",
        Path(config_path).stem,
        f"checkpoint.ckpt_path={checkpoint_path}",
        f"paths.output_dir={eval_output}",
    ]

    process = subprocess.Popen(  # nosec: B603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=env,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    st.session_state.eval_process = process
    st.session_state.eval_log = []
    st.session_state.eval_running = True

    t = threading.Thread(
        target=_stream_output, args=(process, st.session_state.eval_log), daemon=True
    )
    t.start()


def _find_prediction_images(output_dir: str) -> list:
    """Find saved prediction images from the latest evaluation run."""
    eval_dir = Path(output_dir) / "eval_preview"
    if not eval_dir.exists():
        return []

    images = []
    for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
        images.extend(eval_dir.rglob(ext))

    return sorted(images, key=lambda p: p.stat().st_mtime, reverse=True)[:20]


def launch_tensorrt_export(trt_params: dict):
    """Launch TensorRT export as a subprocess."""
    cmd = [
        "python",
        "-u",
        str(PROJECT_ROOT / "scripts" / "export_to_tensorrt.py"),
        "--config",
        trt_params["config_path"],
        "--ckpt",
        trt_params["checkpoint_path"],
        "--output",
        trt_params["output_path"],
        "--input-shape",
    ] + [str(s) for s in trt_params["input_shape"]]

    cmd += ["--precision", trt_params["precision"]]
    cmd += ["--workspace-size", str(trt_params["workspace_size"])]

    if trt_params["dynamic_shapes"]:
        cmd += [
            "--dynamic-shapes",
            "--min-batch",
            str(trt_params["min_batch"]),
            "--max-batch",
            str(trt_params["max_batch"]),
            "--opt-batch",
            str(trt_params["opt_batch"]),
        ]

    if trt_params["precision"] == "int8" and trt_params.get("calibration_data"):
        cmd += ["--calibration-data", trt_params["calibration_data"]]
        cmd += ["--num-calibration-samples", str(trt_params.get("num_calibration_samples", 100))]

    if trt_params["benchmark"]:
        cmd += ["--benchmark"]
        cmd += ["--benchmark-iterations", str(trt_params.get("benchmark_iterations", 100))]

    process = subprocess.Popen(  # nosec: B603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    st.session_state.trt_process = process
    st.session_state.trt_log = []
    st.session_state.trt_running = True

    t = threading.Thread(
        target=_stream_output, args=(process, st.session_state.trt_log), daemon=True
    )
    t.start()
    st.session_state.trt_log_thread = t


def stop_tensorrt():
    _kill_process_group(st.session_state.trt_process)
    st.session_state.trt_running = False


# ---------------------------------------------------------------------------
# AutoKernel
# ---------------------------------------------------------------------------
AUTOKERNEL_VENDOR_DIR = PROJECT_ROOT / "vendor" / "autokernel"
AUTOKERNEL_SENTINEL = AUTOKERNEL_VENDOR_DIR / ".cytodl_ready"
AUTOKERNEL_RESULTS_MARKER = "### AUTOKERNEL_RESULTS_JSON ###"


def autokernel_is_installed() -> bool:
    return AUTOKERNEL_SENTINEL.is_file()


def autokernel_commit() -> str:
    if not AUTOKERNEL_SENTINEL.is_file():
        return ""
    try:
        for line in AUTOKERNEL_SENTINEL.read_text(encoding="utf-8").splitlines():
            if line.startswith("commit="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def launch_autokernel_setup():
    cmd = ["python", "-u", str(PROJECT_ROOT / "scripts" / "setup_autokernel.py")]
    process = subprocess.Popen(  # nosec: B603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    st.session_state.autokernel_setup_process = process
    st.session_state.autokernel_setup_log = []
    st.session_state.autokernel_setup_running = True
    t = threading.Thread(
        target=_stream_output,
        args=(process, st.session_state.autokernel_setup_log),
        daemon=True,
    )
    t.start()


def launch_autokernel(params: dict):
    """Run scripts/autokernel_optimize.py as a subprocess; stream into session log."""
    cmd = [
        "python",
        "-u",
        str(PROJECT_ROOT / "scripts" / "autokernel_optimize.py"),
        "--source", params["source"],
        "--input-shape", *(str(d) for d in params["input_shape"]),
        "--dtype", params["dtype"],
        "--backend", params["backend"],
        "--top-n", str(params["top_n"]),
        "--experiments-per-kernel", str(params["experiments_per_kernel"]),
        "--target-metric", params["target_metric"],
        "--workspace", params["workspace"],
    ]
    if params["source"] == "cytodl_class":
        cmd += ["--module", params["module"], "--class-name", params["class_name"]]
    else:
        cmd += ["--ckpt", params["ckpt"]]
    if params.get("llm_model"):
        cmd += ["--llm-model", params["llm_model"]]
    if params.get("config_path"):
        cmd += ["--config", params["config_path"]]

    process = subprocess.Popen(  # nosec: B603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    st.session_state.autokernel_process = process
    st.session_state.autokernel_log = []
    st.session_state.autokernel_running = True
    st.session_state.autokernel_results = None
    t = threading.Thread(
        target=_stream_output,
        args=(process, st.session_state.autokernel_log),
        daemon=True,
    )
    t.start()
    st.session_state.autokernel_log_thread = t


def stop_autokernel():
    _kill_process_group(st.session_state.autokernel_process)
    st.session_state.autokernel_running = False


def _parse_autokernel_results(log_lines: list) -> dict | None:
    """Find the trailing JSON blob emitted by autokernel_optimize.py."""
    import json as _json

    for idx in range(len(log_lines) - 1, -1, -1):
        if log_lines[idx].strip() == AUTOKERNEL_RESULTS_MARKER:
            payload = "\n".join(log_lines[idx + 1 :]).strip()
            if not payload:
                return None
            with suppress(ValueError):
                return _json.loads(payload)
            return None
    return None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="CytoDL Training GUI", page_icon="🔬", layout="wide")
    _init_state()

    st.title("CytoDL Im2Im Training")
    st.caption(
        "Configure and launch model training without editing YAML files. Optimized for CUDA 13 / Blackwell GPUs."
    )

    # ====================================================================
    # Sidebar
    # ====================================================================
    with st.sidebar:
        st.header("Quick Start")
        st.markdown(
            "1. Choose experiment type\n"
            "2. Preview your data\n"
            "3. Configure model & task heads\n"
            "4. Set training parameters\n"
            "5. Enable GPU optimizations\n"
            "6. Generate config & launch\n"
            "7. Export to TensorRT (optional)\n"
            "8. Optimize kernels with AutoKernel (optional)"
        )
        st.divider()
        st.markdown(f"**Project root:** `{PROJECT_ROOT}`")

    # ====================================================================
    # Tabs
    # ====================================================================
    (
        tab_exp,
        tab_data,
        tab_preview,
        tab_zoo,
        tab_model,
        tab_heads,
        tab_train,
        tab_gpu,
        tab_launch,
        tab_trt,
        tab_autokernel,
    ) = st.tabs(
        [
            "Experiment",
            "Data",
            "Data Preview",
            "Model Zoo",
            "Model",
            "Task Heads",
            "Training",
            "GPU / Perf",
            "Launch",
            "TensorRT Export",
            "AutoKernel",
        ]
    )

    # ------------------------------------------------------------------ #
    # TAB: Experiment
    # ------------------------------------------------------------------ #
    with tab_exp:
        st.subheader("Experiment Setup")
        col1, col2 = st.columns(2)
        with col1:
            exp_type_label = st.selectbox("Experiment type", list(EXPERIMENT_TYPES.keys()))
            experiment_name = st.text_input("Experiment name", value="bf_to_nuclei")
        with col2:
            run_name = st.text_input(
                "Run name", value=datetime.datetime.now().strftime("run_%Y%m%d_%H%M")
            )
            seed = st.number_input(
                "Random seed (0 = none)", min_value=0, max_value=999999, value=12345
            )

        exp_type = EXPERIMENT_TYPES[exp_type_label]

    # ------------------------------------------------------------------ #
    # TAB: Data
    # ------------------------------------------------------------------ #
    with tab_data:
        st.subheader("Data Configuration")

        data_path = st.text_input(
            "Data directory (CSV / manifest path)",
            value=str(PROJECT_ROOT / "data" / "example_experiment_data" / exp_type),
            help="Path to directory containing your training data manifest (CSV with source/target columns).",
        )
        cache_dir = st.text_input("Cache directory (optional)", value="")

        col1, col2 = st.columns(2)
        with col1:
            source_col = st.text_input(
                "Source column name",
                value="brightfield" if exp_type == "labelfree" else "raw",
                help="Column in manifest CSV pointing to input images.",
            )
        with col2:
            target_col = st.text_input(
                "Target column name",
                value="signal" if exp_type == "labelfree" else "seg",
                help="Column in manifest CSV pointing to ground truth images.",
            )

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            spatial_dims = st.selectbox("Spatial dimensions", [3, 2], index=0)
        with col2:
            batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=2)
        with col3:
            num_workers = st.number_input("DataLoader workers", min_value=0, max_value=32, value=8)

        st.markdown("**Patch shape** (training crop size)")
        if spatial_dims == 3:
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pz = st.number_input("Z", min_value=1, value=16)
            with pc2:
                py_ = st.number_input("Y", min_value=1, value=128)
            with pc3:
                px = st.number_input("X", min_value=1, value=128)
            patch_shape = [pz, py_, px]
        else:
            pc1, pc2 = st.columns(2)
            with pc1:
                py_ = st.number_input("Y", min_value=1, value=256)
            with pc2:
                px = st.number_input("X", min_value=1, value=256)
            patch_shape = [py_, px]

    # ------------------------------------------------------------------ #
    # TAB: Data Preview
    # ------------------------------------------------------------------ #
    with tab_preview:
        st.subheader("Data Preview")
        st.caption(
            "Inspect your data directory, view manifest contents, and preview sample images."
        )

        if st.button("Scan Data Directory", type="primary"):
            with st.spinner("Scanning..."):
                info = _scan_data_directory(data_path)
                st.session_state["data_scan_info"] = info

        info = st.session_state.get("data_scan_info")
        if info:
            if not info["exists"]:
                st.error(f"Path does not exist: `{data_path}`")
            else:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total files", info["total_files"])
                with col2:
                    st.metric("Manifest files", len(info["manifest_files"]))
                with col3:
                    st.metric("Image files found", len(info["image_files"]))

                if info["subdirs"]:
                    st.markdown(
                        "**Subdirectories:** " + ", ".join(f"`{d}`" for d in info["subdirs"][:20])
                    )

                # Manifest viewer
                if info["manifest_files"]:
                    st.divider()
                    st.markdown("#### Manifest Viewer")
                    selected_manifest = st.selectbox(
                        "Select manifest file", info["manifest_files"]
                    )

                    if selected_manifest:
                        df = _load_manifest(selected_manifest)
                        if df is not None:
                            st.markdown(f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns")
                            st.markdown(f"**Columns:** {', '.join(f'`{c}`' for c in df.columns)}")

                            # Check for source/target columns
                            if source_col in df.columns:
                                st.success(f"Source column `{source_col}` found")
                            else:
                                st.warning(
                                    f"Source column `{source_col}` NOT found in manifest. Available: {list(df.columns)}"
                                )

                            if target_col in df.columns:
                                st.success(f"Target column `{target_col}` found")
                            else:
                                st.warning(
                                    f"Target column `{target_col}` NOT found in manifest. Available: {list(df.columns)}"
                                )

                            # Split column detection
                            split_candidates = [
                                c
                                for c in df.columns
                                if c.lower() in ("split", "fold", "set", "subset", "partition")
                            ]
                            if split_candidates:
                                for sc in split_candidates:
                                    counts = df[sc].value_counts()
                                    st.markdown(f"**Split column `{sc}`:** {dict(counts)}")

                            # Show dataframe
                            st.dataframe(df.head(50), use_container_width=True)
                        else:
                            st.error("Could not load manifest file.")

                # Image preview
                if info["image_files"]:
                    st.divider()
                    st.markdown("#### Image Preview")
                    st.caption("Select an image file to preview a 2D slice.")

                    # Show first 50 image files to choose from
                    display_files = info["image_files"][:50]
                    selected_image = st.selectbox(
                        "Select image file",
                        display_files,
                        format_func=lambda x: Path(x).name,
                    )

                    if selected_image and st.button("Load Preview"):
                        with st.spinner("Loading image..."):
                            img_arr, err = _try_load_image_preview(selected_image)
                            if img_arr is not None:
                                st.image(
                                    img_arr,
                                    caption=Path(selected_image).name,
                                    use_container_width=True,
                                    clamp=True,
                                )
                                st.markdown(
                                    f"**Shape:** {img_arr.shape}  |  **dtype:** {img_arr.dtype}  |  **range:** [{img_arr.min():.2f}, {img_arr.max():.2f}]"
                                )
                            else:
                                st.error(f"Could not load image: {err}")

                # Also allow preview from manifest paths
                if info["manifest_files"]:
                    st.divider()
                    st.markdown("#### Preview from Manifest Paths")
                    st.caption("Load an image referenced in the manifest CSV.")
                    manifest_for_preview = st.selectbox(
                        "Manifest", info["manifest_files"], key="manifest_preview_select"
                    )
                    df_preview = (
                        _load_manifest(manifest_for_preview) if manifest_for_preview else None
                    )
                    if df_preview is not None and source_col in df_preview.columns:
                        row_idx = st.number_input(
                            "Row index",
                            min_value=0,
                            max_value=max(0, len(df_preview) - 1),
                            value=0,
                            key="manifest_row_idx",
                        )
                        source_path = str(df_preview.iloc[row_idx][source_col])
                        st.text(f"Source path: {source_path}")

                        if target_col in df_preview.columns:
                            target_path_val = str(df_preview.iloc[row_idx][target_col])
                            st.text(f"Target path: {target_path_val}")

                        if st.button("Load Manifest Image", key="load_manifest_img"):
                            with st.spinner("Loading..."):
                                img_arr, err = _try_load_image_preview(source_path)
                                if img_arr is not None:
                                    st.image(
                                        img_arr,
                                        caption=f"Source: {Path(source_path).name}",
                                        use_container_width=True,
                                        clamp=True,
                                    )
                                    st.markdown(
                                        f"**Shape:** {img_arr.shape}  |  **dtype:** {img_arr.dtype}"
                                    )
                                else:
                                    st.warning(f"Could not load source image: {err}")

                                if target_col in df_preview.columns:
                                    target_path_val = str(df_preview.iloc[row_idx][target_col])
                                    timg, terr = _try_load_image_preview(target_path_val)
                                    if timg is not None:
                                        st.image(
                                            timg,
                                            caption=f"Target: {Path(target_path_val).name}",
                                            use_container_width=True,
                                            clamp=True,
                                        )
                                        st.markdown(
                                            f"**Shape:** {timg.shape}  |  **dtype:** {timg.dtype}"
                                        )
                                    else:
                                        st.warning(f"Could not load target image: {terr}")

    # ------------------------------------------------------------------ #
    # TAB: Model Zoo
    # ------------------------------------------------------------------ #
    with tab_zoo:
        st.subheader("Model Zoo & Architecture Explorer")
        st.caption(
            "Browse available architectures, compare task suitability, and get "
            "recommendations for your specific use case. Use the live scanner to "
            "discover all models installed in your environment."
        )

        # ---- Task-Based Recommendations ----
        st.markdown("### Recommendations by Task")
        selected_task = st.selectbox(
            "What are you trying to do?",
            list(TASK_RECOMMENDATIONS.keys()),
            key="zoo_task",
        )

        recs = TASK_RECOMMENDATIONS[selected_task]
        for rank, (model_name, reason) in enumerate(recs, 1):
            medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
            zoo_entry = MODEL_ZOO.get(model_name, {})
            vram = zoo_entry.get("vram_estimate", "?")
            params = zoo_entry.get("params", "?")
            st.markdown(
                f"**{medal}: {model_name}** ({zoo_entry.get('source', '?')}) — {reason}  \n"
                f"*VRAM: {vram} | Params: {params}*"
            )

        st.divider()

        # ---- Full Catalog ----
        st.markdown("### Full Architecture Catalog")

        # Filter by category
        categories = sorted({m["category"] for m in MODEL_ZOO.values()})
        selected_cats = st.multiselect(
            "Filter by category", categories, default=categories, key="zoo_cats"
        )

        for model_name, info in MODEL_ZOO.items():
            if info["category"] not in selected_cats:
                continue

            with st.expander(
                f"{model_name}  —  {info['source']} / {info['category']}", expanded=False
            ):
                st.markdown(f"**{info['description']}**")
                st.markdown(f"**Target class:** `{info['target']}`")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Params:** {info['params']}")
                with col2:
                    st.markdown(f"**VRAM:** {info['vram_estimate']}")
                with col3:
                    st.markdown(f"**Source:** {info['source']}")

                st.markdown(f"**Strengths:** {info['strengths']}")
                st.markdown(f"**Weaknesses:** {info['weaknesses']}")

                # Task suitability table
                st.markdown("**Task Suitability:**")
                task_data = []
                for task, rating in info["tasks"].items():
                    color = {"best": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}.get(
                        rating.split()[0], "⚪"
                    )
                    task_data.append({"Task": task, "Rating": f"{color} {rating}"})
                st.table(task_data)

                # Show recommended config
                if info.get("recommended_config"):
                    st.markdown("**Recommended Config:**")
                    st.code(
                        yaml.dump(info["recommended_config"], default_flow_style=False),
                        language="yaml",
                    )

                # Button to use this model
                if model_name in BACKBONE_ARCHITECTURES or info["target"] in [
                    v["target"] for v in BACKBONE_ARCHITECTURES.values()
                ]:
                    st.success("This architecture is available in the Model tab dropdown.")
                else:
                    st.info(
                        f"To use this model, set the backbone `_target_` to `{info['target']}` "
                        f"in the generated YAML config, or add it to the Model tab."
                    )

        st.divider()

        # ---- Live Environment Scanner ----
        st.markdown("### Live Environment Scanner")
        st.caption("Scan your Python environment to discover all installed model architectures.")

        if st.button("Scan Installed Models", type="primary", key="zoo_scan_btn"):
            with st.spinner("Scanning MONAI and timm installations..."):
                scan_results = _scan_installed_models()
                st.session_state["zoo_scan_results"] = scan_results

        scan = st.session_state.get("zoo_scan_results")
        if scan:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### MONAI Networks")
                if scan["monai_version"]:
                    st.success(f"MONAI {scan['monai_version']} installed")
                    st.markdown(f"**{len(scan['monai_models'])} architectures available:**")

                    # Categorize as known/new
                    known_targets = {
                        v["target"].split(".")[-1]
                        for v in MODEL_ZOO.values()
                        if "monai" in v["target"]
                    }
                    new_models = [m for m in scan["monai_models"] if m not in known_targets]
                    known_models = [m for m in scan["monai_models"] if m in known_targets]

                    if known_models:
                        st.markdown(
                            "**In catalog (configured):** "
                            + ", ".join(f"`{m}`" for m in known_models)
                        )
                    if new_models:
                        st.markdown("**Discovered (not yet in catalog):**")
                        for m in new_models:
                            st.markdown(
                                f"- `monai.networks.nets.{m}` — *try adding as custom backbone*"
                            )
                else:
                    st.warning("MONAI not installed. Install with: `pip install monai`")

            with col2:
                st.markdown("#### timm Models")
                if scan["timm_version"]:
                    st.success(f"timm {scan['timm_version']} installed")
                    st.markdown(f"**{len(scan['timm_models'])} models available**")

                    # Show segmentation-relevant filters
                    st.markdown("**Search timm models:**")
                    timm_search = st.text_input(
                        "Filter (e.g. 'resnet', 'efficientnet', 'convnext')", key="timm_search"
                    )
                    if timm_search:
                        matches = [
                            m for m in scan["timm_models"] if timm_search.lower() in m.lower()
                        ]
                        if matches:
                            st.markdown(f"Found {len(matches)} matches:")
                            st.code("\n".join(matches[:50]), language="text")
                            if len(matches) > 50:
                                st.caption(f"...and {len(matches) - 50} more")
                        else:
                            st.warning("No matches found.")
                    else:
                        # Show curated categories
                        st.markdown("**Popular encoder families for segmentation:**")
                        families = {
                            "ResNet": [m for m in scan["timm_models"] if m.startswith("resnet")],
                            "EfficientNet": [
                                m for m in scan["timm_models"] if m.startswith("efficientnet")
                            ],
                            "ConvNeXt": [
                                m for m in scan["timm_models"] if m.startswith("convnext")
                            ],
                            "RegNet": [m for m in scan["timm_models"] if m.startswith("regnet")],
                        }
                        for family, models in families.items():
                            if models:
                                st.markdown(f"- **{family}:** {len(models)} variants")
                else:
                    st.warning("timm not installed. Install with: `pip install timm`")

        st.divider()

        # ---- Comparison table ----
        st.markdown("### Architecture Comparison")
        st.caption("Side-by-side comparison of all cataloged architectures.")

        comparison_data = []
        for name, info in MODEL_ZOO.items():
            row = {
                "Architecture": name,
                "Source": info["source"],
                "Category": info["category"],
                "Params": info["params"],
                "VRAM": info["vram_estimate"],
            }
            # Add task ratings
            for task in (
                "BF → Nuclei Segmentation",
                "BF → Cell Segmentation",
                "BF → Fluorescence (label-free)",
            ):
                rating = info["tasks"].get(task, "—")
                icon = {"best": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}.get(
                    rating.split()[0] if isinstance(rating, str) else "", "⚪"
                )
                row[task.replace("BF → ", "")] = f"{icon} {rating}"
            comparison_data.append(row)

        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------ #
    # TAB: Model
    # ------------------------------------------------------------------ #
    with tab_model:
        st.subheader("Model Architecture")

        arch_key = st.selectbox("Backbone", list(BACKBONE_ARCHITECTURES.keys()))
        st.info(BACKBONE_ARCHITECTURES[arch_key]["description"])

        col1, col2 = st.columns(2)
        with col1:
            in_channels = st.number_input("Input channels", min_value=1, max_value=64, value=1)
        with col2:
            dropout = st.slider("Dropout", 0.0, 0.5, 0.0, 0.05)

        if arch_key == "DynUNet (MONAI)":
            st.markdown("**Encoder filter sizes** (each level doubles receptive field)")
            filter_str = st.text_input("Filters (comma-separated)", value="32, 64, 128, 256")
            try:
                filters = [int(x.strip()) for x in filter_str.split(",") if x.strip()]
            except ValueError:
                st.warning("Invalid filter values — expected comma-separated integers.")
                filters = [32, 64, 128, 256]
            res_block = st.checkbox("Use residual blocks", value=True)
        elif arch_key == "SwinUNETR (MONAI)":
            feature_size = st.number_input(
                "Feature size", min_value=12, max_value=96, value=48, step=12
            )
            filters = [feature_size]
            res_block = False
        elif arch_key == "AttentionUNet (MONAI)":
            filter_str = st.text_input("Channel sizes (comma-separated)", value="32, 64, 128, 256")
            try:
                filters = [int(x.strip()) for x in filter_str.split(",") if x.strip()]
            except ValueError:
                st.warning("Invalid channel sizes — expected comma-separated integers.")
                filters = [32, 64, 128, 256]
            res_block = False
        else:
            filters = [16, 32, 64]
            res_block = False

        st.divider()
        st.subheader("Optimizer")
        col1, col2 = st.columns(2)
        with col1:
            optimizer_key = st.selectbox("Optimizer", list(OPTIMIZERS.keys()))
        with col2:
            learning_rate = st.number_input(
                "Learning rate", min_value=1e-7, max_value=1.0, value=1e-4, format="%.1e"
            )

        weight_decay = st.number_input(
            "Weight decay", min_value=0.0, max_value=1.0, value=1e-4, format="%.1e"
        )

        st.divider()
        st.subheader("LR Scheduler")
        lr_scheduler_key = st.selectbox("Scheduler", list(LR_SCHEDULERS.keys()))

    # ------------------------------------------------------------------ #
    # TAB: Task Heads (Multi-Task)
    # ------------------------------------------------------------------ #
    with tab_heads:
        st.subheader("Multi-Task Head Configuration")
        st.caption(
            "Configure one or more output task heads. Each head predicts a different target "
            "(e.g., nuclei segmentation + cell boundary + membrane signal). "
            "The backbone is shared; each head has its own loss, postprocessing, and output column."
        )

        # Number of heads
        num_heads = st.number_input("Number of task heads", min_value=1, max_value=8, value=1)

        task_heads_config = []
        for i in range(num_heads):
            st.divider()
            st.markdown(f"### Task Head {i + 1}")

            col1, col2 = st.columns(2)
            with col1:
                task_name = st.text_input(
                    "Target column / task name",
                    value=target_col if i == 0 else f"task_{i + 1}",
                    key=f"head_task_name_{i}",
                    help="Column name in manifest CSV for this task's ground truth.",
                )
            with col2:
                head_type_key = st.selectbox(
                    "Head type",
                    list(HEAD_TYPES.keys()),
                    key=f"head_type_{i}",
                )

            st.info(HEAD_TYPES[head_type_key]["description"])

            col1, col2 = st.columns(2)
            with col1:
                default_loss_idx = 0 if exp_type == "segmentation" else 2
                head_loss = st.selectbox(
                    "Loss function",
                    list(LOSS_FUNCTIONS.keys()),
                    index=default_loss_idx,
                    key=f"head_loss_{i}",
                )
            with col2:
                pred_activation = st.selectbox(
                    "Prediction activation",
                    list(POSTPROCESS_ACTIVATIONS.keys()),
                    index=0 if exp_type == "segmentation" else 3,
                    key=f"head_activation_{i}",
                )

            rescale_dtype = st.selectbox(
                "Output dtype",
                ["numpy.uint8", "numpy.uint16", "numpy.float32"],
                key=f"head_dtype_{i}",
            )

            head_entry = {
                "task_name": task_name,
                "head_type": head_type_key,
                "loss_function": head_loss,
                "prediction_activation": pred_activation,
                "rescale_dtype": rescale_dtype,
                "spatial_dims": spatial_dims,
            }

            # ResBlocksHead extra parameters
            if HEAD_TYPES[head_type_key]["extra_params"]:
                st.markdown("**ResBlocksHead Parameters**")
                rcol1, rcol2, rcol3 = st.columns(3)
                with rcol1:
                    head_entry["in_channels"] = st.number_input(
                        "In channels (backbone out)",
                        min_value=1,
                        value=filters[-1] if filters else 64,
                        key=f"head_in_ch_{i}",
                    )
                    head_entry["out_channels"] = st.number_input(
                        "Out channels",
                        min_value=1,
                        value=1,
                        key=f"head_out_ch_{i}",
                    )
                with rcol2:
                    head_entry["n_convs"] = st.number_input(
                        "Num conv layers",
                        min_value=1,
                        max_value=8,
                        value=2,
                        key=f"head_nconvs_{i}",
                    )
                    head_entry["head_dropout"] = st.slider(
                        "Head dropout",
                        0.0,
                        0.5,
                        0.0,
                        0.05,
                        key=f"head_dropout_{i}",
                    )
                with rcol3:
                    head_entry["dense"] = st.checkbox(
                        "Dense connections", value=False, key=f"head_dense_{i}"
                    )
                    head_entry["resolution"] = st.selectbox(
                        "Resolution",
                        ["lr", "hr"],
                        key=f"head_resolution_{i}",
                        help="'hr' enables upsampling for super-resolution tasks.",
                    )

                if head_entry["resolution"] == "hr":
                    ucol1, ucol2 = st.columns(2)
                    with ucol1:
                        head_entry["upsample_method"] = st.selectbox(
                            "Upsample method",
                            ["pixelshuffle", "deconv", "nontrainable", "deconv_group"],
                            key=f"head_upsample_{i}",
                        )
                    with ucol2:
                        head_entry["upsample_ratio"] = st.number_input(
                            "Upsample ratio",
                            min_value=1,
                            max_value=8,
                            value=2,
                            key=f"head_upsample_ratio_{i}",
                        )

                head_entry["final_activation"] = st.selectbox(
                    "Final activation",
                    ["None", "Sigmoid", "ReLU"],
                    key=f"head_final_act_{i}",
                )

            task_heads_config.append(head_entry)

        st.session_state.task_heads_config = task_heads_config

        # Summary
        st.divider()
        st.markdown("#### Task Heads Summary")
        summary_data = [
            {
                "Task": h["task_name"],
                "Head Type": h["head_type"].split("(")[0].strip(),
                "Loss": h["loss_function"],
                "Activation": h.get("prediction_activation", "—"),
            }
            for h in task_heads_config
        ]
        st.table(summary_data)

    # ------------------------------------------------------------------ #
    # TAB: Training
    # ------------------------------------------------------------------ #
    with tab_train:
        st.subheader("Training Parameters")

        col1, col2 = st.columns(2)
        with col1:
            max_epochs = st.number_input("Max epochs", min_value=1, max_value=10000, value=100)
            val_every_n = st.number_input(
                "Validate every N epochs", min_value=1, max_value=100, value=1
            )
            save_images_every_n = st.number_input(
                "Save images every N epochs", min_value=1, max_value=100, value=5
            )
        with col2:
            gradient_clip = st.number_input(
                "Gradient clip (0=off)", min_value=0.0, max_value=100.0, value=1.0
            )
            num_gpus = st.number_input("Number of GPUs", min_value=1, max_value=8, value=1)
            run_test = st.checkbox("Run test after training", value=True)

        st.divider()
        st.subheader("Callbacks")
        col1, col2 = st.columns(2)
        with col1:
            early_stopping = st.checkbox("Early stopping", value=True)
            if early_stopping:
                early_stopping_patience = st.number_input(
                    "Patience (epochs)", min_value=1, max_value=1000, value=50
                )
            else:
                early_stopping_patience = 50
        with col2:
            lr_monitor = st.checkbox("LR monitor", value=True)

        st.divider()
        st.subheader("Logging")
        logger_key = st.selectbox("Logger", list(LOGGERS.keys()))

        st.divider()
        st.subheader("Resume Training")
        resume_checkpoint = st.text_input("Checkpoint path (leave empty to start fresh)", value="")

        st.divider()
        st.subheader("Output")
        output_dir = st.text_input(
            "Output directory",
            value=str(PROJECT_ROOT / "logs" / experiment_name / run_name),
        )
        log_dir = st.text_input("Log directory", value=str(PROJECT_ROOT / "logs"))

    # ------------------------------------------------------------------ #
    # TAB: GPU / Perf
    # ------------------------------------------------------------------ #
    with tab_gpu:
        st.subheader("GPU Performance Optimizations")
        st.caption("Tuned for CUDA 13 / Blackwell (B200/B100) tensor cores.")

        gpu_optimizations = st.checkbox("Enable GPU optimizations", value=True)

        if gpu_optimizations:
            col1, col2 = st.columns(2)
            with col1:
                precision_key = st.selectbox("Precision", list(PRECISION_OPTIONS.keys()))
                precision = PRECISION_OPTIONS[precision_key]
                matmul_precision = st.selectbox(
                    "Matrix multiply precision", ["high", "medium", "highest"], index=0
                )
                channels_last = st.checkbox(
                    "Channels-last memory format", value=True, help="20-30% speedup on modern GPUs"
                )
            with col2:
                torch_compile = st.checkbox(
                    "torch.compile", value=True, help="JIT compile model for faster execution"
                )
                compile_mode = st.selectbox(
                    "Compile mode",
                    ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    index=0,
                    help="'max-autotune' is best for training (longer compile, best perf)",
                )
                gradient_checkpointing = st.checkbox(
                    "Gradient checkpointing",
                    value=False,
                    help="Trade compute for memory — enable if OOM",
                )
        else:
            precision = "32-true"
            matmul_precision = "highest"
            channels_last = False
            torch_compile = False
            compile_mode = "default"
            gradient_checkpointing = False

    # ------------------------------------------------------------------ #
    # TAB: Launch
    # ------------------------------------------------------------------ #
    with tab_launch:
        st.subheader("Generate Configuration & Launch Training")

        # Use loss_function from first task head if available, else fallback
        fallback_loss = (
            "DiceCE Loss (segmentation)" if exp_type == "segmentation" else "MSE Loss (label-free)"
        )
        loss_fn_for_params = (
            task_heads_config[0]["loss_function"] if task_heads_config else fallback_loss
        )

        params = {
            "experiment_type": exp_type,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "seed": seed,
            "data_path": data_path,
            "cache_dir": cache_dir,
            "source_col": source_col,
            "target_col": target_col,
            "spatial_dims": spatial_dims,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "patch_shape": patch_shape,
            "architecture": arch_key,
            "in_channels": in_channels,
            "dropout": dropout,
            "filters": filters,
            "res_block": res_block,
            "loss_function": loss_fn_for_params,
            "optimizer": optimizer_key,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "lr_scheduler": lr_scheduler_key,
            "max_epochs": max_epochs,
            "val_every_n": val_every_n,
            "save_images_every_n": save_images_every_n,
            "gradient_clip": gradient_clip,
            "num_gpus": num_gpus,
            "precision": precision,
            "run_test": run_test,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "lr_monitor": lr_monitor,
            "logger": logger_key,
            "resume_checkpoint": resume_checkpoint,
            "output_dir": output_dir,
            "log_dir": log_dir,
            "gpu_optimizations": gpu_optimizations,
            "matmul_precision": matmul_precision,
            "channels_last": channels_last,
            "torch_compile": torch_compile,
            "compile_mode": compile_mode,
            "gradient_checkpointing": gradient_checkpointing,
            "task_heads": task_heads_config,
        }

        if st.button("Generate YAML Config", type="primary", use_container_width=True):
            config = build_full_config(params)
            st.session_state.generated_config = config

        if st.session_state.generated_config:
            config_yaml = yaml.dump(
                st.session_state.generated_config, default_flow_style=False, sort_keys=False
            )

            st.markdown("#### Generated Configuration")
            st.code(config_yaml, language="yaml")

            st.download_button(
                "Download YAML",
                data=config_yaml,
                file_name=f"{experiment_name}_{run_name}.yaml",
                mime="text/yaml",
            )

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                if not st.session_state.training_running:
                    if st.button("Launch Training", type="primary", use_container_width=True):
                        config_dir = Path(output_dir) / "configs"
                        config_path = config_dir / "train_gui.yaml"
                        launch_training(st.session_state.generated_config, config_path)
                        st.toast("Training launched!")
                        st.rerun()
                else:
                    st.warning("Training is running...")

            with col2:
                if st.session_state.training_running:
                    if st.button("Stop Training", type="secondary", use_container_width=True):
                        stop_training()
                        st.info("Training stopped.")
                        st.rerun()

            if st.session_state.training_running or st.session_state.training_log:
                st.divider()

                # Check if process is still alive
                proc = st.session_state.training_process
                if proc and proc.poll() is not None and st.session_state.training_running:
                    st.session_state.training_running = False
                    if proc.returncode == 0:
                        st.success("Training completed successfully!")
                    else:
                        st.error(f"Training exited with code {proc.returncode}")

                # ----- Sub-tabs for monitoring -----
                mon_tab_chart, mon_tab_eval, mon_tab_log = st.tabs(
                    ["Loss Monitor", "Evaluate Checkpoint", "Raw Log"]
                )

                # ----- Loss Monitor -----
                with mon_tab_chart:
                    st.markdown("#### Live Loss Curves")

                    # Try CSV metrics first (most reliable)
                    csv_metrics = _load_csv_metrics(output_dir)
                    has_csv_chart = False

                    if csv_metrics is not None and not csv_metrics.empty:
                        try:
                            import pandas as pd

                            loss_cols = [c for c in csv_metrics.columns if "loss" in c.lower()]
                            epoch_col = (
                                "epoch"
                                if "epoch" in csv_metrics.columns
                                else "step" if "step" in csv_metrics.columns else None
                            )

                            if loss_cols and epoch_col:
                                # Group by epoch and get mean (CSVLogger logs per-step)
                                plot_df = csv_metrics[[epoch_col] + loss_cols].dropna(
                                    subset=loss_cols, how="all"
                                )

                                if not plot_df.empty:
                                    # Separate train and val losses for cleaner display
                                    train_cols = [c for c in loss_cols if "train" in c.lower()]
                                    val_cols = [c for c in loss_cols if "val" in c.lower()]
                                    other_cols = [
                                        c
                                        for c in loss_cols
                                        if c not in train_cols and c not in val_cols
                                    ]

                                    if train_cols or val_cols:
                                        chart_col1, chart_col2 = st.columns(2)
                                        with chart_col1:
                                            if train_cols:
                                                train_df = plot_df[
                                                    [epoch_col] + train_cols
                                                ].dropna()
                                                if not train_df.empty:
                                                    st.markdown("**Training Loss**")
                                                    st.line_chart(train_df.set_index(epoch_col))
                                        with chart_col2:
                                            if val_cols:
                                                val_df = plot_df[[epoch_col] + val_cols].dropna()
                                                if not val_df.empty:
                                                    st.markdown("**Validation Loss**")
                                                    st.line_chart(val_df.set_index(epoch_col))

                                    if other_cols:
                                        other_df = plot_df[[epoch_col] + other_cols].dropna()
                                        if not other_df.empty:
                                            st.line_chart(other_df.set_index(epoch_col))

                                    # Show current metrics
                                    last_row = csv_metrics.iloc[-1]
                                    metric_cols = st.columns(min(4, len(loss_cols)))
                                    for idx, col_name in enumerate(loss_cols[:4]):
                                        val = last_row.get(col_name)
                                        if pd.notna(val):
                                            with metric_cols[idx]:
                                                st.metric(col_name, f"{val:.6f}")

                                    has_csv_chart = True
                        except Exception as chart_err:
                            st.caption(f"Could not render CSV metrics chart: {chart_err}")

                    # Fallback: parse from log lines
                    if not has_csv_chart and st.session_state.training_log:
                        parsed = _parse_loss_from_logs(st.session_state.training_log)
                        train_losses = parsed["train_loss"]
                        val_losses = parsed["val_loss"]

                        if train_losses or val_losses:
                            try:
                                import pandas as pd

                                chart_col1, chart_col2 = st.columns(2)
                                with chart_col1:
                                    if train_losses:
                                        tdf = pd.DataFrame(
                                            train_losses, columns=["epoch", "train/loss"]
                                        )
                                        st.markdown("**Training Loss**")
                                        st.line_chart(tdf.set_index("epoch"))
                                        st.metric(
                                            "Latest train/loss", f"{train_losses[-1][1]:.6f}"
                                        )
                                with chart_col2:
                                    if val_losses:
                                        vdf = pd.DataFrame(
                                            val_losses, columns=["epoch", "val/loss"]
                                        )
                                        st.markdown("**Validation Loss**")
                                        st.line_chart(vdf.set_index("epoch"))
                                        st.metric("Latest val/loss", f"{val_losses[-1][1]:.6f}")
                            except ImportError:
                                st.warning("Install pandas for loss charts: `pip install pandas`")
                        else:
                            st.info(
                                "No loss values detected yet. Charts will appear as training progresses."
                            )
                    elif not has_csv_chart:
                        st.info(
                            "No loss data available yet. Start training and charts will appear here."
                        )

                    if st.session_state.training_running:
                        if st.button("Refresh Charts", key="refresh_charts"):
                            st.rerun()

                # ----- Evaluate Checkpoint -----
                with mon_tab_eval:
                    st.markdown("#### Mid-Training Evaluation")
                    st.caption(
                        "Run prediction using the current best or latest checkpoint while "
                        "training continues. View sample outputs to assess model quality."
                    )

                    # Find available checkpoints
                    best_ckpt = _find_best_checkpoint(output_dir)

                    if best_ckpt:
                        st.success(f"Checkpoint found: `{Path(best_ckpt).name}`")
                        st.text(f"Full path: {best_ckpt}")

                        # Show checkpoint metadata
                        ckpt_stat = Path(best_ckpt).stat()
                        ckpt_size_mb = ckpt_stat.st_size / (1024 * 1024)
                        ckpt_time = datetime.datetime.fromtimestamp(ckpt_stat.st_mtime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        st.markdown(
                            f"**Size:** {ckpt_size_mb:.1f} MB  |  **Modified:** {ckpt_time}"
                        )

                        # Also list all available checkpoints
                        all_ckpts = list(Path(output_dir).rglob("*.ckpt"))
                        if len(all_ckpts) > 1:
                            eval_ckpt_choice = st.selectbox(
                                "Select checkpoint to evaluate",
                                [
                                    str(c)
                                    for c in sorted(
                                        all_ckpts, key=lambda p: p.stat().st_mtime, reverse=True
                                    )
                                ],
                                format_func=lambda x: f"{Path(x).name} ({datetime.datetime.fromtimestamp(Path(x).stat().st_mtime).strftime('%H:%M:%S')})",
                                key="eval_ckpt_select",
                            )
                        else:
                            eval_ckpt_choice = best_ckpt

                        # Config path for evaluation
                        config_candidates = list(Path(output_dir).rglob("train_gui.yaml"))
                        eval_config = str(config_candidates[0]) if config_candidates else ""

                        if not eval_config:
                            st.warning(
                                "No training config found in output directory. Generate and launch training first."
                            )

                        # Launch evaluation
                        eval_proc = st.session_state.get("eval_process")
                        eval_running = st.session_state.get("eval_running", False)

                        # Check if eval finished
                        if eval_proc and eval_proc.poll() is not None and eval_running:
                            st.session_state.eval_running = False
                            if eval_proc.returncode == 0:
                                st.success("Evaluation completed!")
                            else:
                                st.error(f"Evaluation exited with code {eval_proc.returncode}")

                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            if not eval_running:
                                if st.button("Run Evaluation", type="primary", key="run_eval_btn"):
                                    if eval_config and eval_ckpt_choice:
                                        launch_evaluation(
                                            eval_config, eval_ckpt_choice, output_dir
                                        )
                                        st.toast("Evaluation launched!")
                                        st.rerun()
                                    else:
                                        st.error("Missing config or checkpoint path.")
                            else:
                                st.warning("Evaluation running...")
                        with col_e2:
                            if eval_running:
                                if st.button("Stop Evaluation", key="stop_eval_btn"):
                                    _kill_process_group(st.session_state.eval_process)
                                    st.session_state.eval_running = False
                                    st.rerun()

                        # Show evaluation log
                        if st.session_state.eval_log:
                            with st.expander("Evaluation Log", expanded=False):
                                st.code(
                                    "\n".join(st.session_state.eval_log[-100:]), language="log"
                                )

                        # Show prediction images
                        pred_images = _find_prediction_images(output_dir)
                        if pred_images:
                            st.divider()
                            st.markdown("#### Prediction Preview")
                            st.caption("Sample outputs from the evaluated checkpoint.")

                            # Display in a grid
                            n_cols = min(3, len(pred_images))
                            for row_start in range(0, min(9, len(pred_images)), n_cols):
                                cols = st.columns(n_cols)
                                for j in range(n_cols):
                                    idx = row_start + j
                                    if idx < len(pred_images):
                                        img_path = pred_images[idx]
                                        with cols[j]:
                                            # Try to load and display the image
                                            try:
                                                img_arr, err = _try_load_image_preview(
                                                    str(img_path)
                                                )
                                                if img_arr is not None:
                                                    st.image(
                                                        img_arr,
                                                        caption=img_path.name,
                                                        use_container_width=True,
                                                        clamp=True,
                                                    )
                                                else:
                                                    st.text(f"{img_path.name}\n({err})")
                                            except (OSError, ValueError):
                                                st.text(f"{img_path.name}\n(load error)")

                        if eval_running:
                            if st.button("Refresh Evaluation", key="refresh_eval"):
                                st.rerun()
                    else:
                        st.info(
                            "No checkpoints found yet. Checkpoints appear after the first "
                            "validation epoch completes."
                        )

                # ----- Raw Log -----
                with mon_tab_log:
                    st.markdown("#### Training Log")
                    if st.session_state.training_log:
                        log_text = "\n".join(st.session_state.training_log[-200:])
                        st.code(log_text, language="log")
                    else:
                        st.info("No log output yet.")

                    if st.session_state.training_running:
                        if st.button("Refresh Log", key="refresh_raw_log"):
                            st.rerun()

    # ------------------------------------------------------------------ #
    # TAB: TensorRT Export
    # ------------------------------------------------------------------ #
    with tab_trt:
        st.subheader("TensorRT Export")
        st.caption(
            "Convert a trained checkpoint to TensorRT for 2-5x faster inference. "
            "Optimized for Blackwell GPUs with FP16/INT8 precision."
        )

        trt_available = (PROJECT_ROOT / "scripts" / "export_to_tensorrt.py").exists()
        if not trt_available:
            st.error("TensorRT export script not found at `scripts/export_to_tensorrt.py`.")
        else:
            st.markdown("#### Model Selection")
            col1, col2 = st.columns(2)
            with col1:
                trt_config_path = st.text_input(
                    "Experiment config path",
                    value="",
                    help="Path to the YAML config used during training (or generated config).",
                    key="trt_config_path",
                )
            with col2:
                trt_ckpt_path = st.text_input(
                    "Checkpoint path (.ckpt)",
                    value="",
                    help="Path to the trained model checkpoint.",
                    key="trt_ckpt_path",
                )

            # Auto-detect checkpoints
            st.markdown("**Auto-detect checkpoints**")
            scan_dir = st.text_input(
                "Scan directory for checkpoints",
                value=str(PROJECT_ROOT / "logs"),
                key="trt_scan_dir",
            )
            if st.button("Scan for Checkpoints", key="trt_scan_btn"):
                scan_path = Path(scan_dir).resolve()
                if not scan_path.is_dir():
                    st.warning(f"Directory does not exist: {scan_dir}")
                    ckpt_files = []
                else:
                    ckpt_files = sorted(str(p) for p in scan_path.rglob("*.ckpt"))
                if ckpt_files:
                    st.session_state["found_checkpoints"] = ckpt_files
                else:
                    st.warning("No .ckpt files found.")
                    st.session_state["found_checkpoints"] = []

            found_ckpts = st.session_state.get("found_checkpoints", [])
            if found_ckpts:
                selected_ckpt = st.selectbox(
                    "Found checkpoints",
                    found_ckpts,
                    format_func=lambda x: f".../{'/'.join(Path(x).parts[-3:])}",
                    key="trt_found_ckpt_select",
                )
                if st.button("Use Selected Checkpoint", key="trt_use_ckpt"):
                    st.session_state["trt_ckpt_selected"] = selected_ckpt
                    st.rerun()

            # Use auto-selected checkpoint if available
            effective_ckpt = st.session_state.get("trt_ckpt_selected", trt_ckpt_path)
            if effective_ckpt and effective_ckpt != trt_ckpt_path:
                st.info(f"Using checkpoint: `{effective_ckpt}`")

            st.divider()
            st.markdown("#### Export Settings")

            col1, col2 = st.columns(2)
            with col1:
                trt_precision = st.selectbox(
                    "TensorRT precision", TRT_PRECISION_OPTIONS, key="trt_precision"
                )
                trt_workspace = st.number_input(
                    "Workspace size (GB)", min_value=1, max_value=32, value=4, key="trt_workspace"
                )
            with col2:
                trt_output = st.text_input(
                    "Output path (.ts)",
                    value=str(PROJECT_ROOT / "models" / "model_trt.ts"),
                    key="trt_output_path",
                )

            st.markdown("**Input shape** (batch, channels, ...spatial dims)")
            if spatial_dims == 3:
                ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                with ic1:
                    trt_b = st.number_input("Batch", min_value=1, value=1, key="trt_b")
                with ic2:
                    trt_c = st.number_input(
                        "Channels", min_value=1, value=in_channels, key="trt_c"
                    )
                with ic3:
                    trt_d = st.number_input("D", min_value=1, value=patch_shape[0], key="trt_d")
                with ic4:
                    trt_h = st.number_input("H", min_value=1, value=patch_shape[1], key="trt_h")
                with ic5:
                    trt_w = st.number_input("W", min_value=1, value=patch_shape[2], key="trt_w")
                trt_input_shape = [trt_b, trt_c, trt_d, trt_h, trt_w]
            else:
                ic1, ic2, ic3, ic4 = st.columns(4)
                with ic1:
                    trt_b = st.number_input("Batch", min_value=1, value=1, key="trt_b")
                with ic2:
                    trt_c = st.number_input(
                        "Channels", min_value=1, value=in_channels, key="trt_c"
                    )
                with ic3:
                    trt_h = st.number_input("H", min_value=1, value=patch_shape[0], key="trt_h")
                with ic4:
                    trt_w = st.number_input("W", min_value=1, value=patch_shape[1], key="trt_w")
                trt_input_shape = [trt_b, trt_c, trt_h, trt_w]

            st.divider()
            st.markdown("#### Dynamic Shapes")
            trt_dynamic = st.checkbox("Enable dynamic batch size", value=False, key="trt_dynamic")
            if trt_dynamic:
                dcol1, dcol2, dcol3 = st.columns(3)
                with dcol1:
                    trt_min_batch = st.number_input(
                        "Min batch", min_value=1, value=1, key="trt_min_batch"
                    )
                with dcol2:
                    trt_opt_batch = st.number_input(
                        "Optimal batch", min_value=1, value=4, key="trt_opt_batch"
                    )
                with dcol3:
                    trt_max_batch = st.number_input(
                        "Max batch", min_value=1, value=16, key="trt_max_batch"
                    )
            else:
                trt_min_batch = trt_opt_batch = trt_max_batch = 1

            st.divider()
            st.markdown("#### INT8 Calibration")
            if trt_precision == "int8":
                st.warning("INT8 requires calibration data for best accuracy.")
                trt_cal_data = st.text_input(
                    "Calibration data directory", value="", key="trt_cal_data"
                )
                trt_cal_samples = st.number_input(
                    "Calibration samples",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    key="trt_cal_samples",
                )
            else:
                trt_cal_data = ""
                trt_cal_samples = 100

            st.divider()
            st.markdown("#### Benchmarking")
            trt_benchmark = st.checkbox(
                "Run benchmark after export (PyTorch vs TensorRT)", value=True, key="trt_benchmark"
            )
            if trt_benchmark:
                trt_bench_iters = st.number_input(
                    "Benchmark iterations",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    key="trt_bench_iters",
                )
            else:
                trt_bench_iters = 100

            st.divider()

            # Launch export
            col1, col2 = st.columns(2)
            with col1:
                if not st.session_state.trt_running:
                    if st.button(
                        "Export to TensorRT",
                        type="primary",
                        use_container_width=True,
                        key="trt_export_btn",
                    ):
                        final_ckpt = effective_ckpt or trt_ckpt_path
                        final_config = trt_config_path

                        if not final_ckpt:
                            st.error("Please provide a checkpoint path.")
                        elif not final_config:
                            st.error("Please provide a config path.")
                        else:
                            # Ensure output directory exists
                            Path(trt_output).parent.mkdir(parents=True, exist_ok=True)

                            trt_params = {
                                "config_path": final_config,
                                "checkpoint_path": final_ckpt,
                                "output_path": trt_output,
                                "input_shape": trt_input_shape,
                                "precision": trt_precision,
                                "workspace_size": trt_workspace,
                                "dynamic_shapes": trt_dynamic,
                                "min_batch": trt_min_batch,
                                "opt_batch": trt_opt_batch,
                                "max_batch": trt_max_batch,
                                "calibration_data": trt_cal_data,
                                "num_calibration_samples": trt_cal_samples,
                                "benchmark": trt_benchmark,
                                "benchmark_iterations": trt_bench_iters,
                            }
                            launch_tensorrt_export(trt_params)
                            st.toast("TensorRT export launched!")
                            st.rerun()
                else:
                    st.warning("TensorRT export is running...")

            with col2:
                if st.session_state.trt_running:
                    if st.button(
                        "Stop Export",
                        type="secondary",
                        use_container_width=True,
                        key="trt_stop_btn",
                    ):
                        stop_tensorrt()
                        st.info("Export stopped.")
                        st.rerun()

            # Export log
            if st.session_state.trt_running or st.session_state.trt_log:
                st.divider()
                st.markdown("#### Export Log")

                trt_proc = st.session_state.trt_process
                if trt_proc and trt_proc.poll() is not None and st.session_state.trt_running:
                    st.session_state.trt_running = False
                    if trt_proc.returncode == 0:
                        st.success("TensorRT export completed successfully!")
                        st.markdown(f"Model saved to: `{trt_output}`")
                    else:
                        st.error(f"Export exited with code {trt_proc.returncode}")

                if st.session_state.trt_log:
                    trt_log_text = "\n".join(st.session_state.trt_log[-200:])
                    st.code(trt_log_text, language="log")

                if st.session_state.trt_running:
                    if st.button("Refresh Export Log", key="trt_refresh_log"):
                        st.rerun()

            # Usage instructions
            st.divider()
            st.markdown("#### Usage After Export")
            st.code(
                """# Load TensorRT model for inference
import torch
model = torch.jit.load("model_trt.ts")
output = model(torch.randn(1, 1, 16, 128, 128).cuda())

# Or use the TensorRTInferenceEngine wrapper
from cyto_dl.utils.tensorrt_utils import TensorRTInferenceEngine
engine = TensorRTInferenceEngine("model_trt.ts", input_shape=(1, 1, 16, 128, 128))
output = engine(input_tensor)""",
                language="python",
            )

    # ------------------------------------------------------------------ #
    # TAB: AutoKernel
    # ------------------------------------------------------------------ #
    with tab_autokernel:
        st.subheader("AutoKernel Optimization")
        st.caption(
            "Profile a trained model and let RightNow-AI's AutoKernel rewrite the "
            "hottest kernels in Triton or CUDA C++. Each experiment ≈ 90 s on a "
            "modern NVIDIA/AMD GPU."
        )

        # --- Setup card ----------------------------------------------------
        st.markdown("#### Setup")
        installed = autokernel_is_installed()
        setup_proc = st.session_state.autokernel_setup_process
        if setup_proc and setup_proc.poll() is not None and st.session_state.autokernel_setup_running:
            st.session_state.autokernel_setup_running = False
            installed = autokernel_is_installed()

        if installed:
            commit = autokernel_commit() or "unknown"
            st.success(f"AutoKernel ready at `vendor/autokernel/` (commit `{commit}`).")
            with st.expander("Reinstall / update AutoKernel"):
                if st.button(
                    "Re-run setup (git pull + uv sync)", key="autokernel_setup_rerun"
                ):
                    launch_autokernel_setup()
                    st.toast("AutoKernel setup re-launched.")
                    st.rerun()
        else:
            st.warning(
                "AutoKernel is not installed. Click below to clone "
                "`https://github.com/RightNow-AI/autokernel` into `vendor/autokernel/` "
                "and run `uv sync`. Requires `uv` and a CUDA/ROCm GPU."
            )
            if not st.session_state.autokernel_setup_running:
                if st.button(
                    "Install AutoKernel", type="primary", key="autokernel_setup_btn"
                ):
                    launch_autokernel_setup()
                    st.toast("AutoKernel setup launched.")
                    st.rerun()
            else:
                st.info("Setup running...")

        if st.session_state.autokernel_setup_log:
            with st.expander("Setup log", expanded=st.session_state.autokernel_setup_running):
                st.code(
                    "\n".join(st.session_state.autokernel_setup_log[-200:]),
                    language="log",
                )
                if st.session_state.autokernel_setup_running:
                    if st.button("Refresh setup log", key="autokernel_setup_refresh"):
                        st.rerun()

        st.divider()

        # --- Model source selection ---------------------------------------
        st.markdown("#### Model Source")
        ak_source_label = st.radio(
            "Where does the model come from?",
            ["cyto-dl class", "Lightning checkpoint (.ckpt)", "Cellpose 4 (.pt)"],
            horizontal=True,
            key="autokernel_source_radio",
        )
        ak_source_map = {
            "cyto-dl class": "cytodl_class",
            "Lightning checkpoint (.ckpt)": "lightning_ckpt",
            "Cellpose 4 (.pt)": "cellpose4",
        }
        ak_source = ak_source_map[ak_source_label]

        ak_module = ""
        ak_class_name = ""
        ak_ckpt = ""
        default_ckpt = _find_best_checkpoint(
            str(PROJECT_ROOT / "logs")
        ) or ""

        if ak_source == "cytodl_class":
            col1, col2 = st.columns(2)
            with col1:
                ak_module = st.text_input(
                    "Python module path",
                    value="cyto_dl.models.im2im.MultiTaskIm2Im",
                    help="Importable Python path to the nn.Module class.",
                    key="autokernel_module",
                )
            with col2:
                ak_class_name = st.text_input(
                    "Class name",
                    value="MultiTaskIm2Im",
                    help="Class within the module above.",
                    key="autokernel_classname",
                )
        elif ak_source == "lightning_ckpt":
            ak_ckpt = st.text_input(
                "Checkpoint path (.ckpt)",
                value=default_ckpt,
                help="Lightning checkpoint produced by cyto-dl training.",
                key="autokernel_ckpt_lightning",
            )
        else:  # cellpose4
            ak_ckpt = st.text_input(
                "Cellpose 4 weights path (.pt)",
                value="",
                help="Path to a Cellpose 4 fine-tuned PyTorch weights file.",
                key="autokernel_ckpt_cellpose",
            )
            st.caption(
                "Requires the `cellpose>=4.0` package "
                "(install with `pip install -e .[autokernel]`)."
            )

        st.markdown("**Input shape**")
        shape_cols = st.columns(5)
        with shape_cols[0]:
            ak_n = st.number_input("N (batch)", min_value=1, value=1, key="autokernel_n")
        with shape_cols[1]:
            ak_c = st.number_input("C (channels)", min_value=1, value=1, key="autokernel_c")
        is_2d = ak_source == "cellpose4"
        if is_2d:
            with shape_cols[2]:
                ak_h = st.number_input("H", min_value=1, value=256, key="autokernel_h2d")
            with shape_cols[3]:
                ak_w = st.number_input("W", min_value=1, value=256, key="autokernel_w2d")
            ak_input_shape = [ak_n, ak_c, ak_h, ak_w]
            with shape_cols[4]:
                st.caption("(2D: N C H W)")
        else:
            with shape_cols[2]:
                ak_z = st.number_input("Z", min_value=1, value=16, key="autokernel_z")
            with shape_cols[3]:
                ak_h = st.number_input("Y/H", min_value=1, value=128, key="autokernel_h3d")
            with shape_cols[4]:
                ak_w = st.number_input("X/W", min_value=1, value=128, key="autokernel_w3d")
            ak_input_shape = [ak_n, ak_c, ak_z, ak_h, ak_w]

        st.divider()

        # --- Tuning controls ----------------------------------------------
        st.markdown("#### Tuning")
        with st.expander("Optimization parameters", expanded=True):
            tcol1, tcol2, tcol3 = st.columns(3)
            with tcol1:
                ak_dtype = st.selectbox(
                    "dtype", ["float16", "float32"], index=0, key="autokernel_dtype"
                )
            with tcol2:
                ak_backend = st.radio(
                    "Backend", ["triton", "cuda"], horizontal=True, key="autokernel_backend"
                )
            with tcol3:
                ak_target = st.radio(
                    "Target metric",
                    ["throughput", "latency", "vram"],
                    horizontal=True,
                    key="autokernel_target",
                )

            tcol4, tcol5 = st.columns(2)
            with tcol4:
                ak_top_n = st.slider(
                    "Top-N kernels to optimize",
                    min_value=1,
                    max_value=20,
                    value=5,
                    key="autokernel_top_n",
                )
            with tcol5:
                ak_experiments = st.slider(
                    "Experiments per kernel",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    key="autokernel_experiments",
                )
            est_minutes = (ak_top_n * ak_experiments * 90) / 60
            st.caption(
                f"Estimated wall time: ≈ {est_minutes:.0f} min "
                f"({ak_top_n} kernels × {ak_experiments} experiments × ~90 s)"
            )

            ak_llm = st.text_input(
                "LLM model (optional)",
                value="",
                help="Pass-through to AutoKernel agent (e.g. claude-opus-4 / gpt-4o).",
                key="autokernel_llm_model",
            )
            ak_workspace = st.text_input(
                "Workspace directory",
                value=str(
                    PROJECT_ROOT
                    / "outputs"
                    / "autokernel"
                    / datetime.datetime.now().strftime("run_%Y%m%d_%H%M")
                ),
                key="autokernel_workspace",
            )

        st.divider()

        # --- Launch / Stop -------------------------------------------------
        st.markdown("#### Run")
        run_cols = st.columns(2)
        with run_cols[0]:
            if not st.session_state.autokernel_running:
                if st.button(
                    "Start AutoKernel optimization",
                    type="primary",
                    use_container_width=True,
                    disabled=not installed,
                    key="autokernel_start_btn",
                ):
                    valid = True
                    if ak_source == "cytodl_class" and not (ak_module and ak_class_name):
                        st.error("Please provide both module path and class name.")
                        valid = False
                    if ak_source in ("lightning_ckpt", "cellpose4") and not ak_ckpt:
                        st.error("Please provide a checkpoint/weights path.")
                        valid = False
                    if valid:
                        params = {
                            "source": ak_source,
                            "module": ak_module,
                            "class_name": ak_class_name,
                            "ckpt": ak_ckpt,
                            "input_shape": ak_input_shape,
                            "dtype": ak_dtype,
                            "backend": ak_backend,
                            "top_n": ak_top_n,
                            "experiments_per_kernel": ak_experiments,
                            "target_metric": ak_target,
                            "workspace": ak_workspace,
                            "llm_model": ak_llm,
                            "config_path": str(
                                PROJECT_ROOT / "configs" / "autokernel" / "default.yaml"
                            ),
                        }
                        launch_autokernel(params)
                        st.toast("AutoKernel launched!")
                        st.rerun()
            else:
                st.warning("AutoKernel is running...")
        with run_cols[1]:
            if st.session_state.autokernel_running:
                if st.button(
                    "Stop", type="secondary", use_container_width=True, key="autokernel_stop_btn"
                ):
                    stop_autokernel()
                    st.info("AutoKernel stopped.")
                    st.rerun()

        # --- Live log + completion handling -------------------------------
        if st.session_state.autokernel_running or st.session_state.autokernel_log:
            st.markdown("#### Run log")
            ak_proc = st.session_state.autokernel_process
            if ak_proc and ak_proc.poll() is not None and st.session_state.autokernel_running:
                st.session_state.autokernel_running = False
                st.session_state.autokernel_results = _parse_autokernel_results(
                    st.session_state.autokernel_log
                )
                if ak_proc.returncode == 0:
                    st.success("AutoKernel run finished.")
                else:
                    st.error(f"AutoKernel exited with code {ak_proc.returncode}")

            if st.session_state.autokernel_log:
                st.code(
                    "\n".join(st.session_state.autokernel_log[-200:]), language="log"
                )
            if st.session_state.autokernel_running:
                if st.button("Refresh log", key="autokernel_refresh_log"):
                    st.rerun()

        # --- Results panel -------------------------------------------------
        results_payload = st.session_state.autokernel_results or _parse_autokernel_results(
            st.session_state.autokernel_log or []
        )
        if results_payload and results_payload.get("results"):
            st.divider()
            st.markdown("#### Results")
            try:
                import pandas as _pd

                df = _pd.DataFrame(results_payload["results"])
                st.dataframe(df, use_container_width=True)
            except ImportError:
                st.json(results_payload["results"])

            tsv_path = Path(results_payload.get("workspace", "")) / "results.tsv"
            if tsv_path.is_file():
                with tsv_path.open("rb") as f:
                    st.download_button(
                        "Download results.tsv",
                        f.read(),
                        file_name=tsv_path.name,
                        mime="text/tab-separated-values",
                        key="autokernel_download_tsv",
                    )

        # --- Footer help ---------------------------------------------------
        st.divider()
        st.markdown("#### Notes")
        st.markdown(
            "- AutoKernel writes optimized kernels to "
            "`vendor/autokernel/workspace/` and logs each experiment in "
            "`results.tsv`.\n"
            "- Use the `Cellpose 4 (.pt)` source to optimize fine-tuned cpsam "
            "checkpoints (requires the optional `cellpose` dependency).\n"
            "- Long runs are CPU-cheap but GPU-busy; stop early if you hit a "
            "speedup ceiling."
        )


if __name__ == "__main__":
    main()
