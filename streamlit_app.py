"""CytoDL Training GUI - Streamlit application for configuring and launching im2im model training.

Replaces manual YAML editing with an interactive GUI for brightfield-to-nuclei/cell prediction
model training. Optimized for CUDA 13 / Blackwell GPU workstations.
"""

import datetime
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st
import yaml

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
    "DiceCE Loss (segmentation)": {"target": "monai.losses.DiceCELoss", "kwargs": {"sigmoid": True}},
    "Dice Loss": {"target": "monai.losses.DiceLoss", "kwargs": {"sigmoid": True}},
    "MSE Loss (label-free)": {"target": "torch.nn.MSELoss", "kwargs": {}},
    "L1 Loss": {"target": "torch.nn.L1Loss", "kwargs": {}},
    "Focal Loss": {"target": "monai.losses.FocalLoss", "kwargs": {}},
    "Tversky Loss": {"target": "monai.losses.TverskyLoss", "kwargs": {"sigmoid": True}},
}

OPTIMIZERS = {
    "AdamW (fused, recommended)": {"target": "torch.optim.AdamW", "fused": True},
    "Adam (fused)": {"target": "torch.optim.Adam", "fused": True},
    "AdamW": {"target": "torch.optim.AdamW", "fused": False},
    "Adam": {"target": "torch.optim.Adam", "fused": False},
    "SGD": {"target": "torch.optim.SGD", "fused": False},
}

LR_SCHEDULERS = {
    "ExponentialLR": {"target": "torch.optim.lr_scheduler.ExponentialLR", "params": {"gamma": 0.995}},
    "CosineAnnealingLR": {
        "target": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "params": {"T_max": 100},
    },
    "ReduceLROnPlateau": {
        "target": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "params": {"factor": 0.5, "patience": 10},
    },
    "StepLR": {"target": "torch.optim.lr_scheduler.StepLR", "params": {"step_size": 30, "gamma": 0.1}},
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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


def build_full_config(params: dict) -> dict:
    """Construct a complete Hydra-compatible config dict from GUI parameters."""
    exp_type = params["experiment_type"]

    # ---- loss ----
    loss_info = LOSS_FUNCTIONS[params["loss_function"]]
    loss_cfg = {"_target_": loss_info["target"]}
    loss_cfg.update(loss_info["kwargs"])

    # ---- postprocess ----
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

    # ---- model ----
    model_cfg = {
        "_target_": "cyto_dl.models.im2im.MultiTaskIm2Im",
        "save_images_every_n_epochs": params["save_images_every_n"],
        "save_dir": "${paths.output_dir}",
        "x_key": params["source_col"],
        "backbone": backbone,
        "task_heads": {
            params["target_col"]: {
                "_target_": "cyto_dl.nn.BaseHead",
                "loss": loss_cfg,
                "postprocess": postprocess,
            },
        },
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

    # ---- saving callback ----
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
    data_cfg = {
        "_target_": "cyto_dl.datamodules.dataframe.DataframeDatamodule",
        "path": params["data_path"],
        "cache_dir": params.get("cache_dir") or None,
        "num_workers": params["num_workers"],
        "batch_size": params["batch_size"],
        "pin_memory": True,
        "persistent_workers": True,
        "split_column": None,
        "columns": [params["source_col"], params["target_col"]],
    }

    # ---- paths ----
    paths_cfg = {
        "root_dir": str(PROJECT_ROOT),
        "data_dir": str(Path(params["data_path"]).parent) if params["data_path"] else str(PROJECT_ROOT / "data"),
        "log_dir": params.get("log_dir") or str(PROJECT_ROOT / "logs"),
        "output_dir": params.get("output_dir") or str(PROJECT_ROOT / "logs" / params["experiment_name"] / params["run_name"]),
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
        "target_col": params["target_col"],
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
# Training launcher
# ---------------------------------------------------------------------------
def _stream_output(process, log_list):
    """Read process stdout line-by-line into the session log list."""
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                log_list.append(line.rstrip("\n"))
        for line in iter(process.stderr.readline, ""):
            if line:
                log_list.append(f"[STDERR] {line.rstrip()}")
    except (ValueError, OSError):
        pass


def launch_training(config: dict, config_path: Path):
    """Write config YAML and launch training subprocess."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
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

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=env,
        preexec_fn=os.setsid,
    )

    st.session_state.training_process = process
    st.session_state.training_log = []
    st.session_state.training_running = True

    t = threading.Thread(target=_stream_output, args=(process, st.session_state.training_log), daemon=True)
    t.start()
    st.session_state.log_thread = t


def stop_training():
    proc = st.session_state.training_process
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    st.session_state.training_running = False


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="CytoDL Training GUI", page_icon="🔬", layout="wide")
    _init_state()

    st.title("CytoDL Im2Im Training")
    st.caption("Configure and launch model training without editing YAML files. Optimized for CUDA 13 / Blackwell GPUs.")

    # ====================================================================
    # Sidebar — quick overview
    # ====================================================================
    with st.sidebar:
        st.header("Quick Start")
        st.markdown(
            "1. Choose experiment type\n"
            "2. Set data paths & columns\n"
            "3. Configure model & training\n"
            "4. Enable GPU optimizations\n"
            "5. Generate config & launch"
        )
        st.divider()
        st.markdown("**Project root:** `{}`".format(PROJECT_ROOT))

    # ====================================================================
    # Tabs
    # ====================================================================
    tab_exp, tab_data, tab_model, tab_train, tab_gpu, tab_launch = st.tabs(
        ["Experiment", "Data", "Model", "Training", "GPU / Perf", "Launch"]
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
            run_name = st.text_input("Run name", value=datetime.datetime.now().strftime("run_%Y%m%d_%H%M"))
            seed = st.number_input("Random seed (0 = none)", min_value=0, max_value=999999, value=12345)

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

        # DynUNet-specific
        if arch_key == "DynUNet (MONAI)":
            st.markdown("**Encoder filter sizes** (each level doubles receptive field)")
            filter_str = st.text_input("Filters (comma-separated)", value="32, 64, 128, 256")
            filters = [int(x.strip()) for x in filter_str.split(",") if x.strip()]
            res_block = st.checkbox("Use residual blocks", value=True)
        elif arch_key == "SwinUNETR (MONAI)":
            feature_size = st.number_input("Feature size", min_value=12, max_value=96, value=48, step=12)
            filters = [feature_size]
            res_block = False
        elif arch_key == "AttentionUNet (MONAI)":
            filter_str = st.text_input("Channel sizes (comma-separated)", value="32, 64, 128, 256")
            filters = [int(x.strip()) for x in filter_str.split(",") if x.strip()]
            res_block = False
        else:
            filters = [16, 32, 64]
            res_block = False

        st.divider()
        st.subheader("Loss Function")
        default_loss_idx = 0 if exp_type == "segmentation" else 2
        loss_fn = st.selectbox("Loss function", list(LOSS_FUNCTIONS.keys()), index=default_loss_idx)

        st.divider()
        st.subheader("Optimizer")
        col1, col2 = st.columns(2)
        with col1:
            optimizer_key = st.selectbox("Optimizer", list(OPTIMIZERS.keys()))
        with col2:
            learning_rate = st.number_input("Learning rate", min_value=1e-7, max_value=1.0, value=1e-4, format="%.1e")

        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-4, format="%.1e")

        st.divider()
        st.subheader("LR Scheduler")
        lr_scheduler_key = st.selectbox("Scheduler", list(LR_SCHEDULERS.keys()))

    # ------------------------------------------------------------------ #
    # TAB: Training
    # ------------------------------------------------------------------ #
    with tab_train:
        st.subheader("Training Parameters")

        col1, col2 = st.columns(2)
        with col1:
            max_epochs = st.number_input("Max epochs", min_value=1, max_value=10000, value=100)
            val_every_n = st.number_input("Validate every N epochs", min_value=1, max_value=100, value=1)
            save_images_every_n = st.number_input("Save images every N epochs", min_value=1, max_value=100, value=5)
        with col2:
            gradient_clip = st.number_input("Gradient clip (0=off)", min_value=0.0, max_value=100.0, value=1.0)
            num_gpus = st.number_input("Number of GPUs", min_value=1, max_value=8, value=1)
            run_test = st.checkbox("Run test after training", value=True)

        st.divider()
        st.subheader("Callbacks")
        col1, col2 = st.columns(2)
        with col1:
            early_stopping = st.checkbox("Early stopping", value=True)
            if early_stopping:
                early_stopping_patience = st.number_input("Patience (epochs)", min_value=1, max_value=1000, value=50)
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
                matmul_precision = st.selectbox("Matrix multiply precision", ["high", "medium", "highest"], index=0)
                channels_last = st.checkbox("Channels-last memory format", value=True, help="20-30% speedup on modern GPUs")
            with col2:
                torch_compile = st.checkbox("torch.compile", value=True, help="JIT compile model for faster execution")
                compile_mode = st.selectbox(
                    "Compile mode",
                    ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    index=0,
                    help="'max-autotune' is best for training (longer compile, best perf)",
                )
                gradient_checkpointing = st.checkbox(
                    "Gradient checkpointing", value=False, help="Trade compute for memory — enable if OOM"
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
            "loss_function": loss_fn,
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
        }

        # Generate button
        if st.button("Generate YAML Config", type="primary", use_container_width=True):
            config = build_full_config(params)
            st.session_state.generated_config = config

        if st.session_state.generated_config:
            config_yaml = yaml.dump(st.session_state.generated_config, default_flow_style=False, sort_keys=False)

            st.markdown("#### Generated Configuration")
            st.code(config_yaml, language="yaml")

            # Download button
            st.download_button(
                "Download YAML",
                data=config_yaml,
                file_name=f"{experiment_name}_{run_name}.yaml",
                mime="text/yaml",
            )

            st.divider()

            # ---- Launch controls ----
            col1, col2 = st.columns(2)
            with col1:
                if not st.session_state.training_running:
                    if st.button("Launch Training", type="primary", use_container_width=True):
                        config_dir = Path(output_dir) / "configs"
                        config_path = config_dir / "train_gui.yaml"
                        launch_training(st.session_state.generated_config, config_path)
                        st.success("Training launched!")
                        st.rerun()
                else:
                    st.warning("Training is running...")

            with col2:
                if st.session_state.training_running:
                    if st.button("Stop Training", type="secondary", use_container_width=True):
                        stop_training()
                        st.info("Training stopped.")
                        st.rerun()

            # ---- Training log ----
            if st.session_state.training_running or st.session_state.training_log:
                st.divider()
                st.markdown("#### Training Log")

                # Check if process is still alive
                proc = st.session_state.training_process
                if proc and proc.poll() is not None and st.session_state.training_running:
                    st.session_state.training_running = False
                    if proc.returncode == 0:
                        st.success("Training completed successfully!")
                    else:
                        st.error(f"Training exited with code {proc.returncode}")

                # Show log
                if st.session_state.training_log:
                    log_text = "\n".join(st.session_state.training_log[-200:])
                    st.code(log_text, language="log")

                if st.session_state.training_running:
                    if st.button("Refresh Log"):
                        st.rerun()


if __name__ == "__main__":
    main()
