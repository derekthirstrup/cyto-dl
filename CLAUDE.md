# CytoDL - AI Assistant Guide

## Project Overview

**CytoDL** is a unified deep learning framework for understanding 2D and 3D biological data (images, point clouds, and tabular data), developed by the Allen Institute for Cell Science. The project aims to enable deep learning approaches for cellular structure analysis, including image-to-image transformations and representation learning.

**Core Technologies:**
- PyTorch 2.0+ and PyTorch Lightning 2.0+
- Hydra 1.3 for hierarchical configuration management
- MONAI for medical/biological image processing
- Based on [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

**Version:** 0.6.1
**Python Support:** 3.9, 3.10, 3.11
**Build System:** PDM with pyproject.toml

---

## Repository Structure

```
cyto-dl/
├── cyto_dl/                  # Main package directory
│   ├── api/                  # Public API (CytoDLModel)
│   ├── callbacks/            # Training callbacks (image saver, layer freeze, etc.)
│   ├── datamodules/          # Data loading modules
│   │   ├── dataframe/        # DataFrame-based datamodules
│   │   ├── array.py          # In-memory array support
│   │   ├── multidim_image.py # Multi-dimensional image handling
│   │   └── ...
│   ├── dataframe/            # DataFrame utilities and transforms
│   ├── image/                # Image I/O and transforms
│   │   ├── io/               # Readers (bioio, ome-zarr, numpy, etc.)
│   │   └── transforms/       # Image augmentation and preprocessing
│   ├── models/               # Model implementations
│   │   ├── im2im/            # Image-to-image (segmentation, GANs, diffusion)
│   │   ├── vae/              # Variational autoencoders
│   │   ├── jepa/             # Joint Embedding Predictive Architecture
│   │   ├── contrastive/      # Contrastive learning (VICReg, etc.)
│   │   ├── classification/   # Classification models
│   │   └── handlers/         # Data handlers
│   ├── nn/                   # Neural network components
│   │   ├── vits/             # Vision Transformers
│   │   ├── discriminators/   # GAN discriminators
│   │   ├── head/             # Task heads
│   │   ├── losses/           # Loss functions
│   │   └── ...
│   ├── point_cloud/          # Point cloud utilities
│   ├── utils/                # Utilities (config, checkpoint, logging, etc.)
│   ├── train.py              # Training entry point
│   └── eval.py               # Evaluation/prediction entry point
├── configs/                  # Hydra configuration files
│   ├── experiment/           # Experiment configurations
│   │   ├── im2im/            # Image-to-image experiments
│   │   ├── classification/   # Classification experiments
│   │   └── contrastive/      # Contrastive learning experiments
│   ├── data/                 # Data configuration
│   ├── model/                # Model architecture configuration
│   ├── callbacks/            # Callback configuration
│   ├── trainer/              # PyTorch Lightning trainer config
│   ├── logger/               # Logger configuration
│   ├── train.yaml            # Default training config
│   └── eval.yaml             # Default evaluation config
├── tests/                    # Test suite
├── scripts/                  # Utility scripts
├── notebooks/                # Jupyter notebooks
├── data/                     # Example data directory
├── docs/                     # Documentation
├── requirements/             # Requirements files
├── pyproject.toml            # Project metadata and dependencies
├── Makefile                  # Development commands
└── .pre-commit-config.yaml   # Pre-commit hooks

```

---

## Key Concepts and Architecture

### 1. Configuration System (Hydra)

CytoDL uses Hydra for hierarchical configuration management with composition and overrides.

**Configuration Hierarchy:**
- `train.yaml` / `eval.yaml` - Top-level configs
- `experiment/` - Complete experiment configs that override defaults
- `data/`, `model/`, `callbacks/`, `trainer/`, `logger/` - Component configs

**Key Patterns:**
- **Variable Interpolation:** `${paths.data_dir}` references other config values
- **_aux Section:** Store reference values for interpolation without instantiation
- **Defaults:** Each config file can specify defaults to inherit from
- **Overrides:** Command-line overrides: `python cyto_dl/train.py trainer.max_epochs=20`

**Example Config Structure:**
```yaml
defaults:
  - override /data: im2im/segmentation.yaml
  - override /model: im2im/segmentation.yaml

experiment_name: my_experiment
run_name: my_run
source_col: raw  # Input column in CSV
target_col: seg  # Target column in CSV

data:
  path: ${paths.data_dir}/my_data
  cache_dir: ${paths.data_dir}/cache
  batch_size: 4
  _aux:
    patch_shape: [16, 32, 32]  # Reference value
```

### 2. Model Architecture Pattern

**Two-Part Structure:**
1. **Backbone:** Processes input images and produces n-channel output
2. **Task Heads:** Transform backbone output to task-specific outputs

**Model Config Components:**
- `backbone`: The main feature extractor (UNet, ViT, etc.)
- `tasks`: Dictionary of task heads, each with:
  - `head`: Transformation layer (conv, projection, identity, etc.)
  - `loss`: Loss function for this task
- `optimizer`: Optimizer configuration
- `lr_scheduler`: Learning rate scheduler
- `postprocessing`: How to save/visualize outputs

**Example:**
```yaml
model:
  backbone:
    _target_: monai.networks.nets.UNet
    spatial_dims: 3
    in_channels: 1
    out_channels: 16
  tasks:
    seg:
      head:
        _target_: cyto_dl.nn.head.BasicHead
      loss:
        _target_: monai.losses.DiceLoss
```

### 3. Data Loading

**Primary System:** DataFrame-based datamodules that wrap MONAI's PersistentDataset

**Key Config Parameters:**
- `path`: CSV file(s) with columns for file paths, or folder with train/val/test CSVs
- `cache_dir`: Location for cached preprocessed images
- `split_column`: Column name for train/val/test splits
- `transforms`: Sequence of MONAI/custom transforms
- Standard DataLoader args: `batch_size`, `num_workers`, `pin_memory`

**Data Input Methods:**
1. **CSV-based:** Paths in CSV files (most common)
2. **In-memory arrays:** Direct numpy arrays via `array.py` datamodule
3. **Folder-based:** Organized directory structure

**Transform Pipeline:**
Transforms are applied sequentially and can include:
- Loading (BioioLoader, NumpyReader, etc.)
- Preprocessing (normalization, resizing, padding)
- Augmentation (rotation, flipping, intensity adjustments)
- Task-specific transforms (binarization for segmentation, etc.)

### 4. Experiment Types

**Image-to-Image (im2im):**
- `segmentation`: Semantic/instance segmentation
- `gan`: Generative adversarial networks
- `labelfree`: Label-free prediction
- `mae`: Masked autoencoder pre-training
- `ijepa` / `iwm`: JEPA-based self-supervised learning
- `diffae`: Diffusion autoencoder

**Representation Learning:**
- `vae/*`: Various VAE architectures
- `contrastive/*`: Contrastive learning (VICReg, etc.)

**Classification:**
- `classification/*`: Classification tasks

---

## Development Workflow

### Initial Setup

```bash
# Clone repository
git clone https://github.com/AllenCellModeling/cyto-dl
cd cyto-dl

# Create conda environment (optional but recommended)
conda create -n cyto-dl python=3.10 fortran-compiler blas-devel
conda activate cyto-dl

# Install dependencies
pip install --no-deps -r requirements/requirements.txt

# Install package in editable mode
pip install -e .

# Optional: Install extras
pip install --no-deps -r requirements/equiv-requirements.txt  # Equivariance
```

### Pre-commit Hooks

**CRITICAL:** Always run pre-commit hooks before committing:

```bash
# Install hooks
pre-commit install

# Run on all files
make format
# OR
pre-commit run -a
```

**Hooks Include:**
- **black** - Code formatting (line length: 99)
- **isort** - Import sorting (black profile)
- **flake8** - Linting (ignores: E203, E402, E501, F401, F841)
- **bandit** - Security linting
- **docformatter** - Docstring formatting
- **pyupgrade** - Syntax upgrades (Python 3.8+)
- **prettier** - YAML formatting
- **mdformat** - Markdown formatting
- **codespell** - Spell checking
- **nbstripout** - Jupyter notebook output clearing
- **refurb** - Additional Python linting

### Testing

```bash
# Run fast tests only
make test
pytest -k "not slow"

# Run all tests
make test-full
pytest

# Run specific test file
pytest tests/test_train.py

# Run with coverage
pytest --cov=cyto_dl
```

**Test Markers:**
- `@pytest.mark.slow` - Marks slow tests (skipped by default)

### Common Make Commands

```bash
make help           # Show all available commands
make clean          # Clean build artifacts and caches
make clean-logs     # Clean log files
make format         # Run pre-commit hooks
make test           # Run fast tests
make test-full      # Run all tests
make gen-docs       # Generate Sphinx documentation
make sync-reqs-files # Sync requirements files from uv.lock
```

### Training Models

**Via CLI:**
```bash
# GPU training
python cyto_dl/train.py experiment=im2im/segmentation trainer=gpu

# CPU training
python cyto_dl/train.py experiment=im2im/segmentation trainer=cpu

# With overrides
python cyto_dl/train.py experiment=im2im/segmentation \
    trainer.max_epochs=100 \
    data.batch_size=8 \
    trainer=gpu
```

**Via API:**
```python
from cyto_dl.api import CytoDLModel

model = CytoDLModel()
model.download_example_data()  # Optional: get example data
model.load_default_experiment(
    "segmentation",
    output_dir="./output",
    overrides=["trainer=gpu"]
)
model.print_config()
model.train()

# For in-memory data
data = {
    "train": [{"raw": np.random.randn(1, 40, 256, 256),
               "seg": np.ones((1, 40, 256, 256))}],
    "val": [{"raw": np.random.randn(1, 40, 256, 256),
             "seg": np.ones((1, 40, 256, 256))}],
}
model.train(data=data)
```

### Prediction/Evaluation

**Via CLI:**
```bash
python cyto_dl/eval.py \
    experiment=im2im/segmentation \
    checkpoint.ckpt_path=/path/to/checkpoint.ckpt
```

**Via API:**
```python
model = CytoDLModel()
model.load_default_experiment(
    "segmentation",
    output_dir="./output",
    train=False,
    overrides=["checkpoint.ckpt_path=/path/to/checkpoint.ckpt"]
)
predictions = model.predict()

# For in-memory prediction
data = [np.random.rand(1, 32, 64, 64), np.random.rand(1, 32, 64, 64)]
_, _, output = model.predict(data=data)
```

---

## Code Style and Conventions

### Python Code Style

- **Formatter:** Black with line length 99
- **Import Sorting:** isort with black profile
- **Linting:** flake8
- **Docstrings:** Google-style, wrapped at 99 characters
- **Type Hints:** Encouraged but not required
- **Python Version:** Minimum 3.9, target 3.10

### File Naming

- **Python modules:** `snake_case.py`
- **Classes:** `PascalCase`
- **Functions/methods:** `snake_case()`
- **Constants:** `UPPER_SNAKE_CASE`
- **Config files:** `snake_case.yaml`

### Import Organization

Following isort/black:
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import os
from pathlib import Path

import torch
import numpy as np
from lightning import LightningModule

from cyto_dl.models import BaseModel
from cyto_dl.utils import get_pylogger
```

### Configuration Naming

- **Experiment names:** Descriptive, lowercase with underscores (e.g., `segmentation_superres`)
- **Column names:** Use `source_col` and `target_col` in experiments
- **Spatial dimensions:** Specify as `spatial_dims: 2` or `spatial_dims: 3`

### Hydra Config Patterns

**Always include `# @package _global_` at the top of experiment configs**

```yaml
# @package _global_
defaults:
  - override /data: im2im/segmentation.yaml
  - override /model: im2im/segmentation.yaml
```

**Use _aux for shared variables:**
```yaml
data:
  _aux:
    patch_shape: [16, 32, 32]
  transforms:
    - _target_: cyto_dl.image.transforms.RandomCrop
      patch_shape: ${data._aux.patch_shape}
```

---

## Common Patterns and Best Practices

### 1. Adding a New Model

1. **Create model class** in appropriate `cyto_dl/models/` subdirectory
2. **Inherit from** `BaseModel` or appropriate parent
3. **Implement required methods:**
   - `__init__()` - Setup architecture
   - `forward()` - Forward pass
   - `training_step()` / `validation_step()` - Training logic
4. **Create config** in `configs/model/`
5. **Create experiment config** in `configs/experiment/`
6. **Add tests** in `tests/`

### 2. Adding New Transforms

1. **Create transform class** in `cyto_dl/image/transforms/` or `cyto_dl/dataframe/transforms/`
2. **Follow MONAI transform patterns** (callable with `__call__()`)
3. **Make configurable** via Hydra (`_target_` instantiation)
4. **Document parameters** clearly
5. **Add to** `__init__.py` for easy imports

### 3. Adding New Experiments

1. **Start from existing experiment** in `configs/experiment/`
2. **Override necessary components:**
   - Data config
   - Model config
   - Trainer settings (epochs, etc.)
3. **Set experiment metadata:**
   - `experiment_name`
   - `run_name`
   - `tags`
4. **Test with small data** first
5. **Document in docstring** or comments

### 4. Working with Callbacks

Callbacks handle cross-cutting concerns during training:

**Common Callbacks:**
- `ImageSaver` - Save images during training/validation
- `ModelCheckpoint` - Save best/latest checkpoints
- `EarlyStopping` - Stop training when metric plateaus
- `LayerFreeze` - Freeze/unfreeze layers during training

**Adding Callbacks:**
```yaml
callbacks:
  my_callback:
    _target_: cyto_dl.callbacks.MyCallback
    param1: value1
```

### 5. Logging and Experiment Tracking

**Supported Loggers:**
- CSV Logger (default)
- MLflow (recommended for experiments)
- TensorBoard
- Weights & Biases (via Lightning)

**MLflow Integration:**
Configure in experiment:
```yaml
logger:
  mlflow:
    _target_: cyto_dl.loggers.MLFlowLogger
    experiment_name: ${experiment_name}
    run_name: ${run_name}
```

### 6. Checkpointing

**Loading Checkpoints:**
```yaml
checkpoint:
  ckpt_path: /path/to/checkpoint.ckpt
  weights_only: false  # Load full training state
  strict: true  # Require exact architecture match
```

**From API:**
```python
model.override_config({"checkpoint.ckpt_path": "/path/to/checkpoint.ckpt"})
```

---

## Important Notes for AI Assistants

### When Making Code Changes

1. **Always read files before modifying them** - Never assume structure
2. **Run pre-commit hooks** after changes: `make format`
3. **Add/update tests** for new functionality
4. **Follow existing patterns** in the codebase
5. **Check that imports resolve** correctly
6. **Consider both 2D and 3D** cases when working with images
7. **Respect the Hydra config system** - changes may require config updates

### When Adding Dependencies

1. **Add to pyproject.toml** under appropriate section:
   - Core deps in `dependencies`
   - Optional deps in `[project.optional-dependencies]`
2. **Update requirements files:** `make sync-reqs-files`
3. **Test that installation works** with new deps
4. **Consider compatibility** with PyTorch/Lightning versions

### When Working with Configs

1. **Don't break variable interpolation** - maintain `${}` references
2. **Keep _aux sections** for shared variables
3. **Test config composition** with different override combinations
4. **Use `_partial_: true`** for partial instantiation when needed
5. **Document custom resolvers** if adding new ones

### When Debugging

1. **Enable debug mode:** `python cyto_dl/train.py debug=default`
2. **Check config resolution:** `model.print_config()` in API
3. **Use smaller data** for faster iteration
4. **Check Lightning logs** in `logs/` directory
5. **Verify data loading** separately from model training

### Common Gotchas

1. **GPU vs CPU:** Many tests/examples use CPU by default
2. **3D vs 2D:** Spatial dimensions affect many components
3. **Patch shapes:** Must be compatible with backbone architecture
4. **Cache directory:** Can cause issues if not cleaned between runs
5. **Config paths:** Relative to config directory, not working directory
6. **MONAI transforms:** Order matters in transform pipelines
7. **OmegaConf:** Configs are not regular dicts, use `OmegaConf` utilities

### Testing Strategy

1. **Unit tests:** Test individual components in isolation
2. **Integration tests:** Test configs and training pipelines
3. **Mark slow tests:** Use `@pytest.mark.slow` decorator
4. **Use small data:** Tests should run quickly
5. **Test both 2D and 3D** when applicable
6. **Mock heavy dependencies** when possible

### Git Workflow

1. **Create feature branches** from main
2. **Make focused commits** with clear messages
3. **Run pre-commit hooks** before committing
4. **Run tests locally** before pushing
5. **Follow PR template** in `.github/PULL_REQUEST_TEMPLATE.md`
6. **Ensure CI passes** before merging

### Pull Request Checklist

From `.github/PULL_REQUEST_TEMPLATE.md`:
- [ ] Title is self-explanatory
- [ ] Description concisely explains the PR
- [ ] PR does only one thing (not bundling changes)
- [ ] List all breaking changes
- [ ] Test locally with `pytest`
- [ ] Run pre-commit hooks with `pre-commit run -a`

---

## Environment Variables

- `PROJECT_ROOT` - Automatically set by pyrootutils
- `CYTODL_CONFIG_PATH` - Override default config directory (defaults to `../configs`)

---

## Useful Resources

- **Lightning-Hydra Template:** https://github.com/ashleve/lightning-hydra-template
- **Hydra Documentation:** https://hydra.cc/
- **PyTorch Lightning:** https://lightning.ai/docs/pytorch/stable/
- **MONAI Transforms:** https://docs.monai.io/en/stable/transforms.html
- **MONAI Losses:** https://docs.monai.io/en/stable/losses.html

---

## Troubleshooting

### Installation Issues

**Problem:** `lie_learn` build failure
**Solution:** Install dependencies first with `uv sync`, then `uv sync --extra equiv`

**Problem:** CUDA errors
**Solution:** Check NVIDIA driver version, use CPU mode if needed: `trainer=cpu`

### Training Issues

**Problem:** Out of memory
**Solution:** Reduce batch size, patch size, or model size

**Problem:** Config resolution errors
**Solution:** Check variable interpolation paths, ensure all referenced keys exist

**Problem:** Transform errors
**Solution:** Verify transform order, check input/output shapes, validate data types

### Data Loading Issues

**Problem:** Cache directory full
**Solution:** Clear cache or use temporary directory (handled automatically unless `persist_cache=true`)

**Problem:** CSV not found
**Solution:** Check paths are relative to `paths.data_dir`, verify file exists

---

## Version History

Current Version: **0.6.1**

For version updates, see:
- `pyproject.toml` - `version` field
- `version.toml` - Version tracking
- `cyto_dl/__init__.py` - `__version__` variable

Version bumping is managed by `bumpver` (see `[tool.bumpver]` in pyproject.toml)

---

*Last Updated: 2025-11-22*
*This file is maintained for AI assistants working with the CytoDL codebase.*
