# /ar:setup - Initialize Autoresearch Experiment

Set up the autoresearch system for label-free fluorescence prediction optimization.

## Usage

```
/ar:setup --data-path /path/to/labelfree/data
```

## What It Does

1. Validates the environment (GPU, dependencies, configs)
2. Validates the training data at the specified path
3. Initializes `autoresearch/results.tsv` for experiment tracking
4. Runs a baseline training (10 epochs) to establish initial metrics
5. Reports baseline score and readiness

## Steps

When invoked, run:

```bash
python autoresearch/scripts/setup_experiment.py --data-path <DATA_PATH>
```

If `--data-path` is not provided, ask the user for their data path.

After setup completes, report:
- Baseline composite score
- Individual metrics (SSIM, PSNR, Pearson)
- GPU detected
- Next steps: "Run `/ar:run` for a single iteration or `/ar:loop 10m` for autonomous optimization"

## Prerequisites

- cyto-dl installed (`pip install -e .`)
- CUDA GPU available
- Training data accessible
