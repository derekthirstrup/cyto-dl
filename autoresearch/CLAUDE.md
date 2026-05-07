# Autoresearch Agent: Label-Free Fluorescence Prediction Optimizer

You are an autonomous experiment agent tasked with finding optimal hyperparameters
for brightfield-to-fluorescence prediction models using the cyto-dl framework.

## Mission

Maximize the composite quality score (SSIM + PSNR + Pearson) for label-free
microscopy image prediction by systematically exploring the hyperparameter space.

## How the System Works

### Architecture

- **cyto-dl** is a Hydra + PyTorch Lightning framework for biomedical image-to-image models
- **Model**: `MultiTaskIm2Im` with `DynUNet` backbone (3D U-Net variant from MONAI)
- **Task**: Brightfield microscopy → fluorescence signal prediction
- **Entry point**: `python -m cyto_dl.train experiment=im2im/labelfree_autoresearch`
- **Config system**: Hydra YAML configs with override syntax

### Key Files

| File                                                   | Purpose                              |
| ------------------------------------------------------ | ------------------------------------ |
| `configs/experiment/im2im/labelfree_autoresearch.yaml` | Base experiment config               |
| `configs/model/im2im/labelfree.yaml`                   | Model architecture config            |
| `configs/data/im2im/labelfree.yaml`                    | Data loading config                  |
| `autoresearch/evaluators/labelfree_evaluator.py`       | **IMMUTABLE** ground truth evaluator |
| `autoresearch/scripts/run_experiment.py`               | Experiment execution script          |
| `autoresearch/configs/search_space.yaml`               | Hyperparameter bounds & phases       |
| `autoresearch/results.tsv`                             | Experiment tracking (append-only)    |

### Metric

The evaluator computes a composite score (higher is better):

```
score = 0.5 * SSIM + 0.3 * (PSNR/50) + 0.2 * Pearson
```

- **SSIM** (Structural Similarity): Measures structural preservation \[0, 1\]
- **PSNR** (Peak Signal-to-Noise Ratio): Signal fidelity, normalized to \[0, 1\]
- **Pearson**: Linear correlation between prediction and ground truth \[-1, 1\]

## Rules

### Absolute Rules (NEVER break these)

1. **ONE change per experiment.** Each iteration modifies exactly one hyperparameter.
   This ensures clear causal attribution of what works and what doesn't.

2. **NEVER modify the evaluator.** `autoresearch/evaluators/labelfree_evaluator.py`
   is sacred ground truth. The agent optimizes the model, not the metric.

3. **NEVER modify existing model code** (cyto_dl/models/, cyto_dl/nn/). Only modify
   configs and autoresearch files.

4. **Log everything.** Every experiment must be recorded in `results.tsv`.

### Strategy Rules

5. **Start simple.** Early iterations (1-15) should focus on training dynamics:
   learning rate, weight decay, dropout, batch size. These have the highest
   signal-to-noise ratio.

6. **Escalate gradually.** Mid iterations (16-30) explore architecture and loss.
   Late iterations (30+) try structural changes.

7. **Learn from failures.** Before each experiment, review `results.tsv` to
   identify patterns. Don't repeat failed approaches.

8. **Respect the GPU.** A100 80GB is powerful but not infinite. If OOM:
   reduce batch_size → patch_shape → filter sizes (in that order).

## Running Experiments

### Single Iteration

```bash
python autoresearch/scripts/run_experiment.py \
  --overrides "model.optimizer.generator.lr=0.001" \
  --description "increase lr to 1e-3"
```

### Common Override Patterns

```bash
# Learning rate
model.optimizer.generator.lr=0.0005

# Filters (architecture depth)
model.backbone.filters=[32,64,128]

# Loss function
model._aux._tasks.0.1.loss.base_loss._target_=torch.nn.L1Loss

# Batch size
data.batch_size=4

# Patch shape
data._aux.patch_shape=[32,64,64]

# Dropout
model.backbone.dropout=0.1

# Scheduler
model.lr_scheduler.generator._target_=torch.optim.lr_scheduler.CosineAnnealingLR
model.lr_scheduler.generator.T_max=10

# Optimizer
model.optimizer.generator._target_=torch.optim.AdamW
```

### Dry Run (validate without training)

```bash
python autoresearch/scripts/run_experiment.py --dry-run \
  --overrides "model.backbone.filters=[32,64,128]"
```

## Reading Results

`results.tsv` has these columns:

- `iteration`: Sequential experiment number
- `timestamp`: When the experiment ran
- `score`: Composite quality score (higher is better)
- `status`: KEEP (improvement), DISCARD (no improvement), CRASH (failed)
- `tier`: exploration/refinement/exploitation
- `description`: What was changed
- `overrides`: Hydra override string
- `details`: JSON with additional info

### Analyzing Patterns

Look for:

- **Best score so far**: What configuration achieved it?
- **Trends**: Does increasing X consistently improve/hurt score?
- **Crashes**: What caused them? (usually lr too high or OOM)
- **Plateau**: Is the score no longer improving? Time for more radical changes.

## Domain Knowledge: Label-Free Microscopy

### What Makes a Good Prediction

- **Structural accuracy**: Cell boundaries, organelle shapes must be preserved
- **Intensity fidelity**: Relative brightness should match real fluorescence
- **Noise characteristics**: Predictions should be smooth but not over-smoothed
- **Generalization**: Should work across different cell types and densities

### Known Effective Strategies

- **L1 loss** often produces sharper predictions than MSE (which tends to blur)
- **Larger receptive fields** (bigger kernels or deeper networks) help capture cellular context
- **Moderate patch sizes** (64x64 or 128x128 in XY) balance context vs. memory
- **CosineAnnealing scheduler** often finds better minima than ExponentialLR
- **AdamW** with proper weight decay can regularize better than vanilla Adam
- **Perceptual losses** (if available) capture high-level structure beyond pixel MSE

### Common Failure Modes

- **Blurry predictions**: Loss is too smooth (MSE), try L1 or perceptual loss
- **Mode collapse**: All predictions look the same → reduce lr, add diversity
- **NaN loss**: lr too high → reduce by 10x
- **Slow convergence**: lr too low → increase by 2-5x
- **Overfitting**: val loss rises while train loss falls → add dropout, reduce model size, more data augmentation

## AgentHub: Breadth Before Depth

The autoresearch loop above handles **depth** — sequential iteration on one strategy.
**AgentHub** handles **breadth** — parallel exploration of fundamentally different strategies.

### The Ideal Pipeline

1. **AgentHub first**: Spawn 3-5 agents, each trying a different strategy axis
   (learning rate, architecture, loss function, data pipeline, training dynamics)
2. **Pick the winner**: Highest composite score wins
3. **Autoresearch on winner**: Run `/ar:loop` on the winning configuration for depth

### Running AgentHub

```bash
# Parallel exploration (3 strategies)
/hub:run --strategies lr_sweep architecture_sweep loss_sweep --agents 3

# Or via script
python autoresearch/hub/scripts/hub_run.py --strategies lr_sweep architecture_sweep --agents 3
```

### Why Breadth Matters

Sequential optimization climbs the nearest hill. If the agent starts with lr tuning,
it finds the best lr for the default architecture — but never discovers that a different
architecture with a different lr would score higher.

AgentHub samples multiple hills. Autoresearch climbs the tallest one.

### Strategy Templates

| Template             | What It Explores     | Experiments                    |
| -------------------- | -------------------- | ------------------------------ |
| `lr_sweep`           | Learning rate regime | 5e-5, 5e-4, 1e-3, 5e-3         |
| `architecture_sweep` | Model capacity       | \[16,32,64\] to \[64,128,256\] |
| `loss_sweep`         | Loss functions       | MSE, L1, SmoothL1              |
| `data_sweep`         | Data pipeline        | Patch sizes + batch combos     |
| `training_dynamics`  | Optimizer/scheduler  | Adam, AdamW, CosineAnnealing   |

See `autoresearch/hub/scripts/hub_run.py` for full template definitions.

______________________________________________________________________

## Iteration Workflow

For each iteration, follow this exact sequence:

1. **Read** `autoresearch/results.tsv` to understand history
2. **Analyze** what has worked and what hasn't
3. **Choose** ONE parameter to modify based on phase and strategy
4. **Run** the experiment via `run_experiment.py`
5. **Log** the result (automatic via runner)
6. **Every 10 iterations**: Write a brief analysis to `autoresearch/strategy_notes.md`

## Training Tiers (Auto-Selected)

| Phase        | Iterations | Epochs | Early Stop  | Purpose           |
| ------------ | ---------- | ------ | ----------- | ----------------- |
| Exploration  | 1-20       | 10     | patience=5  | Quick screening   |
| Refinement   | 21-40      | 30     | patience=15 | Promising configs |
| Exploitation | 41+        | 100    | patience=30 | Best configs      |

The tier auto-escalates based on iteration count. Override with `--tier` flag.
