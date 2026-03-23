# /hub:run - Parallel Strategy Exploration

Spawn multiple agents to explore different optimization strategies simultaneously.

## Usage

```
/hub:run                                    # Default: 3 agents, lr+arch+loss sweep
/hub:run --agents 5                         # 5 parallel agents
/hub:run --strategies lr_sweep loss_sweep    # Specific strategy templates
/hub:run --pattern ensemble                 # Non-competing (different subtasks)
```

## What It Does

1. Selects N fundamentally different optimization strategies
2. Spawns N agents, each in an isolated git worktree
3. Each agent trains a model with its assigned strategy
4. Evaluates all results using the immutable evaluator
5. Picks the winner (highest composite score)
6. Merges the winner's changes, archives the rest as git tags

## Strategy Templates

- `lr_sweep` - Learning rate exploration (5e-5 to 5e-3)
- `architecture_sweep` - Model capacity (small to large U-Net)
- `loss_sweep` - Loss function variants (MSE, L1, SmoothL1)
- `data_sweep` - Patch sizes and batch configurations
- `training_dynamics` - Optimizer/scheduler combinations

## Coordination Patterns

- **fan-out** (default): Same task, different strategies. Agents compete. Best wins.
- **ensemble**: Different subtasks. All results merge together.

## Implementation

Uses Claude Code's Agent tool with `isolation: "worktree"` for true parallel
execution in isolated repository copies.

Alternatively, runs sequentially via:
```bash
python autoresearch/hub/scripts/hub_run.py \
  --strategies lr_sweep architecture_sweep loss_sweep \
  --agents 3
```

## After the Run

- Winner is announced with score and configuration
- Winner's branch can be merged: `git merge <winner-branch>`
- Then run `/ar:loop` on the winner for depth optimization
- Results logged to `autoresearch/hub/hub_results.tsv`
