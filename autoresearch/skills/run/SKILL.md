# /ar:run - Run Single Autoresearch Iteration

Execute one experiment iteration in the autoresearch optimization loop.

## Usage

```
/ar:run
/ar:run "try L1 loss"
/ar:run --override "model.optimizer.generator.lr=0.001"
```

## What It Does

Follows the autoresearch 6-step cycle:

1. **Assess** - Read `autoresearch/results.tsv` and `autoresearch/configs/search_space.yaml`
2. **Analyze** - Identify what has worked, what hasn't, and which phase we're in
3. **Choose** - Select ONE hyperparameter change based on strategy and phase
4. **Execute** - Run training via `autoresearch/scripts/run_experiment.py`
5. **Evaluate** - Score is computed by the immutable evaluator
6. **Report** - Log result and report to user

## Decision Logic

Read `autoresearch/CLAUDE.md` for the full decision framework.

### Quick Reference

- **Iterations 1-15 (early)**: lr, weight_decay, dropout, batch_size
- **Iterations 16-30 (mid)**: optimizer, scheduler, loss, filters, patch_shape
- **Iterations 30+ (late)**: res_block, kernel_size, strides, zoom

### Key Constraint

**ONE change per experiment.** This ensures clear causal attribution.

## After Each Run

Report:

- Iteration number
- What was changed and why
- Score achieved
- Status (KEEP/DISCARD/CRASH)
- Current best score
- Suggestion for next iteration

## Error Handling

- **NaN loss**: Reduce learning rate by 10x
- **OOM**: Reduce batch_size, then patch_shape, then filters
- **5 consecutive crashes**: STOP and report to user
