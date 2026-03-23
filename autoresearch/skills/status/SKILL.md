# /ar:status - Autoresearch Dashboard

Show the current state of the autoresearch optimization.

## Usage

```
/ar:status
/ar:status --full
```

## What It Shows

### Summary
- Total iterations run
- Current best score and its configuration
- Last 5 experiment results
- Current phase (early/mid/late)
- Active loop status

### Detailed (`--full`)
- Full results table
- Score trend (improving/plateauing/declining)
- Per-metric breakdown (SSIM, PSNR, Pearson)
- Crash analysis
- Recommended next steps

## Implementation

When invoked, read and analyze `autoresearch/results.tsv`:

```bash
# Show results summary
cat autoresearch/results.tsv

# Count iterations
wc -l autoresearch/results.tsv
```

Then format a dashboard like:

```
=== Autoresearch Status ===
Iterations:  23
Best Score:  0.7234 (iteration 18)
Best Config: lr=0.0005, filters=[32,64,128], loss=L1
Phase:       mid (iterations 16-30)
Trend:       improving (+0.05 over last 5 runs)

Last 5 Results:
  #23  0.6891  DISCARD  dropout=0.15
  #22  0.7012  DISCARD  patch=[32,64,64]
  #21  0.7234  KEEP     filters=[32,64,128]
  #20  0.6543  DISCARD  AdamW optimizer
  #19  0.7102  KEEP     lr=0.0005
```
