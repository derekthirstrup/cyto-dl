# /ar:loop - Autonomous Optimization Loop

Run the autoresearch optimization loop continuously at a specified interval.

## Usage

```
/ar:loop              # Default: every 10 minutes
/ar:loop 5m           # Every 5 minutes
/ar:loop 30m          # Every 30 minutes
/ar:loop 1h           # Every hour
/ar:loop stop         # Stop the loop
```

## What It Does

Sets up a recurring `/ar:run` execution at the specified interval.
Each iteration autonomously:

1. Reads experiment history
2. Selects the next hyperparameter to try
3. Trains the model
4. Evaluates and logs results
5. Waits for the next interval

## Safety Limits

- **Max consecutive crashes**: 5 (then auto-stop)
- **Auto-expiry**: 3 days (then auto-stop)
- **Max iterations per session**: 100

## Monitoring

While the loop is running:
- `/ar:status` to check progress
- `/ar:loop stop` to stop the loop

## Implementation

This skill uses the `/loop` built-in skill to schedule recurring `/ar:run` calls.

```
/loop <interval> /ar:run
```

## Notes

- Each iteration takes ~5-30 minutes depending on training tier
- With OAuth max plan, you have ample Claude budget for long loops
- Results are always logged to `autoresearch/results.tsv`
- If the model stops improving, the agent will escalate to more radical changes
