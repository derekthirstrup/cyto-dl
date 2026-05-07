# /hub:status - AgentHub Dashboard

Show the state of parallel strategy exploration runs.

## Usage

```
/hub:status
```

## What It Shows

- All hub sessions (past and active)
- Per-session: agents, strategies, scores, winner
- Comparison across sessions
- Active worktrees

## Implementation

Read and display `autoresearch/hub/hub_results.tsv`.
