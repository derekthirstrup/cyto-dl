#!/usr/bin/env python3
"""AgentHub: Parallel strategy exploration for label-free model optimization.

Spawns N agents in isolated git worktrees, each exploring a different
optimization strategy. Evaluates all results and merges the winner.

This is the "breadth" component. Combined with autoresearch (depth),
it covers the optimization space:
  1. AgentHub explores different strategies in parallel
  2. Autoresearch iterates on the winning strategy

Usage:
    python hub_run.py --task "optimize labelfree prediction" --agents 3
    python hub_run.py --strategies lr_sweep architecture_sweep loss_sweep --agents 3
    python hub_run.py --pattern ensemble --agents 3  # non-competing, different subtasks
"""

import argparse
import datetime
import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HUB_RESULTS = PROJECT_ROOT / "autoresearch" / "hub" / "hub_results.tsv"
WORKTREE_BASE = PROJECT_ROOT / ".worktrees"

# Pre-defined strategy templates for label-free optimization
STRATEGY_TEMPLATES = {
    "lr_sweep": {
        "description": "Learning rate and optimizer exploration",
        "experiments": [
            {"override": "model.optimizer.generator.lr=0.00005", "desc": "lr=5e-5"},
            {"override": "model.optimizer.generator.lr=0.0005", "desc": "lr=5e-4"},
            {"override": "model.optimizer.generator.lr=0.001", "desc": "lr=1e-3"},
            {"override": "model.optimizer.generator.lr=0.005", "desc": "lr=5e-3"},
        ],
    },
    "architecture_sweep": {
        "description": "Model architecture exploration",
        "experiments": [
            {"override": "model.backbone.filters=[16,32,64]", "desc": "small-unet"},
            {"override": "model.backbone.filters=[32,64,128]", "desc": "medium-unet"},
            {"override": "model.backbone.filters=[32,64,128,256]", "desc": "large-unet"},
            {"override": "model.backbone.filters=[64,128,256]", "desc": "wide-unet"},
        ],
    },
    "loss_sweep": {
        "description": "Loss function exploration",
        "experiments": [
            {
                "override": "model._aux._tasks.0.1.loss.base_loss._target_=torch.nn.MSELoss",
                "desc": "mse-loss",
            },
            {
                "override": "model._aux._tasks.0.1.loss.base_loss._target_=torch.nn.L1Loss",
                "desc": "l1-loss",
            },
            {
                "override": "model._aux._tasks.0.1.loss.base_loss._target_=torch.nn.SmoothL1Loss",
                "desc": "smooth-l1-loss",
            },
        ],
    },
    "data_sweep": {
        "description": "Data configuration exploration",
        "experiments": [
            {
                "override": "data._aux.patch_shape=[16,32,32] data.batch_size=4",
                "desc": "small-patch-big-batch",
            },
            {
                "override": "data._aux.patch_shape=[16,64,64] data.batch_size=2",
                "desc": "medium-patch",
            },
            {
                "override": "data._aux.patch_shape=[32,64,64] data.batch_size=1",
                "desc": "large-patch",
            },
            {
                "override": "data._aux.patch_shape=[16,128,128] data.batch_size=1",
                "desc": "wide-patch",
            },
        ],
    },
    "training_dynamics": {
        "description": "Optimizer and scheduler exploration",
        "experiments": [
            {
                "override": "model.optimizer.generator._target_=torch.optim.Adam model.lr_scheduler.generator._target_=torch.optim.lr_scheduler.ExponentialLR model.lr_scheduler.generator.gamma=0.995",
                "desc": "adam-exponential",
            },
            {
                "override": "model.optimizer.generator._target_=torch.optim.AdamW model.optimizer.generator.weight_decay=0.01",
                "desc": "adamw-decay",
            },
            {
                "override": "model.lr_scheduler.generator._target_=torch.optim.lr_scheduler.CosineAnnealingLR model.lr_scheduler.generator.T_max=10",
                "desc": "cosine-annealing",
            },
        ],
    },
}


def create_worktree(agent_id, branch_name):
    """Create an isolated git worktree for an agent."""
    worktree_path = WORKTREE_BASE / f"hub-{agent_id}"
    WORKTREE_BASE.mkdir(parents=True, exist_ok=True)

    # Create worktree
    subprocess.run(  # nosec B603, B607
        ["git", "worktree", "add", str(worktree_path), "-b", branch_name],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
    )
    return worktree_path


def cleanup_worktree(worktree_path, branch_name):
    """Remove a git worktree after use."""
    subprocess.run(  # nosec B603, B607
        ["git", "worktree", "remove", str(worktree_path), "--force"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
    )
    # Tag the branch for archival
    subprocess.run(  # nosec B603, B607
        ["git", "tag", f"hub-archive/{branch_name}", branch_name],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
    )


def run_agent_experiment(worktree_path, overrides, description, data_path=None, timeout=120):
    """Run a single experiment in an isolated worktree."""
    runner = worktree_path / "autoresearch" / "scripts" / "run_experiment.py"

    cmd = [
        sys.executable,
        str(runner),
        "--overrides",
        overrides,
        "--description",
        description,
        "--tier",
        "exploration",
    ]
    if data_path:
        cmd.extend(["--data-path", data_path])

    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout * 60,
            cwd=str(worktree_path),
        )

        # Parse score from output
        score = 0.0
        for line in result.stdout.split("\n"):
            if line.startswith("AUTORESEARCH_SCORE="):
                score = float(line.split("=")[1])
                break

        return {
            "score": score,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:],
            "stderr": result.stderr[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "returncode": -1, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"score": 0.0, "returncode": -2, "stdout": "", "stderr": str(e)}


def fan_out_fan_in(strategies, data_path=None, timeout=120):
    """Run multiple strategies in parallel (fan-out), evaluate and pick winner (fan-in).

    Note: In Claude Code, true parallelism comes from the Agent tool's
    isolation="worktree" parameter. This script provides sequential fallback
    and result aggregation.
    """
    results = []
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, strategy in enumerate(strategies):
        agent_id = f"{session_id}-agent{i}"
        branch_name = f"hub/{session_id}/agent-{i}-{strategy['desc']}"

        print(f"\n=== Agent {i+1}/{len(strategies)}: {strategy['desc']} ===")

        # Create isolated worktree
        worktree_path = create_worktree(agent_id, branch_name)

        # Run experiment
        result = run_agent_experiment(
            worktree_path, strategy["override"], strategy["desc"], data_path, timeout
        )
        result["agent_id"] = agent_id
        result["strategy"] = strategy
        result["branch"] = branch_name
        result["worktree"] = str(worktree_path)
        results.append(result)

        print(f"  Score: {result['score']:.6f}")
        print(f"  Status: {'OK' if result['returncode'] == 0 else 'FAILED'}")

    # Pick winner
    valid_results = [r for r in results if r["returncode"] == 0 and r["score"] > 0]
    if not valid_results:
        print("\nAll agents failed. No winner.")
        return None, results

    winner = max(valid_results, key=lambda r: r["score"])
    print(f"\n=== WINNER: Agent {winner['strategy']['desc']} ===")
    print(f"Score: {winner['score']:.6f}")
    print(f"Branch: {winner['branch']}")

    # Log results
    log_hub_results(session_id, results, winner)

    # Cleanup non-winning worktrees (tag them for archival)
    for result in results:
        if result != winner:
            cleanup_worktree(Path(result["worktree"]), result["branch"])

    return winner, results


def log_hub_results(session_id, results, winner):
    """Log hub run results."""
    write_header = not HUB_RESULTS.exists()
    HUB_RESULTS.parent.mkdir(parents=True, exist_ok=True)

    with open(HUB_RESULTS, "a") as f:
        if write_header:
            f.write("session\tagent_id\tscore\tstatus\tstrategy\tbranch\tis_winner\n")
        for result in results:
            is_winner = result == winner
            status = "OK" if result["returncode"] == 0 else "FAILED"
            f.write(
                f"{session_id}\t{result['agent_id']}\t{result['score']:.6f}\t"
                f"{status}\t{result['strategy']['desc']}\t{result['branch']}\t"
                f"{is_winner}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="AgentHub: Parallel strategy exploration")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGY_TEMPLATES.keys()),
        help="Named strategy templates to run",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of agents (selects top N experiments from strategy)",
    )
    parser.add_argument(
        "--pattern",
        choices=["fan-out", "ensemble"],
        default="fan-out",
        help="Coordination pattern",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to training data",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per agent in minutes",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategy templates",
    )
    args = parser.parse_args()

    if args.list_strategies:
        print("Available strategy templates:")
        for name, template in STRATEGY_TEMPLATES.items():
            print(f"\n  {name}: {template['description']}")
            for exp in template["experiments"]:
                print(f"    - {exp['desc']}: {exp['override']}")
        sys.exit(0)

    if not args.strategies:
        # Default: run lr_sweep, architecture_sweep, loss_sweep
        args.strategies = ["lr_sweep", "architecture_sweep", "loss_sweep"]

    # Collect experiments from all requested strategies
    all_experiments = []
    for strategy_name in args.strategies:
        template = STRATEGY_TEMPLATES[strategy_name]
        all_experiments.extend(template["experiments"])

    # Limit to N agents
    experiments = all_experiments[: args.agents]

    print(f"=== AgentHub: {len(experiments)} agents, pattern={args.pattern} ===")
    for i, exp in enumerate(experiments):
        print(f"  Agent {i+1}: {exp['desc']} ({exp['override']})")

    winner, results = fan_out_fan_in(experiments, args.data_path, args.timeout)

    if winner:
        print(f"\nWinner branch: {winner['branch']}")
        print(f"To merge: git merge {winner['branch']}")
        print("To iterate on winner: /ar:loop (in the winning worktree)")
    else:
        print("\nNo winner. Review logs and try different strategies.")


if __name__ == "__main__":
    main()
