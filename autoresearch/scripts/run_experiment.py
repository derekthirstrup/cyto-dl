#!/usr/bin/env python3
"""Run a single autoresearch experiment iteration.

Takes a config override specification, trains a model, evaluates it,
and logs the result to results.tsv.

Usage:
    # Run with Hydra overrides
    python run_experiment.py \
        --overrides "model.optimizer.generator.lr=0.001" \
        --description "lr=1e-3"

    # Dry run (print command without executing)
    python run_experiment.py --dry-run \
        --overrides "model.backbone.filters=[32,64,128]"

    # Specify training tier
    python run_experiment.py --tier exploration \
        --overrides "model.optimizer.generator.lr=0.0005"
"""

import argparse
import datetime
import json
import os
import subprocess  # nosec B404
import sys
from contextlib import suppress
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_FILE = PROJECT_ROOT / "autoresearch" / "results.tsv"
EXPERIMENT_CONFIG = "experiment=im2im/labelfree_autoresearch"

# Training tiers
TIERS = {
    "exploration": {
        "max_epochs": 10,
        "patience": 5,
        "description": "Quick screening (10 epochs)",
    },
    "refinement": {
        "max_epochs": 30,
        "patience": 15,
        "description": "Moderate training (30 epochs)",
    },
    "exploitation": {
        "max_epochs": 100,
        "patience": 30,
        "description": "Full training (100 epochs)",
    },
}


def get_tier_for_iteration(iteration):
    """Auto-select training tier based on iteration number."""
    if iteration <= 20:
        return "exploration"
    if iteration <= 40:
        return "refinement"
    return "exploitation"


def get_current_iteration():
    """Read results.tsv to determine current iteration number."""
    if not RESULTS_FILE.exists():
        return 1
    with RESULTS_FILE.open() as f:
        lines = [
            line for line in f.readlines() if line.strip() and not line.startswith("iteration")
        ]
    return len(lines) + 1


def build_command(overrides, tier, run_name=None, data_path=None):
    """Build the cyto-dl training command with Hydra overrides."""
    tier_config = TIERS[tier]

    cmd = [
        sys.executable,
        "-m",
        "cyto_dl.train",
        EXPERIMENT_CONFIG,
        f"trainer.max_epochs={tier_config['max_epochs']}",
        f"callbacks.early_stopping.patience={tier_config['patience']}",
    ]

    if run_name:
        cmd.append(f"run_name={run_name}")

    if data_path:
        cmd.append(f"data.path={data_path}")

    # Add user-specified overrides
    if overrides:
        for override in overrides.split():
            cmd.append(override)

    return cmd


def log_result(iteration, score, status, description, overrides, tier, details=None):
    """Append result to results.tsv."""
    write_header = not RESULTS_FILE.exists()

    with RESULTS_FILE.open("a") as f:
        if write_header:
            f.write("iteration\ttimestamp\tscore\tstatus\ttier\tdescription\toverrides\tdetails\n")

        timestamp = datetime.datetime.now().isoformat()
        details_str = json.dumps(details) if details else ""
        f.write(
            f"{iteration}\t{timestamp}\t{score:.6f}\t{status}\t{tier}\t"
            f"{description}\t{overrides}\t{details_str}\n"
        )


def run_training(cmd, timeout_minutes=120):
    """Execute the training command and capture output."""
    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", str(e)


def find_output_dir(run_name=None):
    """Find the training output directory for ``run_name`` if given, else the most recent one.

    Looking up by run_name avoids returning a stale run when other training jobs are running
    concurrently (or when the prior run touched its directory after the current one started).
    """
    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        return None

    run_dirs = []
    for task_dir in logs_dir.iterdir():
        if not task_dir.is_dir():
            continue

        runs_dir = task_dir / "runs"
        if runs_dir.exists():
            for exp_dir in runs_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                for run_dir in exp_dir.iterdir():
                    if run_dir.is_dir():
                        run_dirs.append(run_dir)

        multiruns_dir = task_dir / "multiruns"
        if multiruns_dir.exists():
            # multiruns/<date>/<exp>/<run>; descend two levels deeper than runs/
            for date_dir in multiruns_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                for exp_dir in date_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue
                    for run_dir in exp_dir.iterdir():
                        if run_dir.is_dir():
                            run_dirs.append(run_dir)

    if not run_dirs:
        return None

    if run_name:
        matches = [d for d in run_dirs if d.name == run_name or run_name in d.parts]
        if matches:
            return max(matches, key=lambda d: d.stat().st_mtime)

    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Run a single autoresearch experiment")
    parser.add_argument(
        "--overrides",
        "-o",
        default="",
        help="Hydra override string (space-separated key=value pairs)",
    )
    parser.add_argument(
        "--description",
        "-d",
        default="",
        help="Human-readable description of this experiment",
    )
    parser.add_argument(
        "--tier",
        "-t",
        choices=list(TIERS.keys()),
        default=None,
        help="Training tier (default: auto-select based on iteration)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in minutes (default: 120)",
    )
    args = parser.parse_args()

    iteration = get_current_iteration()
    tier = args.tier or get_tier_for_iteration(iteration)
    run_name = f"iter_{iteration:04d}"

    print(f"=== Autoresearch Iteration {iteration} ===")
    print(f"Tier: {tier} ({TIERS[tier]['description']})")
    print(f"Description: {args.description}")
    print(f"Overrides: {args.overrides}")

    cmd = build_command(args.overrides, tier, run_name, args.data_path)

    if args.dry_run:
        print("\nDry run - command would be:")
        print(f"  {' '.join(cmd)}")
        sys.exit(0)

    print("\nRunning training...")
    returncode, stdout, stderr = run_training(cmd, args.timeout)

    if returncode != 0:
        status = "CRASH"
        score = 0.0
        details = {"returncode": returncode, "stderr_tail": stderr[-500:] if stderr else ""}
        print(f"\nTraining FAILED (exit code {returncode})")
        if stderr:
            print(f"Error: {stderr[-300:]}")
        log_result(iteration, score, status, args.description, args.overrides, tier, details)
        sys.exit(1)

    # Find output directory and evaluate (bind to this invocation's run_name).
    output_dir = find_output_dir(run_name=run_name)
    if output_dir is None:
        print("Could not find training output directory")
        log_result(
            iteration,
            0.0,
            "CRASH",
            args.description,
            args.overrides,
            tier,
            {"error": "no output dir found"},
        )
        sys.exit(1)

    print(f"Output directory: {output_dir}")

    # Run evaluator
    evaluator_path = PROJECT_ROOT / "autoresearch" / "evaluators" / "labelfree_evaluator.py"
    eval_returncode = None
    try:
        eval_result = subprocess.run(  # nosec B603
            [sys.executable, str(evaluator_path), str(output_dir), "--verbose"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        eval_returncode = eval_result.returncode
        if eval_result.returncode != 0:
            score = 0.0
            status = "CRASH"
            details_str = eval_result.stderr or eval_result.stdout
        else:
            score = float(eval_result.stdout.strip())
            status = "OK"
            details_str = eval_result.stderr  # verbose output goes to stderr
    except Exception as e:
        score = 0.0
        status = "CRASH"
        details_str = str(e)

    # Decide KEEP vs DISCARD only when the evaluator actually succeeded;
    # never overwrite a CRASH with one of those.
    if status != "CRASH":
        best_score = get_best_score()
        if score > best_score:
            status = "KEEP"
            print(f"\nNEW BEST: {score:.6f} (previous best: {best_score:.6f})")
        else:
            status = "DISCARD"
            print(f"\nNo improvement: {score:.6f} (best: {best_score:.6f})")

    log_result(
        iteration,
        score,
        status,
        args.description,
        args.overrides,
        tier,
        {
            "output_dir": str(output_dir),
            "eval_returncode": eval_returncode,
            "eval_details": details_str[-2000:] if details_str else "",
        },
    )

    print(f"\nScore: {score:.6f} | Status: {status}")
    # Print score for autoresearch agent to parse
    print(f"AUTORESEARCH_SCORE={score:.6f}")


def get_best_score():
    """Get the best score from results.tsv."""
    if not RESULTS_FILE.exists():
        return 0.0

    best = 0.0
    with RESULTS_FILE.open() as f:
        for line in f:
            if line.startswith("iteration") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                with suppress(ValueError):
                    score = float(parts[2])
                    if score > best:
                        best = score
    return best


if __name__ == "__main__":
    main()
