#!/usr/bin/env python3
"""Initialize an autoresearch experiment.

Validates the environment, runs a baseline training, and sets up tracking.

Usage:
    python setup_experiment.py --data-path /path/to/labelfree/data

    # Just validate without running baseline
    python setup_experiment.py --data-path /path/to/data --validate-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = PROJECT_ROOT / "autoresearch" / "results.tsv"


def validate_environment():
    """Check that all dependencies and configs are in place."""
    checks = []

    # Check cyto_dl is importable
    try:
        import cyto_dl
        checks.append(("cyto_dl import", True, "OK"))
    except ImportError as e:
        checks.append(("cyto_dl import", False, str(e)))

    # Check experiment config exists
    config_path = PROJECT_ROOT / "configs" / "experiment" / "im2im" / "labelfree_autoresearch.yaml"
    checks.append(("experiment config", config_path.exists(), str(config_path)))

    # Check evaluator exists
    eval_path = PROJECT_ROOT / "autoresearch" / "evaluators" / "labelfree_evaluator.py"
    checks.append(("evaluator", eval_path.exists(), str(eval_path)))

    # Check runner exists
    runner_path = PROJECT_ROOT / "autoresearch" / "scripts" / "run_experiment.py"
    checks.append(("experiment runner", runner_path.exists(), str(runner_path)))

    # Check torch and CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            checks.append(("GPU", True, f"{gpu_name} ({gpu_mem:.1f} GB)"))
        else:
            checks.append(("GPU", False, "No CUDA GPU available"))
    except ImportError:
        checks.append(("torch", False, "PyTorch not installed"))

    return checks


def validate_data(data_path):
    """Check that training data exists and is accessible."""
    data_path = Path(data_path)

    if not data_path.exists():
        return False, f"Data path does not exist: {data_path}"

    # Check for CSV manifest or image files
    csv_files = list(data_path.glob("*.csv"))
    image_files = list(data_path.rglob("*.tif")) + list(data_path.rglob("*.czi")) + \
                  list(data_path.rglob("*.ome.zarr"))

    if csv_files:
        return True, f"Found {len(csv_files)} CSV manifest(s), {len(image_files)} image file(s)"
    elif image_files:
        return True, f"Found {len(image_files)} image file(s)"
    else:
        return False, f"No recognized data files in {data_path}"


def run_baseline(data_path):
    """Run baseline training (exploration tier) to establish initial metrics."""
    runner = PROJECT_ROOT / "autoresearch" / "scripts" / "run_experiment.py"

    cmd = [
        sys.executable, str(runner),
        "--tier", "exploration",
        "--description", "baseline",
        "--overrides", "",
    ]

    if data_path:
        cmd.extend(["--data-path", str(data_path)])

    print("\nRunning baseline training...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Set up autoresearch experiment")
    parser.add_argument("--data-path", required=True, help="Path to training data")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate, don't run baseline")
    args = parser.parse_args()

    print("=" * 60)
    print("AUTORESEARCH SETUP")
    print("=" * 60)

    # Environment validation
    print("\n--- Environment Checks ---")
    checks = validate_environment()
    all_ok = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not passed:
            all_ok = False

    # Data validation
    print("\n--- Data Validation ---")
    data_ok, data_msg = validate_data(args.data_path)
    print(f"  [{'PASS' if data_ok else 'FAIL'}] {data_msg}")

    if not all_ok or not data_ok:
        print("\nSetup FAILED. Fix the issues above and try again.")
        sys.exit(1)

    # Initialize results.tsv
    if not RESULTS_FILE.exists():
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            f.write("iteration\ttimestamp\tscore\tstatus\ttier\tdescription\toverrides\tdetails\n")
        print(f"\nCreated results tracking: {RESULTS_FILE}")

    if args.validate_only:
        print("\nValidation complete. Ready to run experiments.")
        print(f"\nNext step: python autoresearch/scripts/run_experiment.py \\")
        print(f"  --data-path {args.data_path} \\")
        print(f'  --description "baseline"')
        sys.exit(0)

    # Run baseline
    success = run_baseline(args.data_path)
    if success:
        print("\nBaseline training complete!")
        print("Run '/ar:run' or '/ar:loop' to start the optimization loop.")
    else:
        print("\nBaseline training failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
