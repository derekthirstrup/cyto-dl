"""Idempotent bootstrap for the AutoKernel kernel autotuner.

Clones https://github.com/RightNow-AI/autokernel into ``vendor/autokernel`` and
runs ``uv sync`` so the Streamlit GUI can drive its CLI.

Usage:
    python scripts/setup_autokernel.py [--repo URL] [--ref BRANCH_OR_TAG] [--force]

Exit codes:
    0  ready (already-installed or freshly synced)
    1  hard failure (uv missing, clone failed, sync failed)
    2  installed-but-no-GPU-detected (warning only; GUI may still allow dry runs)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec: B404
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "https://github.com/RightNow-AI/autokernel.git"
DEFAULT_REF = "main"
VENDOR_DIR = PROJECT_ROOT / "vendor" / "autokernel"
SENTINEL = VENDOR_DIR / ".cytodl_ready"


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=False
    ).returncode  # nosec: B603


def _check_uv() -> bool:
    return shutil.which("uv") is not None


def _check_gpu() -> tuple[bool, str]:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            if getattr(torch.version, "hip", None):
                return True, f"ROCm/HIP device: {torch.cuda.get_device_name(0)}"
            return True, f"CUDA device: {torch.cuda.get_device_name(0)}"
        return False, "torch.cuda.is_available() is False"
    except ImportError:
        return False, "torch is not importable"


def _read_commit(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(  # nosec: B603, B607
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_dir), text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument(
        "--force", action="store_true", help="Re-clone even if vendor/autokernel exists."
    )
    parser.add_argument(
        "--skip-sync", action="store_true", help="Skip `uv sync` (clone-only mode)."
    )
    args = parser.parse_args()

    if not args.skip_sync and not _check_uv():
        print(
            "ERROR: `uv` is not on PATH. Install from https://docs.astral.sh/uv/", file=sys.stderr
        )
        return 1

    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)

    if args.force and VENDOR_DIR.exists():
        print(f"--force given; removing {VENDOR_DIR}")
        shutil.rmtree(VENDOR_DIR)

    if not (VENDOR_DIR / ".git").is_dir():
        rc = _run(
            ["git", "clone", "--depth", "1", "--branch", args.ref, args.repo, str(VENDOR_DIR)]
        )
        if rc != 0:
            print("ERROR: git clone failed", file=sys.stderr)
            return 1
    else:
        print(
            f"AutoKernel repo already present at {VENDOR_DIR} (commit {_read_commit(VENDOR_DIR)})"
        )

    if not args.skip_sync:
        rc = _run(["uv", "sync"], cwd=VENDOR_DIR)
        if rc != 0:
            print("ERROR: `uv sync` failed inside vendor/autokernel", file=sys.stderr)
            return 1

    SENTINEL.write_text(f"commit={_read_commit(VENDOR_DIR)}\n", encoding="utf-8")
    print(f"AutoKernel ready at {VENDOR_DIR} (commit {_read_commit(VENDOR_DIR)})")

    gpu_ok, gpu_msg = _check_gpu()
    print(f"GPU check: {gpu_msg}")
    if not gpu_ok:
        print("WARNING: no CUDA/ROCm device detected. AutoKernel runs require a GPU.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
