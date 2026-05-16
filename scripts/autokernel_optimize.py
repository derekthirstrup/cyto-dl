"""Drive AutoKernel against a cyto-dl model, Lightning checkpoint, or Cellpose 4 weights.

Generates a small shim file that exposes the chosen model as a plain ``nn.Module``
class, then invokes the AutoKernel CLI (``profile.py`` -> ``extract.py`` ->
``bench.py``) inside ``vendor/autokernel/``. Streams subprocess output so the
Streamlit GUI's log thread can capture it. On completion, copies AutoKernel's
``results.tsv`` into the run workspace and prints a JSON marker the GUI parses
to populate the results panel.

Run via the GUI's AutoKernel tab or directly:
    python scripts/autokernel_optimize.py \\
        --source lightning_ckpt --ckpt logs/.../last.ckpt \\
        --input-shape 1 1 16 128 128 --dtype float16 \\
        --backend triton --top-n 5 --experiments-per-kernel 50
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess  # nosec: B404
import sys
import textwrap
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "autokernel"
SENTINEL = VENDOR_DIR / ".cytodl_ready"
RESULTS_MARKER = "### AUTOKERNEL_RESULTS_JSON ###"


def _shim_for_lightning(ckpt_path: str, target_module: str) -> str:
    return (
        textwrap.dedent(
            f'''
        """Auto-generated shim. Loads a cyto-dl Lightning checkpoint and exposes
        its underlying network as a plain nn.Module for AutoKernel profiling."""
        from __future__ import annotations
        import torch
        from torch import nn

        from cyto_dl.models.base_model import BaseModel


        class ExportedModel(nn.Module):
            def __init__(self):
                super().__init__()
                lit = BaseModel.load_from_checkpoint({ckpt_path!r}, map_location="cuda", strict=False)
                lit.eval()
                # Prefer .backbone (im2im models) then .net (generic Lightning).
                inner = getattr(lit, "backbone", None) or getattr(lit, "net", None) or lit
                self.inner = inner

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x)
        '''
        ).strip()
        + "\n"
    )


def _shim_for_cellpose(ckpt_path: str) -> str:
    return (
        textwrap.dedent(
            f'''
        """Auto-generated shim. Loads a Cellpose 4 fine-tuned model and exposes
        its underlying torch network as nn.Module for AutoKernel profiling."""
        from __future__ import annotations
        import torch
        from torch import nn


        class ExportedModel(nn.Module):
            def __init__(self):
                super().__init__()
                from cellpose.models import CellposeModel
                cp = CellposeModel(pretrained_model={ckpt_path!r}, gpu=True)
                # Cellpose stores its torch network on .net (CPnet / cpsam).
                self.inner = cp.net.eval()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x)
        '''
        ).strip()
        + "\n"
    )


def _shim_for_class(module: str, class_name: str) -> str:
    return (
        textwrap.dedent(
            f'''
        """Auto-generated shim. Re-exports the user-specified cyto-dl class so
        AutoKernel's --class-name flag finds it at a stable location."""
        from {module} import {class_name} as ExportedModel  # noqa: F401
        '''
        ).strip()
        + "\n"
    )


def _resolve_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.is_file():
        print(f"WARN: --config {config_path} not found; using built-in defaults", flush=True)
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _autokernel_cmd(script: str, *args: str) -> list[str]:
    return ["uv", "run", script, *args]


def _stream(cmd: list[str], cwd: Path) -> int:
    print(f"$ (cwd={cwd}) {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(  # nosec: B603
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return proc.wait()


def _parse_results_tsv(tsv_path: Path) -> list[dict]:
    if not tsv_path.is_file():
        return []
    out: list[dict] = []
    with tsv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["cytodl_class", "lightning_ckpt", "cellpose4"],
        required=True,
    )
    parser.add_argument("--module", help="Python module path for cytodl_class source.")
    parser.add_argument("--class-name", help="Class name within --module.")
    parser.add_argument("--ckpt", help="Path to .ckpt (lightning) or .pt (cellpose4).")
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        required=True,
        help="Input tensor shape, e.g. 1 1 16 128 128 (batch + channels + spatial).",
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default=None)
    parser.add_argument("--backend", choices=["triton", "cuda"], default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--experiments-per-kernel", type=int, default=None)
    parser.add_argument("--target-metric", choices=["throughput", "latency", "vram"], default=None)
    parser.add_argument(
        "--workspace",
        default=None,
        help="Where to copy results, shim, and logs.",
    )
    parser.add_argument(
        "--llm-model", default=None, help="Optional LLM identifier passed through."
    )
    parser.add_argument("--config", default=None, help="Optional YAML defaults override.")
    args = parser.parse_args()

    defaults = _resolve_defaults(args.config)
    backend = args.backend if args.backend is not None else defaults.get("backend", "triton")
    dtype = args.dtype if args.dtype is not None else defaults.get("dtype", "float16")
    top_n = args.top_n if args.top_n is not None else defaults.get("top_n", 5)
    experiments = (
        args.experiments_per_kernel
        if args.experiments_per_kernel is not None
        else defaults.get("experiments_per_kernel", 50)
    )
    target_metric = (
        args.target_metric
        if args.target_metric is not None
        else defaults.get("target_metric", "throughput")
    )
    workspace_arg = (
        args.workspace
        if args.workspace is not None
        else defaults.get("workspace", str(PROJECT_ROOT / "outputs" / "autokernel" / "run"))
    )
    llm_model = args.llm_model if args.llm_model is not None else defaults.get("llm_model", "")

    if not SENTINEL.is_file():
        print(
            "ERROR: AutoKernel is not installed. Run `python scripts/setup_autokernel.py` first.",
            file=sys.stderr,
        )
        return 1

    workspace = Path(workspace_arg).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    shim_path = workspace / "_model_shim.py"
    if args.source == "lightning_ckpt":
        if not args.ckpt:
            print("ERROR: --ckpt required for lightning_ckpt source.", file=sys.stderr)
            return 1
        shim_path.write_text(_shim_for_lightning(args.ckpt, ""), encoding="utf-8")
        module_arg = str(shim_path)
        class_arg = "ExportedModel"
    elif args.source == "cellpose4":
        if not args.ckpt:
            print("ERROR: --ckpt required for cellpose4 source.", file=sys.stderr)
            return 1
        shim_path.write_text(_shim_for_cellpose(args.ckpt), encoding="utf-8")
        module_arg = str(shim_path)
        class_arg = "ExportedModel"
    else:
        if not (args.module and args.class_name):
            print("ERROR: --module and --class-name required for cytodl_class.", file=sys.stderr)
            return 1
        shim_path.write_text(_shim_for_class(args.module, args.class_name), encoding="utf-8")
        module_arg = str(shim_path)
        class_arg = "ExportedModel"

    shape_args = [str(d) for d in args.input_shape]

    profile_cmd = _autokernel_cmd(
        "profile.py",
        "--model",
        module_arg,
        "--class-name",
        class_arg,
        "--input-shape",
        ",".join(shape_args),
        "--dtype",
        dtype,
    )
    extract_cmd = _autokernel_cmd("extract.py", "--top", str(top_n), "--backend", backend)
    bench_cmd = _autokernel_cmd(
        "bench.py",
        "--max-experiments",
        str(experiments),
        "--target-metric",
        target_metric,
    )
    if llm_model:
        bench_cmd += ["--llm-model", llm_model]

    print(f"AutoKernel workspace: {workspace}")
    print(f"Shim file: {shim_path}")
    print(
        f"Backend={backend} dtype={dtype} top_n={top_n} experiments={experiments} target={target_metric}"
    )

    for step_name, cmd in (
        ("profile", profile_cmd),
        ("extract", extract_cmd),
        ("bench", bench_cmd),
    ):
        rc = _stream(cmd, cwd=VENDOR_DIR)
        if rc != 0:
            print(f"ERROR: AutoKernel step `{step_name}` exited with code {rc}", file=sys.stderr)
            return rc

    src_results = VENDOR_DIR / "workspace" / "results.tsv"
    dst_results = workspace / "results.tsv"
    if src_results.is_file():
        shutil.copy2(src_results, dst_results)
        rows = _parse_results_tsv(dst_results)
    else:
        rows = []
        print(f"WARN: {src_results} not found; results panel will be empty.")

    print(RESULTS_MARKER)
    print(json.dumps({"workspace": str(workspace), "results": rows}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
