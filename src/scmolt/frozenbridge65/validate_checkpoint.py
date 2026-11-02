"""Strict-load a checkpoint and run one deterministic forward probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .config import SEED
from .engine import _dataset_batch, configure_context, load_checkpoint_strict
from .io import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--allow-runtime-drift", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError(f"Set PYTHONHASHSEED={SEED} before starting Python")
    torch.set_num_threads(min(20, max(1, os.cpu_count() or 1)))
    work = (
        args.receipt.expanduser().resolve().parent / ".checkpoint_validation_runtime"
        if args.receipt
        else Path.cwd() / ".checkpoint_validation_runtime"
    )
    context = configure_context(
        input_dir=args.input_dir,
        runtime_dir=work,
        device=args.device,
        backend="native_workers",
        allow_runtime_drift=args.allow_runtime_drift,
    )
    model, receipt = load_checkpoint_strict(
        context,
        args.checkpoint,
        fresh_native_modules=False,
        allow_legacy=args.allow_legacy,
    )
    batch = _dataset_batch(context, 0, 0, min(4, len(context.native.datalist[0])))
    with torch.no_grad():
        output = model.forward(batch, cycle_scale=1.0, sample=False)
    receipt["forward_probe"] = {
        "status": "passed",
        "observed_modalities": list(batch["x"]),
        "rows": int(output.joint.mean.shape[0]),
        "joint_shape": list(output.joint.mean.shape),
        "joint_finite": bool(torch.isfinite(output.joint.mean).all().cpu()),
        "reconstruction_shapes": {
            key: list(value.shape) for key, value in output.reconstructions.items()
        },
    }
    if not receipt["forward_probe"]["joint_finite"]:
        raise FloatingPointError("Checkpoint forward probe returned non-finite values")
    if args.receipt:
        atomic_json(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
