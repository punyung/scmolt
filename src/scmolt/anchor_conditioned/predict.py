"""Strict-load a anchor-conditioned-v1 checkpoint and export predictions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .config import SEED
from .engine import configure_context, utc_now
from ..frozenbridge65.io import atomic_json
from .checkpoint import load_checkpoint_strict
from .prediction import export_predictions
from .policy import CONFIGURATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, help="optional separate local runtime storage")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--backend",
        choices=("gpu_resident_v1", "native_workers"),
        default="gpu_resident_v1",
    )
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--hash-input", action="store_true")
    parser.add_argument("--allow-runtime-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError(f"Set PYTHONHASHSEED={SEED} before starting Python")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    context = configure_context(
        configuration=args.configuration,
        input_dir=args.input_dir,
        runtime_dir=args.runtime_dir or output_dir / "runtime",
        device=args.device,
        backend=args.backend,
        allow_runtime_drift=args.allow_runtime_drift,
        hash_input=args.hash_input,
    )
    model, strict_reload = load_checkpoint_strict(
        context,
        args.checkpoint,
        fresh_native_modules=False,
    )
    prediction = export_predictions(
        model=model,
        context=context,
        output_path=output_dir / "predictions.h5",
    )
    atomic_json(output_dir / "STRICT_RELOAD.json", strict_reload)
    atomic_json(output_dir / "PREDICTIONS_FROZEN.json", prediction)
    atomic_json(
        output_dir / "SUCCESS.json",
        {"state": "SUCCESS", "strict_reload": strict_reload, "predictions": prediction, "completed_at": utc_now()},
    )
    print(f"SUCCESS: {output_dir}")


if __name__ == "__main__":
    main()
