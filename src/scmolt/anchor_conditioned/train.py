"""Train anchor-conditioned-v1 for its fixed 65-epoch contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

import torch

from .config import EPOCHS, SEED
from .engine import (
    configure_context,
    preflight_receipt,
    smoke_model,
    train_model,
    utc_now,
)
from ..frozenbridge65.io import atomic_json
from .checkpoint import load_checkpoint_strict
from .prediction import export_predictions
from .policy import CONFIGURATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, help="optional separate local runtime storage")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--backend",
        choices=("gpu_resident_v1", "native_workers"),
        default="gpu_resident_v1",
    )
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--hash-input", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--allow-runtime-drift", action="store_true")
    return parser.parse_args()


def _prepare_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError(
            f"Set PYTHONHASHSEED={SEED} before starting Python; see README.md"
        )
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    output_dir = _prepare_output(args.output_dir)
    runtime_dir = args.runtime_dir or output_dir / "runtime"
    try:
        context = configure_context(
            configuration=args.configuration,
            input_dir=args.input_dir,
            runtime_dir=runtime_dir,
            device=args.device,
            backend=args.backend,
            hash_input=args.hash_input,
            allow_runtime_drift=args.allow_runtime_drift,
        )
        metadata = preflight_receipt(context)
        metadata["formal"] = args.smoke_batches <= 0
        metadata["command"] = [sys.executable, *sys.argv]
        atomic_json(output_dir / "run_metadata.json", metadata)
        if args.smoke_batches > 0:
            result = smoke_model(context, args.smoke_batches)
            atomic_json(output_dir / "SMOKE.json", result)
            atomic_json(
                output_dir / "STATE.json",
                {
                    "state": "SMOKE_PASSED",
                    "formal_epochs_completed": 0,
                    "updated_at": utc_now(),
                },
            )
            print(f"SMOKE PASSED: {output_dir}")
            return

        checkpoint, training = train_model(context, output_dir)
        atomic_json(output_dir / "TRAINING_SUCCESS.json", training)
        reloaded, strict_reload = load_checkpoint_strict(
            context,
            checkpoint,
            fresh_native_modules=True,
        )
        atomic_json(output_dir / "STRICT_RELOAD.json", strict_reload)
        prediction = None
        if not args.skip_predictions:
            prediction = export_predictions(
                model=reloaded,
                context=context,
                output_path=output_dir / "predictions.h5",
            )
            atomic_json(output_dir / "PREDICTIONS_FROZEN.json", prediction)
        success = {
            "state": "SUCCESS",
            "method_name": metadata["experiment"]["method_name"],
            "epochs_completed": EPOCHS,
            "checkpoint": training,
            "strict_reload": strict_reload,
            "predictions": prediction,
            "completed_at": utc_now(),
        }
        atomic_json(output_dir / "SUCCESS.json", success)
        atomic_json(
            output_dir / "STATE.json",
            {"state": "SUCCESS", "epoch_completed": EPOCHS, "updated_at": utc_now()},
        )
        print(f"SUCCESS: {output_dir}")
    except Exception as exc:
        atomic_json(
            output_dir / "FAILED.json",
            {
                "state": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
