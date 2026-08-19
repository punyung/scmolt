"""Command-line verification helpers for scMoLT releases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from .checkpoint import (
    CheckpointError,
    checkpoint_spec,
    inspect_checkpoint,
    load_model_config,
    load_pretrained,
    verify_feature_contract,
)


def _json_output(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _prepare_runtime_cache(root: str | Path) -> dict[str, str]:
    cache_root = Path(root).expanduser().resolve()
    numba = cache_root / "numba"
    matplotlib = cache_root / "matplotlib"
    numba.mkdir(parents=True, exist_ok=True)
    matplotlib.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib))
    return {
        "NUMBA_CACHE_DIR": os.environ["NUMBA_CACHE_DIR"],
        "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
    }


def _command_info(_args: argparse.Namespace) -> None:
    config = load_model_config()
    _json_output(
        {
            "model_id": config["model_id"],
            "display_name": config["display_name"],
            "architecture": config["architecture"],
            "scmidas_version": config["scmidas_version"],
            "training": config["training"],
            "feature_totals": config["feature_totals"],
            "release": config["release"],
        }
    )


def _command_inspect(args: argparse.Namespace) -> None:
    expected = None
    if args.condition is not None:
        expected = checkpoint_spec(args.condition)["sha256"]
    report = inspect_checkpoint(args.checkpoint, expected_sha256=expected)
    if args.condition is not None and report["condition"] != args.condition:
        raise CheckpointError(
            f"Checkpoint declares {report['condition']!r}, not {args.condition!r}"
        )
    _json_output(report)


def _command_verify_assets(args: argparse.Namespace) -> None:
    directory = Path(args.directory).expanduser().resolve()
    config = load_model_config()
    reports = {}
    for condition, spec in config["release"]["checkpoints"].items():
        path = directory / spec["filename"]
        report = inspect_checkpoint(path, expected_sha256=spec["sha256"])
        if report["condition"] != condition:
            raise CheckpointError(
                f"{path.name} declares condition {report['condition']!r}"
            )
        if report["bytes"] != int(spec["bytes"]):
            raise CheckpointError(
                f"{path.name} has {report['bytes']} bytes; expected {spec['bytes']}"
            )
        reports[condition] = report
    _json_output({"status": "pass", "assets": reports})


def _command_verify_features(args: argparse.Namespace) -> None:
    _json_output(
        {
            "status": "pass",
            "features": verify_feature_contract(args.repository_root),
        }
    )


def _command_strict_load(args: argparse.Namespace) -> None:
    caches = _prepare_runtime_cache(args.cache_dir)
    model, metadata = load_pretrained(
        args.checkpoint,
        condition=args.condition,
        device=args.device,
    )
    _json_output(
        {
            "status": "pass",
            "strict_load": True,
            "device": str(next(model.parameters()).device),
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "state_tensor_count": len(model.state_dict()),
            "state_parameter_and_buffer_numel": sum(
                int(value.numel()) for value in model.state_dict().values()
            ),
            "runtime_cache": caches,
            "checkpoint": metadata,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scmolt",
        description="Inspect, verify, and strict-load scMoLT model releases.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show the packaged model contract")
    info.set_defaults(function=_command_info)

    inspect = subparsers.add_parser(
        "inspect", help="safely inspect one model-only checkpoint"
    )
    inspect.add_argument("checkpoint", type=Path)
    inspect.add_argument(
        "--condition", choices=("mask20", "mask40", "mask60", "mask80")
    )
    inspect.set_defaults(function=_command_inspect)

    assets = subparsers.add_parser(
        "verify-assets", help="verify all four staged Release assets"
    )
    assets.add_argument("directory", type=Path)
    assets.set_defaults(function=_command_verify_assets)

    features = subparsers.add_parser(
        "verify-features", help="verify exact RNA and ATAC feature order files"
    )
    features.add_argument("repository_root", type=Path)
    features.set_defaults(function=_command_verify_features)

    strict_load = subparsers.add_parser(
        "strict-load", help="reconstruct the architecture and strict-load weights"
    )
    strict_load.add_argument("checkpoint", type=Path)
    strict_load.add_argument(
        "--condition",
        required=True,
        choices=("mask20", "mask40", "mask60", "mask80"),
    )
    strict_load.add_argument("--device", default="cpu")
    strict_load.add_argument("--cache-dir", default=".cache/scmolt")
    strict_load.set_defaults(function=_command_strict_load)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (CheckpointError, FileNotFoundError, RuntimeError) as exc:
        print(f"scmolt: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
