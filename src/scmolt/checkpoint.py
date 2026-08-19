"""Safe reconstruction and strict loading for released scMoLT checkpoints."""

from __future__ import annotations

import csv
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .trackc import TrackCConfig, TrackCMIDASCycle


CHECKPOINT_FORMAT = "trackc_midas_cycle_model_state_v1"
_REQUIRED_PAYLOAD_KEYS = {
    "format",
    "epoch_completed",
    "profile",
    "method_slug",
    "condition",
    "model_state_dict",
    "optimizer_state_included",
    "replay_state_included",
    "architecture_class",
    "base_w14_config_filename",
}


class CheckpointError(ValueError):
    """Raised when a checkpoint or its release contract is inconsistent."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""

    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the packaged Cell C reconstruction contract."""

    if path is None:
        document = resources.files("scmolt.configs").joinpath("cell_c_bmmc.json")
        with document.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    _validate_model_config(config)
    return config


def checkpoint_spec(
    condition: str, model_config: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Return one immutable Release asset specification."""

    config = dict(model_config) if model_config is not None else load_model_config()
    checkpoints = config["release"]["checkpoints"]
    if condition not in checkpoints:
        raise CheckpointError(
            f"Unknown condition {condition!r}; choose one of {sorted(checkpoints)}"
        )
    return checkpoints[condition]


def inspect_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify and summarize a model-only checkpoint without constructing a model."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual_sha256 = sha256_file(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise CheckpointError(
            f"SHA256 mismatch for {resolved.name}: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    payload = _load_payload(resolved)
    state = payload["model_state_dict"]
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
        "format": payload["format"],
        "condition": payload["condition"],
        "epoch_completed": int(payload["epoch_completed"]),
        "profile": payload["profile"],
        "method_slug": payload["method_slug"],
        "optimizer_state_included": bool(payload["optimizer_state_included"]),
        "replay_state_included": bool(payload["replay_state_included"]),
        "state_tensor_count": len(state),
        "state_parameter_and_buffer_numel": sum(
            int(value.numel()) for value in state.values()
        ),
    }


def build_model(
    model_config: Mapping[str, Any] | str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
) -> TrackCMIDASCycle:
    """Reconstruct the exact Cell C architecture before loading weights."""

    if model_config is None or isinstance(model_config, (str, Path)):
        config = load_model_config(model_config)
    else:
        config = dict(model_config)
        _validate_model_config(config)
    torch.manual_seed(int(config["training"]["seed"]))
    core = TrackCConfig.from_dict(config["trackc_config"])
    model = TrackCMIDASCycle(
        feature_dims=config["feature_dims"],
        n_batches=int(config["n_batches"]),
        batch_class_dims=config["batch_class_dims"],
        native_configs=config["native_config_overrides"],
        config=core,
    )
    return model.to(device)


def load_pretrained(
    checkpoint_path: str | Path,
    *,
    condition: str,
    model_config: Mapping[str, Any] | str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[TrackCMIDASCycle, dict[str, Any]]:
    """Verify hash, reconstruct architecture, and strict-load one checkpoint."""

    if model_config is None or isinstance(model_config, (str, Path)):
        config = load_model_config(model_config)
    else:
        config = dict(model_config)
        _validate_model_config(config)
    spec = checkpoint_spec(condition, config)
    resolved = Path(checkpoint_path).expanduser().resolve()
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != spec["sha256"]:
        raise CheckpointError(
            f"SHA256 mismatch for {resolved.name}: expected {spec['sha256']}, "
            f"found {actual_sha256}"
        )
    if resolved.stat().st_size != int(spec["bytes"]):
        raise CheckpointError(
            f"Byte-size mismatch for {resolved.name}: expected {spec['bytes']}, "
            f"found {resolved.stat().st_size}"
        )
    payload = _load_payload(resolved)
    if payload["condition"] != condition:
        raise CheckpointError(
            f"Checkpoint condition is {payload['condition']!r}, not {condition!r}"
        )
    expected = config["training"]
    if int(payload["epoch_completed"]) != int(expected["epochs"]):
        raise CheckpointError(
            f"Checkpoint stopped at epoch {payload['epoch_completed']}; "
            f"expected {expected['epochs']}"
        )
    for key in ("profile", "method_slug"):
        if payload[key] != expected[key]:
            raise CheckpointError(
                f"Checkpoint {key} is {payload[key]!r}; expected {expected[key]!r}"
            )
    model = build_model(config, device="cpu")
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointError(
            "Strict load unexpectedly returned incompatible state-dict keys"
        )
    model.to(device)
    model.eval()
    metadata = {
        key: value for key, value in payload.items() if key != "model_state_dict"
    }
    metadata.update(
        {
            "sha256": actual_sha256,
            "bytes": resolved.stat().st_size,
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }
    )
    return model, metadata


def verify_feature_contract(
    repository_root: str | Path,
    model_config: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify feature hashes, row counts, headers, and sequential indices."""

    root = Path(repository_root).expanduser().resolve()
    if model_config is None or isinstance(model_config, (str, Path)):
        config = load_model_config(model_config)
    else:
        config = dict(model_config)
        _validate_model_config(config)
    report: dict[str, dict[str, Any]] = {}
    for modality, spec in config["feature_contract"].items():
        path = root / spec["path"]
        actual_sha256 = sha256_file(path)
        if actual_sha256 != spec["sha256"]:
            raise CheckpointError(
                f"{modality} feature SHA256 mismatch: expected {spec['sha256']}, "
                f"found {actual_sha256}"
            )
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != ["", "x"]:
                raise CheckpointError(
                    f"Unexpected {modality} feature header: {header!r}"
                )
            rows = 0
            for rows, row in enumerate(reader, start=1):
                if len(row) != 2 or row[0] != str(rows) or not row[1]:
                    raise CheckpointError(
                        f"Invalid {modality} feature row {rows}: {row!r}"
                    )
        if rows != int(spec["rows"]):
            raise CheckpointError(
                f"{modality} feature count is {rows}; expected {spec['rows']}"
            )
        report[modality] = {
            "path": str(path),
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
            "status": "pass",
        }
    return report


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Could not safely load checkpoint {path.name}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a dictionary")
    missing = _REQUIRED_PAYLOAD_KEYS - set(payload)
    if missing:
        raise CheckpointError(f"Checkpoint is missing keys: {sorted(missing)}")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"Unsupported checkpoint format {payload['format']!r}"
        )
    if payload["optimizer_state_included"] is not False:
        raise CheckpointError("Released checkpoints must not contain optimizer state")
    if payload["replay_state_included"] is not False:
        raise CheckpointError("Released checkpoints must not contain replay state")
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise CheckpointError("model_state_dict must be a non-empty mapping")
    if any(not isinstance(key, str) for key in state):
        raise CheckpointError("All state-dict keys must be strings")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise CheckpointError("All state-dict values must be tensors")
    return payload


def _validate_model_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "checkpoint_format",
        "scmidas_version",
        "training",
        "feature_dims",
        "feature_totals",
        "feature_contract",
        "n_batches",
        "batch_class_dims",
        "native_config_overrides",
        "trackc_config",
        "release",
    }
    missing = required - set(config)
    if missing:
        raise CheckpointError(f"Model config is missing keys: {sorted(missing)}")
    if config["schema_version"] != "scmolt.pretrained.v1":
        raise CheckpointError("Unsupported model-config schema")
    if config["checkpoint_format"] != CHECKPOINT_FORMAT:
        raise CheckpointError("Model config declares an unsupported checkpoint format")
    if config["scmidas_version"] != "0.1.13":
        raise CheckpointError("Cell C requires scmidas==0.1.13")
    for modality, dimensions in config["feature_dims"].items():
        if sum(int(value) for value in dimensions) != int(
            config["feature_totals"][modality]
        ):
            raise CheckpointError(f"{modality} feature dimensions do not sum correctly")
    if int(config["batch_class_dims"]["joint"]) != int(config["n_batches"]):
        raise CheckpointError("joint batch dimension must equal n_batches")
    TrackCConfig.from_dict(config["trackc_config"])


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointError",
    "build_model",
    "checkpoint_spec",
    "inspect_checkpoint",
    "load_model_config",
    "load_pretrained",
    "sha256_file",
    "verify_feature_contract",
]
