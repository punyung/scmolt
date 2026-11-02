"""Model-only checkpoints bound to one policy, topology, and input schema."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ..frozenbridge65.config import EPOCHS, installed_versions
from ..frozenbridge65.engine import _checkpoint_input_contract
from ..frozenbridge65.io import sha256_file
from .config import collect_source_hashes, experiment_receipt
from .policy import policy_sha256
from .registry import assert_model_contract


CHECKPOINT_FORMAT = "anchor_conditioned_v1_portable_model_state_v1"


def save_model_checkpoint(path: Path, *, model: Any, epoch_completed: int, context: Any) -> None:
    assert_model_contract(model, context.configuration)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "epoch_completed": int(epoch_completed),
        "model": model.state_dict(),
        "experiment": experiment_receipt(context.configuration, backend=context.backend),
        "policy_sha256": policy_sha256(),
        "input_contract": _checkpoint_input_contract(context.input_manifest),
        "native_modality_order": list(map(str, model.native_vae.dims_x)),
        "source_sha256": collect_source_hashes(),
        "package_versions": installed_versions(),
        "optimizer_state_included": False,
        "replay_state_included": False,
    }, temporary)
    os.replace(temporary, path)


def load_checkpoint_strict(
    context: Any, checkpoint: Path, *, fresh_native_modules: bool
) -> tuple[Any, dict[str, Any]]:
    path = checkpoint.expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint is not an anchor-conditioned-v1 model-state payload")
    if not isinstance(payload.get("model"), Mapping):
        raise ValueError("Checkpoint lacks a model-state mapping")
    if int(payload.get("epoch_completed", -1)) != EPOCHS:
        raise ValueError(f"Prediction requires the final epoch-{EPOCHS} checkpoint")
    experiment = payload.get("experiment", {})
    backend = experiment.get("training_data_backend")
    if backend not in {"gpu_resident_v1", "native_workers"}:
        raise ValueError("Checkpoint lacks a supported training backend")
    expected = experiment_receipt(context.configuration, backend=backend)
    if dict(experiment) != expected or payload.get("policy_sha256") != policy_sha256():
        raise ValueError("Checkpoint topology, anchor, policy, or objective contract differs")
    if payload.get("source_sha256") != collect_source_hashes():
        raise ValueError("Checkpoint model/runner source hashes differ")
    declared = payload.get("input_contract", {})
    current = _checkpoint_input_contract(context.input_manifest)
    if declared.get("portable_schema_sha256") != current.get("portable_schema_sha256"):
        raise ValueError("Checkpoint and current MIDAS input schemas differ")
    declared_content = declared.get("content_tree_sha256")
    current_content = current.get("content_tree_sha256")
    if declared_content and current_content and declared_content != current_content:
        raise ValueError("Checkpoint and current input content hashes differ")
    if payload.get("native_modality_order") != list(map(str, context.native.dims_x)):
        raise ValueError("Checkpoint native modality order differs")
    if fresh_native_modules:
        from .engine import _build_model

        for name in ("_native_encoder_capture_handle", "_native_batch_capture_handle"):
            handle = getattr(context.model, name, None)
            if handle is not None:
                handle.remove()
        context.model.to(torch.device("cpu"))
        gc.collect()
        if context.device.type == "cuda":
            torch.cuda.empty_cache()
        model = _build_model(context.native, configuration=context.configuration,
                             device=context.device, fresh_native_modules=True)
    else:
        model = context.model
    model.load_state_dict(payload["model"], strict=True)
    assert_model_contract(model, context.configuration)
    model.eval()
    return model, {
        "status": "passed", "strict": True,
        "configuration": context.configuration, "anchor": model.config.anchor,
        "policy_sha256": policy_sha256(),
        "fresh_native_modules_built_during_load": fresh_native_modules,
        "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
        "checkpoint_epoch_completed": EPOCHS,
        "state_dict_key_count": len(payload["model"]),
        "missing_keys": [], "unexpected_keys": [],
    }
