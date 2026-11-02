"""Export joint embeddings and missing modalities with the retained branch routes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from ..frozenbridge65.engine import _dataset_batch
from ..frozenbridge65.io import ordered_hash, sha256_file
from .config import BATCH_SIZE, EPOCHS, MODALITIES, SEED, experiment_receipt
from .policy import MODEL_FAMILY, policy_sha256, uses_direct_adt_route


@torch.no_grad()
def export_predictions(
    *,
    model: Any,
    context: Any,
    output_path: Path,
) -> dict[str, Any]:
    model.eval()
    manifest = context.input_manifest
    batch_order = manifest["batch_order"]
    total_cells = int(manifest["cell_count"])
    target_counts = {
        modality: sum(
            int(manifest["batches"][batch]["cell_count"])
            for batch in batch_order
            if modality in manifest["batches"][batch]["missing_modalities"]
        )
        for modality in MODALITIES
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    string_type = h5py.string_dtype(encoding="utf-8")
    summary: dict[str, Any] = {
        "batch_order": batch_order,
        "joint_embedding_shape": [total_cells, model.config.dim_c],
        "imputed": {},
        "target_specific_routes": {"adt_from_atac_batches": []},
    }
    with h5py.File(temporary, "w") as handle:
        all_ids = [
            cell
            for batch in batch_order
            for cell in manifest["batches"][batch]["cell_ids"]
        ]
        all_batches = [
            batch
            for batch in batch_order
            for _ in manifest["batches"][batch]["cell_ids"]
        ]
        handle.create_dataset("cell_ids", data=np.asarray(all_ids, dtype=object), dtype=string_type)
        handle.create_dataset("batch_ids", data=np.asarray(all_batches, dtype=object), dtype=string_type)
        embedding = handle.create_dataset(
            "joint_embedding",
            shape=(total_cells, model.config.dim_c),
            dtype=np.float32,
            chunks=(min(1024, total_cells), model.config.dim_c),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        imputed_group = handle.create_group("imputed")
        imputed_values: dict[str, Any] = {}
        imputed_offsets = {modality: 0 for modality in MODALITIES}
        for modality in MODALITIES:
            group = imputed_group.create_group(modality)
            ids = [
                cell
                for batch in batch_order
                if modality in manifest["batches"][batch]["missing_modalities"]
                for cell in manifest["batches"][batch]["cell_ids"]
            ]
            batch_ids = [
                batch
                for batch in batch_order
                if modality in manifest["batches"][batch]["missing_modalities"]
                for _ in manifest["batches"][batch]["cell_ids"]
            ]
            group.create_dataset("cell_ids", data=np.asarray(ids, dtype=object), dtype=string_type)
            group.create_dataset("batch_ids", data=np.asarray(batch_ids, dtype=object), dtype=string_type)
            feature_count = sum(model.feature_dims[modality])
            if target_counts[modality] > 0:
                imputed_values[modality] = group.create_dataset(
                    "values",
                    shape=(target_counts[modality], feature_count),
                    dtype=np.float32,
                    chunks=(min(64, target_counts[modality]), min(2048, feature_count)),
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                )
            else:
                imputed_values[modality] = group.create_dataset(
                    "values", shape=(0, feature_count), dtype=np.float32
                )
            target_batches = [
                batch
                for batch in batch_order
                if modality in manifest["batches"][batch]["missing_modalities"]
            ]
            group.attrs["target_batches"] = json.dumps(target_batches)
            summary["imputed"][modality] = {
                "shape": [target_counts[modality], feature_count],
                "target_batches": target_batches,
                "cell_order_sha256": ordered_hash(ids),
            }
        embedding_offset = 0
        for dataset_index, batch_name in enumerate(batch_order):
            record = manifest["batches"][batch_name]
            cell_count = int(record["cell_count"])
            observed = set(record["observed_modalities"])
            missing = tuple(record["missing_modalities"])
            uses_bridge = uses_direct_adt_route(context.configuration, observed)
            if uses_bridge:
                summary["target_specific_routes"]["adt_from_atac_batches"].append(
                    batch_name
                )
            for start in range(0, cell_count, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, cell_count)
                batch = _dataset_batch(context, dataset_index, start, stop)
                output = model.forward(batch, cycle_scale=1.0, sample=False)
                latent = (
                    output.joint.mean[:, : model.config.dim_c]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                if not np.isfinite(latent).all():
                    raise FloatingPointError(f"{batch_name}: non-finite embedding")
                embedding[
                    embedding_offset + start : embedding_offset + stop
                ] = latent
                if missing:
                    raw = dict(output.reconstructions)
                    if uses_bridge:
                        raw["adt"] = model.decode_atac_to_adt(output, sample=False)
                    decoded = model.native_vae.gen_real_data(raw, sampling=False)
                    for modality in missing:
                        values = (
                            decoded[modality]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        if not np.isfinite(values).all():
                            raise FloatingPointError(
                                f"{batch_name}/{modality}: non-finite imputation"
                            )
                        offset = imputed_offsets[modality]
                        imputed_values[modality][offset : offset + len(values)] = values
                        imputed_offsets[modality] += len(values)
            embedding_offset += cell_count
        if embedding_offset != total_cells or imputed_offsets != target_counts:
            raise RuntimeError("Prediction row accounting drifted")
        handle.attrs["method_slug"] = experiment_receipt(context.configuration, backend=context.backend)["method_slug"]
        handle.attrs["model_family"] = MODEL_FAMILY
        handle.attrs["configuration"] = context.configuration
        handle.attrs["anchor"] = model.config.anchor
        handle.attrs["policy_sha256"] = policy_sha256()
        handle.attrs["epochs"] = EPOCHS
        handle.attrs["seed"] = SEED
        handle.attrs["cell_order_sha256"] = ordered_hash(all_ids)
        handle.attrs["technical_dimensions_excluded"] = 2
        handle.flush()
    os.replace(temporary, output_path)
    summary["path"] = str(output_path)
    summary["sha256"] = sha256_file(output_path)
    summary["canonical_cell_order_sha256"] = ordered_hash(all_ids)
    return summary
