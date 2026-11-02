"""Input validation and small receipt helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import natsort
import toml

from .config import MODALITIES


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in map(str, values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _feature_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Malformed feature-name CSV: {path}")
    names = [row[1] for row in rows[1:] if len(row) >= 2]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Feature names are empty or duplicated: {path}")
    return names


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def _cell_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row or not row[0]:
                raise ValueError(f"Blank cell identifier in {path}")
            ids.append(str(row[0]))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Cell identifiers are empty or duplicated in {path}")
    return ids


BATCH_GROUPS_FILE = "batch_groups.toml"


def read_batch_groups(input_dir: Path, batch_names: list[str]) -> dict[str, str] | None:
    """Read the optional ``feat/batch_groups.toml``, which maps batch folders to shared batches.

    Without the file every folder is its own batch, as in native MIDAS. With it,
    folders mapped to the same name share one batch label, for example the cells
    of a batch whose ADT was withheld and the remaining cells of that batch.
    """
    path = input_dir / "feat" / BATCH_GROUPS_FILE
    if not path.is_file():
        return None
    raw = toml.load(path)
    if set(raw) != set(batch_names):
        raise ValueError(
            f"{path} must map exactly the batch folders {batch_names}; got {sorted(raw)}"
        )
    groups = {name: raw[name] for name in batch_names}
    if not all(isinstance(value, str) and value for value in groups.values()):
        raise ValueError(f"{path}: every batch folder must map to a non-empty batch name")
    return groups


def grouped_batch_labels(
    batch_names: list[str],
    observed: Mapping[str, Iterable[str]],
    groups: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    """Batch labels of every folder, assigned as native MIDAS does but counted over shared batches.

    The joint label is the index of the shared batch in order of first appearance;
    the label of a modality is the index among the shared batches that measure it.
    Mapping every folder to its own name reproduces the native labels.
    """
    order = list(dict.fromkeys(groups[name] for name in batch_names))
    measuring: dict[str, list[str]] = {}
    for group in order:
        for name in batch_names:
            if groups[name] != group:
                continue
            for modality in observed[name]:
                if group not in measuring.setdefault(modality, []):
                    measuring[modality].append(group)
    return {
        name: {
            "joint": order.index(groups[name]),
            **{modality: measuring[modality].index(groups[name]) for modality in observed[name]},
        }
        for name in batch_names
    }


def native_batch_names(input_dir: Path) -> list[str]:
    return list(
        natsort.natsorted(
            path.name
            for path in input_dir.iterdir()
            if path.is_dir() and path.name != "feat"
        )
    )


def validate_input(input_dir: Path, *, hash_contents: bool = False) -> dict[str, Any]:
    root = input_dir.expanduser().resolve()
    dims_path = root / "feat" / "feat_dims.toml"
    if not dims_path.is_file():
        raise FileNotFoundError(f"Missing {dims_path}")
    dims_raw = toml.load(dims_path)
    if set(dims_raw) != set(MODALITIES) or len(dims_raw) != len(MODALITIES):
        raise ValueError(
            f"feat_dims.toml must declare exactly {MODALITIES}; got {tuple(dims_raw)}"
        )
    dims: dict[str, list[int]] = {}
    names: dict[str, list[str]] = {}
    for modality in MODALITIES:
        blocks = [int(value) for value in dims_raw[modality]]
        if not blocks or any(value <= 0 for value in blocks):
            raise ValueError(f"Invalid {modality} feature blocks: {blocks}")
        dims[modality] = blocks
        names_path = root / "feat" / f"feat_names_{modality}.csv"
        if not names_path.is_file():
            raise FileNotFoundError(f"Missing {names_path}")
        names[modality] = _feature_names(names_path)
        if len(names[modality]) != sum(blocks):
            raise ValueError(
                f"{modality}: {len(names[modality])} names != {sum(blocks)} dimensions"
            )

    batches: dict[str, Any] = {}
    all_cell_ids: list[str] = []
    modality_batch_counts = {modality: 0 for modality in MODALITIES}
    batch_names = native_batch_names(root)
    if not batch_names:
        raise ValueError("Native MIDAS input has no batch directories")
    for batch_name in batch_names:
        if "/" in batch_name:
            raise ValueError(f"Batch name may not contain '/': {batch_name!r}")
        batch_dir = root / batch_name
        mat_dir = batch_dir / "mat"
        mask_dir = batch_dir / "mask"
        if not mat_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"{batch_name}: expected mat/ and mask/ directories")
        mat_paths = {path.stem: path for path in mat_dir.glob("*.csv")}
        mask_paths = {path.stem: path for path in mask_dir.glob("*.csv")}
        unknown = (set(mat_paths) | set(mask_paths)) - set(MODALITIES)
        if unknown:
            raise ValueError(f"{batch_name}: unknown modalities {sorted(unknown)}")
        observed = tuple(modality for modality in MODALITIES if modality in mat_paths)
        if not observed:
            raise ValueError(f"{batch_name}: no observed modalities")
        if set(mask_paths) != set(observed):
            raise ValueError(
                f"{batch_name}: mask modalities {sorted(mask_paths)} != observed {list(observed)}"
            )
        canonical_ids: list[str] | None = None
        for modality in observed:
            modality_batch_counts[modality] += 1
            expected_header = [""] + names[modality]
            if _header(mat_paths[modality]) != expected_header:
                raise ValueError(f"{batch_name}/{modality}: matrix header differs from feature names")
            if _header(mask_paths[modality]) != expected_header:
                raise ValueError(f"{batch_name}/{modality}: mask header differs from feature names")
            ids = _cell_ids(mat_paths[modality])
            if canonical_ids is None:
                canonical_ids = ids
            elif ids != canonical_ids:
                raise ValueError(f"{batch_name}: cell order differs across observed modalities")
            with mask_paths[modality].open("r", encoding="utf-8", newline="") as handle:
                mask_rows = list(csv.reader(handle))
            if len(mask_rows) != 2 or len(mask_rows[1]) != len(expected_header):
                raise ValueError(f"{batch_name}/{modality}: mask must contain one feature row")
        assert canonical_ids is not None
        duplicates = set(all_cell_ids).intersection(canonical_ids)
        if duplicates:
            raise ValueError(f"Cell identifiers repeat across batches; first duplicate {next(iter(duplicates))}")
        all_cell_ids.extend(canonical_ids)
        batches[batch_name] = {
            "observed_modalities": list(observed),
            "missing_modalities": [m for m in MODALITIES if m not in observed],
            "cell_count": len(canonical_ids),
            "cell_order_sha256": ordered_hash(canonical_ids),
            "cell_ids": canonical_ids,
        }
    absent = [key for key, count in modality_batch_counts.items() if count == 0]
    if absent:
        raise ValueError(f"Every modality must be observed in at least one batch; absent={absent}")
    schema = {
        "schema_version": 1,
        "input_dir": str(root),
        "feature_dims": dims,
        "native_modality_order": list(dims_raw),
        "feature_name_sha256": {
            modality: ordered_hash(names[modality]) for modality in MODALITIES
        },
        "batch_order": batch_names,
        "batch_count": len(batch_names),
        "cell_count": len(all_cell_ids),
        "canonical_cell_order_sha256": ordered_hash(all_cell_ids),
        "modality_batch_counts": modality_batch_counts,
        "paired_atac_adt_batches": [
            batch
            for batch, record in batches.items()
            if {"atac", "adt"}.issubset(record["observed_modalities"])
        ],
        "atac_only_batches": [
            batch for batch, record in batches.items() if record["observed_modalities"] == ["atac"]
        ],
        "batches": batches,
        "content_tree_sha256": tree_hash(root) if hash_contents else None,
    }
    groups = read_batch_groups(root, batch_names)
    if groups is not None:
        # Only present with feat/batch_groups.toml, so inputs without it keep their schema hash.
        schema["batch_groups"] = {
            "groups": groups,
            "labels": grouped_batch_labels(
                batch_names,
                {name: record["observed_modalities"] for name, record in batches.items()},
                groups,
            ),
        }
    digest_view = json.loads(json.dumps(schema))
    digest_view.pop("input_dir")
    # Full content hashing is an optional, potentially expensive audit. It must
    # not change the structural schema identity used for checkpoint reload.
    digest_view.pop("content_tree_sha256")
    for record in digest_view["batches"].values():
        record.pop("cell_ids")
    schema["portable_schema_sha256"] = hashlib.sha256(
        (json.dumps(digest_view, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return schema
