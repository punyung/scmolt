"""Create a deterministic tiny native-MIDAS tri-modal input for smoke tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import toml


FEATURE_DIMS = {"rna": [16], "atac": [8, 8], "adt": [8]}
BATCHES = {
    "batch1_rna_atac": ("rna", "atac"),
    "batch2_rna_adt": ("rna", "adt"),
    "batch3_atac_adt": ("atac", "adt"),
    "batch4_atac_only": ("atac",),
}


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells-per-batch", type=int, default=16)
    args = parser.parse_args()
    if args.cells_per_batch < 4:
        raise ValueError("--cells-per-batch must be at least 4")
    root = args.output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {root}")
    (root / "feat").mkdir(parents=True, exist_ok=True)
    (root / "feat" / "feat_dims.toml").write_text(
        toml.dumps(FEATURE_DIMS), encoding="utf-8"
    )
    feature_names = {
        modality: [f"{modality}_feature_{index:03d}" for index in range(sum(blocks))]
        for modality, blocks in FEATURE_DIMS.items()
    }
    for modality, names in feature_names.items():
        _write_csv(
            root / "feat" / f"feat_names_{modality}.csv",
            [["", "x"], *[[index + 1, name] for index, name in enumerate(names)]],
        )
    rng = np.random.default_rng(1234)
    for batch_index, (batch_name, modalities) in enumerate(BATCHES.items()):
        ids = [f"tiny_b{batch_index + 1}_cell_{index:03d}" for index in range(args.cells_per_batch)]
        for modality in modalities:
            names = feature_names[modality]
            if modality == "atac":
                values = rng.binomial(1, 0.12, size=(len(ids), len(names))).astype(np.float32)
            elif modality == "rna":
                values = rng.poisson(2.5, size=(len(ids), len(names))).astype(np.float32)
            else:
                values = rng.poisson(1.5, size=(len(ids), len(names))).astype(np.float32)
            matrix_rows: list[list[object]] = [["", *names]]
            matrix_rows.extend(
                [[cell_id, *map(float, row)] for cell_id, row in zip(ids, values)]
            )
            _write_csv(root / batch_name / "mat" / f"{modality}.csv", matrix_rows)
            _write_csv(
                root / batch_name / "mask" / f"{modality}.csv",
                [["", *names], [1, *([1] * len(names))]],
            )
    print(root)


if __name__ == "__main__":
    main()
