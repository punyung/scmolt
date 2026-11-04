"""Create a synthetic native input with one exact DOGMA observation topology."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import toml

from ..frozenbridge65.make_tiny_input import FEATURE_DIMS, _write_csv
from .policy import BATCH_ORDER, CONFIGURATIONS, MODALITIES, STRUCTURAL_CONFIGS, selected_anchor


def make_tiny_input(root: Path, configuration: str, *, cells_per_batch: int = 16) -> Path:
    selected_anchor(configuration)
    if cells_per_batch < 4:
        raise ValueError("cells_per_batch must be at least 4")
    root = root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output directory is not empty: {root}")
    (root / "feat").mkdir(parents=True, exist_ok=True)
    (root / "feat/feat_dims.toml").write_text(toml.dumps(FEATURE_DIMS))
    names = {m: [f"{m}_feature_{i:03d}" for i in range(sum(FEATURE_DIMS[m]))]
             for m in MODALITIES}
    for modality in MODALITIES:
        _write_csv(root / "feat" / f"feat_names_{modality}.csv",
                   [["", "x"], *[[i + 1, n] for i, n in enumerate(names[modality])]])
    rng = np.random.default_rng(1234)
    for index, batch in enumerate(BATCH_ORDER):
        ids = [f"tiny_b{index + 1}_cell_{i:03d}" for i in range(cells_per_batch)]
        for modality in MODALITIES:
            if modality not in STRUCTURAL_CONFIGS[configuration][batch]:
                continue
            shape = (cells_per_batch, len(names[modality]))
            values = rng.binomial(1, 0.12, shape) if modality == "atac" else rng.poisson(2, shape)
            _write_csv(root / batch / "mat" / f"{modality}.csv", [
                ["", *names[modality]],
                *[[cell, *map(float, row)] for cell, row in zip(ids, values)],
            ])
            _write_csv(root / batch / "mask" / f"{modality}.csv",
                       [["", *names[modality]], [1, *([1] * len(names[modality]))]])
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells-per-batch", type=int, default=16)
    args = parser.parse_args()
    print(make_tiny_input(args.output_dir, args.configuration,
                          cells_per_batch=args.cells_per_batch))


if __name__ == "__main__":
    main()
