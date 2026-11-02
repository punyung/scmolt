"""Convert AnnData to the native MIDAS input of the scMoLT runners, and read their predictions.

The directory layout is

    OUTPUT/
      feat/feat_dims.toml
      feat/feat_names_{rna,adt,atac}.csv
      <batch>/mat/<modality>.csv     one file per modality measured in the batch
      <batch>/mask/<modality>.csv    one row of ones (all features available)

RNA and ADT are written as raw counts (layer ``counts`` if present, else ``X``) and
ATAC as binary accessibility. A modality that a batch does not measure has no file;
it is never written as a matrix of zeros.
"""

from __future__ import annotations

from collections import OrderedDict
import csv
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import numpy as np

MODALITIES = ("rna", "adt", "atac")
_CHUNK_ROWS = {"rna": 256, "adt": 256, "atac": 64}


def _dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _rows(matrix: Any, indices: np.ndarray) -> Any:
    """Read rows from in-memory or backed matrices (backed reads need sorted indices)."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size and np.all(np.diff(indices) == 1):
        return matrix[int(indices[0]): int(indices[-1]) + 1]
    order = np.argsort(indices)
    block = matrix[indices[order].tolist()]
    return block[np.argsort(order)]


def _values(adata: Any, modality: str, indices: np.ndarray) -> np.ndarray:
    if modality == "atac":
        return (_dense(_rows(adata.X, indices)) > 0).astype(np.uint8)
    source = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return _dense(_rows(source, indices)).astype(np.float32)


def atac_chromosome_blocks(peaks: list[str]) -> list[int]:
    """Number of peaks per chromosome; peaks of one chromosome must be contiguous."""
    blocks: OrderedDict[str, int] = OrderedDict()
    previous = None
    for peak in peaks:
        chromosome = re.split(r"[:\-_]", peak, maxsplit=1)[0]
        if chromosome != previous and chromosome in blocks:
            raise ValueError(f"ATAC peaks of {chromosome} are not contiguous; sort the peaks by chromosome")
        blocks[chromosome] = blocks.get(chromosome, 0) + 1
        previous = chromosome
    return list(blocks.values())


def _write_feature_names(names: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", "x"])
        writer.writerows([index + 1, name] for index, name in enumerate(names))


def _write_matrix(adata: Any, modality: str, indices: np.ndarray, cells: np.ndarray,
                  features: list[str], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["", *features])
    step = _CHUNK_ROWS[modality]
    for start in range(0, len(indices), step):
        stop = min(start + step, len(indices))
        frame = pd.DataFrame(_values(adata, modality, indices[start:stop]),
                             index=cells[start:stop], columns=features)
        frame.to_csv(path, mode="a", header=False)


def _write_mask(features: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", *features])
        writer.writerow([1, *([1] * len(features))])


def read_predictions(path: str | Path, input_dir: str | Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Read the ``predictions.h5`` written by ``scmolt-train``, ``scmolt-predict`` or the anchor runner.

    Returns an AnnData with the 32-dimensional biological embedding in
    ``obsm["X_scmolt"]`` and the batch of every cell in ``obs["batch"]``, and a dict
    with one AnnData of predicted values per modality that some batch does not measure.
    With ``input_dir``, the predicted features are named after the input features.
    """
    import anndata as ad
    import h5py
    import pandas as pd

    def strings(dataset: Any) -> list[str]:
        return [value.decode() if isinstance(value, bytes) else str(value) for value in dataset[:]]

    with h5py.File(path, "r") as handle:
        obs = pd.DataFrame({"batch": strings(handle["batch_ids"])}, index=strings(handle["cell_ids"]))
        embedding = ad.AnnData(obs=obs, obsm={"X_scmolt": handle["joint_embedding"][:]})
        imputed = {}
        for modality, group in handle["imputed"].items():
            if group["values"].shape[0] == 0:
                continue
            var = None
            if input_dir is not None:
                names_path = Path(input_dir) / "feat" / f"feat_names_{modality}.csv"
                var = pd.DataFrame(index=pd.read_csv(names_path, index_col=0)["x"].astype(str).tolist())
            imputed[modality] = ad.AnnData(
                X=group["values"][:], var=var,
                obs=pd.DataFrame({"batch": strings(group["batch_ids"])}, index=strings(group["cell_ids"])))
    return embedding, imputed


def write_midas_input(
    modalities: Mapping[str, Any],
    output_dir: str | Path,
    *,
    batch_key: str = "batch",
    layout: Mapping[str, Any] | None = None,
    batch_groups: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one AnnData per modality as a native MIDAS input directory.

    Parameters
    ----------
    modalities
        ``{"rna": adata, "adt": adata, "atac": adata}``. Every AnnData has a column
        ``batch_key`` in ``obs``; cells are matched across modalities by ``obs_names``.
    output_dir
        The input directory to create.
    layout
        Optional ``{batch: modalities measured in the batch}``, used to build a
        pseudo-mosaic from fully measured data. By default a modality counts as
        measured in a batch whenever its AnnData contains cells of that batch.
    batch_groups
        Optional ``{batch: shared batch}``, written to ``feat/batch_groups.toml``.
        Batches mapped to the same name share one batch label in ``scmolt-train``,
        for example the cells of a batch whose ADT was withheld for evaluation and
        the other cells of that batch. Batch folders are ordered by natural sort
        of their names.
    overwrite
        Replace an existing ``output_dir``.

    Returns
    -------
    The resolved input directory.
    """
    missing = set(MODALITIES) - set(modalities)
    if missing:
        raise ValueError(f"The scMoLT runners need RNA, ADT and ATAC; missing {sorted(missing)}")
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists; pass overwrite=True to replace it")
        shutil.rmtree(root)
    (root / "feat").mkdir(parents=True)

    features = {m: modalities[m].var_names.astype(str).tolist() for m in MODALITIES}
    dims = {"rna": [len(features["rna"])], "adt": [len(features["adt"])],
            "atac": atac_chromosome_blocks(features["atac"])}
    (root / "feat" / "feat_dims.toml").write_text(
        "".join(f"{m} = [{', '.join(map(str, dims[m]))}]\n" for m in MODALITIES), encoding="utf-8")
    for modality in MODALITIES:
        _write_feature_names(features[modality], root / "feat" / f"feat_names_{modality}.csv")

    batch_of = {m: modalities[m].obs[batch_key].astype(str).to_numpy() for m in MODALITIES}
    if layout is None:
        batches = sorted(set().union(*(set(values) for values in batch_of.values())))
        layout = {b: [m for m in MODALITIES if np.any(batch_of[m] == b)] for b in batches}
    for batch, measured in sorted(layout.items()):
        measured = [m for m in MODALITIES if m in set(measured)]
        if not measured:
            raise ValueError(f"Batch {batch!r} measures no modality")
        first = modalities[measured[0]]
        cells = first.obs_names[batch_of[measured[0]] == str(batch)].astype(str).to_numpy()
        if cells.size == 0:
            raise ValueError(f"No {measured[0]} cells in batch {batch!r}")
        for modality in measured:
            adata = modalities[modality]
            positions = adata.obs_names.get_indexer(cells)
            if np.any(positions < 0) or np.any(batch_of[modality][positions] != str(batch)):
                raise ValueError(f"Batch {batch!r}: {modality} does not contain the same cells as {measured[0]}")
            _write_matrix(adata, modality, positions, cells, features[modality],
                          root / str(batch) / "mat" / f"{modality}.csv")
            _write_mask(features[modality], root / str(batch) / "mask" / f"{modality}.csv")
    if batch_groups is not None:
        if set(map(str, batch_groups)) != set(map(str, layout)):
            raise ValueError("batch_groups must map every batch to a shared batch")
        (root / "feat" / "batch_groups.toml").write_text(
            "".join(f'"{batch}" = "{batch_groups[batch]}"\n' for batch in sorted(batch_groups)),
            encoding="utf-8")
    return root
