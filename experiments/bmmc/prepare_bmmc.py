"""Write the bone marrow CITE-seq + Multiome mosaic of the paper as a scMoLT input directory.

usage: python experiments/bmmc/prepare_bmmc.py --cite CITE_H5AD --multiome MULTIOME_H5AD --output-dir DIR

CITE_H5AD and MULTIOME_H5AD are the processed GEO GSE194122 files
GSE194122_openproblems_neurips2021_{cite,multiome}_BMMC_processed.h5ad.

cells.csv.gz lists the 36,951 cells of the paper in training order with the input folder
of every cell, and features.csv.gz the 4,000 genes, 134 proteins and 50,000 peaks.
Each of the six assay-by-block batches is written as two folders: the cells that keep
both modalities, and the holdout cells, whose ADT (CITE-seq) or ATAC (Multiome) was
withheld before training and which keep only RNA. feat/batch_groups.toml gives the
two folders of a batch one shared batch label, as in the paper's run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

try:
    from anndata.io import read_elem, sparse_dataset
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem, sparse_dataset

import scmolt

HERE = Path(__file__).resolve().parent


def _subset(counts, obs_names: pd.Index, var: pd.DataFrame, feature_type: str, names: list[str],
            cells: pd.DataFrame) -> ad.AnnData:
    """Counts of the given cells and features, in the given order, read from the h5ad without loading it."""
    of_type = var["feature_types"].to_numpy() == feature_type
    kind = var.index[of_type]
    if kind.has_duplicates:
        raise ValueError(f"{feature_type} feature names are not unique")
    positions = kind.get_indexer(names)
    rows = obs_names.get_indexer(cells["obs_name"])
    if (rows < 0).any() or (positions < 0).any():
        raise ValueError("Cells or features of the manifest are missing from the GEO file")
    columns = np.flatnonzero(of_type)[positions]
    order = np.argsort(rows)
    block = counts[rows[order]][np.argsort(order)][:, columns]
    return ad.AnnData(block.tocsr().astype(np.float32),
                      obs=pd.DataFrame({"batch": cells["folder"].to_numpy()}, index=cells["cell_id"].to_numpy()),
                      var=pd.DataFrame(index=names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cite", type=Path, required=True)
    parser.add_argument("--multiome", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cells = pd.read_csv(HERE / "cells.csv.gz")
    features = pd.read_csv(HERE / "features.csv.gz")
    names = {m: features.loc[features.modality == m, "name"].tolist() for m in ("rna", "adt", "atac")}
    cells["assay"] = cells["cell_id"].str.split("#").str[0]
    cells["obs_name"] = cells["cell_id"].str.split("#", n=1).str[1]
    cells["holdout"] = cells["folder"].str.endswith("_holdout")

    parts = {"rna": [], "adt": [], "atac": []}
    for assay, path, second in (("cite", args.cite, "adt"), ("multiome", args.multiome, "atac")):
        with h5py.File(path, "r") as handle:
            obs_names = pd.Index(read_elem(handle["obs"]).index.astype(str))
            var = read_elem(handle["var"])
            var.index = var.index.astype(str)
            counts = sparse_dataset(handle["layers/counts"])
            rows = cells[cells.assay == assay]
            parts["rna"].append(_subset(counts, obs_names, var, "GEX", names["rna"], rows))
            parts[second].append(_subset(counts, obs_names, var, second.upper(), names[second],
                                         rows[~rows.holdout]))
    adatas = {m: ad.concat(p) if len(p) > 1 else p[0] for m, p in parts.items()}
    groups = {folder: folder.split("_", 1)[1].removesuffix("_holdout") for folder in cells.folder.unique()}
    root = scmolt.write_midas_input(adatas, args.output_dir, batch_groups=groups)
    print(root)


if __name__ == "__main__":
    main()
