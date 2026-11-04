"""AnnData to native input conversion."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ad = pytest.importorskip("anndata")

from scmolt.anchor_conditioned.policy import STRUCTURAL_CONFIGS, validate_topology
from scmolt.data import atac_chromosome_blocks, write_midas_input
from scmolt.frozenbridge65.io import validate_input


def _dogma_like(cells_per_batch=5):
    rng = np.random.default_rng(0)
    batches = ["lll_ctrl", "lll_stim", "dig_ctrl", "dig_stim"]
    obs = pd.DataFrame({"batch": np.repeat(batches, cells_per_batch)},
                       index=[f"{b}:cell{i}" for b in batches for i in range(cells_per_batch)])
    peaks = [f"chr1-{i * 100}-{i * 100 + 50}" for i in range(6)] + [f"chr2-{i * 100}-{i * 100 + 50}" for i in range(4)]
    rna = ad.AnnData(sp.csr_matrix(rng.poisson(2.0, (len(obs), 7)).astype(np.float32)), obs=obs.copy(),
                     var=pd.DataFrame(index=[f"gene{i}" for i in range(7)]))
    rna.layers["counts"] = rna.X.copy()
    rna.X = rna.X * 0.5  # a normalized X must not be written
    adt = ad.AnnData(rng.poisson(5.0, (len(obs), 3)).astype(np.float32), obs=obs.copy(),
                     var=pd.DataFrame(index=["CD3", "CD4", "CD8"]))
    atac = ad.AnnData(sp.csr_matrix(rng.poisson(0.4, (len(obs), len(peaks))).astype(np.float32)), obs=obs.copy(),
                      var=pd.DataFrame(index=peaks))
    return {"rna": rna, "adt": adt, "atac": atac}


def test_chromosome_blocks():
    assert atac_chromosome_blocks(["chr1-1-2", "chr1-5-9", "chr2:1-2", "chrX-3-4"]) == [2, 1, 1]
    with pytest.raises(ValueError, match="not contiguous"):
        atac_chromosome_blocks(["chr1-1-2", "chr2-1-2", "chr1-5-9"])


def test_pseudo_mosaic_matches_the_dogma_topology(tmp_path):
    adatas = _dogma_like()
    layout = {batch: sorted(measured) for batch, measured in STRUCTURAL_CONFIGS["paired-ab"].items()}
    root = write_midas_input(adatas, tmp_path / "input", layout=layout)
    manifest = validate_input(root)
    validate_topology(manifest, "paired-ab")
    assert manifest["feature_dims"] == {"rna": [7], "adt": [3], "atac": [6, 4]}
    assert manifest["native_modality_order"] == ["rna", "adt", "atac"]
    rna = pd.read_csv(root / "lll_ctrl/mat/rna.csv", index_col=0)
    expected = adatas["rna"][adatas["rna"].obs["batch"] == "lll_ctrl"].layers["counts"].toarray()
    np.testing.assert_array_equal(rna.to_numpy(), expected)
    atac = pd.read_csv(root / "lll_ctrl/mat/atac.csv", index_col=0).to_numpy()
    assert set(np.unique(atac)) <= {0, 1}
    assert not (root / "lll_stim/mat/rna.csv").exists()
    with pytest.raises(FileExistsError):
        write_midas_input(adatas, root, layout=layout)


def test_measured_modalities_are_inferred_from_the_cells(tmp_path):
    adatas = _dogma_like()
    adatas["adt"] = adatas["adt"][adatas["adt"].obs["batch"].isin(["dig_ctrl", "dig_stim"])].copy()
    adatas["atac"] = adatas["atac"][adatas["atac"].obs["batch"].isin(["lll_ctrl", "lll_stim"])].copy()
    manifest = validate_input(write_midas_input(adatas, tmp_path / "input"))
    observed = {batch: record["observed_modalities"] for batch, record in manifest["batches"].items()}
    assert observed == {"dig_ctrl": ["rna", "adt"], "dig_stim": ["rna", "adt"],
                        "lll_ctrl": ["rna", "atac"], "lll_stim": ["rna", "atac"]}
