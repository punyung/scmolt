"""Batch folders that share one batch label (feat/batch_groups.toml)."""

import numpy as np
import pandas as pd
import pytest
import torch

ad = pytest.importorskip("anndata")
pytest.importorskip("scmidas")

from scmolt.data import write_midas_input
from scmolt.frozenbridge65.io import grouped_batch_labels, validate_input

FOLDERS = ["01_cite_a", "02_cite_a_holdout", "03_cite_b", "04_multiome_a", "05_multiome_a_holdout"]
GROUPS = {"01_cite_a": "cite_a", "02_cite_a_holdout": "cite_a", "03_cite_b": "cite_b",
          "04_multiome_a": "multiome_a", "05_multiome_a_holdout": "multiome_a"}
OBSERVED = {"01_cite_a": ["rna", "adt"], "02_cite_a_holdout": ["rna"], "03_cite_b": ["rna", "adt"],
            "04_multiome_a": ["rna", "atac"], "05_multiome_a_holdout": ["rna"]}


def test_identity_groups_give_the_native_labels():
    labels = grouped_batch_labels(FOLDERS, OBSERVED, {name: name for name in FOLDERS})
    assert [labels[name]["joint"] for name in FOLDERS] == [0, 1, 2, 3, 4]
    assert [labels[name]["rna"] for name in FOLDERS] == [0, 1, 2, 3, 4]
    assert labels["03_cite_b"]["adt"] == 1 and labels["04_multiome_a"]["atac"] == 0


def test_holdout_folders_share_the_label_of_their_batch():
    labels = grouped_batch_labels(FOLDERS, OBSERVED, GROUPS)
    assert labels["01_cite_a"] == {"joint": 0, "rna": 0, "adt": 0}
    assert labels["02_cite_a_holdout"] == {"joint": 0, "rna": 0}
    assert labels["03_cite_b"] == {"joint": 1, "rna": 1, "adt": 1}
    assert labels["04_multiome_a"] == {"joint": 2, "rna": 2, "atac": 0}
    assert labels["05_multiome_a_holdout"] == {"joint": 2, "rna": 2}


def _grouped_input(root, cells=6):
    rng = np.random.default_rng(0)
    peaks = [f"chr1-{i * 100}-{i * 100 + 50}" for i in range(8)] + [f"chr2-{i * 100}-{i * 100 + 50}" for i in range(8)]
    features = {"rna": [f"gene{i}" for i in range(16)], "adt": [f"prot{i}" for i in range(8)], "atac": peaks}
    adatas = {}
    for modality, names in features.items():
        folders = [f for f in FOLDERS if modality in OBSERVED[f]]
        obs = pd.DataFrame({"batch": np.repeat(folders, cells)},
                           index=[f"{f}:{i}" for f in folders for i in range(cells)])
        values = rng.binomial(1, 0.2, (len(obs), len(names))) if modality == "atac" else rng.poisson(2, (len(obs), len(names)))
        adatas[modality] = ad.AnnData(values.astype(np.float32), obs=obs, var=pd.DataFrame(index=names))
    return write_midas_input(adatas, root, batch_groups=GROUPS)


def test_grouped_input_is_validated_and_hashed(tmp_path):
    root = _grouped_input(tmp_path / "input")
    manifest = validate_input(root)
    assert manifest["batch_groups"]["groups"] == GROUPS
    assert manifest["batch_groups"]["labels"] == grouped_batch_labels(FOLDERS, OBSERVED, GROUPS)
    (root / "feat" / "batch_groups.toml").unlink()
    assert "batch_groups" not in validate_input(root)
    assert validate_input(root)["portable_schema_sha256"] != manifest["portable_schema_sha256"]


def test_runner_trains_with_shared_labels(tmp_path):
    from scmolt.frozenbridge65.engine import configure_context, smoke_model

    torch.set_num_threads(1)
    root = _grouped_input(tmp_path / "input")
    context = configure_context(input_dir=root, runtime_dir=tmp_path / "runtime", device="cpu",
                                backend="native_workers", allow_runtime_drift=True)
    assert dict(context.native.dims_s) == {"rna": 3, "adt": 2, "atac": 1, "joint": 3}
    labels = grouped_batch_labels(FOLDERS, OBSERVED, GROUPS)
    assert [dataset.mod_id_dict for dataset in context.native.datalist] == [labels[name] for name in FOLDERS]
    assert context.model.native_vae.dims_s == context.native.dims_s
    assert smoke_model(context, 4)["smoke_batches_completed"] == 4


def test_anchor_runner_rejects_batch_groups(tmp_path):
    from scmolt.anchor_conditioned.engine import configure_context
    from scmolt.anchor_conditioned.make_tiny_input import make_tiny_input

    root = make_tiny_input(tmp_path / "input", "paired-ab", cells_per_batch=4)
    (root / "feat" / "batch_groups.toml").write_text(
        "".join(f'"{b}" = "all"\n' for b in ("lll_ctrl", "lll_stim", "dig_ctrl", "dig_stim")))
    with pytest.raises(ValueError, match="scmolt-train only"):
        configure_context(configuration="paired-ab", input_dir=root, runtime_dir=tmp_path / "runtime",
                          device="cpu", backend="native_workers", allow_runtime_drift=True)
