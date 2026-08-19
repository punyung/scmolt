from pathlib import Path

from scmolt import load_model_config, verify_feature_contract
from scmolt.trackc import TrackCConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_cell_c_contract() -> None:
    config = load_model_config()
    assert config["training"]["epochs"] == 1000
    assert config["training"]["seed"] == 1234
    assert config["feature_totals"] == {"rna": 4000, "atac": 50000}
    assert sum(config["feature_dims"]["atac"]) == 50000
    assert config["batch_class_dims"] == {"rna": 3, "atac": 3, "joint": 3}
    core = TrackCConfig.from_dict(config["trackc_config"])
    assert core.variant.value == "C3"
    assert core.missingness_batch_ce_scale == 1.25
    assert core.paired_rna_anchor_consistency_weight == 5.0


def test_feature_contract() -> None:
    report = verify_feature_contract(REPOSITORY_ROOT)
    assert report["rna"]["rows"] == 4000
    assert report["atac"]["rows"] == 50000
