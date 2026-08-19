import json
from pathlib import Path

from scmolt import load_model_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_release_index_matches_packaged_config() -> None:
    config = load_model_config()
    index = json.loads((REPOSITORY_ROOT / "weights/release-index.json").read_text())
    indexed = {asset["condition"]: asset for asset in index["assets"]}
    expected = config["release"]["checkpoints"]
    assert set(indexed) == set(expected)
    for condition, spec in expected.items():
        assert indexed[condition]["filename"] == spec["filename"]
        assert indexed[condition]["bytes"] == spec["bytes"]
        assert indexed[condition]["sha256"] == spec["sha256"]


def test_repository_does_not_track_weight_assets() -> None:
    assert not list(REPOSITORY_ROOT.rglob("*.pt"))
    assert not list(REPOSITORY_ROOT.rglob("*.pth"))
