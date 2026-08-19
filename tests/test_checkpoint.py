from pathlib import Path

import pytest
import torch

from scmolt.checkpoint import CHECKPOINT_FORMAT, CheckpointError, inspect_checkpoint


def _payload() -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "epoch_completed": 1000,
        "profile": "d40_missadv125_pairanchor_feature_remask25_tune",
        "method_slug": "trackc_w14_d40_c3_missadv125_pairanchor_feature_remask25_tune",
        "condition": "mask20",
        "model_state_dict": {"weight": torch.ones(2, 3)},
        "optimizer_state_included": False,
        "replay_state_included": False,
        "architecture_class": "scmolt.trackc.TrackCMIDASCycle",
        "base_w14_config_filename": "cell_c_bmmc.json",
    }


def test_safe_checkpoint_inspection(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(_payload(), path)
    report = inspect_checkpoint(path)
    assert report["format"] == CHECKPOINT_FORMAT
    assert report["state_tensor_count"] == 1
    assert report["state_parameter_and_buffer_numel"] == 6


def test_optimizer_state_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    payload = _payload()
    payload["optimizer_state_included"] = True
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="optimizer"):
        inspect_checkpoint(path)


def test_wrong_hash_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(_payload(), path)
    with pytest.raises(CheckpointError, match="SHA256"):
        inspect_checkpoint(path, expected_sha256="0" * 64)
