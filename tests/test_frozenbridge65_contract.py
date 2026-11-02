from pathlib import Path

from scmolt.frozenbridge65.config import (
    BRIDGE_WEIGHTS,
    EPOCHS,
    EXPECTED_CORE_SOURCE_SHA256,
    bridge_config,
    collect_core_source_hashes,
    core_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_frozenbridge65_exact_epoch_and_schedule() -> None:
    config = core_config()
    assert config.epochs == EPOCHS == 65
    assert config.cycle_scale(29) == 0.0
    assert config.cycle_scale(30) == 1.0 / 30.0
    assert config.cycle_scale(59) == 1.0
    assert config.cycle_scale(64) == 1.0


def test_frozenbridge65_exact_bridge_contract() -> None:
    assert bridge_config().to_dict() == BRIDGE_WEIGHTS
    assert BRIDGE_WEIGHTS == {
        "paired_consistency_weight": 5.0,
        "cycle_weight": 30.0,
        "identity_weight": 15.0,
        "cross_decode_weight": 1.0,
        "seed_offset": 500_003,
    }


def test_frozenbridge65_source_lock() -> None:
    assert collect_core_source_hashes() == EXPECTED_CORE_SOURCE_SHA256


def test_frozenbridge65_contains_no_checkpoint() -> None:
    assert not list((REPOSITORY_ROOT / "src/scmolt/frozenbridge65").rglob("*.pt"))
