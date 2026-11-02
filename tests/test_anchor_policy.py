import pytest

from scmolt.anchor_conditioned.config import collect_source_hashes, experiment_receipt
from scmolt.anchor_conditioned.policy import (
    CONFIGURATIONS, STRUCTURAL_CONFIGS, load_policy, policy_sha256,
    selected_anchor, uses_direct_adt_route, validate_topology,
)


def test_policy_is_the_one_used_in_the_paper():
    assert policy_sha256() == "f876e0207ee925447b1a10ac62f55d57468d04294e0b6e41e611f9a87ef56773"
    assert [selected_anchor(c) for c in CONFIGURATIONS] == [
        "rna", "atac", "rna", "atac", "rna", "adt", "rna", "rna",
    ]
    assert load_policy()["independent_policy_validation"] is False


def test_policy_rejects_unknown_topologies_and_wrong_observed_files():
    with pytest.raises(ValueError, match="Unknown topology"):
        selected_anchor("unseen")
    manifest = {"batches": {batch: {"observed_modalities": sorted(mods)}
                           for batch, mods in STRUCTURAL_CONFIGS["paired-ab"].items()}}
    validate_topology(manifest, "paired-ab")
    with pytest.raises(ValueError, match="Input topology differs"):
        validate_topology(manifest, "paired-ac")


def test_retained_branch_export_routes_are_distinct():
    assert uses_direct_adt_route("diagonal", {"atac"})
    assert not uses_direct_adt_route("paired-ac", {"rna", "atac"})
    assert uses_direct_adt_route("paired-ab", {"rna", "atac"})
    assert uses_direct_adt_route("paired+full", {"rna", "atac"})
    assert not uses_direct_adt_route("paired-bc", {"atac", "adt"})


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_branch_receipt_preserves_schedule_and_objectives(configuration):
    receipt = experiment_receipt(configuration, backend="native_workers")
    assert (receipt["seed"], receipt["epochs"], receipt["batch_size"]) == (1234, 65, 256)
    assert receipt["warmup_epochs"] == receipt["ramp_epochs"] == 30
    assert receipt["batch_expert_scope"] == "native_full_z_c_plus_u"
    assert receipt["paired_anchor_teacher"] == selected_anchor(configuration)
    assert receipt["missingness_conditioned_on_anchor_absence"] == selected_anchor(configuration)
    if selected_anchor(configuration) == "rna":
        assert receipt["atac_adt_bridge_weights"]["cycle_weight"] == 30
    else:
        assert "atac_adt_bridge_weights" not in receipt
        assert receipt["extra_atac_adt_cycle_weight"] == 0


def test_packaged_sources_match_the_source_lock():
    hashes = collect_source_hashes()
    assert "trackc/model.py" in hashes and "anchor_conditioned/engine.py" in hashes
