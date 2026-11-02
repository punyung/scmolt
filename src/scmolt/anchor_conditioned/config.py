"""Fixed branch recipes, preserving the common 65-epoch training contract."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from typing import Any

from ..frozenbridge65.config import (
    BATCH_SIZE, CHECKPOINT_EVERY_EPOCHS, EPOCHS, GPU_RESIDENT_PRIVATE_SEED,
    MODALITIES, N_MAX, REMASK_PRIVATE_SEED, SEED, bridge_config, core_config,
    experiment_receipt as rna_experiment_receipt,
)
from ..trackc_atacgraph import ATACADTEdgeConfig
from .policy import MODEL_FAMILY, REVISION, policy_sha256, selected_anchor, topology_alias


BRANCH_METHODS = {
    "rna": "frozen_old_bridge_equal_horizon_e65",
    "atac": "c3_ataccentered_graph_clean_constructioncompat_e65",
    "adt": "c3_adtcentered_graph_pair_crossdecode_e65",
}


def edge_config() -> ATACADTEdgeConfig:
    return ATACADTEdgeConfig(
        paired_consistency_weight=5.0, cross_decode_weight=1.0, seed_offset=500_003
    )


def experiment_receipt(configuration: str, *, backend: str) -> dict[str, Any]:
    anchor = selected_anchor(configuration)
    result = rna_experiment_receipt(
        backend=backend, topology_alias=topology_alias(configuration)
    )
    if anchor != "rna":
        result.pop("atac_adt_bridge_weights")
        result.pop("atac_only_adt_prediction_route")
        result["atac_adt_edge_weights"] = edge_config().to_dict()
        result["extra_atac_adt_cycle_weight"] = 0.0
        result["extra_atac_adt_identity_weight"] = 0.0
    result.update({
        "model_family": MODEL_FAMILY,
        "policy_revision": REVISION,
        "policy_sha256": policy_sha256(),
        "configuration": configuration,
        "topology": configuration,
        "core_topology_alias": topology_alias(configuration),
        "anchor": anchor,
        "method_slug": f"{MODEL_FAMILY}__{configuration}__{anchor}",
        "method_name": f"DOGMA {MODEL_FAMILY}: {configuration}, {anchor.upper()} anchor",
        "research_branch_method_slug": BRANCH_METHODS[anchor],
        "paired_anchor_teacher": anchor,
        "missingness_conditioned_on_anchor_absence": anchor,
        "separate_atac_adt_bridge_module": anchor == "rna",
        "retained_directed_translators": 6 if anchor == "rna" else 4,
        "batch_expert_scope": "native_full_z_c_plus_u",
        "remask_visibility_protection": "historical RNA behavior retained",
        "anchor_fixed_for_entire_run": True,
        "jointly_trained_shared_parameter_router": False,
        "independent_policy_validation": False,
    })
    if anchor == "atac":
        result.update({
            "historical_constructor_cuda_seed_compatible": True,
            "historical_main_parameter_order_compatible": True,
            "historical_zero_weight_call_schedule_compatible": False,
        })
    return result


def collect_source_hashes() -> dict[str, str]:
    resource = files("scmolt.configs").joinpath("anchor_conditioned_sources.json")
    expected = json.loads(resource.read_text())["files"]
    root = files("scmolt")
    observed = {}
    for relative, digest in expected.items():
        actual = hashlib.sha256(root.joinpath(relative).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Published anchor-conditioned source drifted: {relative}")
        observed[relative] = actual
    return observed
