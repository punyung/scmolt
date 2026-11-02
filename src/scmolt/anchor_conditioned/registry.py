"""Construct one complete selected branch, without mixing independent weights."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..trackc import TrackCMIDASCycle
from ..trackc_adbridge import TrackCADBridgeMIDASCycle
from ..trackc_adtgraph import TrackCADTCenteredMIDASCycle
from ..trackc_atacgraph_replaycompat import TrackCATACCenteredConstructionCompatMIDASCycle
from .config import bridge_config, core_config, edge_config
from .policy import selected_anchor, topology_alias


def build_model(
    configuration: str,
    *,
    feature_dims: Mapping[str, int | Sequence[int]],
    n_batches: int,
    **native_kwargs: Any,
) -> TrackCMIDASCycle:
    anchor = selected_anchor(configuration)
    kwargs = dict(
        feature_dims=feature_dims,
        n_batches=n_batches,
        config=core_config(topology_alias=topology_alias(configuration)),
        **native_kwargs,
    )
    if anchor == "rna":
        model = TrackCADBridgeMIDASCycle(**kwargs, bridge_config=bridge_config())
    elif anchor == "atac":
        model = TrackCATACCenteredConstructionCompatMIDASCycle(
            **kwargs, edge_config=edge_config()
        )
    else:
        model = TrackCADTCenteredMIDASCycle(**kwargs, edge_config=edge_config())
    assert_model_contract(model, configuration)
    return model


def assert_model_contract(model: TrackCMIDASCycle, configuration: str) -> None:
    anchor = selected_anchor(configuration)
    if model.config.anchor != anchor or tuple(model.modalities) != ("rna", "atac", "adt"):
        raise RuntimeError("Executable branch differs from the selected policy")
    expected = {
        f"{source}__{target}"
        for modality in ("rna", "atac", "adt") if modality != anchor
        for source, target in ((modality, anchor), (anchor, modality))
    }
    if set(model.transport.translators) != expected:
        raise RuntimeError("Canonical transport graph differs from the selected anchor")
    if hasattr(model, "atac_adt_bridge") != (anchor == "rna"):
        raise RuntimeError("Auxiliary bridge registration differs from the selected branch")
    if anchor == "rna":
        if model.bridge_config.to_dict() != bridge_config().to_dict():
            raise RuntimeError("RNA auxiliary bridge objective drifted")
    elif model.edge_config.to_dict() != edge_config().to_dict():
        raise RuntimeError("Integrated ATAC-ADT edge objective drifted")
    if anchor == "atac":
        receipt = model.anchor_strategy_receipt()
        for key, expected_value in {
            "historical_constructor_cuda_seed_compatible": True,
            "historical_main_parameter_order_compatible": True,
            "historical_zero_weight_call_schedule_compatible": False,
        }.items():
            if receipt.get(key) is not expected_value:
                raise RuntimeError(f"ATAC construction compatibility drifted: {key}")
