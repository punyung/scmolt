"""Controlled compatibility variants for the ATAC-centered graph refactor.

The delegated implementation initialized its separately owned ATAC--ADT edge
before rebuilding the ATAC-anchor graph.  Each CPU-only initialization block
used ``torch.manual_seed`` inside ``fork_rng(devices=[])``.  Although CPU RNG
was restored, ``torch.manual_seed`` also reset CUDA RNG, so the final hidden
side effect was the RNA--ATAC graph seed.  Moving all edges into one graph
reversed that final CUDA seed.  It also registered the two ATAC--ADT directions
in reverse optimizer order.

These classes make both historical side effects explicit while retaining one
graph-owned translator set and no separate bridge module.  The construction-
compatible class keeps the clean objective.  The replay-compatible class also
preserves the historical zero-weight stochastic calls and is diagnostic only.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from scmolt.trackc_atacgraph import (
    TrackCATACCenteredMIDASCycle,
)
from scmolt.trackc_atacgraph_rngcompat import (
    TrackCATACCenteredRNGCompatMIDASCycle,
)


_HISTORICAL_TRANSLATOR_REGISTRATION_ORDER = (
    "rna__atac",
    "atac__rna",
    "atac__adt",
    "adt__atac",
)


def _install_historical_construction_compatibility(model: Any) -> None:
    translators = model.transport.translators
    if set(translators) != set(_HISTORICAL_TRANSLATOR_REGISTRATION_ORDER):
        raise RuntimeError("Integrated translator keys drifted")
    model.transport.translators = nn.ModuleDict(
        {
            key: translators[key]
            for key in _HISTORICAL_TRANSLATOR_REGISTRATION_ORDER
        }
    )
    # Match the delegated constructor's final CUDA-only side effect without
    # changing the caller's CPU RNG state.  Slurm exposes only the task-owned
    # device, but manual_seed_all is retained for exact torch.manual_seed
    # semantics if more than one device is intentionally visible.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(model.config.seed) + 300_003)


class TrackCATACCenteredConstructionCompatMIDASCycle(
    TrackCATACCenteredMIDASCycle
):
    """Clean integrated objective with historical construction side effects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _install_historical_construction_compatibility(self)

    def anchor_strategy_receipt(self) -> dict[str, Any]:
        receipt = super().anchor_strategy_receipt()
        receipt.update(
            {
                "historical_constructor_cuda_seed_compatible": True,
                "historical_main_parameter_order_compatible": True,
                "historical_zero_weight_call_schedule_compatible": False,
                "publication_model": True,
            }
        )
        return receipt


class TrackCATACCenteredReplayCompatMIDASCycle(
    TrackCATACCenteredRNGCompatMIDASCycle
):
    """Fully replay-compatible integrated graph; diagnostic use only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _install_historical_construction_compatibility(self)

    def anchor_strategy_receipt(self) -> dict[str, Any]:
        receipt = super().anchor_strategy_receipt()
        receipt.update(
            {
                "historical_constructor_cuda_seed_compatible": True,
                "historical_main_parameter_order_compatible": True,
                "historical_zero_weight_call_schedule_compatible": True,
                "publication_model": False,
            }
        )
        return receipt


__all__ = [
    "TrackCATACCenteredConstructionCompatMIDASCycle",
    "TrackCATACCenteredReplayCompatMIDASCycle",
]
