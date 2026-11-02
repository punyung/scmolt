"""Integrated graph diagnostic that preserves the delegated-minimal RNG calls.

This class is not the clean publication model.  It exists to isolate graph
ownership from stochastic call scheduling.  The historical delegated-minimal
implementation evaluated ATAC--ADT cycle and identity expressions even when
their scalar weights were zero.  Translator dropout in those dead expressions
advanced the training RNG before paired cross-decoding.  Here the translators
remain owned by the single ATAC-centered graph, while those zero-weight calls
are reproduced in the same order for a controlled replication diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor

from scmolt.trackc.gaussian import symmetric_kl
from scmolt.trackc.model import (
    TrackCForward,
    TrackCLosses,
    TrackCMIDASCycle,
)
from scmolt.trackc.modules import identity_loss
from scmolt.trackc_atacgraph import (
    TrackCATACCenteredMIDASCycle,
)


@dataclass(frozen=True)
class TrackCATACCenteredRNGCompatLosses:
    """Integrated edge losses with explicitly discarded RNG-padding terms."""

    base: TrackCLosses
    main_loss: Tensor
    atac_adt_edge_total_weighted: Tensor
    atac_adt_pair_raw: Tensor
    atac_adt_pair_weighted: Tensor
    atac_adt_discarded_cycle_raw: Tensor
    atac_adt_discarded_cycle_weighted: Tensor
    atac_adt_discarded_identity_raw: Tensor
    atac_adt_discarded_identity_weighted: Tensor
    atac_to_adt_cross_decode_raw: Tensor
    adt_to_atac_cross_decode_raw: Tensor
    atac_adt_cross_decode_raw: Tensor
    atac_adt_cross_decode_weighted: Tensor
    atac_adt_active_directions: Tensor

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def as_dict(self) -> Dict[str, Tensor]:
        result = dict(self.base.as_dict())
        result.update(
            {
                "main_loss": self.main_loss,
                "atac_adt_edge_total_weighted": self.atac_adt_edge_total_weighted,
                "atac_adt_pair_raw": self.atac_adt_pair_raw,
                "atac_adt_pair_weighted": self.atac_adt_pair_weighted,
                "atac_adt_discarded_cycle_raw": (
                    self.atac_adt_discarded_cycle_raw
                ),
                "atac_adt_discarded_cycle_weighted": (
                    self.atac_adt_discarded_cycle_weighted
                ),
                "atac_adt_discarded_identity_raw": (
                    self.atac_adt_discarded_identity_raw
                ),
                "atac_adt_discarded_identity_weighted": (
                    self.atac_adt_discarded_identity_weighted
                ),
                "atac_to_adt_cross_decode_raw": self.atac_to_adt_cross_decode_raw,
                "adt_to_atac_cross_decode_raw": self.adt_to_atac_cross_decode_raw,
                "atac_adt_cross_decode_raw": self.atac_adt_cross_decode_raw,
                "atac_adt_cross_decode_weighted": (
                    self.atac_adt_cross_decode_weighted
                ),
                "atac_adt_active_directions": self.atac_adt_active_directions,
            }
        )
        return result


class TrackCATACCenteredRNGCompatMIDASCycle(TrackCATACCenteredMIDASCycle):
    """Integrated graph with the historical zero-weight stochastic call order."""

    def anchor_strategy_receipt(self) -> dict[str, Any]:
        receipt = super().anchor_strategy_receipt()
        receipt.update(
            {
                "rng_compatibility_diagnostic": True,
                "discarded_cycle_identity_calls_executed": True,
                "discarded_cycle_identity_scalar_weights": {
                    "cycle": 0.0,
                    "identity": 0.0,
                },
                "publication_model": False,
            }
        )
        return receipt

    def compute_main_loss(
        self,
        output: TrackCForward,
        *,
        cycle_scale: float = 1.0,
    ) -> TrackCATACCenteredRNGCompatLosses:
        # Deliberately bypass the clean integrated override.  Everything below
        # mirrors the delegated-minimal call order exactly while using graph-
        # owned ATAC--ADT translators.
        base = TrackCMIDASCycle.compute_main_loss(
            self, output, cycle_scale=cycle_scale
        )
        zero = output.latent.sum() * 0.0
        scale = output.latent.new_tensor(float(cycle_scale))
        paired = (
            output.transport_active
            and "atac" in output.native_biological
            and "adt" in output.native_biological
        )
        if paired:
            atac = output.native_biological["atac"]
            adt = output.native_biological["adt"]
            atac_to_adt = self.transport.translate(atac, "atac", "adt")
            adt_to_atac = self.transport.translate(adt, "adt", "atac")
            pair_raw = 0.5 * (
                symmetric_kl(atac_to_adt, adt.detach())
                + symmetric_kl(adt_to_atac, atac.detach())
            )
            cycle_raw = 0.5 * (
                symmetric_kl(
                    atac,
                    self.transport.cycle(atac, "atac", "adt"),
                )
                + symmetric_kl(
                    adt,
                    self.transport.cycle(adt, "adt", "atac"),
                )
            )
            identity_raw = 0.5 * (
                identity_loss(self.transport, adt, "atac", "adt")
                + identity_loss(self.transport, atac, "adt", "atac")
            )
            atac_to_adt_cross = self._edge_cross_decode_one(
                output,
                source="atac",
                target="adt",
            )
            adt_to_atac_cross = self._edge_cross_decode_one(
                output,
                source="adt",
                target="atac",
            )
            cross_raw = 0.5 * (atac_to_adt_cross + adt_to_atac_cross)
            active = torch.tensor(2, dtype=torch.long, device=output.latent.device)
        else:
            pair_raw = cycle_raw = identity_raw = zero
            atac_to_adt_cross = adt_to_atac_cross = cross_raw = zero
            active = torch.zeros((), dtype=torch.long, device=output.latent.device)

        pair_weighted = (
            scale * self.edge_config.paired_consistency_weight * pair_raw
        )
        cycle_weighted = scale * 0.0 * cycle_raw
        identity_weighted = scale * 0.0 * identity_raw
        cross_weighted = scale * self.edge_config.cross_decode_weight * cross_raw
        edge_total = (
            pair_weighted + cycle_weighted + identity_weighted + cross_weighted
        )
        return TrackCATACCenteredRNGCompatLosses(
            base=base,
            main_loss=base.main_loss + edge_total,
            atac_adt_edge_total_weighted=edge_total,
            atac_adt_pair_raw=pair_raw,
            atac_adt_pair_weighted=pair_weighted,
            atac_adt_discarded_cycle_raw=cycle_raw,
            atac_adt_discarded_cycle_weighted=cycle_weighted,
            atac_adt_discarded_identity_raw=identity_raw,
            atac_adt_discarded_identity_weighted=identity_weighted,
            atac_to_adt_cross_decode_raw=atac_to_adt_cross,
            adt_to_atac_cross_decode_raw=adt_to_atac_cross,
            atac_adt_cross_decode_raw=cross_raw,
            atac_adt_cross_decode_weighted=cross_weighted,
            atac_adt_active_directions=active,
        )


__all__ = [
    "TrackCATACCenteredRNGCompatLosses",
    "TrackCATACCenteredRNGCompatMIDASCycle",
]
