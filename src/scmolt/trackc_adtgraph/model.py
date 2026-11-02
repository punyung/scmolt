"""ADT-centered Track C graph for an ADT-observed cross-batch hub.

The canonical biological graph is::

    RNA <-> ADT <-> ATAC

All four directed translators are owned by one graph.  The ATAC--ADT edge
retains the paired posterior-consistency and paired cross-decoding terms used
by the clean ATAC-centered graph, but there is no separately registered bridge
module.  Native scMIDAS encoders, decoders, likelihoods, and the full-z batch
expert are unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from scmolt.trackc.config import MODALITIES, TrackCConfig
from scmolt.trackc.gaussian import GaussianPosterior
from scmolt.trackc.model import TrackCMIDASCycle
from scmolt.trackc.modules import (
    DenseMLP,
    GaussianResidualTranslator,
    GaussianTransportGraph,
)
from scmolt.trackc_atacgraph import (
    ATACADTEdgeConfig,
    TrackCATACCenteredMIDASCycle,
)


def _direction(source: str, target: str) -> str:
    return f"{source}__{target}"


class ADTCenteredTransportGraph(nn.Module):
    """One graph whose canonical biological space is ADT."""

    ANCHOR = "adt"
    DIRECTIONS = (
        ("rna", "adt"),
        ("adt", "rna"),
        ("atac", "adt"),
        ("adt", "atac"),
    )
    _ATAC_ADT_INITIALIZATION_ORDER = (
        ("atac", "adt"),
        ("adt", "atac"),
    )

    def __init__(
        self,
        config: TrackCConfig,
        *,
        edge_config: ATACADTEdgeConfig,
        modalities: Sequence[str] = MODALITIES,
    ) -> None:
        super().__init__()
        requested = tuple(name for name in MODALITIES if name in set(modalities))
        unknown = set(modalities) - set(MODALITIES)
        if requested != MODALITIES or unknown:
            raise ValueError(
                "ADT-centered transport requires RNA, ATAC, and ADT; "
                f"observed={requested}, unknown={sorted(unknown)}"
            )
        if config.anchor != self.ANCHOR:
            raise ValueError("ADT-centered transport requires config.anchor='adt'")
        self.modalities = requested

        # Reproduce the historical RNA-star initialization before selecting
        # its RNA--ADT pair.  Constructing the unused RNA--ATAC pair here keeps
        # the selected RNA--ADT initial tensors equal to the frozen baseline.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 300_003)
            historical_anchor = {
                _direction(source, target): GaussianResidualTranslator(config)
                for source, target in GaussianTransportGraph.FULL_DIRECTIONS
            }
            discriminators = {
                modality: DenseMLP(
                    (config.dim_c, *config.domain_discriminator_hidden, 1),
                    dropout=config.dropout,
                )
                for modality in self.modalities
            }

        # Reproduce the historical ATAC--ADT bridge initialization while
        # moving both directions into the graph-owned namespace.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + edge_config.seed_offset)
            atac_adt = {
                _direction(source, target): GaussianResidualTranslator(config)
                for source, target in self._ATAC_ADT_INITIALIZATION_ORDER
            }

        self.translators = nn.ModuleDict(
            {
                _direction("rna", "adt"): historical_anchor[
                    _direction("rna", "adt")
                ],
                _direction("adt", "rna"): historical_anchor[
                    _direction("adt", "rna")
                ],
                _direction("atac", "adt"): atac_adt[
                    _direction("atac", "adt")
                ],
                _direction("adt", "atac"): atac_adt[
                    _direction("adt", "atac")
                ],
            }
        )
        self.discriminators = nn.ModuleDict(discriminators)

    def translate(
        self,
        posterior: GaussianPosterior,
        source: str,
        target: str,
    ) -> GaussianPosterior:
        if source == target:
            return posterior
        key = _direction(source, target)
        if key not in self.translators:
            raise ValueError(f"Unsupported ADT-centered direction {source}->{target}")
        return self.translators[key](posterior)

    def to_anchor(
        self,
        posterior: GaussianPosterior,
        source: str,
    ) -> GaussianPosterior:
        return (
            posterior
            if source == self.ANCHOR
            else self.translate(posterior, source, self.ANCHOR)
        )

    def domain_logits(self, values: torch.Tensor, target: str) -> torch.Tensor:
        return self.discriminators[target](values)

    def cycle(
        self,
        posterior: GaussianPosterior,
        source: str,
        target: str,
    ) -> GaussianPosterior:
        return self.translate(
            self.translate(posterior, source, target), target, source
        )


class TrackCADTCenteredMIDASCycle(TrackCATACCenteredMIDASCycle):
    """Tri-modal Track C whose canonical biological anchor is ADT.

    The inherited edge-specific implementation remains valid because it acts
    on the ATAC--ADT pair and canonicalizes through ``transport.to_anchor``.
    ``TrackCMIDASCycle.__init__`` is called directly to avoid constructing and
    discarding an ATAC-centered graph before installing the ADT graph.
    """

    EXECUTABLE_ANCHOR = "adt"

    def __init__(
        self,
        *args: Any,
        config: TrackCConfig,
        edge_config: ATACADTEdgeConfig,
        **kwargs: Any,
    ) -> None:
        if config.anchor != "rna":
            raise ValueError("ADT-centered model expects the audited RNA bootstrap")
        TrackCMIDASCycle.__init__(self, *args, config=config, **kwargs)
        object.__setattr__(self.config, "anchor", self.EXECUTABLE_ANCHOR)
        self.edge_config = edge_config
        self.transport = ADTCenteredTransportGraph(
            self.config,
            edge_config=edge_config,
            modalities=self.modalities,
        )
        if self.config.anchor != self.EXECUTABLE_ANCHOR:
            raise RuntimeError("Executable Track C anchor did not switch to ADT")
        expected = {
            _direction(source, target) for source, target in self.transport.DIRECTIONS
        }
        if set(self.transport.translators) != expected:
            raise RuntimeError("Integrated ADT graph translator directions drifted")
        if hasattr(self, "atac_adt_bridge"):
            raise RuntimeError("ADT graph unexpectedly retained a bridge module")

    def anchor_strategy_receipt(self) -> dict[str, Any]:
        return {
            "executable_anchor": self.config.anchor,
            "canonical_anchor": self.transport.ANCHOR,
            "transport_directions": [
                list(value) for value in self.transport.DIRECTIONS
            ],
            "rna_adt_parameterization": "integrated_trackc_transport",
            "atac_adt_parameterization": "integrated_trackc_transport",
            "rna_atac_parameterization": "absent_from_adt_star",
            "atac_adt_separately_registered_bridge": False,
            "native_scmidas_modules_modified": False,
            "batch_expert_distribution": "diagonal_gaussian",
            "batch_expert_scope": "native_scmidas_full_z_c_plus_u",
            "paired_anchor_teacher": "adt",
            "legacy_config_field_alias": {
                "paired_rna_anchor_consistency_weight": (
                    "paired_adt_anchor_consistency_weight"
                )
            },
            "edge_objective": self.edge_config.to_dict(),
            "extra_atac_adt_cycle_weight": 0.0,
            "extra_atac_adt_identity_weight": 0.0,
            "publication_model": False,
            "screening_candidate": True,
        }


__all__ = ["ADTCenteredTransportGraph", "TrackCADTCenteredMIDASCycle"]
