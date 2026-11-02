"""ATAC-centered Track C graph with RNA and ADT edges owned in one module.

This is the publication-oriented refactor of the delegated v2 implementation.
The graph itself owns all four biological translators::

    RNA <-> ATAC <-> ADT

There is no separately registered ATAC--ADT bridge module.  The selected
minimal objective retains paired ATAC--ADT posterior consistency and paired
cross-decoding while relying on the common Track C cycle/identity objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from scmolt.trackc.config import MODALITIES, TrackCConfig
from scmolt.trackc.gaussian import (
    GaussianPosterior,
    gaussian_poe,
    symmetric_kl,
)
from scmolt.trackc.model import (
    TrackCForward,
    TrackCLosses,
    TrackCMIDASCycle,
)
from scmolt.trackc.modules import (
    DenseMLP,
    GaussianResidualTranslator,
)


def _direction(source: str, target: str) -> str:
    return f"{source}__{target}"


@dataclass(frozen=True)
class ATACADTEdgeConfig:
    """Weights retained specifically for the integrated ATAC--ADT edge."""

    paired_consistency_weight: float = 5.0
    cross_decode_weight: float = 1.0
    seed_offset: int = 500_003

    def __post_init__(self) -> None:
        for name in ("paired_consistency_weight", "cross_decode_weight"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite non-negative number") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, value)
        if int(self.seed_offset) < 0:
            raise ValueError("seed_offset must be non-negative")
        object.__setattr__(self, "seed_offset", int(self.seed_offset))

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "paired_consistency_weight": self.paired_consistency_weight,
            "cross_decode_weight": self.cross_decode_weight,
            "seed_offset": self.seed_offset,
        }


class ATACCenteredTransportGraph(nn.Module):
    """One ATAC-centered graph that owns every biological transport edge."""

    ANCHOR = "atac"
    DIRECTIONS = (
        ("rna", "atac"),
        ("atac", "rna"),
        ("adt", "atac"),
        ("atac", "adt"),
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
                "ATAC-centered transport requires RNA, ATAC, and ADT; "
                f"observed={requested}, unknown={sorted(unknown)}"
            )
        if config.anchor != self.ANCHOR:
            raise ValueError("ATAC-centered transport requires config.anchor='atac'")
        self.modalities = requested

        # Reproduce the delegated v2 RNA--ATAC graph initialization exactly.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 300_003)
            rna_atac = {
                _direction("rna", "atac"): GaussianResidualTranslator(config),
                _direction("atac", "rna"): GaussianResidualTranslator(config),
            }
            discriminators = {
                modality: DenseMLP(
                    (config.dim_c, *config.domain_discriminator_hidden, 1),
                    dropout=config.dropout,
                )
                for modality in self.modalities
            }

        # Reproduce the delegated v2 ATAC--ADT bridge initialization exactly,
        # while making these translators graph-owned parameters.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + edge_config.seed_offset)
            atac_adt = {
                _direction(source, target): GaussianResidualTranslator(config)
                for source, target in self._ATAC_ADT_INITIALIZATION_ORDER
            }

        self.translators = nn.ModuleDict(
            {
                _direction("rna", "atac"): rna_atac[_direction("rna", "atac")],
                _direction("atac", "rna"): rna_atac[_direction("atac", "rna")],
                _direction("adt", "atac"): atac_adt[_direction("adt", "atac")],
                _direction("atac", "adt"): atac_adt[_direction("atac", "adt")],
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
            raise ValueError(
                f"Unsupported ATAC-centered direction {source}->{target}"
            )
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

    def domain_logits(self, values: Tensor, target: str) -> Tensor:
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


@dataclass(frozen=True)
class TrackCATACCenteredLosses:
    """Common Track C losses plus the retained ATAC--ADT edge terms."""

    base: TrackCLosses
    main_loss: Tensor
    atac_adt_edge_total_weighted: Tensor
    atac_adt_pair_raw: Tensor
    atac_adt_pair_weighted: Tensor
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


class TrackCATACCenteredMIDASCycle(TrackCMIDASCycle):
    """Tri-modal Track C whose complete transport graph is ATAC-centered."""

    EXECUTABLE_ANCHOR = "atac"

    def __init__(
        self,
        *args: Any,
        config: TrackCConfig,
        edge_config: ATACADTEdgeConfig,
        **kwargs: Any,
    ) -> None:
        if config.anchor != "rna":
            raise ValueError("ATAC-centered model expects the audited RNA bootstrap")
        super().__init__(*args, config=config, **kwargs)
        object.__setattr__(self.config, "anchor", self.EXECUTABLE_ANCHOR)
        self.edge_config = edge_config
        self.transport = ATACCenteredTransportGraph(
            self.config,
            edge_config=edge_config,
            modalities=self.modalities,
        )
        if self.config.anchor != self.EXECUTABLE_ANCHOR:
            raise RuntimeError("Executable Track C anchor did not switch to ATAC")
        if set(self.transport.translators) != {
            _direction(source, target) for source, target in self.transport.DIRECTIONS
        }:
            raise RuntimeError("Integrated graph translator directions drifted")
        if hasattr(self, "atac_adt_bridge"):
            raise RuntimeError("Integrated graph unexpectedly retained a bridge module")

    def anchor_strategy_receipt(self) -> dict[str, Any]:
        return {
            "executable_anchor": self.config.anchor,
            "transport_directions": [list(value) for value in self.transport.DIRECTIONS],
            "rna_atac_parameterization": "integrated_trackc_transport",
            "atac_adt_parameterization": "integrated_trackc_transport",
            "atac_adt_separately_registered_bridge": False,
            "native_scmidas_modules_modified": False,
            "edge_objective": self.edge_config.to_dict(),
            "extra_atac_adt_cycle_weight": 0.0,
            "extra_atac_adt_identity_weight": 0.0,
            "legacy_config_field_alias": {
                "paired_rna_anchor_consistency_weight": (
                    "paired_atac_anchor_consistency_weight"
                )
            },
        }

    def _edge_latent(
        self,
        output: TrackCForward,
        canonical_biological: GaussianPosterior,
        *,
        source: str,
        sample: bool,
    ) -> Tensor:
        c_experts = [canonical_biological]
        u_experts = [output.native_technical[source]]
        if output.batch_expert is not None:
            batch_c, batch_u = output.batch_expert.split(
                (self.config.dim_c, self.config.dim_u)
            )
            c_experts.append(batch_c)
            u_experts.append(batch_u)
        joint = GaussianPosterior.cat((gaussian_poe(c_experts), gaussian_poe(u_experts)))
        return joint.rsample() if sample else joint.mean

    def _edge_target_reconstruction(
        self,
        output: TrackCForward,
        *,
        source: str,
        target: str,
        sample: bool,
    ) -> Tensor:
        translated_target = self.transport.translate(
            output.native_biological[source], source, target
        )
        canonical = self.transport.to_anchor(translated_target, target)
        latent = self._edge_latent(
            output,
            canonical,
            source=source,
            sample=sample,
        )
        return self.native_vae.decoder(latent)[target]

    def decode_atac_to_adt(
        self,
        output: TrackCForward,
        *,
        sample: bool = False,
    ) -> Tensor:
        if not output.transport_active:
            raise ValueError("ATAC-to-ADT decoding requires active transport")
        if "atac" not in output.native_biological:
            raise ValueError("ATAC-to-ADT decoding requires observed ATAC")
        return self._edge_target_reconstruction(
            output,
            source="atac",
            target="adt",
            sample=sample,
        )

    def _edge_cross_decode_one(
        self,
        output: TrackCForward,
        *,
        source: str,
        target: str,
    ) -> Tensor:
        reconstruction = self._edge_target_reconstruction(
            output,
            source=source,
            target=target,
            sample=True,
        )
        target_values = output.inputs[target]
        raw_mask = output.event_masks[target]
        if target == "atac":
            values = F.binary_cross_entropy(
                reconstruction,
                target_values,
                reduction="none",
            )
        elif target == "adt":
            values = F.poisson_nll_loss(
                reconstruction,
                target_values,
                log_input=True,
                full=True,
                reduction="none",
            )
        else:
            raise ValueError(f"Unsupported edge reconstruction target {target!r}")
        return (values * raw_mask).sum() / target_values.shape[0]

    def compute_main_loss(
        self,
        output: TrackCForward,
        *,
        cycle_scale: float = 1.0,
    ) -> TrackCATACCenteredLosses:
        base = super().compute_main_loss(output, cycle_scale=cycle_scale)
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
            pair_raw = atac_to_adt_cross = adt_to_atac_cross = cross_raw = zero
            active = torch.zeros((), dtype=torch.long, device=output.latent.device)

        pair_weighted = (
            scale * self.edge_config.paired_consistency_weight * pair_raw
        )
        cross_weighted = scale * self.edge_config.cross_decode_weight * cross_raw
        edge_total = pair_weighted + cross_weighted
        return TrackCATACCenteredLosses(
            base=base,
            main_loss=base.main_loss + edge_total,
            atac_adt_edge_total_weighted=edge_total,
            atac_adt_pair_raw=pair_raw,
            atac_adt_pair_weighted=pair_weighted,
            atac_to_adt_cross_decode_raw=atac_to_adt_cross,
            adt_to_atac_cross_decode_raw=adt_to_atac_cross,
            atac_adt_cross_decode_raw=cross_raw,
            atac_adt_cross_decode_weighted=cross_weighted,
            atac_adt_active_directions=active,
        )


def migrate_delegated_atac_anchor_state_dict(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Move delegated v2 ATAC--ADT keys into the integrated graph namespace."""

    prefix = "atac_adt_bridge.translators."
    migrated: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            target = "transport.translators." + key[len(prefix) :]
        else:
            target = key
        if target in migrated:
            raise RuntimeError(f"State migration produced duplicate key {target}")
        migrated[target] = value
    return migrated


__all__ = [
    "ATACADTEdgeConfig",
    "ATACCenteredTransportGraph",
    "TrackCATACCenteredLosses",
    "TrackCATACCenteredMIDASCycle",
    "migrate_delegated_atac_anchor_state_dict",
]
