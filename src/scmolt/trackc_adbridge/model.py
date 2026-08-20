"""Bidirectional ATAC--ADT bridge layered on the frozen Track C model.

The parent Track C graph remains RNA-centred and byte-for-byte unchanged.  This
module adds a separately weighted, paired-data bridge trained only when ATAC
and ADT are genuinely co-observed.  For ATAC-only ADT imputation, the bridge is
also part of the actual prediction path::

    ATAC posterior -> ATAC-to-ADT -> ADT-to-RNA canonical -> native ADT decoder

No labels, feature maps, or held-out modality truth enter these losses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..trackc.gaussian import (
    GaussianPosterior,
    gaussian_poe,
    symmetric_kl,
)
from ..trackc.model import (
    TrackCForward,
    TrackCLosses,
    TrackCMIDASCycle,
)
from ..trackc.modules import (
    GaussianResidualTranslator,
    identity_loss,
)


@dataclass(frozen=True)
class ATACADTBridgeConfig:
    """Explicit weights for the new paired ATAC--ADT transport objective."""

    paired_consistency_weight: float = 5.0
    cycle_weight: float = 30.0
    identity_weight: float = 15.0
    cross_decode_weight: float = 1.0
    seed_offset: int = 500_003

    def __post_init__(self) -> None:
        for name in (
            "paired_consistency_weight",
            "cycle_weight",
            "identity_weight",
            "cross_decode_weight",
        ):
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
            "cycle_weight": self.cycle_weight,
            "identity_weight": self.identity_weight,
            "cross_decode_weight": self.cross_decode_weight,
            "seed_offset": self.seed_offset,
        }


class ATACADTGaussianBridge(nn.Module):
    """Two residual Gaussian translators with an explicit reverse direction."""

    DIRECTIONS = (("atac", "adt"), ("adt", "atac"))

    def __init__(self, trackc_config: Any) -> None:
        super().__init__()
        self.translators = nn.ModuleDict(
            {
                f"{source}__{target}": GaussianResidualTranslator(trackc_config)
                for source, target in self.DIRECTIONS
            }
        )

    def translate(
        self,
        posterior: GaussianPosterior,
        source: str,
        target: str,
    ) -> GaussianPosterior:
        if source == target:
            return posterior
        key = f"{source}__{target}"
        if key not in self.translators:
            raise ValueError(f"Unsupported ATAC--ADT bridge direction {source}->{target}")
        return self.translators[key](posterior)

    def cycle(
        self,
        posterior: GaussianPosterior,
        source: str,
        target: str,
    ) -> GaussianPosterior:
        return self.translate(self.translate(posterior, source, target), target, source)


@dataclass(frozen=True)
class TrackCADBridgeLosses:
    """Parent Track C losses plus separately attributable bridge terms."""

    base: TrackCLosses
    main_loss: Tensor
    atac_adt_bridge_total_weighted: Tensor
    atac_adt_pair_raw: Tensor
    atac_adt_pair_weighted: Tensor
    atac_adt_cycle_raw: Tensor
    atac_adt_cycle_weighted: Tensor
    atac_adt_identity_raw: Tensor
    atac_adt_identity_weighted: Tensor
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
                "atac_adt_bridge_total_weighted": self.atac_adt_bridge_total_weighted,
                "atac_adt_pair_raw": self.atac_adt_pair_raw,
                "atac_adt_pair_weighted": self.atac_adt_pair_weighted,
                "atac_adt_cycle_raw": self.atac_adt_cycle_raw,
                "atac_adt_cycle_weighted": self.atac_adt_cycle_weighted,
                "atac_adt_identity_raw": self.atac_adt_identity_raw,
                "atac_adt_identity_weighted": self.atac_adt_identity_weighted,
                "atac_to_adt_cross_decode_raw": self.atac_to_adt_cross_decode_raw,
                "adt_to_atac_cross_decode_raw": self.adt_to_atac_cross_decode_raw,
                "atac_adt_cross_decode_raw": self.atac_adt_cross_decode_raw,
                "atac_adt_cross_decode_weighted": self.atac_adt_cross_decode_weighted,
                "atac_adt_active_directions": self.atac_adt_active_directions,
            }
        )
        return result


class TrackCADBridgeMIDASCycle(TrackCMIDASCycle):
    """Track C with an independently weighted, bidirectional ATAC--ADT bridge."""

    def __init__(self, *args: Any, bridge_config: ATACADTBridgeConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not {"rna", "atac", "adt"}.issubset(self.modalities):
            raise ValueError("ATAC--ADT bridge requires the full tri-modal model")
        self.bridge_config = bridge_config
        # Do not advance the global RNG consumed by native scMIDAS or the parent
        # Track C graph.  The bridge has its own deterministic initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed + self.bridge_config.seed_offset)
            self.atac_adt_bridge = ATACADTGaussianBridge(self.config)

    def _bridge_latent(
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

    def _bridge_target_reconstruction(
        self,
        output: TrackCForward,
        *,
        source: str,
        target: str,
        sample: bool,
    ) -> Tensor:
        translated_target = self.atac_adt_bridge.translate(
            output.native_biological[source], source, target
        )
        canonical = self.transport.to_anchor(translated_target, target)
        latent = self._bridge_latent(
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
        """Return native ADT decoder parameters through the deployed bridge route."""

        if not output.transport_active:
            raise ValueError("ATAC--ADT bridge decoding requires active transport")
        if "atac" not in output.native_biological:
            raise ValueError("ATAC--ADT bridge decoding requires observed ATAC")
        return self._bridge_target_reconstruction(
            output,
            source="atac",
            target="adt",
            sample=sample,
        )

    def _bridge_cross_decode_one(
        self,
        output: TrackCForward,
        *,
        source: str,
        target: str,
    ) -> Tensor:
        reconstruction = self._bridge_target_reconstruction(
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
            raise ValueError(f"Unsupported bridge reconstruction target {target!r}")
        return (values * raw_mask).sum() / target_values.shape[0]

    def compute_main_loss(
        self,
        output: TrackCForward,
        *,
        cycle_scale: float = 1.0,
    ) -> TrackCADBridgeLosses:
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
            atac_to_adt = self.atac_adt_bridge.translate(atac, "atac", "adt")
            adt_to_atac = self.atac_adt_bridge.translate(adt, "adt", "atac")
            pair_raw = 0.5 * (
                symmetric_kl(atac_to_adt, adt.detach())
                + symmetric_kl(adt_to_atac, atac.detach())
            )
            cycle_raw = 0.5 * (
                symmetric_kl(
                    atac,
                    self.atac_adt_bridge.cycle(atac, "atac", "adt"),
                )
                + symmetric_kl(
                    adt,
                    self.atac_adt_bridge.cycle(adt, "adt", "atac"),
                )
            )
            identity_raw = 0.5 * (
                identity_loss(self.atac_adt_bridge, adt, "atac", "adt")
                + identity_loss(self.atac_adt_bridge, atac, "adt", "atac")
            )
            atac_to_adt_cross = self._bridge_cross_decode_one(
                output,
                source="atac",
                target="adt",
            )
            adt_to_atac_cross = self._bridge_cross_decode_one(
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
            scale * self.bridge_config.paired_consistency_weight * pair_raw
        )
        cycle_weighted = scale * self.bridge_config.cycle_weight * cycle_raw
        identity_weighted = scale * self.bridge_config.identity_weight * identity_raw
        cross_weighted = (
            scale * self.bridge_config.cross_decode_weight * cross_raw
        )
        bridge_total = (
            pair_weighted + cycle_weighted + identity_weighted + cross_weighted
        )
        return TrackCADBridgeLosses(
            base=base,
            main_loss=base.main_loss + bridge_total,
            atac_adt_bridge_total_weighted=bridge_total,
            atac_adt_pair_raw=pair_raw,
            atac_adt_pair_weighted=pair_weighted,
            atac_adt_cycle_raw=cycle_raw,
            atac_adt_cycle_weighted=cycle_weighted,
            atac_adt_identity_raw=identity_raw,
            atac_adt_identity_weighted=identity_weighted,
            atac_to_adt_cross_decode_raw=atac_to_adt_cross,
            adt_to_atac_cross_decode_raw=adt_to_atac_cross,
            atac_adt_cross_decode_raw=cross_raw,
            atac_adt_cross_decode_weighted=cross_weighted,
            atac_adt_active_directions=active,
        )


__all__ = [
    "ATACADTBridgeConfig",
    "ATACADTGaussianBridge",
    "TrackCADBridgeLosses",
    "TrackCADBridgeMIDASCycle",
]
