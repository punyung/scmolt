"""Typed, immutable configuration for isolated Track C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, Mapping, Optional, Tuple


MODALITIES: Tuple[str, ...] = ("rna", "atac", "adt")
SUPPORTED_TOPOLOGIES: Tuple[str, ...] = ("full", "paired-abc", "diagonal")


class TrackCVariant(str, Enum):
    """The four locked causal variants in the Track C protocol."""

    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"

    @classmethod
    def coerce(cls, value: "TrackCVariant | str") -> "TrackCVariant":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ValueError("variant must be one of C0, C1, C2, or C3") from exc


@dataclass(frozen=True)
class TrackCConfig:
    """Faithful scMIDAS-0.1.13 defaults plus Track C transport controls.

    ``lam_alignment=None`` selects the protocol value (50 for C0 and zero for
    C1-C3).  Passing a contradictory explicit value is rejected instead of
    silently changing the requested causal variant.
    """

    variant: TrackCVariant | str = TrackCVariant.C1
    topology: str = "full"
    seed: int = 1234
    epochs: int = 300
    anchor: str = "rna"

    dim_c: int = 32
    dim_u: int = 2
    encoder_hidden: Tuple[int, int] = (1024, 128)
    decoder_hidden: Tuple[int, int] = (128, 1024)
    batch_hidden: Tuple[int, int] = (16, 16)
    batch_discriminator_hidden: Tuple[int, int] = (128, 64)
    translator_hidden: Tuple[int, int] = (128, 128)
    domain_discriminator_hidden: Tuple[int, int] = (128, 128)
    atac_adapter_hidden: Tuple[int, int] = (128, 32)
    dropout: float = 0.2
    s_drop_rate: float = 0.1
    # Used only by standalone component tests; the executable model wraps the
    # native scMIDAS batch modules, whose internal dropout is `drop=0.2`.
    batch_dropout: float = 0.1
    logvar_min: float = -8.0
    logvar_max: float = 8.0

    lam_kld: float = 1.0
    lam_kld_c: float = 1.0
    lam_kld_u: float = 5.0
    lam_recon: float = 1.0
    lam_recon_s: float = 1000.0
    lam_alignment: Optional[float] = None
    lam_adv: float = 1.0
    lam_dsc: float = 30.0
    translated_batch_ce_scale: float = 1.0
    # Optional BMMC missing-modality specialization.  A value above one
    # strengthens only translated/joint batch confusion for minibatches where
    # RNA is physically absent; paired/native modality terms retain D40's
    # original weight.  The default is an exact no-op for historical waves.
    missingness_batch_ce_scale: float = 1.0
    # On genuinely paired RNA+ATAC rows, align ATAC->RNA with the native RNA
    # posterior while treating RNA as a stop-gradient teacher.  No paired RNA
    # means no loss, so masked RNA truth can never enter this term.
    paired_rna_anchor_consistency_weight: float = 0.0
    # Runtime-owned schema switch. Legacy waves leave this false so their
    # history/diagnostic key sets remain byte-for-byte compatible.
    emit_translated_batch_ce_metrics: bool = False
    lam_cycle: float = 10.0
    lam_identity: float = 1.0
    lam_cross_decode: float = 1.0
    lam_domain: float = 1.0

    n_iter_disc: int = 3
    learning_rate: float = 1e-4
    batch_discriminator_learning_rate: float = 1e-4
    domain_discriminator_learning_rate: float = 1e-4
    # scMIDAS invokes torch.optim.AdamW without overriding its default.
    weight_decay: float = 0.01
    warmup_epochs: int = 30
    ramp_epochs: int = 30
    replay_capacity: int = 4096

    def __post_init__(self) -> None:
        variant = TrackCVariant.coerce(self.variant)
        topology = str(self.topology).strip().lower()
        anchor = str(self.anchor).strip().lower()
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "anchor", anchor)

        if topology not in SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {SUPPORTED_TOPOLOGIES}; got {topology!r}"
            )
        if anchor != "rna":
            raise ValueError("Track C v1 fixes the real canonical anchor to 'rna'.")

        positive_ints = {
            "epochs": self.epochs,
            "dim_c": self.dim_c,
            "dim_u": self.dim_u,
            "n_iter_disc": self.n_iter_disc,
            "ramp_epochs": self.ramp_epochs,
            "replay_capacity": self.replay_capacity,
        }
        invalid_ints = {key: value for key, value in positive_ints.items() if int(value) <= 0}
        if invalid_ints:
            raise ValueError(f"Track C integer parameters must be positive: {invalid_ints}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.warmup_epochs + self.ramp_epochs > self.epochs:
            raise ValueError("warmup_epochs + ramp_epochs cannot exceed epochs")

        for name, widths in {
            "encoder_hidden": self.encoder_hidden,
            "decoder_hidden": self.decoder_hidden,
            "batch_hidden": self.batch_hidden,
            "batch_discriminator_hidden": self.batch_discriminator_hidden,
            "translator_hidden": self.translator_hidden,
            "domain_discriminator_hidden": self.domain_discriminator_hidden,
            "atac_adapter_hidden": self.atac_adapter_hidden,
        }.items():
            if not widths or any(int(width) <= 0 for width in widths):
                raise ValueError(f"{name} must contain positive widths")

        if (
            not 0.0 <= self.dropout < 1.0
            or not 0.0 <= self.batch_dropout < 1.0
            or not 0.0 <= self.s_drop_rate < 1.0
        ):
            raise ValueError("dropout values must be in [0, 1)")
        if not self.logvar_min < self.logvar_max:
            raise ValueError("logvar_min must be less than logvar_max")

        try:
            translated_batch_ce_scale = float(self.translated_batch_ce_scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "translated_batch_ce_scale must be a finite non-negative number"
            ) from exc
        if (
            not math.isfinite(translated_batch_ce_scale)
            or translated_batch_ce_scale < 0.0
            or translated_batch_ce_scale > 1.0
        ):
            raise ValueError(
                "translated_batch_ce_scale must be a finite number in [0, 1]"
            )
        object.__setattr__(
            self, "translated_batch_ce_scale", translated_batch_ce_scale
        )
        for name in (
            "missingness_batch_ce_scale",
            "paired_rna_anchor_consistency_weight",
        ):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite non-negative number") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, value)
        if type(self.emit_translated_batch_ce_metrics) is not bool:
            raise ValueError("emit_translated_batch_ce_metrics must be boolean")
        if variant not in (TrackCVariant.C2, TrackCVariant.C3):
            if translated_batch_ce_scale != 1.0:
                raise ValueError(
                    "translated_batch_ce_scale is Cycle-only; C0/C1 require 1.0"
                )
            if self.emit_translated_batch_ce_metrics:
                raise ValueError(
                    "translated batch-CE metrics are Cycle-only"
                )
        if variant is not TrackCVariant.C3:
            if self.missingness_batch_ce_scale != 1.0:
                raise ValueError(
                    "missingness_batch_ce_scale is C3-only; non-C3 variants require 1.0"
                )
            if self.paired_rna_anchor_consistency_weight != 0.0:
                raise ValueError(
                    "paired_rna_anchor_consistency_weight is C3-only; "
                    "non-C3 variants require 0.0"
                )

        non_negative = {
            "lam_kld": self.lam_kld,
            "lam_kld_c": self.lam_kld_c,
            "lam_kld_u": self.lam_kld_u,
            "lam_recon": self.lam_recon,
            "lam_recon_s": self.lam_recon_s,
            "lam_adv": self.lam_adv,
            "lam_dsc": self.lam_dsc,
            "translated_batch_ce_scale": self.translated_batch_ce_scale,
            "missingness_batch_ce_scale": self.missingness_batch_ce_scale,
            "paired_rna_anchor_consistency_weight": (
                self.paired_rna_anchor_consistency_weight
            ),
            "lam_cycle": self.lam_cycle,
            "lam_identity": self.lam_identity,
            "lam_cross_decode": self.lam_cross_decode,
            "lam_domain": self.lam_domain,
            "weight_decay": self.weight_decay,
        }
        if self.lam_alignment is not None:
            non_negative["lam_alignment"] = self.lam_alignment
        invalid_weights = {key: value for key, value in non_negative.items() if value < 0.0}
        if invalid_weights:
            raise ValueError(f"Track C weights must be non-negative: {invalid_weights}")
        for name in (
            "learning_rate",
            "batch_discriminator_learning_rate",
            "domain_discriminator_learning_rate",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

        expected_alignment = 50.0 if variant is TrackCVariant.C0 else 0.0
        if self.lam_alignment is not None and float(self.lam_alignment) != expected_alignment:
            raise ValueError(
                f"{variant.value} requires lam_alignment={expected_alignment:g}; "
                f"got {self.lam_alignment:g}"
            )

    @property
    def dim_z(self) -> int:
        return self.dim_c + self.dim_u

    @property
    def uses_cycle(self) -> bool:
        return self.variant in (TrackCVariant.C2, TrackCVariant.C3)

    @property
    def uses_translated_fusion(self) -> bool:
        return self.variant is TrackCVariant.C3

    @property
    def alignment_weight(self) -> float:
        # With no co-observed rows the native MIDAS consistency term is
        # mathematically zero; spelling this out avoids a redundant C0 run.
        if self.topology == "diagonal":
            return 0.0
        return 50.0 if self.variant is TrackCVariant.C0 else 0.0

    def cycle_scale(self, epoch_index: int) -> float:
        return cycle_weight_at_epoch(
            epoch_index,
            warmup_epochs=self.warmup_epochs,
            ramp_epochs=self.ramp_epochs,
        ) if self.uses_cycle else 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["variant"] = self.variant.value
        result["dim_z"] = self.dim_z
        result["effective_lam_alignment"] = self.alignment_weight
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrackCConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown Track C config fields: {sorted(unknown)}")
        return cls(**dict(values))


def cycle_weight_at_epoch(
    epoch_index: int,
    *,
    warmup_epochs: int = 30,
    ramp_epochs: int = 30,
) -> float:
    """Return the C2/C3 multiplier for a zero-based epoch index.

    Epochs 1-30 are zero, epochs 31-60 linearly move from ``1/30`` to
    ``1``, and epoch 61 onward remains one.
    """

    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    if warmup_epochs < 0 or ramp_epochs <= 0:
        raise ValueError("warmup_epochs must be non-negative and ramp_epochs positive")
    if epoch_index < warmup_epochs:
        return 0.0
    return min(1.0, float(epoch_index - warmup_epochs + 1) / float(ramp_epochs))


__all__ = [
    "MODALITIES",
    "SUPPORTED_TOPOLOGIES",
    "TrackCConfig",
    "TrackCVariant",
    "cycle_weight_at_epoch",
]
