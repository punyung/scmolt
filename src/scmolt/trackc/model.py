"""Joint Cycle transport for isolated Track C on the native modules."""

from __future__ import annotations

from dataclasses import dataclass, fields
from importlib import metadata
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import MODALITIES, TrackCConfig
from .gaussian import GaussianPosterior, gaussian_poe, symmetric_kl
from .modules import (
    GaussianTransportGraph,
    cycle_consistency_loss,
    frozen_parameters,
    identity_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
    normalize_feature_dims,
)
from .replay import GaussianReplayBank


REQUIRED_SCMIDAS_VERSION = "0.1.13"


@dataclass(frozen=True)
class TrackCForward:
    """One native forward plus the variant-specific joint posterior."""

    inputs: Mapping[str, Tensor]
    event_masks: Mapping[str, Tensor]
    availability: Mapping[str, Tensor]
    batch_targets: Mapping[str, Tensor]
    batch_index: Tensor
    native: Mapping[str, GaussianPosterior]
    native_biological: Mapping[str, GaussianPosterior]
    native_technical: Mapping[str, GaussianPosterior]
    native_biological_samples: Mapping[str, Tensor]
    native_unimodal_samples: Mapping[str, Tensor]
    translated_biological: Mapping[str, GaussianPosterior]
    batch_expert: Optional[GaussianPosterior]
    native_joint: GaussianPosterior
    joint: GaussianPosterior
    unimodal: Mapping[str, GaussianPosterior]
    latent: Tensor
    reconstructions: Mapping[str, Tensor]
    batch_reconstruction_logits: Optional[Tensor]
    transport_active: bool

    @property
    def joint_embedding(self) -> Tensor:
        first = next(iter(self.native_biological.values()))
        return self.joint.mean[:, : first.mean.shape[1]]


@dataclass(frozen=True)
class TrackCDiscriminatorLosses:
    batch_discriminator_loss: Tensor
    domain_discriminator_loss: Tensor
    batch_raw_native: Tensor
    batch_raw_translated: Tensor
    batch_raw_joint: Tensor
    domain_raw: Tensor
    active_domain_directions: Tensor
    domain_optimizer_active: bool
    batch_weighted_native: Optional[Tensor] = None
    batch_weighted_translated: Optional[Tensor] = None
    batch_weighted_joint: Optional[Tensor] = None
    batch_missingness_scale: Optional[Tensor] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


@dataclass(frozen=True)
class TrackCBatchDiscriminatorLosses:
    batch_discriminator_loss: Tensor
    batch_raw_native: Tensor
    batch_raw_translated: Tensor
    batch_raw_joint: Tensor
    batch_weighted_native: Optional[Tensor] = None
    batch_weighted_translated: Optional[Tensor] = None
    batch_weighted_joint: Optional[Tensor] = None
    batch_missingness_scale: Optional[Tensor] = None

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


@dataclass(frozen=True)
class TrackCDomainDiscriminatorLosses:
    domain_discriminator_loss: Tensor
    domain_raw: Tensor
    active_domain_directions: Tensor
    domain_optimizer_active: bool

    def as_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class TrackCLosses:
    main_loss: Tensor
    batch_discriminator_loss: Tensor
    domain_discriminator_loss: Tensor
    reconstruction: Tensor
    reconstruction_rna: Tensor
    reconstruction_atac: Tensor
    reconstruction_adt: Tensor
    batch_reconstruction_raw: Tensor
    batch_reconstruction_weighted: Tensor
    kld_c_raw: Tensor
    kld_u_raw: Tensor
    kld_weighted: Tensor
    alignment_raw: Tensor
    alignment_weighted: Tensor
    batch_confusion_native_raw: Tensor
    batch_confusion_translated_raw: Tensor
    batch_confusion_joint_raw: Tensor
    batch_confusion_weighted: Tensor
    cycle_raw: Tensor
    identity_raw: Tensor
    cross_decode_raw: Tensor
    domain_generator_raw: Tensor
    cycle_weighted: Tensor
    identity_weighted: Tensor
    cross_decode_weighted: Tensor
    domain_generator_weighted: Tensor
    cycle_scale: Tensor
    active_domain_directions: Tensor
    batch_confusion_native_weighted: Optional[Tensor] = None
    batch_confusion_translated_weighted: Optional[Tensor] = None
    batch_confusion_joint_weighted: Optional[Tensor] = None
    batch_confusion_missingness_scale: Optional[Tensor] = None
    paired_rna_anchor_consistency_raw: Optional[Tensor] = None
    paired_rna_anchor_consistency_weighted: Optional[Tensor] = None
    paired_rna_anchor_active_sources: Optional[Tensor] = None

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


class TrackCMIDASCycle(nn.Module):
    """Wrap the native VAE and add only Track C Gaussian transport.

    C0/C1 call the installed native ``VAE.forward`` unchanged. Forward hooks
    observe the exact posterior parameters from that same pass, so no encoder
    call, dropout mask, or RNG draw is duplicated. C2 keeps native fusion; C3
    replaces only the biological fusion with RNA-canonical translated PoE and
    retains native technical experts, full batch expert, prior, and decoder.

    The accepted batch format is the native contract::

        {"x": {mod: [B,D]}, "e": {mod: event mask},
         "s": {mod: batch target, "joint": global batch target}}

    ``replay_bank`` is a plain device-local object and is absent from
    ``state_dict``.
    """

    def __init__(
        self,
        feature_dims: Mapping[str, int | Sequence[int]],
        n_batches: int,
        config: Optional[TrackCConfig] = None,
        *,
        batch_class_dims: Optional[Mapping[str, int]] = None,
        native_configs: Optional[Mapping[str, Any]] = None,
        native_vae: Optional[nn.Module] = None,
        native_discriminator: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config or TrackCConfig()
        if int(n_batches) <= 0:
            raise ValueError("n_batches must be positive")
        self.n_batches = int(n_batches)
        self.feature_dims = normalize_feature_dims(feature_dims)
        self.modalities = tuple(self.feature_dims)
        if self.config.anchor not in self.feature_dims:
            raise ValueError("Track C feature_dims must contain the RNA anchor")
        self.batch_class_dims = self._normalize_batch_class_dims(batch_class_dims)

        VAE, Discriminator, load_config, midas_class = _load_native_scmidas()
        base_configs = dict(load_config())
        if native_configs is not None:
            base_configs.update(dict(native_configs))
        self._validate_native_architecture(base_configs)
        self.native_configs = base_configs
        dims_x = {key: list(value) for key, value in self.feature_dims.items()}
        self.native_vae = native_vae or VAE(
            dims_x, self.batch_class_dims, **self.native_configs
        )
        self.native_discriminator = native_discriminator or Discriminator(
            dims_x, self.batch_class_dims, **self.native_configs
        )
        self._midas_loss_api = midas_class

        # Native construction ends after VAE + discriminator. Building
        # Track C-only modules must not advance the global RNG seen by the
        # first native forward (critical for exact C0 parity).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed + 300_003)
            self.transport = GaussianTransportGraph(
                self.config, modalities=self.modalities
            )
        self.replay_bank = GaussianReplayBank(
            self.config.dim_c,
            capacity=self.config.replay_capacity,
            modalities=self.modalities,
        )
        # Keep the two observation hooks for the lifetime of the wrapper.
        # Registering/removing Python hooks every minibatch is pure overhead and
        # does not change what the native forward computes.
        self._native_forward_capture: Optional[Dict[str, Any]] = None
        self._native_encoder_capture_handle = (
            self.native_vae.encoder.register_forward_hook(
                self._capture_native_modalities
            )
        )
        self._native_batch_capture_handle = (
            self.native_vae.s_encoder.register_forward_hook(
                self._capture_native_batch
            )
        )

    def _capture_native_modalities(
        self, _module: nn.Module, _args: Any, value: Any
    ) -> None:
        if self._native_forward_capture is not None:
            self._native_forward_capture["modalities"] = value

    def _capture_native_batch(
        self, _module: nn.Module, _args: Any, value: Tensor
    ) -> None:
        if self._native_forward_capture is not None:
            self._native_forward_capture["batch"] = value

    def forward(
        self,
        batch_or_inputs: Mapping[str, Any],
        availability: Optional[Mapping[str, Tensor]] = None,
        batch_index: Optional[Tensor | Mapping[str, Tensor]] = None,
        *,
        cycle_scale: float = 1.0,
        sample: Optional[bool] = None,
    ) -> TrackCForward:
        cycle_scale_value = float(cycle_scale)
        if not 0.0 <= cycle_scale_value <= 1.0:
            raise ValueError("cycle_scale must be in [0,1]")
        transport_active = self.config.uses_cycle and cycle_scale_value > 0.0
        batch = self._native_batch(batch_or_inputs, availability, batch_index)
        if self._native_forward_capture is not None:
            raise RuntimeError("Track C does not support reentrant native forwards")
        captured: Dict[str, Any] = {}
        original_training = self.native_vae.training
        original_transport_training = self.transport.training
        changed_native_mode = False
        changed_transport_mode = False
        if sample is not None:
            requested_training = bool(sample)
            if original_training != requested_training:
                self.native_vae.train(requested_training)
                changed_native_mode = True
            if original_transport_training != requested_training:
                self.transport.train(requested_training)
                changed_transport_mode = True
        self._native_forward_capture = captured
        try:
            native_result = self.native_vae(batch)
        finally:
            self._native_forward_capture = None

        (
            native_reconstruction,
            native_batch_logits,
            native_joint_mean,
            native_joint_logvar,
            native_latent,
            _native_c,
            _native_u,
            native_unimodal_samples,
            native_c_samples,
        ) = native_result
        if "modalities" not in captured:
            raise RuntimeError("Failed to capture native scMIDAS modality posteriors")
        modality_means, modality_logvars = captured["modalities"]
        native = {
            modality: GaussianPosterior(
                modality_means[modality], modality_logvars[modality]
            )
            for modality in modality_means
        }
        native_c: Dict[str, GaussianPosterior] = {}
        native_u: Dict[str, GaussianPosterior] = {}
        for modality, posterior in native.items():
            native_c[modality], native_u[modality] = posterior.split(
                (self.config.dim_c, self.config.dim_u)
            )

        batch_posterior: Optional[GaussianPosterior] = None
        if "batch" in captured:
            batch_mean, batch_logvar = captured["batch"].split(
                self.config.dim_z, dim=1
            )
            batch_posterior = GaussianPosterior(batch_mean, batch_logvar)

        unimodal = self._native_unimodal_posteriors(native, batch_posterior)
        if transport_active:
            translated_c = {
                modality: self.transport.to_anchor(posterior, modality)
                for modality, posterior in native_c.items()
            }
        else:
            # Do not even call the dormant translators: their dropout would
            # consume RNG and break the native C0/C1 trajectory.
            translated_c = dict(native_c)
        native_joint = GaussianPosterior(native_joint_mean, native_joint_logvar)
        joint = native_joint
        latent = native_latent
        reconstructions = native_reconstruction
        batch_logits = native_batch_logits
        if self.config.uses_translated_fusion and transport_active:
            c_experts = list(translated_c.values())
            u_experts = list(native_u.values())
            if batch_posterior is not None:
                batch_c, batch_u = batch_posterior.split(
                    (self.config.dim_c, self.config.dim_u)
                )
                c_experts.append(batch_c)
                u_experts.append(batch_u)
            joint_c = gaussian_poe(c_experts, include_standard_prior=True)
            joint_u = gaussian_poe(u_experts, include_standard_prior=True)
            joint = GaussianPosterior.cat((joint_c, joint_u))
            should_sample = self.native_vae.training if sample is None else bool(sample)
            latent = joint.rsample() if should_sample else joint.mean
            reconstructions = self.native_vae.decoder(latent)
            batch_logits = (
                self.native_vae.s_decoder(latent[:, self.config.dim_c :])
                if batch_posterior is not None
                else None
            )

        # Native MultiBatchSampler expresses topology through present x keys.
        # `e` is a feature mask and must never be reinterpreted as row
        # availability.
        availability_rows = {
            modality: torch.ones(
                batch["x"][modality].shape[0],
                dtype=torch.bool,
                device=batch["x"][modality].device,
            )
            for modality in batch["x"]
        }
        result = TrackCForward(
            inputs=batch["x"],
            event_masks=batch["e"],
            availability=availability_rows,
            batch_targets=batch["s"],
            batch_index=_flat_labels(batch["s"]["joint"]),
            native=native,
            native_biological=native_c,
            native_technical=native_u,
            native_biological_samples={
                modality: native_c_samples[modality]
                for modality in native_c_samples
                if modality != "joint"
            },
            native_unimodal_samples=dict(native_unimodal_samples),
            translated_biological=translated_c,
            batch_expert=batch_posterior,
            native_joint=native_joint,
            joint=joint,
            unimodal=unimodal,
            latent=latent,
            reconstructions=reconstructions,
            batch_reconstruction_logits=batch_logits,
            transport_active=transport_active,
        )
        if changed_native_mode:
            self.native_vae.train(original_training)
        if changed_transport_mode:
            self.transport.train(original_transport_training)
        return result

    def encode_joint(
        self,
        batch_or_inputs: Mapping[str, Any],
        availability: Optional[Mapping[str, Tensor]] = None,
        batch_index: Optional[Tensor | Mapping[str, Tensor]] = None,
        *,
        use_mean: bool = True,
    ) -> GaussianPosterior:
        """Return only the 32D biological joint posterior for export."""

        output = self.forward(
            batch_or_inputs,
            availability,
            batch_index,
            sample=not use_mean,
        )
        biological, _technical = output.joint.split(
            (self.config.dim_c, self.config.dim_u)
        )
        return biological

    @torch.no_grad()
    def update_replay_bank(self, output: TrackCForward) -> Mapping[str, int]:
        """Insert once per minibatch; the bank is never checkpointed."""

        return {
            modality: self.replay_bank.add(
                modality,
                output.native_biological[modality],
                output.availability[modality],
            )
            for modality in output.native_biological
        }

    def compute_discriminator_losses(
        self,
        output: TrackCForward,
        *,
        cycle_scale: float = 1.0,
    ) -> TrackCDiscriminatorLosses:
        batch = self.compute_batch_discriminator_losses(
            output, cycle_scale=cycle_scale
        )
        domain = self.compute_domain_discriminator_losses(
            output, cycle_scale=cycle_scale
        )
        return TrackCDiscriminatorLosses(
            batch_discriminator_loss=batch.batch_discriminator_loss,
            domain_discriminator_loss=domain.domain_discriminator_loss,
            batch_raw_native=batch.batch_raw_native,
            batch_raw_translated=batch.batch_raw_translated,
            batch_raw_joint=batch.batch_raw_joint,
            domain_raw=domain.domain_raw,
            active_domain_directions=domain.active_domain_directions,
            domain_optimizer_active=domain.domain_optimizer_active,
            batch_weighted_native=batch.batch_weighted_native,
            batch_weighted_translated=batch.batch_weighted_translated,
            batch_weighted_joint=batch.batch_weighted_joint,
            batch_missingness_scale=batch.batch_missingness_scale,
        )

    def compute_batch_discriminator_losses(
        self,
        output: TrackCForward,
        *,
        cycle_scale: float = 1.0,
    ) -> TrackCBatchDiscriminatorLosses:
        """Fresh batch-D-only graph; never samples a domain critic/replay."""

        native, translated, joint, _total = self._batch_losses(
            output, detach_inputs=True, freeze_discriminator=False
        )
        scale_value = self._scalar_value(cycle_scale)
        scale = output.latent.new_tensor(scale_value)
        if (scale_value > 0.0) != output.transport_active and self.config.uses_cycle:
            raise ValueError("forward and batch-D losses must use the same cycle_scale")
        translated_scaled = self._scaled_translated_batch_ce(translated, scale)
        missingness_scale = self._missingness_batch_scale(output)
        missingness_applies = (
            self.config.anchor not in output.inputs
            and self.config.missingness_batch_ce_scale != 1.0
        )
        if missingness_applies:
            translated_scaled = translated_scaled * missingness_scale
            joint_scaled = joint * missingness_scale
        else:
            joint_scaled = joint
        total = native + joint_scaled + translated_scaled
        batch_total = self.config.lam_dsc * total
        emit_components = self.config.emit_translated_batch_ce_metrics
        emit_missingness = self.config.missingness_batch_ce_scale != 1.0
        return TrackCBatchDiscriminatorLosses(
            batch_discriminator_loss=batch_total,
            batch_raw_native=native,
            batch_raw_translated=translated,
            batch_raw_joint=joint,
            batch_weighted_native=(
                self.config.lam_dsc * native if emit_components else None
            ),
            batch_weighted_translated=(
                self.config.lam_dsc * translated_scaled
                if emit_components
                else None
            ),
            batch_weighted_joint=(
                self.config.lam_dsc * joint_scaled if emit_components else None
            ),
            batch_missingness_scale=(
                missingness_scale if emit_missingness else None
            ),
        )

    def compute_domain_discriminator_losses(
        self,
        output: TrackCForward,
        *,
        cycle_scale: float = 1.0,
    ) -> TrackCDomainDiscriminatorLosses:
        """Fresh domain-D-only graph and an explicit optimizer-step gate."""

        scale_value = self._scalar_value(cycle_scale)
        scale = output.latent.new_tensor(scale_value)
        if (scale_value > 0.0) != output.transport_active and self.config.uses_cycle:
            raise ValueError("forward and domain-D losses must use the same cycle_scale")
        if output.transport_active:
            domain_raw, active, optimizer_active = self._domain_discriminator_loss(
                output
            )
        else:
            domain_raw = self._domain_parameter_zero()
            active = _count_tensor(0, output.latent)
            optimizer_active = False
        domain_total = self.config.lam_domain * scale * domain_raw
        return TrackCDomainDiscriminatorLosses(
            domain_discriminator_loss=domain_total,
            domain_raw=domain_raw,
            active_domain_directions=active,
            domain_optimizer_active=(output.transport_active and optimizer_active),
        )

    def compute_batch_discriminator_loss(
        self, output: TrackCForward, *, cycle_scale: float = 1.0
    ) -> Tensor:
        return self.compute_batch_discriminator_losses(
            output, cycle_scale=cycle_scale
        ).batch_discriminator_loss

    def compute_domain_discriminator_loss(
        self, output: TrackCForward, *, cycle_scale: float = 1.0
    ) -> Tensor:
        return self.compute_domain_discriminator_losses(
            output, cycle_scale=cycle_scale
        ).domain_discriminator_loss

    def compute_main_loss(
        self,
        output: TrackCForward,
        *,
        cycle_scale: float = 1.0,
    ) -> TrackCLosses:
        scale_value = self._scalar_value(cycle_scale)
        scale = output.latent.new_tensor(scale_value)
        if (scale_value > 0.0) != output.transport_active and self.config.uses_cycle:
            raise ValueError("forward and main loss must use the same cycle_scale")
        zero = output.latent.sum() * 0.0
        native_reconstruction_total, reconstruction_parts, batch_recon_raw, batch_recon_weighted = (
            self._native_reconstruction_losses(output)
        )
        reconstruction = sum(reconstruction_parts.values(), zero) * self.config.lam_recon

        joint_c, joint_u = output.joint.split((self.config.dim_c, self.config.dim_u))
        batch_size = output.latent.shape[0]
        # The native loss sums KL over rows and coordinates, then divides by B.
        kld_c = self._midas_loss_api.calc_kld_loss(joint_c.mean, joint_c.logvar)
        kld_u = self._midas_loss_api.calc_kld_loss(joint_u.mean, joint_u.logvar)
        kld_weighted = self.config.lam_kld * self._midas_loss_api.calc_kld_z_loss(
            self.config.dim_c,
            self.config.dim_u,
            self.config.lam_kld_c,
            self.config.lam_kld_u,
            output.joint.mean,
            output.joint.logvar,
        )
        # This is the exact z_uni dictionary returned by the same native
        # forward call: no posterior reconstruction and no additional RNG.
        alignment_raw = (
            self._midas_loss_api.calc_consistency_loss(
                dict(output.native_unimodal_samples)
            )
            if len(output.native_unimodal_samples) > 1
            else zero
        )
        alignment_weighted = self.config.alignment_weight * alignment_raw

        batch_native, batch_translated, batch_joint, _batch_total_raw = self._batch_losses(
            output, detach_inputs=False, freeze_discriminator=True
        )
        batch_translated_scaled = self._scaled_translated_batch_ce(
            batch_translated, scale
        )
        missingness_scale = self._missingness_batch_scale(output)
        missingness_applies = (
            self.config.anchor not in output.inputs
            and self.config.missingness_batch_ce_scale != 1.0
        )
        if missingness_applies:
            batch_translated_scaled = batch_translated_scaled * missingness_scale
            batch_joint_scaled = batch_joint * missingness_scale
        else:
            batch_joint_scaled = batch_joint
        batch_total_raw = batch_native + batch_joint_scaled + batch_translated_scaled
        batch_discriminator_weighted = batch_total_raw * self.config.lam_dsc
        batch_confusion_weighted = -(
            batch_discriminator_weighted * self.config.lam_adv
        )
        emit_components = self.config.emit_translated_batch_ce_metrics
        batch_confusion_native_weighted = (
            -(batch_native * self.config.lam_dsc * self.config.lam_adv)
            if emit_components
            else None
        )
        batch_confusion_translated_weighted = (
            -(
                batch_translated_scaled
                * self.config.lam_dsc
                * self.config.lam_adv
            )
            if emit_components
            else None
        )
        batch_confusion_joint_weighted = (
            -(batch_joint_scaled * self.config.lam_dsc * self.config.lam_adv)
            if emit_components
            else None
        )

        if output.transport_active:
            cycle_raw = self._cycle_loss(output)
            identity_raw = self._identity_loss(output)
            cross_decode_raw = self._cross_decode_loss(output)
            domain_generator_raw, active = self._domain_generator_loss(output)
        else:
            cycle_raw = identity_raw = cross_decode_raw = domain_generator_raw = zero
            active = torch.zeros((), dtype=torch.long, device=zero.device)
        cycle_weighted = scale * self.config.lam_cycle * cycle_raw
        identity_weighted = scale * self.config.lam_identity * identity_raw
        cross_decode_weighted = scale * self.config.lam_cross_decode * cross_decode_raw
        domain_generator_weighted = scale * self.config.lam_domain * domain_generator_raw
        paired_anchor_enabled = (
            self.config.paired_rna_anchor_consistency_weight > 0.0
        )
        if paired_anchor_enabled:
            paired_anchor_raw, paired_anchor_sources = (
                self._paired_rna_anchor_consistency_loss(output)
            )
            paired_anchor_weighted = (
                scale
                * self.config.paired_rna_anchor_consistency_weight
                * paired_anchor_raw
            )
        else:
            paired_anchor_raw = paired_anchor_weighted = None
            paired_anchor_sources = None
        # Preserve the native arithmetic/order for C0/C1 parity. The native
        # reconstruction total already includes the weighted s reconstruction.
        base_main = (
            native_reconstruction_total * self.config.lam_recon
            + kld_weighted
            + alignment_weighted
        )
        main = (
            base_main
            - batch_discriminator_weighted * self.config.lam_adv
            + cycle_weighted
            + identity_weighted
            + cross_decode_weighted
            + domain_generator_weighted
        )
        if paired_anchor_weighted is not None:
            main = main + paired_anchor_weighted
        # Reuse the already-computed batch raw terms. Calling the
        # discriminator helper here would re-sample translated posteriors and
        # consume RNG purely for diagnostics.
        batch_discriminator_diagnostic = batch_discriminator_weighted
        domain_discriminator_diagnostic = self._domain_parameter_zero()
        return TrackCLosses(
            main_loss=main,
            batch_discriminator_loss=batch_discriminator_diagnostic.detach(),
            domain_discriminator_loss=domain_discriminator_diagnostic.detach(),
            reconstruction=reconstruction,
            reconstruction_rna=reconstruction_parts.get("rna", zero),
            reconstruction_atac=reconstruction_parts.get("atac", zero),
            reconstruction_adt=reconstruction_parts.get("adt", zero),
            batch_reconstruction_raw=batch_recon_raw,
            batch_reconstruction_weighted=batch_recon_weighted,
            kld_c_raw=kld_c,
            kld_u_raw=kld_u,
            kld_weighted=kld_weighted,
            alignment_raw=alignment_raw,
            alignment_weighted=alignment_weighted,
            batch_confusion_native_raw=batch_native,
            batch_confusion_translated_raw=batch_translated,
            batch_confusion_joint_raw=batch_joint,
            batch_confusion_weighted=batch_confusion_weighted,
            cycle_raw=cycle_raw,
            identity_raw=identity_raw,
            cross_decode_raw=cross_decode_raw,
            domain_generator_raw=domain_generator_raw,
            cycle_weighted=cycle_weighted,
            identity_weighted=identity_weighted,
            cross_decode_weighted=cross_decode_weighted,
            domain_generator_weighted=domain_generator_weighted,
            cycle_scale=scale,
            active_domain_directions=active,
            batch_confusion_native_weighted=(
                batch_confusion_native_weighted
            ),
            batch_confusion_translated_weighted=(
                batch_confusion_translated_weighted
            ),
            batch_confusion_joint_weighted=batch_confusion_joint_weighted,
            batch_confusion_missingness_scale=(
                missingness_scale
                if self.config.missingness_batch_ce_scale != 1.0
                else None
            ),
            paired_rna_anchor_consistency_raw=paired_anchor_raw,
            paired_rna_anchor_consistency_weighted=paired_anchor_weighted,
            paired_rna_anchor_active_sources=paired_anchor_sources,
        )

    def compute_losses(
        self,
        batch_or_output: TrackCForward | Mapping[str, Any],
        availability: Optional[Mapping[str, Tensor]] = None,
        batch_index: Optional[Tensor | Mapping[str, Tensor]] = None,
        *,
        cycle_scale: float = 1.0,
    ) -> TrackCLosses:
        output = (
            batch_or_output
            if isinstance(batch_or_output, TrackCForward)
            else self.forward(
                batch_or_output,
                availability,
                batch_index,
                cycle_scale=cycle_scale,
            )
        )
        return self.compute_main_loss(output, cycle_scale=cycle_scale)

    def modality_reconstruction_loss(
        self,
        output: TrackCForward,
        *,
        modality: str,
        target: Tensor,
        event_mask: Optional[Tensor] = None,
        normalize_by_event_count: bool = False,
    ) -> Tensor:
        """Native-distribution reconstruction loss for a held-out modality.

        The target is detached defensively, so gradients flow only through the
        model.  Whole-modality re-masking retains native batch
        normalization.  Feature-level re-masking can additionally request a
        mean over artificially hidden entries so RNA and ATAC losses remain
        comparable despite their very different feature counts.
        """

        if modality not in self.feature_dims:
            raise ValueError(f"Unknown reconstruction modality {modality!r}")
        target_value = torch.as_tensor(target, device=output.latent.device).detach()
        expected = (output.latent.shape[0], sum(self.feature_dims[modality]))
        if target_value.ndim != 2 or tuple(target_value.shape) != expected:
            raise ValueError(
                f"{modality} target must have shape {expected}; "
                f"got {tuple(target_value.shape)}"
            )
        masks: Dict[str, Tensor] = {}
        normalization_mask: Optional[Tensor] = None
        if event_mask is not None:
            mask = torch.as_tensor(event_mask, device=target_value.device)
            try:
                torch.broadcast_shapes(target_value.shape, mask.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"{modality} target mask shape {tuple(mask.shape)} is not "
                    f"broadcastable to {tuple(target_value.shape)}"
                ) from exc
            normalization_mask = torch.broadcast_to(mask, target_value.shape).to(
                target_value.dtype
            )
            masks[modality] = mask.to(target_value.dtype)
        elif normalize_by_event_count:
            raise ValueError(
                "normalize_by_event_count requires an explicit event_mask"
            )
        dist = {
            key: value
            for key, value in self.native_configs.items()
            if key.startswith("distribution_dec_")
        }
        lam = {
            "lam_recon_rna": 1.0,
            "lam_recon_atac": 1.0,
            "lam_recon_adt": 1.0,
            "lam_recon_s": 0.0,
        }
        total, _raw = self._midas_loss_api.calc_recon_loss(
            {modality: target_value},
            output.batch_targets["joint"],
            masks,
            output.reconstructions,
            None,
            dist,
            lam,
        )
        if normalize_by_event_count:
            assert normalization_mask is not None
            event_count = normalization_mask.sum()
            if event_count.device.type == "cuda":
                torch._assert_async(
                    event_count > 0,
                    f"{modality} feature re-masking selected no target entries",
                )
            elif not bool(event_count > 0):
                raise ValueError(
                    f"{modality} feature re-masking selected no target entries"
                )
            total = total / (event_count / target_value.shape[0])
        return total

    def optimizer_parameter_groups(self) -> Mapping[str, list[nn.Parameter]]:
        batch_ids = {id(parameter) for parameter in self.native_discriminator.parameters()}
        domain_ids = {
            id(parameter) for parameter in self.transport.discriminators.parameters()
        }
        main = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in batch_ids and id(parameter) not in domain_ids
        ]
        groups = {
            "main": main,
            "batch_discriminator": list(self.native_discriminator.parameters()),
            "domain_discriminator": list(self.transport.discriminators.parameters()),
        }
        flattened = [id(parameter) for group in groups.values() for parameter in group]
        if len(flattened) != len(set(flattened)):
            raise RuntimeError("Track C optimizer parameter groups overlap")
        if set(flattened) != {id(parameter) for parameter in self.parameters()}:
            raise RuntimeError("Track C optimizer parameter groups are incomplete")
        return groups

    def build_optimizers(self) -> Mapping[str, torch.optim.Optimizer]:
        groups = self.optimizer_parameter_groups()
        return {
            "main": torch.optim.AdamW(
                groups["main"],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            ),
            "batch_discriminator": torch.optim.AdamW(
                groups["batch_discriminator"],
                lr=self.config.batch_discriminator_learning_rate,
                weight_decay=self.config.weight_decay,
            ),
            "domain_discriminator": torch.optim.AdamW(
                groups["domain_discriminator"],
                lr=self.config.domain_discriminator_learning_rate,
                weight_decay=self.config.weight_decay,
            ),
        }

    def _native_batch(
        self,
        batch_or_inputs: Mapping[str, Any],
        availability: Optional[Mapping[str, Tensor]],
        batch_index: Optional[Tensor | Mapping[str, Tensor]],
    ) -> Dict[str, Dict[str, Tensor]]:
        if "x" in batch_or_inputs:
            raw_inputs = batch_or_inputs["x"]
            availability = batch_or_inputs.get("e", availability)
            batch_index = batch_or_inputs.get("s", batch_index)
        else:
            raw_inputs = batch_or_inputs
        if not isinstance(raw_inputs, Mapping) or not raw_inputs:
            raise ValueError("Track C requires at least one native modality input")
        owns_inputs = bool(getattr(batch_or_inputs, "trackc_owns_inputs", False))
        inputs = (
            dict(raw_inputs)
            if owns_inputs
            else {key: value.clone() for key, value in raw_inputs.items()}
        )
        unknown = set(inputs) - set(self.feature_dims)
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        first = next(iter(inputs.values()))
        if first.ndim != 2:
            raise ValueError("Modality inputs must be rank two")
        batch_size = first.shape[0]
        for modality, values in inputs.items():
            if values.ndim != 2 or values.shape != (
                batch_size,
                sum(self.feature_dims[modality]),
            ):
                raise ValueError(
                    f"{modality} input must have shape "
                    f"[{batch_size},{sum(self.feature_dims[modality])}]"
                )
        event_masks: Dict[str, Tensor] = {}
        for modality, values in inputs.items():
            raw_mask = None if availability is None else availability.get(modality)
            if raw_mask is None:
                event_masks[modality] = torch.ones(
                    (values.shape[1],), device=values.device, dtype=values.dtype
                )
            else:
                mask = torch.as_tensor(raw_mask, device=values.device)
                try:
                    torch.broadcast_shapes(values.shape, mask.shape)
                except RuntimeError as exc:
                    raise ValueError(
                        f"{modality} feature mask shape {tuple(mask.shape)} is not "
                        f"broadcastable to input {tuple(values.shape)}"
                    ) from exc
                event_masks[modality] = mask.to(values.dtype)

        if isinstance(batch_index, Mapping):
            targets = dict(batch_index)
        elif batch_index is not None:
            targets = {"joint": batch_index}
        else:
            raise ValueError("Track C requires native batch targets in s")
        if "joint" not in targets:
            raise ValueError("Track C requires s['joint']")
        joint = torch.as_tensor(targets["joint"], device=first.device)
        if joint.ndim == 1:
            joint = joint.unsqueeze(1)
        if joint.shape != (batch_size, 1):
            raise ValueError("s['joint'] must have shape [B] or [B,1]")
        targets["joint"] = joint.long()
        for modality in inputs:
            value = torch.as_tensor(targets.get(modality, joint), device=first.device)
            if value.ndim == 1:
                value = value.unsqueeze(1)
            if value.shape != (batch_size, 1):
                raise ValueError(f"s['{modality}'] must have shape [B] or [B,1]")
            targets[modality] = value.long()
        return {"x": inputs, "e": event_masks, "s": targets}

    def _native_unimodal_posteriors(
        self,
        native: Mapping[str, GaussianPosterior],
        batch_posterior: Optional[GaussianPosterior],
    ) -> Dict[str, GaussianPosterior]:
        result = {}
        for modality, posterior in native.items():
            experts = [posterior]
            if batch_posterior is not None:
                experts.append(batch_posterior)
            result[modality] = gaussian_poe(experts, include_standard_prior=True)
        return result

    def _native_reconstruction_losses(
        self, output: TrackCForward
    ) -> tuple[Tensor, Dict[str, Tensor], Tensor, Tensor]:
        dist = {
            key: value
            for key, value in self.native_configs.items()
            if key.startswith("distribution_dec_")
        }
        lam = {
            "lam_recon_rna": 1.0,
            "lam_recon_atac": 1.0,
            "lam_recon_adt": 1.0,
            "lam_recon_s": self.config.lam_recon_s,
        }
        total, raw = self._midas_loss_api.calc_recon_loss(
            output.inputs,
            output.batch_targets["joint"],
            output.event_masks,
            output.reconstructions,
            output.batch_reconstruction_logits,
            dist,
            lam,
        )
        batch_size = output.latent.shape[0]
        zero = output.latent.sum() * 0.0
        parts = {
            modality: raw.get(f"recon_loss/{modality}", zero) / batch_size
            for modality in output.inputs
        }
        batch_weighted = raw.get("recon_loss/s", zero) / batch_size
        batch_raw = (
            batch_weighted / self.config.lam_recon_s
            if self.config.lam_recon_s > 0.0
            else zero
        )
        return total, parts, batch_raw, batch_weighted

    def _batch_losses(
        self,
        output: TrackCForward,
        *,
        detach_inputs: bool,
        freeze_discriminator: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        context = (
            frozen_parameters(self.native_discriminator)
            if freeze_discriminator
            else _NullContext()
        )
        zero = (
            next(self.native_discriminator.parameters()).sum() * 0.0
            if detach_inputs
            else output.latent.sum() * 0.0
        )
        batch_size = output.latent.shape[0]

        def prepare(values: Tensor) -> Tensor:
            return values.detach() if detach_inputs else values

        with context:
            native_terms = []
            translated_terms = []
            for modality, values in output.native_biological_samples.items():
                labels = _flat_labels(
                    output.batch_targets.get(modality, output.batch_targets["joint"])
                )
                logits = self.native_discriminator.predictors[modality](prepare(values))
                native_terms.append(F.cross_entropy(logits, labels, reduction="sum"))
                if output.transport_active:
                    translated_values = output.translated_biological[
                        modality
                    ].sample_or_mean(self.native_vae.training)
                    logits = self.native_discriminator.predictors[modality](
                        prepare(translated_values)
                    )
                    translated_terms.append(
                        F.cross_entropy(logits, labels, reduction="sum")
                    )
            joint_values = output.latent[:, : self.config.dim_c]
            joint_logits = self.native_discriminator.predictors["joint"](
                prepare(joint_values)
            )
            joint_term = F.cross_entropy(
                joint_logits,
                _flat_labels(output.batch_targets["joint"]),
                reduction="sum",
            )
            native = sum(native_terms, zero) / batch_size
            translated = sum(translated_terms, zero) / batch_size
            joint = joint_term / batch_size
            # For C0/C1 this has the same ordered sum as the native
            # discriminator loss: modality predictors followed by joint.
            ordered_terms = native_terms + translated_terms + [joint_term]
            total = sum(ordered_terms, zero) / batch_size
        return native, translated, joint, total

    def _scaled_translated_batch_ce(
        self, translated: Tensor, cycle_scale: Tensor
    ) -> Tensor:
        """Scale only Cycle translated CE while preserving alpha=1 arithmetic."""

        scaled = cycle_scale * translated
        if self.config.translated_batch_ce_scale != 1.0:
            scaled = scaled * self.config.translated_batch_ce_scale
        return scaled

    def _missingness_batch_scale(self, output: TrackCForward) -> Tensor:
        """Return the predeclared adversary multiplier from observed keys only."""

        value = 1.0
        if self.config.anchor not in output.inputs:
            value = self.config.missingness_batch_ce_scale
        return output.latent.new_tensor(value)

    def _paired_rna_anchor_consistency_loss(
        self, output: TrackCForward
    ) -> tuple[Tensor, Tensor]:
        """Align paired non-RNA posteriors to an observed, detached RNA anchor.

        Native minibatches are homogeneous modality configurations, so
        the presence of ``rna`` in ``output.inputs`` proves that every row in
        this minibatch has observed RNA.  ATAC-only masked groups therefore
        return an exact zero without consulting any manifest or truth file.
        """

        zero = output.latent.sum() * 0.0
        if not output.transport_active or self.config.anchor not in output.native_biological:
            return zero, _count_tensor(0, output.latent)
        teacher = output.native_biological[self.config.anchor].detach()
        terms = [
            symmetric_kl(output.translated_biological[source], teacher)
            for source in output.native_biological
            if source != self.config.anchor
            and source in output.translated_biological
        ]
        if not terms:
            return zero, _count_tensor(0, output.latent)
        return torch.stack(terms).mean(), _count_tensor(len(terms), output.latent)

    def _domain_discriminator_loss(
        self, output: TrackCForward
    ) -> tuple[Tensor, Tensor, bool]:
        terms = []
        for source, target in self.transport.DIRECTIONS:
            if source not in output.native_biological or not self._domain_target_available(
                output, target
            ):
                continue
            with torch.no_grad():
                fake = self.transport.translate(
                    output.native_biological[source].detach(), source, target
                ).rsample()
            if target in output.native_biological:
                real_posterior = output.native_biological[target].detach()
            else:
                real_posterior = self.replay_bank.sample(
                    target,
                    fake.shape[0],
                    device=fake.device,
                    dtype=fake.dtype,
                )
            if real_posterior is None:
                continue
            count = min(fake.shape[0], real_posterior.mean.shape[0])
            terms.append(
                lsgan_discriminator_loss(
                    self.transport.domain_logits(real_posterior.rsample()[:count], target),
                    self.transport.domain_logits(fake[:count], target),
                )
            )
        if terms:
            return (
                torch.stack(terms).mean(),
                _count_tensor(len(terms), output.latent),
                True,
            )
        parameter = next(self.transport.discriminators.parameters())
        return parameter.sum() * 0.0, _count_tensor(0, output.latent), False

    def _domain_parameter_zero(self) -> Tensor:
        parameter = next(self.transport.discriminators.parameters())
        return parameter.sum() * 0.0

    def _domain_generator_loss(
        self, output: TrackCForward
    ) -> tuple[Tensor, Tensor]:
        terms = []
        with frozen_parameters(self.transport.discriminators):
            for source, target in self.transport.DIRECTIONS:
                if source not in output.native_biological or not self._domain_target_available(
                    output, target
                ):
                    continue
                fake = self.transport.translate(
                    output.native_biological[source], source, target
                ).rsample()
                terms.append(
                    lsgan_generator_loss(self.transport.domain_logits(fake, target))
                )
        if terms:
            return torch.stack(terms).mean(), _count_tensor(len(terms), output.latent)
        return output.latent.sum() * 0.0, _count_tensor(0, output.latent)

    def _domain_target_available(self, output: TrackCForward, target: str) -> bool:
        return target in output.native_biological or self.replay_bank.primed(target)

    def _cycle_loss(self, output: TrackCForward) -> Tensor:
        terms = [
            cycle_consistency_loss(
                self.transport, output.native_biological[source], source, target
            )
            for source, target in self.transport.DIRECTIONS
            if source in output.native_biological
        ]
        return torch.stack(terms).mean() if terms else output.latent.sum() * 0.0

    def _identity_loss(self, output: TrackCForward) -> Tensor:
        terms = [
            identity_loss(
                self.transport, output.native_biological[target], source, target
            )
            for source, target in self.transport.DIRECTIONS
            if target in output.native_biological
        ]
        return torch.stack(terms).mean() if terms else output.latent.sum() * 0.0

    def _cross_decode_loss(self, output: TrackCForward) -> Tensor:
        terms = []
        for source, target in self.transport.DIRECTIONS:
            if source not in output.native or target not in output.inputs:
                continue
            source_c = self.transport.to_anchor(
                output.native_biological[source], source
            )
            c_experts = [source_c]
            u_experts = [output.native_technical[source]]
            if output.batch_expert is not None:
                batch_c, batch_u = output.batch_expert.split(
                    (self.config.dim_c, self.config.dim_u)
                )
                c_experts.append(batch_c)
                u_experts.append(batch_u)
            latent = GaussianPosterior.cat(
                (gaussian_poe(c_experts), gaussian_poe(u_experts))
            ).rsample()
            decoded = self.native_vae.decoder(latent)[target]
            target_values = output.inputs[target]
            raw_mask = output.event_masks[target]
            if target == "atac":
                values = F.binary_cross_entropy(decoded, target_values, reduction="none")
            else:
                values = F.poisson_nll_loss(
                    decoded,
                    target_values,
                    log_input=True,
                    full=True,
                    reduction="none",
                )
            terms.append((values * raw_mask).sum() / target_values.shape[0])
        return torch.stack(terms).mean() if terms else output.latent.sum() * 0.0

    def _normalize_batch_class_dims(
        self, values: Optional[Mapping[str, int]]
    ) -> Dict[str, int]:
        result = {key: self.n_batches for key in (*self.modalities, "joint")}
        if values is not None:
            unknown = set(values) - set(result)
            if unknown:
                raise ValueError(f"Unknown batch class dimensions: {sorted(unknown)}")
            result.update({key: int(value) for key, value in values.items()})
        if any(value <= 0 for value in result.values()):
            raise ValueError("All batch class dimensions must be positive")
        if result["joint"] != self.n_batches:
            raise ValueError("batch_class_dims['joint'] must equal n_batches")
        return result

    def _validate_native_architecture(self, values: Mapping[str, Any]) -> None:
        expected = {
            "dim_c": self.config.dim_c,
            "dim_u": self.config.dim_u,
            "dims_shared_enc": list(self.config.encoder_hidden),
            "dims_shared_dec": list(self.config.decoder_hidden),
            "dims_enc_s": list(self.config.batch_hidden),
            "dims_dec_s": list(self.config.batch_hidden),
            "dims_dsc": list(self.config.batch_discriminator_hidden),
            "dims_before_enc_atac": list(self.config.atac_adapter_hidden),
            "dims_after_dec_atac": list(reversed(self.config.atac_adapter_hidden)),
            "s_drop_rate": self.config.s_drop_rate,
            "drop": self.config.dropout,
            "norm": "ln",
            "out_trans": "mish",
            "optim_net": "AdamW",
            "optim_dsc": "AdamW",
        }
        expected.update(
            {
                "distribution_dec_{}".format(modality): {
                    "rna": "POISSON",
                    "atac": "BERNOULLI",
                    "adt": "POISSON",
                }[modality]
                for modality in self.modalities
            }
        )
        mismatches = {
            key: (values.get(key), expected_value)
            for key, expected_value in expected.items()
            if values.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(
                f"Native scMIDAS architecture differs from locked Track C: {mismatches}"
            )

    @staticmethod
    def _scalar_value(value: float | Tensor) -> float:
        if isinstance(value, Tensor) and value.ndim != 0:
            raise ValueError("cycle_scale must be a scalar in [0,1]")
        try:
            scalar = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("cycle_scale must be a scalar in [0,1]") from error
        if not 0.0 <= scalar <= 1.0:
            raise ValueError("cycle_scale must be a scalar in [0,1]")
        return scalar

    @staticmethod
    def _scalar(value: float | Tensor, reference: Tensor) -> Tensor:
        """Compatibility helper with host validation before device creation."""

        return reference.new_tensor(TrackCMIDASCycle._scalar_value(value))


def _load_native_scmidas() -> tuple[Any, Any, Any, Any]:
    try:
        installed = metadata.version("scmidas")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Track C requires the scmidas conda environment with scmidas==0.1.13"
        ) from exc
    if installed != REQUIRED_SCMIDAS_VERSION:
        raise RuntimeError(
            f"Track C requires scmidas=={REQUIRED_SCMIDAS_VERSION}; found {installed}"
        )
    try:
        from scmidas.config import load_config
        from scmidas.model import Discriminator, MIDAS, VAE
    except Exception as exc:
        raise ImportError(
            "Failed to import scmidas 0.1.13. Set NUMBA_CACHE_DIR and MPLCONFIGDIR "
            "to writable job-local paths before importing Track C."
        ) from exc
    return VAE, Discriminator, load_config, MIDAS


def _availability_rows(mask: Optional[Tensor], values: Tensor) -> Tensor:
    if mask is None:
        return torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    mask = torch.as_tensor(mask, device=values.device)
    if mask.ndim == 1:
        return mask.bool()
    return mask.reshape(mask.shape[0], -1).any(dim=1)


def _flat_labels(values: Tensor) -> Tensor:
    values = torch.as_tensor(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("Batch labels must have shape [B] or [B,1]")
    return values.long()


def _count_tensor(count: int, reference: Tensor) -> Tensor:
    return torch.tensor(count, dtype=torch.long, device=reference.device)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


__all__ = [
    "REQUIRED_SCMIDAS_VERSION",
    "TrackCBatchDiscriminatorLosses",
    "TrackCDiscriminatorLosses",
    "TrackCDomainDiscriminatorLosses",
    "TrackCForward",
    "TrackCLosses",
    "TrackCMIDASCycle",
]
