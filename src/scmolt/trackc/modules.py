"""Neural components for the isolated Cycle Track C model."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from .config import MODALITIES, TrackCConfig
from .gaussian import GaussianPosterior, symmetric_kl


def _hidden_block(width: int, dropout: float) -> list[nn.Module]:
    return [nn.LayerNorm(width), nn.Mish(), nn.Dropout(dropout)]


class DenseMLP(nn.Module):
    """Linear/LayerNorm/Mish/Dropout MLP."""

    def __init__(
        self,
        dimensions: Sequence[int],
        *,
        dropout: float,
        activate_output: bool = False,
    ) -> None:
        super().__init__()
        dimensions = tuple(int(value) for value in dimensions)
        if len(dimensions) < 2 or any(value <= 0 for value in dimensions):
            raise ValueError("MLP dimensions must contain at least two positive values")
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2 or activate_output:
                layers.extend(_hidden_block(output_dim, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2:
            raise ValueError("DenseMLP input must be rank two")
        return self.net(values)


def normalize_feature_dims(
    feature_dims: Mapping[str, int | Sequence[int]],
) -> Dict[str, Tuple[int, ...]]:
    extra = set(feature_dims) - set(MODALITIES)
    if not feature_dims or extra:
        raise ValueError(
            f"feature_dims must contain a non-empty subset of {MODALITIES}; "
            f"extra={sorted(extra)}"
        )
    result: Dict[str, Tuple[int, ...]] = {}
    for modality in (name for name in MODALITIES if name in feature_dims):
        value = feature_dims[modality]
        if isinstance(value, int):
            dimensions = (value,)
        else:
            dimensions = tuple(int(item) for item in value)
        if not dimensions or any(item <= 0 for item in dimensions):
            raise ValueError(f"{modality} feature dimensions must be positive")
        result[modality] = dimensions
    return result


class GaussianResidualTranslator(nn.Module):
    """64->128->128->64 residual map over ``(mu_c, logvar_c)``.

    The output layer is initialized to zero, so every direction starts as the
    identity.  Only biological Gaussian parameters are accepted; no technical
    ``u`` tensor appears in this API.
    """

    def __init__(self, config: TrackCConfig) -> None:
        super().__init__()
        self.dim_c = config.dim_c
        self.logvar_min = float(config.logvar_min)
        self.logvar_max = float(config.logvar_max)
        self.backbone = DenseMLP(
            (2 * config.dim_c, *config.translator_hidden),
            dropout=config.dropout,
        )
        self.residual = nn.Linear(config.translator_hidden[-1], 2 * config.dim_c)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, posterior: GaussianPosterior) -> GaussianPosterior:
        if posterior.mean.shape[1] != self.dim_c:
            raise ValueError(f"Translator expects biological dimension {self.dim_c}")
        parameters = torch.cat([posterior.mean, posterior.logvar], dim=1)
        delta_mean, delta_logvar = self.residual(self.backbone(parameters)).split(
            self.dim_c, dim=1
        )
        return GaussianPosterior(
            posterior.mean + delta_mean,
            (posterior.logvar + delta_logvar).clamp(
                min=self.logvar_min, max=self.logvar_max
            ),
        )


def _direction(source: str, target: str) -> str:
    return f"{source}__{target}"


class GaussianTransportGraph(nn.Module):
    """RNA-anchor Gaussian translators and one LSGAN critic per modality."""

    FULL_DIRECTIONS: Tuple[Tuple[str, str], ...] = (
        ("atac", "rna"),
        ("rna", "atac"),
        ("adt", "rna"),
        ("rna", "adt"),
    )

    def __init__(
        self,
        config: TrackCConfig,
        modalities: Sequence[str] = MODALITIES,
    ) -> None:
        super().__init__()
        requested = tuple(name for name in MODALITIES if name in set(modalities))
        unknown = set(modalities) - set(MODALITIES)
        if not requested or unknown:
            raise ValueError(
                f"modalities must contain a non-empty subset of {MODALITIES}; "
                f"unknown={sorted(unknown)}"
            )
        if config.anchor not in requested:
            raise ValueError("Track C transport modalities must contain the RNA anchor")
        self.modalities = requested
        self.DIRECTIONS = tuple(
            direction
            for direction in self.FULL_DIRECTIONS
            if direction[0] in requested and direction[1] in requested
        )
        self.translators = nn.ModuleDict(
            {_direction(source, target): GaussianResidualTranslator(config) for source, target in self.DIRECTIONS}
        )
        self.discriminators = nn.ModuleDict(
            {
                modality: DenseMLP(
                    (config.dim_c, *config.domain_discriminator_hidden, 1),
                    dropout=config.dropout,
                )
                for modality in self.modalities
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
        key = _direction(source, target)
        if key not in self.translators:
            raise ValueError(f"Unsupported Track C translation direction {source}->{target}")
        return self.translators[key](posterior)

    def to_anchor(self, posterior: GaussianPosterior, source: str) -> GaussianPosterior:
        return posterior if source == "rna" else self.translate(posterior, source, "rna")

    def domain_logits(self, values: Tensor, target: str) -> Tensor:
        return self.discriminators[target](values)

    def cycle(self, posterior: GaussianPosterior, source: str, target: str) -> GaussianPosterior:
        return self.translate(self.translate(posterior, source, target), target, source)


def cycle_consistency_loss(
    graph: GaussianTransportGraph,
    posterior: GaussianPosterior,
    source: str,
    target: str,
) -> Tensor:
    return symmetric_kl(posterior, graph.cycle(posterior, source, target))


def identity_loss(
    graph: GaussianTransportGraph,
    posterior: GaussianPosterior,
    source: str,
    target: str,
) -> Tensor:
    """CycleGAN identity: feed real target-space values through G_source->target."""

    return symmetric_kl(posterior, graph.translate(posterior, source, target))


def lsgan_discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return 0.5 * ((real_logits - 1.0).square().mean() + fake_logits.square().mean())


def lsgan_generator_loss(fake_logits: Tensor) -> Tensor:
    return 0.5 * (fake_logits - 1.0).square().mean()


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    """Freeze module weights for a generator pass while preserving input grads."""

    parameters = tuple(module.parameters())
    states = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(parameters, states):
            parameter.requires_grad_(state)


__all__ = [
    "DenseMLP",
    "GaussianResidualTranslator",
    "GaussianTransportGraph",
    "cycle_consistency_loss",
    "frozen_parameters",
    "identity_loss",
    "lsgan_discriminator_loss",
    "lsgan_generator_loss",
    "normalize_feature_dims",
]
