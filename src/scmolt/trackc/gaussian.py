"""Diagonal-Gaussian primitives used by Track C transport and fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianPosterior:
    """A batch of diagonal Gaussian posterior parameters."""

    mean: Tensor
    logvar: Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 2 or self.logvar.ndim != 2:
            raise ValueError("Gaussian mean and logvar must be rank-two tensors")
        if self.mean.shape != self.logvar.shape:
            raise ValueError("Gaussian mean and logvar must have identical shapes")

    @property
    def mu(self) -> Tensor:
        """Alias of ``mean``."""

        return self.mean

    @property
    def variance(self) -> Tensor:
        return self.logvar.exp()

    @property
    def stddev(self) -> Tensor:
        return (0.5 * self.logvar).exp()

    def rsample(self) -> Tensor:
        return self.mean + torch.randn_like(self.mean) * self.stddev

    def sample_or_mean(self, sample: bool) -> Tensor:
        return self.rsample() if sample else self.mean

    def detach(self) -> "GaussianPosterior":
        return GaussianPosterior(self.mean.detach(), self.logvar.detach())

    def to(self, *args, **kwargs) -> "GaussianPosterior":
        return GaussianPosterior(
            self.mean.to(*args, **kwargs),
            self.logvar.to(*args, **kwargs),
        )

    def select(self, mask: Tensor) -> "GaussianPosterior":
        mask = _row_mask(mask, self.mean.shape[0], self.mean.device)
        return GaussianPosterior(self.mean[mask], self.logvar[mask])

    def split(self, sizes: Sequence[int]) -> tuple["GaussianPosterior", ...]:
        means = self.mean.split(tuple(int(size) for size in sizes), dim=1)
        logvars = self.logvar.split(tuple(int(size) for size in sizes), dim=1)
        return tuple(GaussianPosterior(mu, lv) for mu, lv in zip(means, logvars))

    @staticmethod
    def cat(parts: Sequence["GaussianPosterior"]) -> "GaussianPosterior":
        if not parts:
            raise ValueError("At least one Gaussian part is required")
        return GaussianPosterior(
            torch.cat([part.mean for part in parts], dim=1),
            torch.cat([part.logvar for part in parts], dim=1),
        )


def gaussian_poe(
    experts: Sequence[GaussianPosterior],
    masks: Optional[Sequence[Tensor]] = None,
    *,
    include_standard_prior: bool = True,
    eps: float = 1e-8,
) -> GaussianPosterior:
    """Prior-inclusive Product-of-Experts matching the native fusion.

    Masks are row masks.  An inactive expert contributes exactly zero
    precision, while the standard-normal prior contributes unit precision.
    """

    if not experts:
        raise ValueError("gaussian_poe requires at least one expert")
    first = experts[0]
    batch_size, latent_dim = first.mean.shape
    if masks is not None and len(masks) != len(experts):
        raise ValueError("masks must contain one row mask per expert")
    for expert in experts:
        if expert.mean.shape != (batch_size, latent_dim):
            raise ValueError("All Gaussian experts must have the same shape")

    precision_sum = torch.ones_like(first.mean) if include_standard_prior else torch.zeros_like(first.mean)
    weighted_mean = torch.zeros_like(first.mean)
    for index, expert in enumerate(experts):
        precision = torch.exp(-expert.logvar)
        if masks is not None:
            active = _row_mask(masks[index], batch_size, first.mean.device)
            precision = precision * active.to(first.mean.dtype).unsqueeze(1)
        precision_sum = precision_sum + precision
        weighted_mean = weighted_mean + precision * expert.mean

    precision_sum = precision_sum.clamp_min(float(eps))
    return GaussianPosterior(
        weighted_mean / precision_sum,
        -torch.log(precision_sum),
    )


def diagonal_gaussian_kl(
    first: GaussianPosterior,
    second: Optional[GaussianPosterior] = None,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute ``KL(first || second)`` or ``KL(first || N(0,I))``."""

    if second is None:
        values = 0.5 * (
            first.mean.square() + first.logvar.exp() - 1.0 - first.logvar
        ).sum(dim=1)
    else:
        if first.mean.shape != second.mean.shape:
            raise ValueError("KL posteriors must have the same shape")
        variance_ratio = torch.exp(first.logvar - second.logvar)
        mean_term = (first.mean - second.mean).square() * torch.exp(-second.logvar)
        values = 0.5 * (
            second.logvar - first.logvar + variance_ratio + mean_term - 1.0
        ).sum(dim=1)
    return _reduce(values, reduction)


def symmetric_kl(
    first: GaussianPosterior,
    second: GaussianPosterior,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Symmetric diagonal-Gaussian KL, averaged across both directions."""

    forward = diagonal_gaussian_kl(first, second, reduction="none")
    reverse = diagonal_gaussian_kl(second, first, reduction="none")
    return _reduce(0.5 * (forward + reverse), reduction)


def _reduce(values: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean() if values.numel() else values.sum()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def _row_mask(mask: Tensor, batch_size: int, device: torch.device) -> Tensor:
    result = torch.as_tensor(mask, device=device)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1 or result.shape[0] != batch_size:
        raise ValueError(f"Expected a row mask with shape ({batch_size},)")
    return result.bool()


__all__ = [
    "GaussianPosterior",
    "diagonal_gaussian_kl",
    "gaussian_poe",
    "symmetric_kl",
]
