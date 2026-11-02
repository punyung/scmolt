"""Leakage-safe re-masking helpers for Track C.

Historical BMMC profiles remove complete RNA observations from selected paired
rows.  New anchor-preserving profiles instead hide feature entries inside RNA
and ATAC while retaining both modalities in the encoder input.  In both cases,
targets are returned separately and detached from the model input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class RemaskedModalityBatch:
    """A model batch with one observed modality removed plus its held-out target."""

    model_batch: Dict[str, Dict[str, Tensor]]
    target: Tensor
    target_event_mask: Optional[Tensor]
    row_indices: Tensor
    target_modality: str
    conditioning_modalities: Tuple[str, ...]
    original_batch_size: int

    @property
    def remasked_rows(self) -> int:
        return int(self.row_indices.numel())

    @property
    def row_mask(self) -> Tensor:
        """Return selected rows as a full-batch boolean mask."""

        mask = torch.zeros(
            self.original_batch_size,
            dtype=torch.bool,
            device=self.row_indices.device,
        )
        mask[self.row_indices] = True
        return mask


@dataclass(frozen=True)
class FeatureRemaskedBatch:
    """A paired batch with partially hidden features in observed modalities.

    ``model_batch`` retains every original modality key.  Artificially hidden
    values are zeroed in ``x`` and marked unavailable in ``e``.  The detached
    originals and loss-only masks live in ``targets`` and
    ``target_event_masks``; they are never inserted into the model input.
    """

    model_batch: Dict[str, Dict[str, Tensor]]
    targets: Mapping[str, Tensor]
    target_event_masks: Mapping[str, Tensor]
    feature_masks: Mapping[str, Tensor]
    row_indices: Tensor
    target_modalities: Tuple[str, ...]
    anchor_modality: str
    original_batch_size: int

    @property
    def remasked_rows(self) -> int:
        return int(self.row_indices.numel())

    @property
    def row_mask(self) -> Tensor:
        mask = torch.zeros(
            self.original_batch_size,
            dtype=torch.bool,
            device=self.row_indices.device,
        )
        mask[self.row_indices] = True
        return mask

    @property
    def masked_elements(self) -> Dict[str, int]:
        return {
            modality: int(mask.count_nonzero().item())
            for modality, mask in self.feature_masks.items()
        }

    @property
    def available_elements(self) -> Dict[str, int]:
        return {
            modality: int(
                (
                    torch.as_tensor(self.model_batch["e"][modality]).bool()
                    | torch.as_tensor(self.feature_masks[modality]).bool()
                )
                .count_nonzero()
                .item()
            )
            for modality in self.target_modalities
        }

    @property
    def realized_feature_fractions(self) -> Dict[str, float]:
        available = self.available_elements
        return {
            modality: (
                self.masked_elements[modality] / available[modality]
                if available[modality] > 0
                else 0.0
            )
            for modality in self.target_modalities
        }


def _row_slice(value: Tensor, rows: Tensor, batch_size: int) -> Tensor:
    """Slice a row-dependent tensor while retaining feature-only masks."""

    if value.ndim >= 2 and value.shape[0] == batch_size:
        return value.index_select(0, rows)
    return value


def _broadcast_event_mask(
    event_mask: Optional[Tensor], target: Tensor
) -> Tensor:
    if event_mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    mask = torch.as_tensor(event_mask, device=target.device)
    try:
        torch.broadcast_shapes(target.shape, mask.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"event mask shape {tuple(mask.shape)} is not broadcastable to "
            f"target shape {tuple(target.shape)}"
        ) from exc
    return torch.broadcast_to(mask, target.shape).bool()


def _sample_partial_feature_mask(
    available: Tensor,
    *,
    fraction: float,
    generator: Optional[torch.Generator],
    modality: str,
) -> Tensor:
    """Sample entry masks while retaining supervision and visible features."""

    sufficient = torch.all(available.sum(dim=1) >= 2)
    if sufficient.device.type == "cuda":
        torch._assert_async(
            sufficient,
            f"feature re-masking requires at least two observed {modality} "
            "features per selected row",
        )
    elif not bool(sufficient):
        raise ValueError(
            f"feature re-masking requires at least two observed {modality} "
            "features per selected row"
        )
    random_values = torch.rand(
        available.shape,
        device=available.device,
        generator=generator,
    )
    sampled = (random_values < fraction) & available
    # Force one hidden and one visible observed feature without per-row GPU
    # synchronization.  Available random values are <1 while unavailable
    # entries are replaced by 2, so the two selected indices are always valid
    # and distinct after the assertion above.
    mandatory = torch.topk(
        random_values.masked_fill(~available, 2.0),
        k=2,
        dim=1,
        largest=False,
    ).indices
    rows = torch.arange(available.shape[0], device=available.device)
    sampled[rows, mandatory[:, 0]] = True
    sampled[rows, mandatory[:, 1]] = False
    return sampled


def remask_observed_features(
    batch: Mapping[str, Any],
    *,
    feature_fractions: Mapping[str, float],
    row_fraction: float = 1.0,
    anchor_modality: str = "rna",
    generator: Optional[torch.Generator] = None,
) -> FeatureRemaskedBatch:
    """Partially hide RNA/ATAC features without removing the RNA anchor.

    A uniform subset of paired rows is selected without replacement.  Within
    every selected row, each requested modality receives an independent
    Bernoulli feature mask restricted to originally observed entries.  At
    least one target and one visible feature are enforced per row and modality,
    so this helper can never degrade into whole-modality dropout.
    """

    try:
        row_fraction_value = float(row_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("row_fraction must be a finite number in (0, 1]") from exc
    if (
        not math.isfinite(row_fraction_value)
        or not 0.0 < row_fraction_value <= 1.0
    ):
        raise ValueError("row_fraction must be a finite number in (0, 1]")
    if not isinstance(feature_fractions, Mapping) or not feature_fractions:
        raise ValueError("feature_fractions must define at least one modality")
    normalized_fractions: Dict[str, float] = {}
    for modality, value in feature_fractions.items():
        try:
            fraction_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"feature fraction for {modality!r} must be finite and in (0, 1)"
            ) from exc
        if not math.isfinite(fraction_value) or not 0.0 < fraction_value < 1.0:
            raise ValueError(
                f"feature fraction for {modality!r} must be finite and in (0, 1)"
            )
        normalized_fractions[str(modality)] = fraction_value

    inputs = batch.get("x")
    event_masks = batch.get("e")
    batch_targets = batch.get("s")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("feature re-masking requires a non-empty batch['x'] mapping")
    if not isinstance(event_masks, Mapping):
        raise ValueError("feature re-masking requires batch['e']")
    if not isinstance(batch_targets, Mapping) or "joint" not in batch_targets:
        raise ValueError("feature re-masking requires batch['s']['joint']")
    if anchor_modality not in inputs:
        raise ValueError(f"anchor modality {anchor_modality!r} is not observed")
    missing_modalities = set(normalized_fractions) - set(inputs)
    if missing_modalities:
        raise ValueError(
            f"feature re-masking modalities are not observed: {sorted(missing_modalities)}"
        )

    first = torch.as_tensor(inputs[anchor_modality])
    if first.ndim != 2 or first.shape[0] <= 0:
        raise ValueError("anchor modality must be a non-empty rank-two tensor")
    batch_size = int(first.shape[0])
    for modality, value in inputs.items():
        tensor = torch.as_tensor(value)
        if tensor.ndim != 2 or tensor.shape[0] != batch_size:
            raise ValueError(
                f"observed modality {modality!r} does not share the anchor rows"
            )

    remasked_rows = max(
        1, min(batch_size, int(round(batch_size * row_fraction_value)))
    )
    rows = torch.randperm(
        batch_size,
        device=first.device,
        generator=generator,
    )[:remasked_rows]
    rows = rows.sort().values

    remasked_inputs = {
        modality: torch.as_tensor(value).index_select(0, rows).clone()
        for modality, value in inputs.items()
    }
    remasked_events = {
        modality: _row_slice(
            torch.as_tensor(value), rows, batch_size
        ).clone()
        for modality, value in event_masks.items()
        if modality in remasked_inputs
    }
    remasked_targets = {
        key: _row_slice(torch.as_tensor(value), rows, batch_size).clone()
        for key, value in batch_targets.items()
        if key == "joint" or key in remasked_inputs
    }
    targets: Dict[str, Tensor] = {}
    target_event_masks: Dict[str, Tensor] = {}
    feature_masks: Dict[str, Tensor] = {}
    for modality, fraction in normalized_fractions.items():
        target = remasked_inputs[modality].detach().clone()
        available = _broadcast_event_mask(
            remasked_events.get(modality), target
        )
        feature_mask = _sample_partial_feature_mask(
            available,
            fraction=fraction,
            generator=generator,
            modality=modality,
        )
        corrupted = remasked_inputs[modality]
        corrupted.masked_fill_(feature_mask, 0)
        remasked_events[modality] = (available & ~feature_mask).to(target.dtype)
        targets[modality] = target
        target_event_masks[modality] = feature_mask.to(target.dtype)
        feature_masks[modality] = feature_mask

    if anchor_modality not in remasked_inputs or anchor_modality not in remasked_events:
        raise RuntimeError("feature re-masking removed the anchor modality")
    anchor_visible = _broadcast_event_mask(
        remasked_events[anchor_modality], remasked_inputs[anchor_modality]
    ).sum(dim=1)
    if not bool(torch.all(anchor_visible > 0).item()):
        raise RuntimeError("feature re-masking removed every RNA anchor feature")
    model_batch = {
        "x": remasked_inputs,
        "e": remasked_events,
        "s": remasked_targets,
    }
    return FeatureRemaskedBatch(
        model_batch=model_batch,
        targets=targets,
        target_event_masks=target_event_masks,
        feature_masks=feature_masks,
        row_indices=rows,
        target_modalities=tuple(normalized_fractions),
        anchor_modality=str(anchor_modality),
        original_batch_size=batch_size,
    )


def remask_observed_modality(
    batch: Mapping[str, Any],
    *,
    target_modality: str,
    fraction: float,
    generator: Optional[torch.Generator] = None,
) -> RemaskedModalityBatch:
    """Uniformly re-mask complete rows of an observed target modality.

    Sampling is without replacement.  At least one row is selected whenever
    the function is called, and the selected target is detached so gradients
    can only enter through the conditioning modalities and decoder.
    """

    try:
        fraction_value = float(fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fraction must be a finite number in (0, 1]") from exc
    if not math.isfinite(fraction_value) or not 0.0 < fraction_value <= 1.0:
        raise ValueError("fraction must be a finite number in (0, 1]")

    inputs = batch.get("x")
    event_masks = batch.get("e")
    batch_targets = batch.get("s")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("re-masking requires a non-empty batch['x'] mapping")
    if not isinstance(event_masks, Mapping):
        raise ValueError("re-masking requires batch['e']")
    if not isinstance(batch_targets, Mapping) or "joint" not in batch_targets:
        raise ValueError("re-masking requires batch['s']['joint']")
    if target_modality not in inputs:
        raise ValueError(f"target modality {target_modality!r} is not observed")

    target = torch.as_tensor(inputs[target_modality])
    if target.ndim != 2 or target.shape[0] <= 0:
        raise ValueError("target modality must be a non-empty rank-two tensor")
    batch_size = int(target.shape[0])
    conditioning = tuple(key for key in inputs if key != target_modality)
    if not conditioning:
        raise ValueError("re-masking requires at least one conditioning modality")
    for modality in conditioning:
        value = torch.as_tensor(inputs[modality])
        if value.ndim != 2 or value.shape[0] != batch_size:
            raise ValueError(
                f"conditioning modality {modality!r} does not share the target rows"
            )

    remasked_rows = max(1, min(batch_size, int(round(batch_size * fraction_value))))
    rows = torch.randperm(
        batch_size,
        device=target.device,
        generator=generator,
    )[:remasked_rows]
    # Sorting does not change the uniformly sampled set and makes subsequent
    # index_select operations deterministic and memory-coalesced.
    rows = rows.sort().values

    remasked_inputs = {
        modality: torch.as_tensor(inputs[modality]).index_select(0, rows)
        for modality in conditioning
    }
    remasked_events = {
        modality: _row_slice(
            torch.as_tensor(event_masks[modality]), rows, batch_size
        )
        for modality in conditioning
        if modality in event_masks
    }
    remasked_targets = {
        key: _row_slice(torch.as_tensor(value), rows, batch_size)
        for key, value in batch_targets.items()
        if key == "joint" or key in conditioning
    }
    if "joint" not in remasked_targets:
        raise ValueError("re-masked batch lost the joint batch target")

    target_event_mask = None
    if target_modality in event_masks:
        target_event_mask = _row_slice(
            torch.as_tensor(event_masks[target_modality]), rows, batch_size
        )
    model_batch = {
        "x": remasked_inputs,
        "e": remasked_events,
        "s": remasked_targets,
    }
    if any(target_modality in model_batch[group] for group in ("x", "e", "s")):
        raise RuntimeError("target modality leaked into the re-masked model batch")
    return RemaskedModalityBatch(
        model_batch=model_batch,
        target=target.index_select(0, rows).detach(),
        target_event_mask=(
            None if target_event_mask is None else target_event_mask.detach()
        ),
        row_indices=rows,
        target_modality=target_modality,
        conditioning_modalities=conditioning,
        original_batch_size=batch_size,
    )


__all__ = [
    "FeatureRemaskedBatch",
    "RemaskedModalityBatch",
    "remask_observed_features",
    "remask_observed_modality",
]
