"""Non-persistent deterministic replay for unpaired diagonal training."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor

from .config import MODALITIES
from .gaussian import GaussianPosterior


class GaussianReplayBank:
    """Bounded device-local ring buffers of detached posterior parameters.

    This deliberately is *not* an ``nn.Module`` and owns no registered tensor,
    so model ``state_dict`` and checkpoints cannot persist replay samples. Both
    insertion and sampling use fixed circular cursors; no RNG state is involved.
    Buffers are allocated lazily on the producer device, avoiding a synchronous
    CUDA-to-host copy on every training minibatch.
    """

    def __init__(
        self,
        latent_dim: int,
        capacity: int = 4096,
        modalities: Sequence[str] = MODALITIES,
    ) -> None:
        if latent_dim <= 0 or capacity <= 0:
            raise ValueError("Replay latent_dim and capacity must be positive")
        requested = tuple(name for name in MODALITIES if name in set(modalities))
        unknown = set(modalities) - set(MODALITIES)
        if not requested or unknown:
            raise ValueError(
                f"modalities must contain a non-empty subset of {MODALITIES}; "
                f"unknown={sorted(unknown)}"
            )
        self.latent_dim = int(latent_dim)
        self.capacity = int(capacity)
        self.modalities = requested
        self._means: Dict[str, Tensor] = {}
        self._logvars: Dict[str, Tensor] = {}
        self._write: Dict[str, int] = {modality: 0 for modality in self.modalities}
        self._read: Dict[str, int] = {modality: 0 for modality in self.modalities}
        self._size: Dict[str, int] = {modality: 0 for modality in self.modalities}

    def clear(self) -> None:
        self._means.clear()
        self._logvars.clear()
        for modality in self.modalities:
            self._write[modality] = 0
            self._read[modality] = 0
            self._size[modality] = 0

    def __len__(self) -> int:
        return sum(self._size.values())

    def size(self, modality: str) -> int:
        self._validate_modality(modality)
        return self._size[modality]

    def primed(self, modality: str, minimum: int = 1) -> bool:
        if minimum <= 0:
            raise ValueError("minimum must be positive")
        return self.size(modality) >= minimum

    @torch.no_grad()
    def add(
        self,
        modality: str,
        posterior: GaussianPosterior,
        mask: Optional[Tensor] = None,
    ) -> int:
        self._validate_modality(modality)
        if posterior.mean.shape[1] != self.latent_dim:
            raise ValueError(
                f"Replay expects latent dimension {self.latent_dim}; "
                f"got {posterior.mean.shape[1]}"
            )
        mean = posterior.mean.detach()
        logvar = posterior.logvar.detach()
        if mask is not None:
            active = torch.as_tensor(mask, device=mean.device)
            if active.ndim == 2 and active.shape[1] == 1:
                active = active[:, 0]
            if active.ndim != 1 or active.shape[0] != mean.shape[0]:
                raise ValueError("Replay mask must contain one value per posterior row")
            mean = mean[active.bool()]
            logvar = logvar[active.bool()]
        if mean.shape[0] == 0:
            return 0
        mean = mean.to(dtype=torch.float32)
        logvar = logvar.to(dtype=torch.float32)
        if modality not in self._means:
            self._means[modality] = torch.empty(
                (self.capacity, self.latent_dim),
                device=mean.device,
                dtype=torch.float32,
            )
            self._logvars[modality] = torch.empty_like(self._means[modality])
        elif self._means[modality].device != mean.device:
            # A bank normally stays on one training device. Supporting an
            # explicit device move keeps the plain-object API robust without
            # registering replay in model/checkpoint state.
            self._means[modality] = self._means[modality].to(mean.device)
            self._logvars[modality] = self._logvars[modality].to(mean.device)

        count = min(mean.shape[0], self.capacity)
        # If a batch itself exceeds capacity, retain its final deterministic
        # suffix, which is also what repeated ring insertion would leave.
        mean = mean[-count:]
        logvar = logvar[-count:]
        write = self._write[modality]
        first = min(count, self.capacity - write)
        self._means[modality][write : write + first].copy_(mean[:first])
        self._logvars[modality][write : write + first].copy_(logvar[:first])
        remainder = count - first
        if remainder:
            self._means[modality][:remainder].copy_(mean[first:])
            self._logvars[modality][:remainder].copy_(logvar[first:])
        self._write[modality] = (write + count) % self.capacity
        self._size[modality] = min(self.capacity, self._size[modality] + count)
        return count

    def sample(
        self,
        modality: str,
        count: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Optional[GaussianPosterior]:
        self._validate_modality(modality)
        if count <= 0:
            raise ValueError("Replay sample count must be positive")
        size = self._size[modality]
        if size == 0:
            return None
        count = min(int(count), size)
        start = self._read[modality] % size
        storage_mean = self._means[modality]
        storage_logvar = self._logvars[modality]
        indices = (
            torch.arange(count, dtype=torch.long, device=storage_mean.device) + start
        ) % size
        self._read[modality] = int((start + count) % size)
        return GaussianPosterior(
            storage_mean.index_select(0, indices).to(device=device, dtype=dtype),
            storage_logvar.index_select(0, indices).to(device=device, dtype=dtype),
        )

    def sizes(self) -> Mapping[str, int]:
        return dict(self._size)

    def _validate_modality(self, modality: str) -> None:
        if modality not in self.modalities:
            raise ValueError(f"Unknown replay modality: {modality!r}")


__all__ = ["GaussianReplayBank"]
