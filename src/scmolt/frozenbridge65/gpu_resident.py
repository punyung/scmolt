"""GPU-resident training batches with the exact native scMIDAS sampler.

This module is deliberately standalone until the W0 baselines are complete.
It does not replace or patch the installed ``scmidas`` package.  The native
``MultiBatchSampler`` remains the sole authority for sample order; this file
only materializes its indices into fresh, contiguous device batches.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


def _identity(value: Any) -> Any:
    return value


class OwnedResidentBatch(dict):
    """A native-contract mapping whose ``x`` tensors are fresh and disposable.

    The marker is an attribute rather than a mapping key, so the exact
    ``{"x", "s", "e"}`` scMIDAS input contract is unchanged.  Track C can
    consequently avoid cloning an already-fresh ``index_select`` result.
    """

    trackc_owns_inputs = True


class _ZeroDataset(Dataset):
    """A zero-length map dataset used only to invoke PyTorch base-seed logic."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Any:  # pragma: no cover - never fetched
        raise IndexError(index)


def _consume_native_first_iterator_base_seed() -> None:
    """Consume exactly the global CPU base seed used by a native DataLoader.

    scMIDAS uses a persistent-worker DataLoader.  Its first iterator consumes
    one global CPU-torch random integer for worker base seeding, while later
    epoch resets do not.  The resident DataLoader has a private generator, so
    this one-time dummy iterator reproduces the native global-RNG transition
    using PyTorch's own implementation rather than duplicating internals.
    """

    loader = DataLoader(_ZeroDataset(), batch_size=None, num_workers=0)
    iterator = iter(loader)
    del iterator, loader


def _numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


@dataclass(frozen=True)
class _RNGState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...]


def _capture_rng_state() -> _RNGState:
    cuda = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    return _RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda,
    )


def _assert_same_rng_state(before: _RNGState, after: _RNGState) -> None:
    if before.python != after.python:
        raise RuntimeError("GPU-resident cache construction changed Python RNG state.")
    if not _numpy_rng_equal(before.numpy, after.numpy):
        raise RuntimeError("GPU-resident cache construction changed NumPy RNG state.")
    if not torch.equal(before.torch_cpu, after.torch_cpu):
        raise RuntimeError("GPU-resident cache construction changed CPU torch RNG state.")
    if len(before.torch_cuda) != len(after.torch_cuda) or any(
        not torch.equal(left, right)
        for left, right in zip(before.torch_cuda, after.torch_cuda)
    ):
        raise RuntimeError("GPU-resident cache construction changed CUDA RNG state.")


def _as_float_matrix(dataset: Any, modality: str) -> np.ndarray:
    source = dataset.data[modality]
    if hasattr(source, "data_frame"):
        values = source.data_frame
    elif hasattr(source, "get_all"):
        values = source.get_all()
    else:
        raise TypeError(
            "Unsupported native modality dataset for GPU residence: {}".format(
                type(source).__name__
            )
        )
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(dataset):
        raise ValueError(
            "Invalid native {} matrix shape {} for {} rows.".format(
                modality, tuple(array.shape), len(dataset)
            )
        )
    return array


@dataclass
class ResidentDataset:
    """One native batch cached on a single device."""

    x: Dict[str, Tensor]
    s: Dict[str, int]
    e: Dict[str, Tensor]
    size: int
    _versions: Dict[str, int] = field(repr=False)

    @property
    def bytes(self) -> int:
        tensors = list(self.x.values()) + list(self.e.values())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def assert_unmodified(self) -> None:
        for modality, tensor in self.x.items():
            if tensor._version != self._versions["x:" + modality]:  # noqa: SLF001
                raise RuntimeError(
                    "Resident x cache was modified in place for {}.".format(modality)
                )
        for modality, tensor in self.e.items():
            if tensor._version != self._versions["e:" + modality]:  # noqa: SLF001
                raise RuntimeError(
                    "Resident e cache was modified in place for {}.".format(modality)
                )


@dataclass
class GpuResidentCache:
    """Device-resident matrices plus compact feature masks and batch labels."""

    datasets: tuple[ResidentDataset, ...]
    cumulative_sizes: tuple[int, ...]
    device: torch.device
    source_datasets: tuple[Any, ...] = field(default=(), repr=False)

    @property
    def bytes(self) -> int:
        return sum(dataset.bytes for dataset in self.datasets)

    @property
    def total_rows(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def dataset_index(self, global_index: int) -> int:
        value = int(global_index)
        if value < 0 or value >= self.total_rows:
            raise IndexError("Global native index is out of range: {}".format(value))
        return bisect_right(self.cumulative_sizes, value)

    def dataset_offset(self, dataset_index: int) -> int:
        return 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]

    def assert_unmodified(self) -> None:
        for dataset in self.datasets:
            dataset.assert_unmodified()


def build_gpu_cache(
    datalist: Sequence[Any], device: torch.device | str
) -> GpuResidentCache:
    """Copy the native datalist to a device without advancing any RNG."""

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU residence was requested but CUDA is unavailable.")
    before = _capture_rng_state()
    resident = []
    cumulative = []
    total = 0
    for native in datalist:
        transform = dict(getattr(native, "transform", {}))
        if transform != {"atac": "binarize"}:
            raise ValueError(
                "Track C resident cache requires native transform "
                "{{'atac': 'binarize'}}; found {}.".format(transform)
            )
        x: Dict[str, Tensor] = {}
        for modality in native.data:
            tensor = torch.as_tensor(_as_float_matrix(native, modality)).to(
                device=target, dtype=torch.float32, copy=True
            )
            if modality == "atac":
                # This exactly matches scmidas TransformRegistry.binarize.
                tensor = (tensor > 0.5).to(dtype=torch.float32)
            x[modality] = tensor.contiguous()

        labels = {key: int(value) for key, value in native.mod_id_dict.items()}
        expected_labels = set(x) | {"joint"}
        if set(labels) != expected_labels:
            raise ValueError(
                "Native batch labels differ from observed modalities: {} vs {}.".format(
                    sorted(labels), sorted(expected_labels)
                )
            )

        e: Dict[str, Tensor] = {}
        for modality, values in (native.mask or {}).items():
            mask = torch.as_tensor(np.asarray(values, dtype=np.float32)).to(
                device=target, dtype=torch.float32, copy=True
            )
            mask = mask.reshape(-1).contiguous()
            if modality not in x or mask.numel() != x[modality].shape[1]:
                raise ValueError(
                    "Native feature mask does not match cached {} matrix.".format(
                        modality
                    )
                )
            e[modality] = mask

        versions = {
            **{"x:" + key: value._version for key, value in x.items()},  # noqa: SLF001
            **{"e:" + key: value._version for key, value in e.items()},  # noqa: SLF001
        }
        size = int(len(native))
        resident.append(ResidentDataset(x=x, s=labels, e=e, size=size, _versions=versions))
        total += size
        cumulative.append(total)

    cache = GpuResidentCache(
        datasets=tuple(resident),
        cumulative_sizes=tuple(cumulative),
        device=target,
        source_datasets=tuple(datalist),
    )
    _assert_same_rng_state(before, _capture_rng_state())
    return cache


class GpuBatchDataset(Dataset):
    """Map-style dataset whose vectorized fetch returns a complete GPU batch."""

    def __init__(self, cache: GpuResidentCache, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.cache = cache
        self.batch_size = int(batch_size)
        # Labels and feature masks are immutable under the native VAE contract.
        # Materialize their full contiguous batch shapes once instead of on
        # every minibatch; only x needs a fresh index_select allocation.
        self._batch_targets = tuple(
            {
                modality: torch.full(
                    (self.batch_size, 1),
                    source.s[modality],
                    device=self.cache.device,
                    dtype=torch.int64,
                )
                for modality in (*source.x, "joint")
            }
            for source in self.cache.datasets
        )
        self._event_masks = tuple(
            {
                modality: values.unsqueeze(0)
                .expand(self.batch_size, -1)
                .contiguous()
                for modality, values in source.e.items()
            }
            for source in self.cache.datasets
        )

    def __len__(self) -> int:
        return self.cache.total_rows

    def __getitem__(self, index: int) -> Any:  # pragma: no cover - guarded path
        raise RuntimeError("GpuBatchDataset requires vectorized __getitems__ access.")

    def __getitems__(self, global_indices: Sequence[int]) -> Dict[str, Dict[str, Tensor]]:
        indices = [int(value) for value in global_indices]
        if len(indices) != self.batch_size:
            raise RuntimeError(
                "Native resident batch has {} rows, expected {}.".format(
                    len(indices), self.batch_size
                )
            )
        dataset_index = self.cache.dataset_index(indices[0])
        if any(self.cache.dataset_index(value) != dataset_index for value in indices):
            raise RuntimeError("Native sampler produced a cross-dataset minibatch.")
        source = self.cache.datasets[dataset_index]
        offset = self.cache.dataset_offset(dataset_index)
        local_indices = [value - offset for value in indices]
        if min(local_indices) < 0 or max(local_indices) >= source.size:
            raise RuntimeError("Native sampler produced an invalid local index.")
        local = torch.tensor(
            local_indices, dtype=torch.long, device=self.cache.device
        )

        # index_select is mandatory: the native encoder masks x in place.
        x = {
            modality: values.index_select(0, local)
            for modality, values in source.x.items()
        }
        return OwnedResidentBatch(
            x=x,
            s=dict(self._batch_targets[dataset_index]),
            e=dict(self._event_masks[dataset_index]),
        )


class NativeScheduleSampler(Sampler[int]):
    """Audit and delegate one native MultiBatchSampler schedule per epoch."""

    def __init__(
        self,
        native_sampler: Sampler[int],
        cache: GpuResidentCache,
        batch_size: int,
        *,
        stream_name: str = "training",
    ) -> None:
        if int(getattr(native_sampler, "batch_size", batch_size)) != int(batch_size):
            raise ValueError("Native sampler batch size differs from resident loader.")
        self.native_sampler = native_sampler
        self.cache = cache
        self.batch_size = int(batch_size)
        self.stream_name = str(stream_name)
        self._consumed_first_base_seed = False
        self.epoch_records: list[Dict[str, Any]] = []

    def __len__(self) -> int:
        return int(len(self.native_sampler))

    @property
    def last_record(self) -> Mapping[str, Any] | None:
        return self.epoch_records[-1] if self.epoch_records else None

    def __iter__(self) -> Iterator[int]:
        if not self._consumed_first_base_seed:
            _consume_native_first_iterator_base_seed()
            self._consumed_first_base_seed = True

        indices = [int(value) for value in iter(self.native_sampler)]
        expected_rows = len(self)
        if len(indices) != expected_rows or expected_rows % self.batch_size:
            raise RuntimeError(
                "Native sampler yielded {} rows; expected {} divisible by {}.".format(
                    len(indices), expected_rows, self.batch_size
                )
            )
        counts = [0 for _ in self.cache.datasets]
        for start in range(0, len(indices), self.batch_size):
            block = indices[start : start + self.batch_size]
            dataset_index = self.cache.dataset_index(block[0])
            if any(self.cache.dataset_index(value) != dataset_index for value in block):
                raise RuntimeError("Native sampler mixed datasets inside a minibatch.")
            counts[dataset_index] += len(block)
        if len(set(counts)) != 1:
            raise RuntimeError(
                "Native MultiBatchSampler draw counts are unbalanced: {}.".format(counts)
            )
        encoded = np.asarray(indices, dtype="<i8").tobytes(order="C")
        self.epoch_records.append(
            {
                "stream": self.stream_name,
                "epoch_index": len(self.epoch_records),
                "schedule_sha256": hashlib.sha256(encoded).hexdigest(),
                "rows": len(indices),
                "batches": len(indices) // self.batch_size,
                "dataset_draw_counts": counts,
            }
        )
        return iter(indices)


@dataclass
class GpuResidentLoaderBundle:
    loader: DataLoader
    cache: GpuResidentCache
    schedule_sampler: NativeScheduleSampler


def schedule_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compactly bind an ordered sequence of native sampler schedules."""

    copied = [dict(record) for record in records]
    digest = hashlib.sha256()
    for record in copied:
        digest.update(bytes.fromhex(str(record["schedule_sha256"])))
    return {
        "record_count": len(copied),
        "schedule_sequence_sha256": digest.hexdigest(),
        "first": copied[0] if copied else None,
        "last": copied[-1] if copied else None,
    }


def build_gpu_resident_loader(
    datalist: Sequence[Any],
    native_sampler: Sampler[int],
    device: torch.device | str,
    *,
    batch_size: int,
    requested_n_max: int = 10_000,
    private_seed: int = 9_731_337,
    require_scmidas_contract: bool = True,
    cache: GpuResidentCache | None = None,
    stream_name: str = "training",
) -> GpuResidentLoaderBundle:
    """Build a standard DataLoader with device batches and native semantics."""

    sampler_data = getattr(native_sampler, "data", None)
    sampler_datasets = getattr(sampler_data, "datasets", None)
    sampler_cumulative = getattr(sampler_data, "cumulative_sizes", None)
    if sampler_datasets is None or sampler_cumulative is None:
        raise TypeError("Resident loader requires a native ConcatDataset sampler.")
    if len(sampler_datasets) != len(datalist) or any(
        sampled is not requested
        for sampled, requested in zip(sampler_datasets, datalist)
    ):
        raise ValueError("Native sampler is not bound to the requested datalist.")
    expected_cumulative = tuple(np.cumsum([len(data) for data in datalist]).tolist())
    if tuple(map(int, sampler_cumulative)) != expected_cumulative:
        raise ValueError("Native sampler cumulative sizes differ from the datalist.")
    if require_scmidas_contract:
        if int(requested_n_max) <= 0:
            raise ValueError("requested_n_max must be positive.")
        contract = {
            "type": type(native_sampler).__name__,
            "batch_size": int(getattr(native_sampler, "batch_size", -1)),
            "n_max": int(getattr(native_sampler, "n_max", -1)),
            "shuffle": bool(getattr(native_sampler, "shuffle", False)),
        }
        required = {
            "type": "MultiBatchSampler",
            "batch_size": int(batch_size),
            # scmidas.data.MultiBatchSampler stores the effective cap rather
            # than the requested one. Small datasets therefore legitimately
            # expose max(dataset sizes), while the formal protocol still
            # requests n_max=10,000.
            "n_max": min(
                max(len(dataset) for dataset in datalist),
                int(requested_n_max),
            ),
            "shuffle": True,
        }
        if contract != required:
            raise ValueError(
                "Installed scMIDAS sampler contract drifted: {} != {}.".format(
                    contract, required
                )
            )

    target = torch.device(device)
    if cache is None:
        cache = build_gpu_cache(datalist, target)
    else:
        if cache.device != target:
            raise ValueError(
                "Resident cache device {} differs from requested {}.".format(
                    cache.device, target
                )
            )
        if len(cache.source_datasets) != len(datalist) or any(
            cached is not requested
            for cached, requested in zip(cache.source_datasets, datalist)
        ):
            raise ValueError("Resident cache is not bound to the requested datalist.")
        if cache.cumulative_sizes != expected_cumulative:
            raise ValueError("Resident cache cumulative sizes differ from the datalist.")
    dataset = GpuBatchDataset(cache, batch_size=batch_size)
    schedule = NativeScheduleSampler(
        native_sampler,
        cache,
        batch_size=batch_size,
        stream_name=stream_name,
    )
    generator = torch.Generator()
    generator.manual_seed(int(private_seed))
    loader = DataLoader(
        dataset,
        sampler=schedule,
        batch_size=int(batch_size),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=_identity,
        generator=generator,
    )
    if len(loader) * int(batch_size) != len(schedule):
        raise RuntimeError("Resident loader changed native step accounting.")
    return GpuResidentLoaderBundle(loader=loader, cache=cache, schedule_sampler=schedule)


__all__ = [
    "GpuBatchDataset",
    "GpuResidentCache",
    "GpuResidentLoaderBundle",
    "NativeScheduleSampler",
    "OwnedResidentBatch",
    "ResidentDataset",
    "build_gpu_cache",
    "build_gpu_resident_loader",
    "schedule_summary",
]
