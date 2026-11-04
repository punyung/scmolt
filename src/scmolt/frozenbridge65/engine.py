"""Portable native-model construction, training, reload, and export."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import platform
import random
import resource
import time
from typing import Any, Iterator, Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..trackc import remask_observed_features
from ..trackc_adbridge import TrackCADBridgeMIDASCycle

from .config import (
    BATCH_SIZE,
    BRIDGE_WEIGHTS,
    CHECKPOINT_EVERY_EPOCHS,
    EPOCHS,
    GPU_RESIDENT_PRIVATE_SEED,
    METHOD_NAME,
    METHOD_SLUG,
    MODALITIES,
    N_MAX,
    REMASK_FEATURE_FRACTIONS,
    REMASK_PRIVATE_SEED,
    REMASK_RECONSTRUCTION_WEIGHTS,
    SEED,
    assert_model_contract,
    bridge_config,
    collect_core_source_hashes,
    core_config,
    experiment_receipt,
    installed_versions,
    version_drift,
)
from .gpu_resident import (
    GpuResidentLoaderBundle,
    build_gpu_cache,
    build_gpu_resident_loader,
    schedule_summary,
)
from .io import append_jsonl, atomic_json, ordered_hash, sha256_file, validate_input


@dataclass
class NativeContext:
    native: Any
    model: TrackCADBridgeMIDASCycle
    loader: Any
    bundle: GpuResidentLoaderBundle | None
    native_config: dict[str, Any]
    input_dir: Path
    input_manifest: dict[str, Any]
    device: torch.device
    backend: str


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: move_batch(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch(item, device) for item in value)
    if isinstance(value, list):
        return [move_batch(item, device) for item in value]
    return value


def _assert_native_defaults(config: Mapping[str, Any]) -> None:
    expected = {
        "dim_c": 32,
        "dim_u": 2,
        "dims_shared_enc": [1024, 128],
        "dims_shared_dec": [128, 1024],
        "drop": 0.2,
        "s_drop_rate": 0.1,
        "lam_kld_c": 1,
        "lam_kld_u": 5,
        "lam_recon_s": 1000,
        "lam_adv": 1,
        "n_iter_disc": 3,
        "optim_net": "AdamW",
        "optim_dsc": "AdamW",
        "lr_net": 1e-4,
        "lr_dsc": 1e-4,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "n_max": N_MAX,
        "norm": "ln",
        "out_trans": "mish",
        "distribution_dec_rna": "POISSON",
        "distribution_dec_atac": "BERNOULLI",
        "distribution_dec_adt": "POISSON",
    }
    drift = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if drift:
        raise RuntimeError(f"Installed scMIDAS defaults drifted: {drift}")


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _build_model(
    native: Any,
    *,
    device: torch.device,
    topology_alias: str,
    fresh_native_modules: bool,
) -> TrackCADBridgeMIDASCycle:
    if fresh_native_modules:
        ordered_dims_x = {
            str(modality): list(blocks) for modality, blocks in native.dims_x.items()
        }
        dims_s = {str(key): int(value) for key, value in native.dims_s.items()}
        native_configs = dict(native.configs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 900_003)
            native_vae = type(native.net)(ordered_dims_x, dims_s, **native_configs)
            native_discriminator = type(native.dsc)(
                ordered_dims_x, dims_s, **native_configs
            )
    else:
        native_vae = native.net
        native_discriminator = native.dsc
    model = TrackCADBridgeMIDASCycle(
        feature_dims=native.dims_x,
        n_batches=int(native.dims_s["joint"]),
        config=core_config(topology_alias=topology_alias),
        bridge_config=bridge_config(),
        batch_class_dims=native.dims_s,
        native_configs=native.configs,
        native_vae=native_vae,
        native_discriminator=native_discriminator,
    ).to(device)
    assert_model_contract(model)
    expected_native_order = tuple(map(str, native.dims_x))
    for name, module in {
        "vae": model.native_vae,
        "encoder": model.native_vae.encoder,
        "decoder": model.native_vae.decoder,
        "discriminator": model.native_discriminator,
    }.items():
        if tuple(map(str, module.dims_x)) != expected_native_order:
            raise RuntimeError(f"Native modality order drifted in {name}")
    return model


def _configure_grouped_data(
    midas: Any,
    native_config: Mapping[str, Any],
    input_dir: Path,
    runtime_dir: Path,
    manifest: Mapping[str, Any],
) -> Any:
    """``configure_data_from_dir`` with the shared batch labels of ``feat/batch_groups.toml``.

    The datasets keep their label dictionaries by reference, so relabelling them
    before ``configure_data`` sets both the labels of every minibatch and the
    batch dimensions of the native modules built there.
    """
    root = str(Path(input_dir).expanduser().resolve())
    data, mask, dims_x = midas.get_info_from_dir(root, "csv")
    datalist, dims_s, s_joint, combs = midas.get_datasets_from_dir(
        data, mask, {"atac": "binarize"}, "csv"
    )
    labels = manifest["batch_groups"]["labels"]
    batch_names = list(manifest["batch_order"])
    if len(batch_names) != len(s_joint):
        raise RuntimeError("Native batch folders differ from the validated input")
    for name, record in zip(batch_names, s_joint):
        if set(record) != set(labels[name]):
            raise RuntimeError(f"{name}: native modalities differ from the validated input")
        record.update(labels[name])
    dims_s = {
        key: 1 + max(record[key] for record in s_joint if key in record) for key in dims_s
    }
    midas.start_epoch = 0
    midas.load_optimizer_state = False
    return midas.configure_data(
        native_config,
        datalist,
        dims_x,
        dims_s,
        s_joint,
        combs,
        save_model_path=str((runtime_dir / "native_checkpoints").resolve()),
        batch_size=BATCH_SIZE,
        n_save=EPOCHS + 1,
        sampler_type="auto",
        viz_umap_tb=False,
    )


def configure_context(
    *,
    input_dir: Path,
    runtime_dir: Path,
    device: str,
    backend: str,
    topology_alias: str = "full",
    hash_input: bool = False,
    allow_runtime_drift: bool = False,
) -> NativeContext:
    if backend not in {"gpu_resident_v1", "native_workers"}:
        raise ValueError("backend must be gpu_resident_v1 or native_workers")
    resolved_device = _resolve_device(device)
    if backend == "gpu_resident_v1" and resolved_device.type != "cuda":
        # The implementation can reside on CPU, but calling it gpu_resident in
        # a formal receipt would be misleading.
        backend = "native_workers"
    drift = version_drift()
    if drift and not allow_runtime_drift:
        raise RuntimeError(
            f"Runtime package versions differ from the validated environment: {drift}. "
            "Use the bundled environment.yml or pass --allow-runtime-drift for an exploratory run."
        )
    manifest = validate_input(input_dir, hash_contents=hash_input)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str((runtime_dir / "numba_cache").resolve())
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    import lightning
    from scmidas.config import load_config
    from scmidas.model import MIDAS

    lightning.seed_everything(SEED, workers=True)
    installed = dict(load_config())
    installed.update(
        {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    )
    _assert_native_defaults(installed)
    native_config = dict(installed)
    native_config["lam_alignment"] = 0.0
    native_config["lam_dsc"] = 40.0
    if manifest.get("batch_groups") is None:
        native = MIDAS.configure_data_from_dir(
            native_config,
            str(Path(input_dir).expanduser().resolve()),
            transform={"atac": "binarize"},
            save_model_path=str((runtime_dir / "native_checkpoints").resolve()),
            format="csv",
            batch_size=BATCH_SIZE,
            n_save=EPOCHS + 1,
            sampler_type="auto",
            viz_umap_tb=False,
        )
    else:
        native = _configure_grouped_data(
            MIDAS, native_config, input_dir, runtime_dir, manifest
        )
    native_loader = native.train_dataloader()
    sampler = native_loader.sampler
    required_sampler = {
        "type": "MultiBatchSampler",
        "batch_size": BATCH_SIZE,
        "n_max": min(max(len(dataset) for dataset in native.datalist), N_MAX),
        "shuffle": True,
    }
    observed_sampler = {
        "type": type(sampler).__name__,
        "batch_size": int(getattr(sampler, "batch_size", -1)),
        "n_max": int(getattr(sampler, "n_max", -1)),
        "shuffle": bool(getattr(sampler, "shuffle", False)),
    }
    if observed_sampler != required_sampler:
        raise RuntimeError(
            f"Native MultiBatchSampler contract drifted: {observed_sampler} != {required_sampler}"
        )
    bundle: GpuResidentLoaderBundle | None = None
    loader: Any = native_loader
    if backend == "gpu_resident_v1":
        cache = build_gpu_cache(native.datalist, resolved_device)
        bundle = build_gpu_resident_loader(
            native.datalist,
            sampler,
            resolved_device,
            batch_size=BATCH_SIZE,
            requested_n_max=N_MAX,
            private_seed=GPU_RESIDENT_PRIVATE_SEED,
            cache=cache,
            stream_name="training_and_diagnostic",
        )
        loader = bundle.loader
    model = _build_model(
        native,
        device=resolved_device,
        topology_alias=topology_alias,
        fresh_native_modules=False,
    )
    return NativeContext(
        native=native,
        model=model,
        loader=loader,
        bundle=bundle,
        native_config=native_config,
        input_dir=Path(input_dir).expanduser().resolve(),
        input_manifest=manifest,
        device=resolved_device,
        backend=backend,
    )


def _iter_training_batches(context: NativeContext) -> Iterator[Mapping[str, Any]]:
    for batch in context.loader:
        yield batch if context.bundle is not None else move_batch(batch, context.device)


def finite(value: torch.Tensor, name: str) -> None:
    if value.ndim != 0 or not bool(torch.isfinite(value).detach().cpu()):
        raise FloatingPointError(f"{name} is not a finite scalar: {value}")


def _remask_generator(model: TrackCADBridgeMIDASCycle) -> torch.Generator:
    generator = torch.Generator(device=next(model.parameters()).device)
    generator.manual_seed(REMASK_PRIVATE_SEED)
    return generator


def feature_remask_objective(
    *,
    model: TrackCADBridgeMIDASCycle,
    batch: Mapping[str, Any],
    cycle_scale: float,
    generator: torch.Generator,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    inputs = batch.get("x", {})
    observed = tuple(modality for modality in MODALITIES if modality in inputs)
    rows = int(batch["s"]["joint"].shape[0])
    if not observed or cycle_scale <= 0.0:
        return zero, {
            "remask_active": False,
            "remask_eligible_rows": rows if observed else 0,
            "remask_target_rows": 0,
            "remask_target_modalities": [],
            "rna_genuinely_observed": "rna" in observed,
        }
    fractions = {key: REMASK_FEATURE_FRACTIONS[key] for key in observed}
    visibility_anchor = "rna" if "rna" in observed else observed[0]
    remasked = remask_observed_features(
        batch,
        feature_fractions=fractions,
        row_fraction=0.25,
        anchor_modality=visibility_anchor,
        generator=generator,
    )
    output = model.forward(remasked.model_batch, cycle_scale=cycle_scale, sample=True)
    if set(output.inputs) != set(inputs):
        raise RuntimeError("Feature re-masking changed observed modality keys")
    total = zero
    diagnostics: dict[str, Any] = {
        "remask_active": True,
        "remask_eligible_rows": remasked.original_batch_size,
        "remask_target_rows": remasked.remasked_rows,
        "remask_target_modalities": list(remasked.target_modalities),
        "rna_genuinely_observed": "rna" in observed,
        "visibility_anchor": visibility_anchor,
    }
    for modality in remasked.target_modalities:
        raw = model.modality_reconstruction_loss(
            output,
            modality=modality,
            target=remasked.targets[modality],
            event_mask=remasked.target_event_masks[modality],
            normalize_by_event_count=True,
        )
        weighted = raw * float(cycle_scale) * REMASK_RECONSTRUCTION_WEIGHTS[modality]
        total = total + weighted
        diagnostics[f"remask_{modality}_masked_elements"] = int(
            remasked.masked_elements[modality]
        )
        diagnostics[f"remask_{modality}_reconstruction_raw"] = raw
        diagnostics[f"remask_{modality}_reconstruction_weighted"] = weighted
    diagnostics["remask_total_weighted"] = total
    return total, diagnostics


def _optimizer_step(
    *,
    model: TrackCADBridgeMIDASCycle,
    optimizers: Mapping[str, torch.optim.Optimizer],
    batch: Mapping[str, Any],
    cycle_scale: float,
    generator: torch.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    forward = model.forward(batch, cycle_scale=cycle_scale, sample=True)
    model.update_replay_bank(forward)
    batch_disc_snapshot: dict[str, Any] = {}
    for _ in range(model.config.n_iter_disc):
        disc = model.compute_batch_discriminator_losses(forward, cycle_scale=cycle_scale)
        finite(disc.batch_discriminator_loss, "batch_discriminator_loss")
        optimizers["batch_discriminator"].zero_grad(set_to_none=True)
        disc.batch_discriminator_loss.backward()
        optimizers["batch_discriminator"].step()
        batch_disc_snapshot = disc.as_dict()
    domain = model.compute_domain_discriminator_losses(forward, cycle_scale=cycle_scale)
    finite(domain.domain_discriminator_loss, "domain_discriminator_loss")
    if domain.domain_optimizer_active:
        optimizers["domain_discriminator"].zero_grad(set_to_none=True)
        domain.domain_discriminator_loss.backward()
        optimizers["domain_discriminator"].step()
    optimizers["main"].zero_grad(set_to_none=True)
    losses = model.compute_main_loss(forward, cycle_scale=cycle_scale)
    remask_weighted, remask = feature_remask_objective(
        model=model,
        batch=batch,
        cycle_scale=cycle_scale,
        generator=generator,
        zero=forward.latent.sum() * 0.0,
    )
    main_loss = losses.main_loss + remask_weighted
    finite(main_loss, "main_loss_with_feature_remask")
    main_loss.backward()
    optimizers["main"].step()
    values: dict[str, Any] = dict(losses.as_dict())
    values["main_loss_without_remask"] = losses.main_loss
    values["main_loss"] = main_loss
    values.update(remask)
    values.update({f"batch_disc_{key}": value for key, value in batch_disc_snapshot.items()})
    values.update({f"domain_disc_{key}": value for key, value in domain.as_dict().items()})
    diagnostics = {
        "observed_modalities": list(batch["x"]),
        "bridge_active_directions": int(
            float(losses.atac_adt_active_directions.detach().cpu())
        ),
        "bridge_gradient_l1": sum(
            float(parameter.grad.detach().abs().sum().cpu())
            for parameter in model.atac_adt_bridge.parameters()
            if parameter.grad is not None
        ),
        "decoder_gradient_l1": sum(
            float(parameter.grad.detach().abs().sum().cpu())
            for parameter in model.native_vae.decoder.parameters()
            if parameter.grad is not None
        ),
        "remask_target_modalities": remask["remask_target_modalities"],
        "remask_target_rows": remask["remask_target_rows"],
        "domain_optimizer_active": bool(domain.domain_optimizer_active),
    }
    return values, diagnostics


def _checkpoint_input_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(manifest)))
    result.pop("input_dir", None)
    for record in result.get("batches", {}).values():
        record.pop("cell_ids", None)
    return result


def save_model_checkpoint(
    path: Path,
    *,
    model: TrackCADBridgeMIDASCycle,
    epoch_completed: int,
    context: NativeContext,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "frozenbridge65_portable_model_state_v1",
            "epoch_completed": int(epoch_completed),
            "model": model.state_dict(),
            "experiment": experiment_receipt(backend=context.backend),
            "input_contract": _checkpoint_input_contract(context.input_manifest),
            "native_modality_order": list(map(str, model.native_vae.dims_x)),
            "source_sha256": collect_core_source_hashes(),
            "package_versions": installed_versions(),
            "optimizer_state_included": False,
            "replay_state_included": False,
        },
        temporary,
    )
    os.replace(temporary, path)


def _runtime_snapshot(context: NativeContext) -> dict[str, Any]:
    gpu = None
    if context.device.type == "cuda":
        gpu = torch.cuda.get_device_name(context.device)
    return {
        "python": platform.python_version(),
        "hostname": platform.node(),
        "device": str(context.device),
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "package_versions": installed_versions(),
    }


def train_model(context: NativeContext, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    model = context.model
    model.train()
    optimizers = model.build_optimizers()
    generator = _remask_generator(model)
    runtime_dir = output_dir / "runtime"
    history_path = output_dir / "history.jsonl"
    rolling = runtime_dir / "rolling_model.pt"
    started = time.monotonic()
    total_steps = 0
    total_rows = 0
    for epoch in range(EPOCHS):
        epoch_started = time.monotonic()
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        scale = model.config.cycle_scale(epoch)
        sums: dict[str, float] = {}
        rows_seen = 0
        steps = 0
        active_bridge_steps = 0
        for batch in _iter_training_batches(context):
            batch_rows = int(batch["s"]["joint"].shape[0])
            values, diagnostics = _optimizer_step(
                model=model,
                optimizers=optimizers,
                batch=batch,
                cycle_scale=scale,
                generator=generator,
            )
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_rows
                elif isinstance(value, (int, float, bool)):
                    sums[key] = sums.get(key, 0.0) + float(value) * batch_rows
            active_bridge_steps += int(diagnostics["bridge_active_directions"] > 0)
            rows_seen += batch_rows
            steps += 1
            total_steps += 1
            total_rows += batch_rows
        if rows_seen <= 0:
            raise RuntimeError("Native MultiBatchSampler yielded no rows")
        elapsed = time.monotonic() - started
        row = {
            "epoch_index": epoch,
            "epoch_completed": epoch + 1,
            "cycle_scale": scale,
            "steps": steps,
            "rows_seen": rows_seen,
            "active_bridge_steps": active_bridge_steps,
            "epoch_seconds": time.monotonic() - epoch_started,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": elapsed / (epoch + 1) * (EPOCHS - epoch - 1),
            "rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2,
            "gpu_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(context.device) / 1024**3
                if context.device.type == "cuda"
                else None
            ),
            "replay_sizes": dict(model.replay_bank.sizes()),
            "metrics": {key: value / rows_seen for key, value in sums.items()},
            "resident_schedule": (
                dict(context.bundle.schedule_sampler.last_record or {})
                if context.bundle is not None
                else None
            ),
        }
        append_jsonl(history_path, row)
        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0 or epoch + 1 == EPOCHS:
            save_model_checkpoint(
                rolling,
                model=model,
                epoch_completed=epoch + 1,
                context=context,
            )
        status = {
            "state": "TRAINING" if epoch + 1 < EPOCHS else "TRAINING_COMPLETE",
            "epoch_completed": epoch + 1,
            "epochs": EPOCHS,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": row["estimated_remaining_seconds"],
            "updated_at": utc_now(),
        }
        atomic_json(output_dir / "STATE.json", status)
        print(
            f"epoch {epoch + 1}/{EPOCHS} {row['epoch_seconds']:.2f}s "
            f"main={row['metrics'].get('main_loss', float('nan')):.6g}",
            flush=True,
        )
    final_path = output_dir / f"model_state_epoch{EPOCHS}.pt"
    os.replace(rolling, final_path)
    summary = {
        "status": "passed",
        "epochs_completed": EPOCHS,
        "optimizer_steps": total_steps,
        "rows_seen": total_rows,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(final_path),
        "checkpoint_sha256": sha256_file(final_path),
        "resident_schedule": (
            schedule_summary(context.bundle.schedule_sampler.epoch_records)
            if context.bundle is not None
            else None
        ),
    }
    return final_path, summary


def _dataset_batch(
    context: NativeContext, dataset_index: int, start: int, stop: int
) -> Mapping[str, Any]:
    if context.bundle is not None:
        dataset = context.bundle.cache.datasets[dataset_index]
        count = stop - start
        return {
            "x": {key: value[start:stop] for key, value in dataset.x.items()},
            "e": dict(dataset.e),
            "s": {
                **{
                    key: torch.full(
                        (count, 1), int(dataset.s[key]), dtype=torch.long, device=context.device
                    )
                    for key in dataset.x
                },
                "joint": torch.full(
                    (count, 1), int(dataset.s["joint"]), dtype=torch.long, device=context.device
                ),
            },
        }
    native_dataset = context.native.datalist[dataset_index]
    indices = list(range(start, stop))
    samples = [native_dataset[index] for index in indices]
    from torch.utils.data._utils.collate import default_collate

    return move_batch(default_collate(samples), context.device)


def smoke_model(context: NativeContext, smoke_batches: int) -> dict[str, Any]:
    if smoke_batches <= 0:
        raise ValueError("smoke_batches must be positive")
    model = context.model
    model.train()
    optimizers = model.build_optimizers()
    generator = _remask_generator(model)
    iterator = iter(_iter_training_batches(context))
    rows: list[dict[str, Any]] = []
    paired_bridge_steps = 0
    for index in range(smoke_batches):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Smoke requested {smoke_batches} batches but the loader yielded only {index}"
            ) from exc
        _, diagnostics = _optimizer_step(
            model=model,
            optimizers=optimizers,
            batch=batch,
            cycle_scale=1.0,
            generator=generator,
        )
        if diagnostics["decoder_gradient_l1"] <= 0.0:
            raise RuntimeError("Feature re-masking did not reach the native decoder")
        if diagnostics["bridge_active_directions"]:
            paired_bridge_steps += 1
            if diagnostics["bridge_gradient_l1"] <= 0.0:
                raise RuntimeError("Active ATAC-ADT bridge received no gradient")
        rows.append({"index": index, **diagnostics})
    route_probes: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for dataset_index, (batch_name, record) in enumerate(
            zip(context.input_manifest["batch_order"], context.input_manifest["batches"].values())
        ):
            if record["observed_modalities"] != ["atac"]:
                continue
            count = min(4, int(record["cell_count"]))
            batch = _dataset_batch(context, dataset_index, 0, count)
            output = model.forward(batch, cycle_scale=1.0, sample=False)
            raw = dict(output.reconstructions)
            raw["adt"] = model.decode_atac_to_adt(output, sample=False)
            decoded = model.native_vae.gen_real_data(raw, sampling=False)["adt"]
            if decoded.shape != (count, sum(model.feature_dims["adt"])):
                raise RuntimeError("ATAC-only ADT route returned an invalid shape")
            if not bool(torch.isfinite(decoded).all().cpu()):
                raise RuntimeError("ATAC-only ADT route returned non-finite values")
            route_probes.append(
                {"batch": batch_name, "rows": count, "shape": list(decoded.shape)}
            )
    return {
        "status": "passed",
        "formal_epochs_completed": 0,
        "smoke_batches_completed": smoke_batches,
        "paired_atac_adt_batches_declared": context.input_manifest[
            "paired_atac_adt_batches"
        ],
        "paired_bridge_steps_exercised": paired_bridge_steps,
        "atac_only_adt_route_probes": route_probes,
        "rows": rows,
    }


def load_checkpoint_strict(
    context: NativeContext,
    checkpoint: Path,
    *,
    fresh_native_modules: bool,
    allow_legacy: bool = False,
) -> tuple[TrackCADBridgeMIDASCycle, dict[str, Any]]:
    path = checkpoint.expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError("Checkpoint is not a FrozenBridge model-state mapping")
    if int(payload.get("epoch_completed", -1)) != EPOCHS:
        raise ValueError(f"Checkpoint must contain epoch {EPOCHS} weights")
    experiment = payload.get("experiment", {})
    required = {
        "variant": "C3",
        "epochs": EPOCHS,
        "seed": SEED,
        "lam_dsc": 40.0,
        "translated_batch_ce_scale": 0.625,
        "missingness_batch_ce_scale": 1.25,
        "paired_rna_anchor_consistency_weight": 5.0,
    }
    drift = {
        key: {"expected": value, "observed": experiment.get(key)}
        for key, value in required.items()
        if experiment.get(key) != value
    }
    if drift:
        raise ValueError(f"Checkpoint experiment contract drifted: {drift}")
    declared_input = payload.get("input_contract")
    current_input = _checkpoint_input_contract(context.input_manifest)
    if declared_input is None and not allow_legacy:
        raise ValueError("Checkpoint lacks the portable input contract")
    if declared_input is not None:
        if declared_input.get("portable_schema_sha256") != current_input.get(
            "portable_schema_sha256"
        ):
            raise ValueError("Checkpoint and current MIDAS input schemas differ")
    if fresh_native_modules:
        for handle_name in (
            "_native_encoder_capture_handle",
            "_native_batch_capture_handle",
        ):
            handle = getattr(context.model, handle_name, None)
            if handle is not None:
                handle.remove()
        context.model.to(torch.device("cpu"))
        gc.collect()
        if context.device.type == "cuda":
            torch.cuda.empty_cache()
    model = (
        _build_model(
            context.native,
            device=context.device,
            topology_alias="full",
            fresh_native_modules=True,
        )
        if fresh_native_modules
        else context.model
    )
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    receipt = {
        "status": "passed",
        "strict": True,
        "fresh_model_instance": True,
        "fresh_native_modules_built_during_load": bool(fresh_native_modules),
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_epoch_completed": EPOCHS,
        "state_dict_key_count": len(payload["model"]),
        "missing_keys": [],
        "unexpected_keys": [],
        "native_modality_order": list(map(str, model.native_vae.dims_x)),
        "legacy_checkpoint_without_portable_input_contract": declared_input is None,
    }
    return model, receipt


@torch.no_grad()
def export_predictions(
    *,
    model: TrackCADBridgeMIDASCycle,
    context: NativeContext,
    output_path: Path,
) -> dict[str, Any]:
    model.eval()
    manifest = context.input_manifest
    batch_order = manifest["batch_order"]
    total_cells = int(manifest["cell_count"])
    target_counts = {
        modality: sum(
            int(manifest["batches"][batch]["cell_count"])
            for batch in batch_order
            if modality in manifest["batches"][batch]["missing_modalities"]
        )
        for modality in MODALITIES
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    string_type = h5py.string_dtype(encoding="utf-8")
    summary: dict[str, Any] = {
        "batch_order": batch_order,
        "joint_embedding_shape": [total_cells, model.config.dim_c],
        "imputed": {},
        "target_specific_routes": {"adt_from_atac_bridge_batches": []},
    }
    with h5py.File(temporary, "w") as handle:
        all_ids = [
            cell
            for batch in batch_order
            for cell in manifest["batches"][batch]["cell_ids"]
        ]
        all_batches = [
            batch
            for batch in batch_order
            for _ in manifest["batches"][batch]["cell_ids"]
        ]
        handle.create_dataset("cell_ids", data=np.asarray(all_ids, dtype=object), dtype=string_type)
        handle.create_dataset("batch_ids", data=np.asarray(all_batches, dtype=object), dtype=string_type)
        embedding = handle.create_dataset(
            "joint_embedding",
            shape=(total_cells, model.config.dim_c),
            dtype=np.float32,
            chunks=(min(1024, total_cells), model.config.dim_c),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        imputed_group = handle.create_group("imputed")
        imputed_values: dict[str, Any] = {}
        imputed_offsets = {modality: 0 for modality in MODALITIES}
        for modality in MODALITIES:
            group = imputed_group.create_group(modality)
            ids = [
                cell
                for batch in batch_order
                if modality in manifest["batches"][batch]["missing_modalities"]
                for cell in manifest["batches"][batch]["cell_ids"]
            ]
            batch_ids = [
                batch
                for batch in batch_order
                if modality in manifest["batches"][batch]["missing_modalities"]
                for _ in manifest["batches"][batch]["cell_ids"]
            ]
            group.create_dataset("cell_ids", data=np.asarray(ids, dtype=object), dtype=string_type)
            group.create_dataset("batch_ids", data=np.asarray(batch_ids, dtype=object), dtype=string_type)
            feature_count = sum(model.feature_dims[modality])
            if target_counts[modality] > 0:
                imputed_values[modality] = group.create_dataset(
                    "values",
                    shape=(target_counts[modality], feature_count),
                    dtype=np.float32,
                    chunks=(min(64, target_counts[modality]), min(2048, feature_count)),
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                )
            else:
                imputed_values[modality] = group.create_dataset(
                    "values", shape=(0, feature_count), dtype=np.float32
                )
            target_batches = [
                batch
                for batch in batch_order
                if modality in manifest["batches"][batch]["missing_modalities"]
            ]
            group.attrs["target_batches"] = json.dumps(target_batches)
            summary["imputed"][modality] = {
                "shape": [target_counts[modality], feature_count],
                "target_batches": target_batches,
                "cell_order_sha256": ordered_hash(ids),
            }
        embedding_offset = 0
        for dataset_index, batch_name in enumerate(batch_order):
            record = manifest["batches"][batch_name]
            cell_count = int(record["cell_count"])
            observed = set(record["observed_modalities"])
            missing = tuple(record["missing_modalities"])
            uses_bridge = "adt" in missing and observed == {"atac"}
            if uses_bridge:
                summary["target_specific_routes"]["adt_from_atac_bridge_batches"].append(
                    batch_name
                )
            for start in range(0, cell_count, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, cell_count)
                batch = _dataset_batch(context, dataset_index, start, stop)
                output = model.forward(batch, cycle_scale=1.0, sample=False)
                latent = (
                    output.joint.mean[:, : model.config.dim_c]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                if not np.isfinite(latent).all():
                    raise FloatingPointError(f"{batch_name}: non-finite embedding")
                embedding[
                    embedding_offset + start : embedding_offset + stop
                ] = latent
                if missing:
                    raw = dict(output.reconstructions)
                    if uses_bridge:
                        raw["adt"] = model.decode_atac_to_adt(output, sample=False)
                    decoded = model.native_vae.gen_real_data(raw, sampling=False)
                    for modality in missing:
                        values = (
                            decoded[modality]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        if not np.isfinite(values).all():
                            raise FloatingPointError(
                                f"{batch_name}/{modality}: non-finite imputation"
                            )
                        offset = imputed_offsets[modality]
                        imputed_values[modality][offset : offset + len(values)] = values
                        imputed_offsets[modality] += len(values)
            embedding_offset += cell_count
        if embedding_offset != total_cells or imputed_offsets != target_counts:
            raise RuntimeError("Prediction row accounting drifted")
        handle.attrs["method_slug"] = METHOD_SLUG
        handle.attrs["method_name"] = METHOD_NAME
        handle.attrs["epochs"] = EPOCHS
        handle.attrs["seed"] = SEED
        handle.attrs["cell_order_sha256"] = ordered_hash(all_ids)
        handle.attrs["technical_dimensions_excluded"] = 2
        handle.flush()
    os.replace(temporary, output_path)
    summary["path"] = str(output_path)
    summary["sha256"] = sha256_file(output_path)
    summary["canonical_cell_order_sha256"] = ordered_hash(all_ids)
    return summary


def preflight_receipt(context: NativeContext) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "PREFLIGHT_PASSED",
        "method_slug": METHOD_SLUG,
        "method_name": METHOD_NAME,
        "seed": SEED,
        "requested_epochs": EPOCHS,
        "started_at": utc_now(),
        "experiment": experiment_receipt(backend=context.backend),
        "runtime": _runtime_snapshot(context),
        "input": _checkpoint_input_contract(context.input_manifest),
        "source_sha256": collect_core_source_hashes(),
        "truth_opened_during_training_or_prediction": False,
    }
