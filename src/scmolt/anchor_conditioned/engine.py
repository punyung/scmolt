"""Portable training of one policy-selected, complete anchor branch."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from ..frozenbridge65.config import installed_versions, version_drift
from ..frozenbridge65.engine import (
    NativeContext, _assert_native_defaults, _checkpoint_input_contract,
    _dataset_batch, _iter_training_batches, _remask_generator, _resolve_device,
    _runtime_snapshot, feature_remask_objective, finite, set_seed, utc_now,
)
from ..frozenbridge65.gpu_resident import (
    build_gpu_cache, build_gpu_resident_loader, schedule_summary,
)
from ..frozenbridge65.io import append_jsonl, atomic_json, sha256_file, validate_input
from .checkpoint import save_model_checkpoint
from .config import (
    BATCH_SIZE, CHECKPOINT_EVERY_EPOCHS, EPOCHS, GPU_RESIDENT_PRIVATE_SEED,
    N_MAX, SEED, collect_source_hashes, experiment_receipt,
)
from .policy import policy_sha256, selected_anchor, uses_direct_adt_route, validate_topology
from .registry import build_model


@dataclass
class AnchorContext(NativeContext):
    configuration: str
    runtime_dir: Path


def _build_model(
    native: Any, *, configuration: str, device: torch.device, fresh_native_modules: bool
) -> Any:
    if fresh_native_modules:
        dims_x = {str(key): list(value) for key, value in native.dims_x.items()}
        dims_s = {str(key): int(value) for key, value in native.dims_s.items()}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 900_003)
            native_vae = type(native.net)(dims_x, dims_s, **native.configs)
            native_discriminator = type(native.dsc)(dims_x, dims_s, **native.configs)
    else:
        native_vae, native_discriminator = native.net, native.dsc
    model = build_model(
        configuration, feature_dims=native.dims_x,
        n_batches=int(native.dims_s["joint"]), batch_class_dims=native.dims_s,
        native_configs=native.configs, native_vae=native_vae,
        native_discriminator=native_discriminator,
    ).to(device)
    order = tuple(map(str, native.dims_x))
    for module in (model.native_vae, model.native_vae.encoder,
                   model.native_vae.decoder, model.native_discriminator):
        if tuple(map(str, module.dims_x)) != order:
            raise RuntimeError("Native modality registration order drifted")
    return model


def configure_context(
    *, configuration: str, input_dir: Path, runtime_dir: Path, device: str,
    backend: str, hash_input: bool = False, allow_runtime_drift: bool = False,
) -> AnchorContext:
    selected_anchor(configuration)
    collect_source_hashes()
    if backend not in {"gpu_resident_v1", "native_workers"}:
        raise ValueError("Unsupported training backend")
    resolved_device = _resolve_device(device)
    if resolved_device.type != "cuda":
        backend = "native_workers"
    drift = version_drift()
    if drift and not allow_runtime_drift:
        raise RuntimeError(f"Runtime versions differ from environment.yml: {drift}")
    manifest = validate_input(input_dir, hash_contents=hash_input)
    validate_topology(manifest, configuration)
    if runtime_dir.exists() and any(runtime_dir.iterdir()):
        raise RuntimeError(f"Runtime directory is not empty: {runtime_dir}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for variable, name in (("NUMBA_CACHE_DIR", "numba_cache"),
                           ("MPLCONFIGDIR", "matplotlib_cache")):
        cache = (runtime_dir / name).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    set_seed(SEED)
    import lightning
    from scmidas.config import load_config
    from scmidas.model import MIDAS

    lightning.seed_everything(SEED, workers=True)
    native_config = dict(load_config())
    native_config.update(num_workers=0, pin_memory=False, persistent_workers=False)
    _assert_native_defaults(native_config)
    native_config.update(lam_alignment=0.0, lam_dsc=40.0)
    native = MIDAS.configure_data_from_dir(
        native_config, str(input_dir.expanduser().resolve()),
        transform={"atac": "binarize"},
        save_model_path=str((runtime_dir / "native_checkpoints").resolve()),
        format="csv", batch_size=BATCH_SIZE, n_save=EPOCHS + 1,
        sampler_type="auto", viz_umap_tb=False,
    )
    loader = native.train_dataloader()
    sampler = loader.sampler
    expected = ("MultiBatchSampler", BATCH_SIZE,
                min(max(len(dataset) for dataset in native.datalist), N_MAX), True)
    observed = (type(sampler).__name__, int(sampler.batch_size),
                int(sampler.n_max), bool(sampler.shuffle))
    if observed != expected:
        raise RuntimeError(f"Native sampler drifted: {observed} != {expected}")
    bundle = None
    if backend == "gpu_resident_v1":
        bundle = build_gpu_resident_loader(
            native.datalist, sampler, resolved_device,
            batch_size=BATCH_SIZE, requested_n_max=N_MAX,
            private_seed=GPU_RESIDENT_PRIVATE_SEED,
            cache=build_gpu_cache(native.datalist, resolved_device),
            stream_name="training_and_diagnostic",
        )
        loader = bundle.loader
    model = _build_model(native, configuration=configuration, device=resolved_device,
                         fresh_native_modules=False)
    return AnchorContext(native=native, model=model, loader=loader, bundle=bundle,
                         native_config=native_config, input_dir=input_dir.resolve(),
                         input_manifest=manifest, device=resolved_device,
                         backend=backend, configuration=configuration,
                         runtime_dir=runtime_dir.resolve())


def optimizer_step(
    *, model: Any, optimizers: Mapping[str, Any], batch: Mapping[str, Any],
    cycle_scale: float, generator: torch.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    forward = model.forward(batch, cycle_scale=cycle_scale, sample=True)
    model.update_replay_bank(forward)
    disc_values = {}
    for _ in range(model.config.n_iter_disc):
        disc = model.compute_batch_discriminator_losses(forward, cycle_scale=cycle_scale)
        finite(disc.batch_discriminator_loss, "batch_discriminator_loss")
        optimizers["batch_discriminator"].zero_grad(set_to_none=True)
        disc.batch_discriminator_loss.backward()
        optimizers["batch_discriminator"].step()
        disc_values = disc.as_dict()
    domain = model.compute_domain_discriminator_losses(forward, cycle_scale=cycle_scale)
    finite(domain.domain_discriminator_loss, "domain_discriminator_loss")
    if domain.domain_optimizer_active:
        optimizers["domain_discriminator"].zero_grad(set_to_none=True)
        domain.domain_discriminator_loss.backward()
        optimizers["domain_discriminator"].step()
    optimizers["main"].zero_grad(set_to_none=True)
    losses = model.compute_main_loss(forward, cycle_scale=cycle_scale)
    remask_loss, remask = feature_remask_objective(
        model=model, batch=batch, cycle_scale=cycle_scale,
        generator=generator, zero=forward.latent.sum() * 0.0,
    )
    loss = losses.main_loss + remask_loss
    finite(loss, "main_loss_with_feature_remask")
    loss.backward()
    optimizers["main"].step()
    values = dict(losses.as_dict())
    values.update(main_loss_without_remask=losses.main_loss, main_loss=loss)
    values.update(remask)
    values.update({f"batch_disc_{key}": value for key, value in disc_values.items()})
    values.update({f"domain_disc_{key}": value for key, value in domain.as_dict().items()})
    edge = model.atac_adt_bridge if hasattr(model, "atac_adt_bridge") else model.transport
    diagnostics = {
        "observed_modalities": list(batch["x"]),
        "atac_adt_active_directions": int(losses.atac_adt_active_directions.detach().cpu()),
        "transport_gradient_l1": sum(float(p.grad.detach().abs().sum().cpu())
                                     for p in edge.parameters() if p.grad is not None),
        "decoder_gradient_l1": sum(float(p.grad.detach().abs().sum().cpu())
                                   for p in model.native_vae.decoder.parameters()
                                   if p.grad is not None),
        "remask_target_modalities": remask["remask_target_modalities"],
    }
    return values, diagnostics


def train_model(context: AnchorContext, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    model = context.model
    model.train()
    optimizers = model.build_optimizers()
    generator = _remask_generator(model)
    rolling = context.runtime_dir / "rolling_model.pt"
    started = time.monotonic()
    total_steps = 0
    for epoch in range(EPOCHS):
        scale = model.config.cycle_scale(epoch)
        sums: dict[str, float] = {}
        rows = 0
        for batch in _iter_training_batches(context):
            count = int(batch["s"]["joint"].shape[0])
            values, _ = optimizer_step(model=model, optimizers=optimizers, batch=batch,
                                       cycle_scale=scale, generator=generator)
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count
                elif isinstance(value, (int, float, bool)):
                    sums[key] = sums.get(key, 0.0) + float(value) * count
            rows += count
            total_steps += 1
        if rows <= 0:
            raise RuntimeError("Native sampler yielded no rows")
        append_jsonl(output_dir / "history.jsonl", {
            "epoch_completed": epoch + 1, "cycle_scale": scale, "rows_seen": rows,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": {key: value / rows for key, value in sums.items()},
            "replay_sizes": dict(model.replay_bank.sizes()),
        })
        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0 or epoch + 1 == EPOCHS:
            save_model_checkpoint(rolling, model=model, epoch_completed=epoch + 1,
                                  context=context)
        atomic_json(output_dir / "STATE.json", {
            "state": "TRAINING" if epoch + 1 < EPOCHS else "TRAINING_COMPLETE",
            "epoch_completed": epoch + 1, "epochs": EPOCHS, "updated_at": utc_now(),
        })
        print(f"epoch {epoch + 1}/{EPOCHS} main={sums['main_loss'] / rows:.6g}", flush=True)
    final = output_dir / f"model_state_epoch{EPOCHS}.pt"
    os.replace(rolling, final)
    return final, {
        "status": "passed", "configuration": context.configuration,
        "anchor": model.config.anchor, "epochs_completed": EPOCHS,
        "optimizer_steps": total_steps, "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(final), "checkpoint_sha256": sha256_file(final),
        "resident_schedule": schedule_summary(context.bundle.schedule_sampler.epoch_records)
        if context.bundle is not None else None,
    }


def smoke_model(context: AnchorContext, smoke_batches: int) -> dict[str, Any]:
    if smoke_batches <= 0:
        raise ValueError("smoke_batches must be positive")
    model = context.model
    model.train()
    optimizers = model.build_optimizers()
    generator = _remask_generator(model)
    iterator = iter(_iter_training_batches(context))
    rows = []
    for index in range(smoke_batches):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"Smoke requested more batches than the loader provides ({index})") from exc
        _, diagnostics = optimizer_step(model=model, optimizers=optimizers, batch=batch,
                                        cycle_scale=1.0, generator=generator)
        if diagnostics["decoder_gradient_l1"] <= 0:
            raise RuntimeError("Smoke optimizer step did not reach the native decoder")
        if diagnostics["atac_adt_active_directions"] and diagnostics["transport_gradient_l1"] <= 0:
            raise RuntimeError("Active transport received no gradient")
        rows.append({"index": index, **diagnostics})
    probes = []
    model.eval()
    with torch.no_grad():
        for index, batch_name in enumerate(context.input_manifest["batch_order"]):
            record = context.input_manifest["batches"][batch_name]
            if not uses_direct_adt_route(context.configuration, set(record["observed_modalities"])):
                continue
            count = min(4, record["cell_count"])
            batch = _dataset_batch(context, index, 0, count)
            output = model.forward(batch, cycle_scale=1.0, sample=False)
            raw = dict(output.reconstructions)
            raw["adt"] = model.decode_atac_to_adt(output, sample=False)
            decoded = model.native_vae.gen_real_data(raw, sampling=False)["adt"]
            if decoded.shape != (count, sum(model.feature_dims["adt"])) or not torch.isfinite(decoded).all():
                raise RuntimeError("Direct ADT route smoke returned invalid predictions")
            probes.append({"batch": batch_name, "shape": list(decoded.shape)})
    return {
        "status": "passed", "configuration": context.configuration,
        "anchor": model.config.anchor, "formal_epochs_completed": 0,
        "smoke_batches_completed": smoke_batches, "rows": rows,
        "direct_adt_route_probes": probes,
    }


def preflight_receipt(context: AnchorContext) -> dict[str, Any]:
    return {
        "schema_version": 1, "state": "PREFLIGHT_PASSED",
        "configuration": context.configuration, "anchor": context.model.config.anchor,
        "policy_sha256": policy_sha256(), "requested_epochs": EPOCHS,
        "started_at": utc_now(),
        "experiment": experiment_receipt(context.configuration, backend=context.backend),
        "input": _checkpoint_input_contract(context.input_manifest),
        "runtime": _runtime_snapshot(context), "source_sha256": collect_source_hashes(),
        "truth_opened_during_training_or_prediction": False,
    }
