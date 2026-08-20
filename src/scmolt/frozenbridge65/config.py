"""The immutable FrozenBridge65 model and optimization contract."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from ..trackc import TrackCConfig
from ..trackc_adbridge import ATACADTBridgeConfig


METHOD_SLUG = "frozen_old_bridge_65"
METHOD_NAME = (
    "Frozen Old-Bridge TrackC W14-D40/C3 + MissAdv1.25 + PairAnchor5 + "
    "ATAC-ADT bridge + RNA/ATAC/ADT remask25 (65 epochs)"
)
MODALITIES = ("rna", "atac", "adt")
SEED = 1234
EPOCHS = 65
BATCH_SIZE = 256
N_MAX = 10_000
GPU_RESIDENT_PRIVATE_SEED = 9_731_337
REMASK_PRIVATE_SEED = 7_311_337
CHECKPOINT_EVERY_EPOCHS = 25

REMASK_FEATURE_FRACTIONS = {modality: 0.25 for modality in MODALITIES}
REMASK_RECONSTRUCTION_WEIGHTS = {modality: 0.5 for modality in MODALITIES}
BRIDGE_WEIGHTS = {
    "paired_consistency_weight": 5.0,
    "cycle_weight": 30.0,
    "identity_weight": 15.0,
    "cross_decode_weight": 1.0,
    "seed_offset": 500_003,
}

EXPECTED_VERSIONS = {
    "scmidas": "0.1.13",
    "torch": "2.8.0+cu128",
    "lightning": "2.6.0",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "h5py": "3.14.0",
    "scipy": "1.13.1",
    "scikit-learn": "1.6.1",
    "pyyaml": "6.0.3",
    "toml": "0.10.2",
}

EXPECTED_CORE_SOURCE_SHA256 = {
    "scmolt/trackc/__init__.py": "344915df67c385b38d4952e9c9157eb950319980b68df89092fef4b534ad359e",
    "scmolt/trackc/config.py": "8c3e0c56e4ee9900ac75b2e313c5144b8498c27f96023b39f6f5291eac6b742a",
    "scmolt/trackc/gaussian.py": "dcef797352954213ad959f182f7b3e83b9af5e6fe0354deb160259f001f8357b",
    "scmolt/trackc/model.py": "f2b32f9be47265242067f19dc7de284ae0e79fd2862528d4f405aeaca4583cbb",
    "scmolt/trackc/modules.py": "af8b6332ace4ccc1a3634874a4e4bffca56486a4478fc46c801c9c022b78a0a7",
    "scmolt/trackc/remasking.py": "4f46650e6d338daa12389999177a6b4bb980604df089fe3f8fb871db9ed3ceb0",
    "scmolt/trackc/replay.py": "b5f1d6f0549b0c65e3992ea58aa01469a1caa9e36e31fb510e1e241616a13d38",
    "scmolt/trackc_adbridge/__init__.py": "45195dccbec8e0c1778a627aba5a6220f8a132f736733ce7ad3ca69b50498d9a",
    "scmolt/trackc_adbridge/model.py": "38eaf0b290fd0e9465fb729e8e7af6f3b2c3c36f6a7a41b8a00560b6bcf625d3",
}


def core_config(*, topology_alias: str = "full") -> TrackCConfig:
    """Return the exact 65-epoch TrackC configuration.

    The C3 alignment weight is zero for every alias. The alias is recorded for
    provenance; actual missingness is determined solely by the files present in
    each native MIDAS batch directory.
    """

    return TrackCConfig(
        variant="C3",
        topology=topology_alias,
        seed=SEED,
        epochs=EPOCHS,
        anchor="rna",
        dim_c=32,
        dim_u=2,
        encoder_hidden=(1024, 128),
        decoder_hidden=(128, 1024),
        translator_hidden=(128, 128),
        dropout=0.2,
        s_drop_rate=0.1,
        batch_dropout=0.1,
        logvar_min=-8.0,
        logvar_max=8.0,
        lam_kld=1.0,
        lam_kld_c=1.0,
        lam_kld_u=5.0,
        lam_recon=1.0,
        lam_recon_s=1000.0,
        lam_alignment=0.0,
        lam_adv=1.0,
        lam_dsc=40.0,
        translated_batch_ce_scale=0.625,
        missingness_batch_ce_scale=1.25,
        paired_rna_anchor_consistency_weight=5.0,
        emit_translated_batch_ce_metrics=True,
        lam_cycle=30.0,
        lam_identity=15.0,
        lam_cross_decode=1.0,
        lam_domain=0.1,
        n_iter_disc=3,
        learning_rate=1e-4,
        batch_discriminator_learning_rate=3e-4,
        domain_discriminator_learning_rate=1e-4,
        weight_decay=0.01,
        warmup_epochs=30,
        ramp_epochs=30,
    )


def bridge_config() -> ATACADTBridgeConfig:
    return ATACADTBridgeConfig(**BRIDGE_WEIGHTS)


def experiment_receipt(*, backend: str, topology_alias: str = "full") -> dict[str, Any]:
    config = core_config(topology_alias=topology_alias)
    result = config.to_dict()
    result.update(
        {
            "method_slug": METHOD_SLUG,
            "method_name": METHOD_NAME,
            "batch_size": BATCH_SIZE,
            "n_max": N_MAX,
            "training_sampler": "midas_multibatch",
            "training_data_backend": backend,
            "gpu_resident_private_seed": GPU_RESIDENT_PRIVATE_SEED,
            "remask_private_seed": REMASK_PRIVATE_SEED,
            "remask_feature_row_fraction": 0.25,
            "remask_feature_fractions": dict(REMASK_FEATURE_FRACTIONS),
            "remask_feature_reconstruction_weights": dict(
                REMASK_RECONSTRUCTION_WEIGHTS
            ),
            "atac_adt_bridge_weights": dict(BRIDGE_WEIGHTS),
            "atac_only_adt_prediction_route": (
                "ATAC posterior -> ATAC-to-ADT bridge -> ADT-to-RNA canonical "
                "translator -> native ADT decoder"
            ),
            "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
            "early_stopping": False,
        }
    )
    return result


def installed_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in EXPECTED_VERSIONS:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def version_drift(*, permit_torch_build_drift: bool = True) -> dict[str, dict[str, str]]:
    observed = installed_versions()
    result: dict[str, dict[str, str]] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        actual = observed[package]
        matches = actual == expected
        if package == "torch" and permit_torch_build_drift:
            matches = actual.split("+", 1)[0] == expected.split("+", 1)[0]
        if not matches:
            result[package] = {"expected": expected, "observed": actual}
    return result


def assert_model_contract(model: Any) -> None:
    if tuple(model.modalities) != MODALITIES:
        raise RuntimeError(f"FrozenBridge65 requires modality order {MODALITIES}; got {model.modalities}")
    if tuple(model.transport.DIRECTIONS) != (
        ("atac", "rna"),
        ("rna", "atac"),
        ("adt", "rna"),
        ("rna", "adt"),
    ):
        raise RuntimeError("RNA-centered TrackC transport graph drifted")
    if tuple(model.atac_adt_bridge.DIRECTIONS) != (("atac", "adt"), ("adt", "atac")):
        raise RuntimeError("ATAC-ADT bridge directions drifted")
    if model.config.to_dict() != core_config(topology_alias=model.config.topology).to_dict():
        raise RuntimeError("FrozenBridge65 TrackC configuration drifted")
    if model.bridge_config.to_dict() != BRIDGE_WEIGHTS:
        raise RuntimeError("FrozenBridge65 bridge configuration drifted")


def package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def collect_core_source_hashes() -> dict[str, str]:
    scmolt_package = Path(__file__).resolve().parents[1]
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_CORE_SOURCE_SHA256.items():
        logical = Path(relative)
        if not logical.parts or logical.parts[0] != "scmolt":
            raise RuntimeError(f"Invalid package-relative source lock path: {relative}")
        source_path = scmolt_package.joinpath(*logical.parts[1:])
        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Frozen core source drifted: {relative}: {actual} != {expected}"
            )
        observed[relative] = actual
    return observed


def canonical_config_mapping() -> Mapping[str, Any]:
    return experiment_receipt(backend="gpu_resident_v1")
