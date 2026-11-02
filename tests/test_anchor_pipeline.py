"""Synthetic execution and format checks; these are not trained DOGMA scores."""

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("scmidas")

from scmolt.anchor_conditioned.checkpoint import load_checkpoint_strict, save_model_checkpoint
from scmolt.anchor_conditioned.engine import configure_context, smoke_model
from scmolt.anchor_conditioned.make_tiny_input import make_tiny_input
from scmolt.anchor_conditioned.policy import CONFIGURATIONS, selected_anchor
from scmolt.anchor_conditioned.prediction import export_predictions


@pytest.fixture
def context_factory(tmp_path):
    torch.set_num_threads(1)

    def create(configuration):
        root = tmp_path / configuration
        input_dir = make_tiny_input(root / "input", configuration, cells_per_batch=4)
        return configure_context(configuration=configuration, input_dir=input_dir,
                                 runtime_dir=root / "runtime", device="cpu",
                                 backend="native_workers", allow_runtime_drift=True)

    return create


def test_separate_runtime_directory_is_not_overwritten(tmp_path):
    root = make_tiny_input(tmp_path / "input", "full", cells_per_batch=4)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    existing = runtime / "rolling_model.pt"
    existing.write_bytes(b"existing run")
    with pytest.raises(RuntimeError, match="Runtime directory is not empty"):
        configure_context(configuration="full", input_dir=root, runtime_dir=runtime,
                          device="cpu", backend="native_workers", allow_runtime_drift=True)
    assert existing.read_bytes() == b"existing run"


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_all_topologies_execute_selected_branch_and_export(context_factory, tmp_path, configuration):
    import h5py
    import numpy as np

    context = context_factory(configuration)
    model = context.model
    assert model.config.anchor == selected_anchor(configuration)
    assert len(model.transport.translators) == 4
    assert hasattr(model, "atac_adt_bridge") == (model.config.anchor == "rna")
    result = smoke_model(context, 4)
    assert result["formal_epochs_completed"] == 0
    assert result["smoke_batches_completed"] == 4
    path = tmp_path / "predictions.h5"
    prediction = export_predictions(model=model, context=context, output_path=path)
    assert prediction["joint_embedding_shape"] == [16, 32]
    with h5py.File(path) as handle:
        assert handle.attrs["anchor"] == selected_anchor(configuration)
        assert handle.attrs["configuration"] == configuration
        assert np.isfinite(handle["joint_embedding"][:]).all()
        if configuration == "paired-bc":
            assert handle["imputed/adt/values"].shape == (0, 8)
    if configuration == "paired-ab":
        assert len(prediction["target_specific_routes"]["adt_from_atac_batches"]) == 2
    if configuration == "diagonal":
        assert prediction["target_specific_routes"]["adt_from_atac_batches"] == ["lll_ctrl"]


@pytest.mark.parametrize("configuration", ["full", "paired-ab", "paired-bc"])
def test_synthetic_state_roundtrip_and_contract_rejection(context_factory, tmp_path, configuration):
    context = context_factory(configuration)
    context.model.eval()
    batch = next(iter(context.loader))
    with torch.no_grad():
        expected = context.model.forward(batch, sample=False).joint.mean.clone()
    path = tmp_path / "synthetic-model-state.pt"
    # Epoch 65 here tests the checkpoint envelope on synthetic initialized
    # weights; it does not assert that a training run completed.
    save_model_checkpoint(path, model=context.model, epoch_completed=65, context=context)
    reloaded, receipt = load_checkpoint_strict(context, path, fresh_native_modules=True)
    with torch.no_grad():
        actual = reloaded.forward(batch, sample=False).joint.mean
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert receipt["strict"]
    original = torch.load(path, weights_only=True)
    for field, replacement in [("policy_sha256", "wrong"), ("epoch_completed", 64)]:
        changed = dict(original)
        changed[field] = replacement
        torch.save(changed, path)
        with pytest.raises(ValueError):
            load_checkpoint_strict(context, path, fresh_native_modules=False)
    changed = dict(original)
    changed["experiment"] = dict(original["experiment"], anchor="other")
    torch.save(changed, path)
    with pytest.raises(ValueError, match="contract differs"):
        load_checkpoint_strict(context, path, fresh_native_modules=False)
    changed = dict(original)
    changed["input_contract"] = dict(original["input_contract"], portable_schema_sha256="wrong")
    torch.save(changed, path)
    with pytest.raises(ValueError, match="schemas differ"):
        load_checkpoint_strict(context, path, fresh_native_modules=False)
