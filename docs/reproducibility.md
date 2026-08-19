# Reproducibility and evidence boundary

## What is frozen

The public contract freezes model architecture, feature order, RNA mask tiers,
three original batches, preprocessing assumptions, 1,000 epochs, seed 1234,
metric directions, and the scMIDAS-1000 comparison. The checkpoint payloads
freeze exact tensor state and are bound by byte size plus SHA256.

## What is published

- Installable model and strict-load code.
- Sanitized architecture/configuration contract.
- Exact feature-order files.
- Aggregate integration and imputation tables.
- Leakage status, checkpoint metadata, and hashes.
- Four model-only checkpoint Release assets.

## What is intentionally excluded

- Source data and transformed matrices.
- Cell identifiers and cell-level results.
- Cell-type labels and hidden RNA truth.
- Dense predictions and latent AnnData files.
- Optimizer state, replay state, logs, and scratch checkpoints.
- Cluster paths, usernames, environment executables, and scheduler metadata.

Hidden RNA truth was unavailable to the training and prediction process and
was opened only by the separate evaluator. This truth-sealing statement is
part of the benchmark receipt, not an assumption inferred from metric values.

## Verification sequence

```bash
python -m pip install -e '.[test]'
pytest
python scripts/audit_repository.py .
scmolt verify-features .
scmolt verify-assets /path/to/release-assets
scmolt strict-load /path/to/one.pt --condition mask20 --device cpu
```

The initial `scmidas` import can be slow on a shared filesystem because Scanpy
and Numba initialize caches. `scmolt strict-load` creates writable cache
directories and must be allowed to finish before its result is interpreted.
