# Pretrained input contract

The BMMC checkpoints are architecture- and vocabulary-specific.

## Features

- RNA: 4,000 raw-count genes in `features/feat_names_rna.csv` order.
- ATAC: 50,000 binary peaks in `features/feat_names_atac.csv` order.
- ATAC is passed as one `[cells, 50000]` tensor. The model splits it into the
  27 architecture chunks recorded in `FEATURE_MANIFEST.json`.
- Feature files are part of the checkpoint contract. Matching dimensions with
  a different order is invalid.

Run `scmolt verify-features .` before preparing inputs.

## Native batch dictionary

The model accepts the scMIDAS dictionary contract:

```python
batch = {
    "x": {modality: values},
    "e": {modality: observed_feature_mask},
    "s": {"joint": batch_ids, modality: batch_ids},
}
```

Every tensor in a minibatch shares the same row order. `x` contains only
modalities genuinely observed for that homogeneous block. `e` is a
broadcastable feature-observation mask and must not be repurposed as a row
availability flag. If a modality-specific entry is omitted from `s`, scMoLT
uses `s["joint"]` for that observed modality.

The released model has three batch classes, encoded as integer labels 0, 1,
and 2. It has not been validated on new batch classes. A checkpoint may still
produce numbers for shifted biological data, but that is outside the released
evidence contract.

## Deterministic export

Call `model.eval()` and use `sample=False` or
`model.encode_joint(..., use_mean=True)` for a deterministic posterior mean.
The exported biological embedding is 32-dimensional; do not append or publish
the two-dimensional technical latent as biological representation.
