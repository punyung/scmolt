# FrozenBridge65 tri-modal application runner

This repository also contains the runnable 65-epoch tri-modal application
model used in the equal-horizon comparison:

> Frozen Old-Bridge TrackC W14-D40/C3 + MissAdv1.25 + PairAnchor5 +
> bidirectional ATAC-ADT bridge + 25% RNA/ATAC/ADT feature re-masking.

It reuses the repository's byte-identical `scmolt.trackc` core and adds
`scmolt.trackc_adbridge` plus the portable trainer under
`scmolt.frozenbridge65`. No real data, labels, held-out truth, cell-level
predictions, or FrozenBridge checkpoint is included.

## Install and validate

```bash
conda env create -f environment.yml
conda activate scmolt
python -m pip install -e '.[model]'
python -m scmolt.frozenbridge65.validate
```

Without Conda, install the pinned application extra with
`python -m pip install -e '.[frozenbridge65]'` after selecting the appropriate
PyTorch 2.8.0 CUDA wheel for the target machine.

The validated runtime is Python 3.10, scMIDAS 0.1.13, PyTorch 2.8.0, Lightning
2.6.0, NumPy 1.26.4, pandas 2.3.3, SciPy 1.13.1, and h5py 3.14.0.

## Input format

Pass the native MIDAS directory that directly contains `feat/` and the batch
directories:

```text
INPUT/
  feat/
    feat_dims.toml
    feat_names_rna.csv
    feat_names_atac.csv
    feat_names_adt.csv
  batch1/
    mat/rna.csv
    mat/atac.csv
    mask/rna.csv
    mask/atac.csv
  batch2/
    mat/adt.csv
    mask/adt.csv
```

`feat_dims.toml` must declare `rna`, `atac`, and `adt` exactly once. Its native
modality order is preserved and checkpoint-bound. Matrices are cells × features
with cell IDs in column 1 and feature names in row 1. A batch contains
files only for modalities genuinely observed in training; never represent a
structurally missing modality with a zero-filled matrix. Every modality must be
observed in at least one batch. Labels and hidden modality truth must remain
outside this input tree and are evaluator-only.

The ATAC-ADT bridge receives direct paired supervision only from batches where
ATAC and ADT are genuinely co-observed. The input receipt records whether such
batches exist.

## Functional smoke

The smoke executes four optimizer steps, exercises all three remasking targets,
checks an active paired ATAC-ADT bridge gradient, and probes the ATAC-only→ADT
deployment route. It completes zero formal epochs.

```bash
python -m scmolt.frozenbridge65.make_tiny_input \
  --output-dir /tmp/fb65_tiny_input

PYTHONHASHSEED=1234 python -m scmolt.frozenbridge65.train \
  --input-dir /tmp/fb65_tiny_input \
  --output-dir /tmp/fb65_smoke \
  --device cpu \
  --backend native_workers \
  --smoke-batches 4
```

## Formal PBMC or BMMC application

Use one fresh output directory and one independently trained checkpoint per
dataset:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=1234 \
python -m scmolt.frozenbridge65.train \
  --input-dir /data/PBMC/midas/input \
  --output-dir /results/PBMC/frozenbridge65 \
  --device cuda \
  --backend gpu_resident_v1 \
  --hash-input

CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=1234 \
python -m scmolt.frozenbridge65.train \
  --input-dir /data/BMMC/midas/input \
  --output-dir /results/BMMC/frozenbridge65 \
  --device cuda \
  --backend gpu_resident_v1 \
  --hash-input
```

`gpu_resident_v1` is the frozen formal backend and retains the native scMIDAS
MultiBatchSampler schedule. `native_workers` is an explicitly recorded fallback
for a dataset that cannot fit in GPU memory; do not describe that fallback as a
bitwise-identical formal run.

The CLI intentionally has no epoch, seed, architecture, optimizer, bridge-loss,
or remask-fraction override. It also has no resume option because the replay
bank is intentionally not persisted.

## Outputs

A successful formal run contains:

- `model_state_epoch65.pt`;
- `history.jsonl`;
- `STRICT_RELOAD.json`;
- `predictions.h5` with a 32-D biological embedding and actual-missing-only
  imputation blocks;
- `run_metadata.json`, `PREDICTIONS_FROZEN.json`, and `SUCCESS.json`.

The ATAC-only ADT prediction block uses the deployed bridge route:

```text
ATAC posterior -> ATAC-to-ADT bridge -> ADT-to-RNA canonical translator
-> native ADT decoder
```

Keep cells, preprocessing, structural masks, feature order, seed, and metrics
fixed when comparing against scMIDAS or other methods. Evaluate integration and
imputation separately and release evaluator truth only after predictions have
been frozen.
