# scMoLT

scMoLT is a research implementation of biological-latent transport for mosaic
single-cell multi-omics. The first checkpoint family is the audited **Cell C**
model for BMMC Multiome RNA-missingness sensitivity: Track C W14-D40/C3 with a
missingness-conditioned batch adversary (1.25), paired RNA-anchor consistency
(5.0), and 25% RNA/ATAC feature re-masking.

The repository also includes the **FrozenBridge65** tri-modal training and
prediction runner (RNA + ATAC + ADT), including its bidirectional ATAC-ADT
bridge and all-modality re-masking contract. See
[FROZENBRIDGE65.md](FROZENBRIDGE65.md). This runner is code-only; it does not
turn the v0.1.0 Cell C checkpoints into tri-modal or cross-dataset weights.

This repository deliberately separates source and small scientific receipts
from weights. The four 94 MB checkpoints belong in the GitHub Release
`v0.1.0`; no checkpoint, private input, cell-level prediction, label, or
evaluator-only truth file is stored in Git.

## Evidence status

- Dataset: GSE194122 BMMC Multiome, 22,211 cells, three original batches.
- Protocol: four nested RNA-mask conditions (20%, 40%, 60%, 80%), 4,000 RNA
  genes and 50,000 ATAC peaks, 1,000 epochs, seed 1234.
- Leakage control: training and prediction did not open hidden RNA truth;
  `cell_type_l1` was used only for stratification and post-training evaluation.
- Compared with scMIDAS-1000 under the same cells, features, masks, labels, and
  metrics, Cell C won 39/56 directional condition-metric comparisons, including
  28/32 imputation comparisons.
- This is one-seed evidence, not a universal or statistically replicated win.
  Bio conservation decreases at some masks even when imputation improves. See
  [MODEL_CARD.md](MODEL_CARD.md) for the full integration/imputation trade-off.

## Installation

The exact validated environment used Python 3.10.20, PyTorch 2.8.0+cu128, and
scMIDAS 0.1.13. A clean environment can be created with:

```bash
conda env create -f environment.yml
conda activate scmolt
```

Or install the package into an existing compatible environment:

```bash
python -m pip install -e .
```

scMIDAS imports Scanpy/Numba. Its cache directories must be writable on shared
clusters; the `scmolt strict-load` command creates local cache directories by
default.

## Download and verify weights

After `v0.1.0` is published, download exactly one checkpoint or all four:

```bash
gh release download v0.1.0 \
  --repo punyung/scmolt \
  --pattern 'scmolt-cell-c-bmmc-mask40-seed1234-epoch1000.pt' \
  --dir weights/assets
```

Verify the model-only payload and its published hash:

```bash
scmolt inspect \
  weights/assets/scmolt-cell-c-bmmc-mask40-seed1234-epoch1000.pt \
  --condition mask40

scmolt strict-load \
  weights/assets/scmolt-cell-c-bmmc-mask40-seed1234-epoch1000.pt \
  --condition mask40 \
  --device cpu
```

To verify a directory containing all four Release assets:

```bash
scmolt verify-assets weights/assets
scmolt verify-features .
```

The four checkpoints are condition-specific models, not interchangeable
replicates. Select the checkpoint matching the RNA-mask condition being
reproduced. No checkpoint has been validated as a general-purpose model for an
unknown missingness rate or unseen batch vocabulary.

## Python API

```python
import torch
from scmolt import load_pretrained

model, receipt = load_pretrained(
    "weights/assets/scmolt-cell-c-bmmc-mask40-seed1234-epoch1000.pt",
    condition="mask40",
    device="cuda",
)

# ATAC must be binary and in features/feat_names_atac.csv order.
# Batch IDs use the frozen three-class BMMC vocabulary (0, 1, 2).
batch = {
    "x": {"atac": atac_binary_float32},
    "e": {"atac": torch.ones_like(atac_binary_float32)},
    "s": {"joint": batch_ids[:, None]},
}

with torch.no_grad():
    output = model(batch, cycle_scale=1.0, sample=False)
    embedding_32d = output.joint_embedding
    imputed_rna_mean = output.reconstructions["rna"]
```

RNA inputs, when present, are raw counts in the exact RNA feature order. See
[docs/input_contract.md](docs/input_contract.md) before applying a checkpoint.

## Repository map

- `src/scmolt/trackc/`: model, losses, Gaussian transport, replay, and
  leakage-safe re-masking primitives.
- `src/scmolt/trackc_adbridge/`: the bidirectional ATAC-ADT Gaussian bridge.
- `src/scmolt/frozenbridge65/`: fixed 65-epoch tri-modal train, strict-load,
  prediction, input-audit, and smoke-test entrypoints.
- `src/scmolt/checkpoint.py`: safe payload inspection, model reconstruction,
  feature validation, and strict loading.
- `src/scmolt/configs/cell_c_bmmc.json`: sanitized architecture and Release
  contract.
- `features/`: exact RNA/ATAC feature-order lists.
- `benchmarks/bmmc/`: aggregate scIB and RNA-imputation comparisons only.
- `weights/release-index.json`: hashes and names for Release assets.
- `docs/`: input contract, architecture, reproducibility, and publication
  boundaries.

## Scope and license

The checkpoint release is intended for research use. It depends on, but does
not vendor or modify, `scmidas==0.1.13`; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). scMoLT source code is
released under the [MIT License](LICENSE); third-party dependencies retain
their own licenses.
