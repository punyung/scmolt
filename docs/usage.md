# Running scMoLT

scMoLT has two training commands, which share the model code and the training settings of the paper.

| Command | Anchor | Used in the paper for |
|---|---|---|
| `scmolt-train`, `scmolt-predict` | RNA | the BMMC and PBMC applications, and any RNA/ADT/ATAC mosaic |
| `scmolt-anchor-train`, `scmolt-anchor-predict` | RNA, ATAC or ADT, fixed per configuration | the eight DOGMA-seq configurations |

Both set the seed to 1234 and train for 65 epochs: 30 warm-up epochs that fuse the untransported posteriors, then 30
epochs over which the weight $\alpha_t$ of the transport terms rises to 1, then 5 more epochs. Minibatches have 256 cells
and are drawn as in MIDAS (every batch contributes $\lceil n/256\rceil$ minibatches per epoch, where $n$ is the size of
the largest batch capped at 10,000 cells). The commands have no options that change the model or the schedule.

## Input

Both commands read the native input directory of MIDAS.

```text
INPUT/
  feat/
    feat_dims.toml          rna = [4000]  adt = [134]  atac = [5091, 2226, ...]   (peaks per chromosome)
    feat_names_rna.csv
    feat_names_adt.csv
    feat_names_atac.csv
  batch1/
    mat/rna.csv             cells × features; cell IDs in column 1, feature names in row 1
    mat/atac.csv
    mask/rna.csv            one row of ones (all features available)
    mask/atac.csv
  batch2/
    mat/rna.csv
    mat/adt.csv
    mask/rna.csv
    mask/adt.csv
```

- A batch contains files only for the modalities it measures. A modality that a batch does not measure has no file and
  is never written as zeros.
- Every one of RNA, ADT and ATAC is measured in at least one batch.
- Within a batch, every modality lists the same cells in the same order. Cell IDs are unique across batches.
- RNA and ADT are raw counts. ATAC is binarized when it is read.
- Batches are ordered by their folder names (natural sort). Each folder is one batch of the batch-label likelihood and
  the batch adversary, unless `feat/batch_groups.toml` maps several folders to one shared batch (below).

### Folders that share a batch

All cells of a folder measure the same modalities. To withhold a modality from some cells of a batch, for example to
evaluate its prediction, put those cells in a folder of their own and list both folders in `feat/batch_groups.toml`:

```toml
"01_cite_s1d1" = "cite_s1d1"
"02_cite_s1d1_holdout" = "cite_s1d1"
"03_multiome_s1d1" = "multiome_s1d1"
"04_multiome_s1d1_holdout" = "multiome_s1d1"
```

The file maps every folder to a batch name, and folders with the same name share one batch label: the joint label is
the index of the batch in order of first appearance, and the label of a modality is the index among the batches that
measure it. Without the file, every folder is its own batch, as in native MIDAS. `write_midas_input(...,
batch_groups=...)` writes the file. Only `scmolt-train` reads it; `scmolt-anchor-train` stops if it is present.

`scmolt.write_midas_input` (Python) and `scmolt prepare-input` (command line) write this directory from one AnnData
per modality and check these rules. Labels and held-out measurements must stay outside the input directory; the
runners never read cell-type annotations.

`scmolt-anchor-train` additionally needs the four DOGMA-seq batches `lll_ctrl`, `lll_stim`, `dig_ctrl` and `dig_stim`,
measured as in the chosen `--configuration`, and stops if the input measures other modalities.

## Training

```bash
PYTHONHASHSEED=1234 scmolt-train --input-dir INPUT --output-dir RUN --device cuda
PYTHONHASHSEED=1234 scmolt-anchor-train --configuration paired-ab --input-dir INPUT --output-dir RUN --device cuda
```

- `PYTHONHASHSEED=1234` must be set before Python starts; the commands stop otherwise.
- `--output-dir` must be new or empty, so a finished run is never overwritten.
- `--backend gpu_resident_v1` (default) keeps the training data on the GPU, as in the paper. `--backend
  native_workers` loads minibatches with the scMIDAS data loader and is used automatically on the CPU.
- `--hash-input` records a SHA-256 of all input files in the run metadata.
- `--allow-runtime-drift` runs with package versions other than those of `environment.yml` and records them. Results
  may then differ slightly from the paper.
- `--smoke-batches N` runs N optimizer steps instead of the 65 epochs, to check an input quickly.
- `--runtime-dir` (anchor runner) puts temporary files on separate local storage.

A run writes

| File | Content |
|---|---|
| `model_state_epoch65.pt` | model weights after epoch 65 with the input schema and the settings of the run (no optimizer state) |
| `history.jsonl` | losses of every epoch |
| `STRICT_RELOAD.json` | check that the saved model reloads and reproduces the predictions |
| `predictions.h5` | embedding and predicted modalities (below) |
| `run_metadata.json`, `SUCCESS.json` | settings, package versions, input schema and timings |

## Prediction

```bash
PYTHONHASHSEED=1234 scmolt-predict --input-dir INPUT --checkpoint RUN/model_state_epoch65.pt --output-dir PRED
PYTHONHASHSEED=1234 scmolt-anchor-predict --configuration paired-ab --input-dir INPUT \
  --checkpoint RUN/model_state_epoch65.pt --output-dir PRED
```

Prediction checks that the input has the same batches, modalities, features and cell order as the training input, and
that the model was trained by the same version of the code.

`predictions.h5` contains

- `cell_ids`, `batch_ids`: all cells, in input order;
- `joint_embedding`: the 32-dimensional mean of the fused biological posterior, the embedding used for integration;
- `imputed/<modality>/values`, `cell_ids`, `batch_ids`: the modality decoded from the posterior means, without
  sampling, for the cells of every batch that does not measure it.

`scmolt.read_predictions("predictions.h5", input_dir="INPUT")` returns the embedding as an AnnData
(`obsm["X_scmolt"]`) and the predictions as one AnnData per modality, with the input feature names.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests build small synthetic inputs, run a few optimizer steps of every DOGMA-seq configuration on the CPU, export
predictions and check that saved models reload exactly. They check the software, not the results of the paper.
