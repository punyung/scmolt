# Experiments

All experiments train one model per dataset with the fixed settings of the package (seed 1234, 65 epochs), on one
NVIDIA GPU, in the environment of [`environment.yml`](../environment.yml). Every run ends with `predictions.h5`, which
holds the 32-dimensional embedding used for the integration scores and the predicted masked modalities used for the
imputation scores.

## DOGMA-seq pseudo-mosaics (Fig. 2)

**Data.** DOGMA-seq of 32,975 stimulated and control peripheral blood cells in four batches (`lll_ctrl`, `lll_stim`,
`dig_ctrl`, `dig_stim`) with RNA, ATAC and ADT measured in every cell (Mimitou et al., *Nat. Biotechnol.* 2021; GEO
[GSE156478](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156478)), preprocessed as in MIDAS (He et al.,
*Nat. Biotechnol.* 2024; [`reproducibility`](https://github.com/labomics/midas/tree/reproducibility) branch of the MIDAS
repository): 4,045 genes, 30,589 peaks in 22 chromosomes and 208 proteins. Save the three modalities as h5ad files with
RNA and ADT counts, ATAC peaks sorted by chromosome, and the batch of every cell in `obs["batch"]`.

**Run.**

```bash
bash experiments/dogma/run_dogma.sh rna.h5ad adt.h5ad atac.h5ad dogma_runs            # paired-ab, paired-bc, paired-ac
bash experiments/dogma/run_dogma.sh rna.h5ad adt.h5ad atac.h5ad dogma_runs full diagonal   # other configurations
```

For every configuration, the script masks the data to the configuration (`scmolt prepare-input --configuration`) and
trains the model with the anchor of the paper (`scmolt-anchor-train`). The masked measurements never enter the input
directory, so they can be used as the ground truth of imputation. We checked that `scmolt prepare-input` reproduces
the input files of the paper's `paired-ab` run byte for byte.

## Bone marrow CITE-seq and Multiome (Fig. 3)

**Data.** The processed CITE-seq and Multiome bone marrow data of the NeurIPS 2021 benchmark (Luecken et al.; GEO
[GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122),
`GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad` and `..._multiome_BMMC_processed.h5ad`), restricted to the
donor/site blocks s1d1, s2d4 and s4d8 that both assays measured: 14,740 CITE-seq cells (RNA and ADT) and 22,211 Multiome
cells (RNA and ATAC), six assay-by-block batches.

**Preprocessing** (Supplementary Material). The 4,000 genes shared by both assays are ranked by count variance
divided by mean, keeping 72 predefined hematopoietic markers. Peaks detected in fewer than 12 cells are removed and the
50,000 most frequently accessible peaks are kept. The 134 ADT features exclude the isotype controls. Before training,
the ADT of 2,948 CITE-seq cells and the ATAC of 4,442 Multiome cells are withheld (seed 2026) for the holdout
evaluation; these cells keep their RNA. [`bmmc/cells.csv.gz`](bmmc/cells.csv.gz) and
[`bmmc/features.csv.gz`](bmmc/features.csv.gz) list the selected cells, with their holdout status, and the selected
features.

**Run.**

```bash
python experiments/bmmc/prepare_bmmc.py \
  --cite GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad \
  --multiome GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad \
  --output-dir bmmc_input
PYTHONHASHSEED=1234 scmolt-train --input-dir bmmc_input --output-dir bmmc_run --device cuda --hash-input
```

`prepare_bmmc.py` writes each assay-by-block batch as two folders, the cells that keep both modalities and the holdout
cells with RNA only, and `feat/batch_groups.toml` gives the two folders one shared batch label, as in the paper's run.
We checked that the input it writes from the GEO files is byte-identical to the training data of the paper's run. On
this input, `scmolt-train`'s model construction strict-loads the paper's trained model, and its prediction export
reproduces the paper's embedding of all 36,951 cells (largest difference 2.4e-6) and its predicted ADT and ATAC (relative
difference below 1.4e-5). The withheld ADT and ATAC are read from the same GEO files for evaluation.

## PBMC Multiome and TotalSeq-C (Supplementary Material)

**Data.** The 10x Genomics PBMC10k Multiome dataset (granulocyte-depleted nuclei; 9,631 cells, RNA and ATAC) and the
10x Genomics 5′ whole-cell PBMC dataset with TotalSeq-C antibodies (7,725 cells after quality control, RNA and ADT).

**Preprocessing.** TotalSeq-C cells need at least 200 detected genes, 500 RNA counts and at most 20% mitochondrial
counts. The 4,000 shared genes are selected by count dispersion with immune markers retained, peaks detected in fewer
than 10 cells are removed before the 50,000 most frequently accessible peaks are kept, and the three isotype controls
are removed from the 17 ADT channels. All measurements are used for training.

**Run.** Write the two cohorts as two batches with `scmolt.write_midas_input` and train with `scmolt-train`, as for BMMC.

## Evaluation

Integration is scored on `joint_embedding` with the scIB metrics, using the cell-type annotations and the batches;
annotations are never part of the training input. Imputation is scored on `imputed/<modality>/values` against the
masked or withheld measurements. The settings of every metric are given in the Supplementary Material.
