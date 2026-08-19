# Model card: scMoLT Cell C BMMC checkpoint family

## Model details

| Field | Value |
|---|---|
| Architecture | `scmolt.trackc.TrackCMIDASCycle` |
| Variant | C3, paired-ABC topology, RNA canonical anchor |
| Biological / technical latent | 32 / 2 dimensions |
| Native backbone | scMIDAS 0.1.13 |
| Training | 1,000 epochs, seed 1234 |
| Modalities | RNA + ATAC; no synthetic ADT |
| Inputs | 4,000 RNA genes, 50,000 ATAC peaks, three batch classes |
| Checkpoints | One model for each 20%, 40%, 60%, and 80% RNA-mask condition |

The specialization combines W14-D40 (`lam_dsc=40`) with translated batch CE
scale 0.625, missing-RNA batch-adversary scale 1.25, paired RNA-anchor
consistency weight 5.0, and independent 25% feature re-masking of observed RNA
and ATAC entries.

## Intended use

The released weights reproduce the frozen BMMC sensitivity experiment and
support research inspection, strict loading, embedding export, and RNA
imputation under its exact feature and batch contract. They are not validated
for clinical decisions, unseen feature vocabularies, new organisms, or unseen
batch classes.

## Evaluation contract

- Same 22,211 cells, nested masks, selected features, and three original
  batches as the scMIDAS-1000 baseline.
- scIB uses `cell_type_l1` and reports Bio conservation, Batch correction, and
  Total on all cells and masked cells.
- RNA imputation reports per-cell Pearson, cell-type pseudobulk Pearson,
  per-cell/global RMSE, per-cell/global MAE, cosine similarity, and mean
  library-size ratio.
- Higher is better except RMSE, MAE, and absolute library-size-ratio error.
- Hidden RNA truth is evaluator-only. Cell-type labels are never model inputs.

## Headline results versus scMIDAS-1000

The format below is `scMIDAS -> scMoLT (scMoLT minus scMIDAS)`; for RMSE, a
negative raw delta is better.

| RNA mask | All-cell Bio | All-cell Batch | All-cell Total | RNA Pearson | RNA global RMSE | RNA cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 20% | 0.5940 -> 0.5494 (-0.0446) | 0.7031 -> 0.7685 (+0.0654) | 0.6377 -> 0.6370 (-0.0006) | 0.5731 -> 0.5718 (-0.0012) | 1.2794 -> 1.2898 (+0.0104) | 0.5940 -> 0.5928 (-0.0012) |
| 40% | 0.5704 -> 0.5662 (-0.0043) | 0.7420 -> 0.7641 (+0.0221) | 0.6391 -> 0.6454 (+0.0063) | 0.5643 -> 0.5688 (+0.0044) | 1.2644 -> 1.2399 (-0.0246) | 0.5856 -> 0.5899 (+0.0043) |
| 60% | 0.5632 -> 0.5595 (-0.0037) | 0.7403 -> 0.7312 (-0.0091) | 0.6340 -> 0.6281 (-0.0059) | 0.5548 -> 0.5646 (+0.0099) | 1.2877 -> 1.2723 (-0.0153) | 0.5766 -> 0.5860 (+0.0094) |
| 80% | 0.5505 -> 0.5683 (+0.0178) | 0.7109 -> 0.6957 (-0.0152) | 0.6147 -> 0.6192 (+0.0046) | 0.5402 -> 0.5537 (+0.0135) | 1.2992 -> 1.2870 (-0.0122) | 0.5631 -> 0.5757 (+0.0126) |

Across all pre-existing metrics, scMoLT wins 39/56 condition-metric
comparisons; within imputation it wins 28/32. This count is descriptive and
was not a predeclared composite score. Complete machine-readable values are in
`benchmarks/bmmc/comparison_long.csv` and `comparison_wide.csv`.

## Integration/imputation trade-off

The imputation gains do not imply uniformly better biological conservation.
The reconstruction objective directly rewards recovery of held-out expression,
whereas scIB Bio measures neighborhood/label structure in the joint embedding.
The batch adversary and translated fusion can improve technical mixing and
cross-modal prediction while moving some biologically similar neighborhoods.
For that reason, Bio, Batch, Total, and imputation are reported separately and
no single aggregate number is used to claim universal superiority.

## Limitations

- One training seed; no confidence intervals or cross-seed stability claim.
- Four condition-specific checkpoints; no interpolation claim between mask
  rates.
- Fixed human BMMC feature order and three-class training batch vocabulary.
- Results are from GSE194122 BMMC Multiome only.
- Dense predictions, cells, truth, labels, optimizer state, and replay state
  are intentionally excluded from the release.
