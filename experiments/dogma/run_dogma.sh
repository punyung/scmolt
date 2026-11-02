#!/usr/bin/env bash
# Train and predict scMoLT on DOGMA-seq pseudo-mosaics.
#
# usage: run_dogma.sh RNA_H5AD ADT_H5AD ATAC_H5AD OUTPUT_ROOT [CONFIGURATION ...]
#
# The three h5ad files hold all 32,975 DOGMA-seq cells with RNA counts, ADT counts and
# ATAC peaks, and the batch of every cell (lll_ctrl, lll_stim, dig_ctrl, dig_stim) in
# obs["batch"]. Without CONFIGURATION arguments, the three mosaics of the main text are
# run; the other configurations are full, paired+full, paired-abc, diagonal+full, diagonal.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  sed -n '4,9p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
rna=$1 adt=$2 atac=$3 root=$4
shift 4
configurations=("$@")
if [[ ${#configurations[@]} -eq 0 ]]; then
  configurations=(paired-ab paired-bc paired-ac)
fi

export PYTHONHASHSEED=1234
for configuration in "${configurations[@]}"; do
  input="$root/$configuration/input"
  scmolt prepare-input --rna "$rna" --adt "$adt" --atac "$atac" \
    --configuration "$configuration" --output-dir "$input"
  # Training ends with predictions.h5: the embedding of all cells and the masked modalities.
  scmolt-anchor-train --configuration "$configuration" --input-dir "$input" \
    --output-dir "$root/$configuration/run" --device cuda --hash-input
done
