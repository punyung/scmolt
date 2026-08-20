#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 INPUT_DIR OUTPUT_DIR CUDA_DEVICE_INDEX" >&2
  exit 2
fi

input_dir=$1
output_dir=$2
cuda_index=$3

CUDA_VISIBLE_DEVICES="$cuda_index" PYTHONHASHSEED=1234 \
python -m scmolt.frozenbridge65.train \
  --input-dir "$input_dir" \
  --output-dir "$output_dir" \
  --device cuda \
  --backend gpu_resident_v1 \
  --hash-input
