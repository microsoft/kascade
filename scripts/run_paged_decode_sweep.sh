#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BLOCK_SIZE="${BLOCK_SIZE:-16}"
RUN_TYPE="${RUN_TYPE:-benchmark}"
TOPK="${TOPK:-10}"

configs=(
  "256 1024"
  "128 2048"
  "64 4096"
  "32 8192"
  "16 16384"
  "8 32768"
  "4 65536"
  "2 131072"
)

for cfg in "${configs[@]}"; do
  read -r max_batch max_cache_seqlen <<< "${cfg}"
  python -m src.kernels.flash_decoding.gqa_decode_paged \
    --run_type "${RUN_TYPE}" \
    --max_batch "${max_batch}" \
    --max_cache_seqlen "${max_cache_seqlen}" \
    --block_size "${BLOCK_SIZE}"
done

for layer in 0 1; do
  for cfg in "${configs[@]}"; do
    read -r max_batch max_cache_seqlen <<< "${cfg}"
    python -m src.kernels.flash_decoding.recompute_kascade_gqa_decode_paged \
      --run_type "${RUN_TYPE}" \
      --max_batch "${max_batch}" \
      --max_cache_seqlen "${max_cache_seqlen}" \
      --block_size "${BLOCK_SIZE}" \
      --topk "${TOPK}" \
      --layer "${layer}"
  done
done

for cfg in "${configs[@]}"; do
  read -r max_batch max_cache_seqlen <<< "${cfg}"
  python -m src.kernels.flash_decoding.reuse_kascade_gqa_decode_paged \
    --run_type "${RUN_TYPE}" \
    --max_batch "${max_batch}" \
    --max_cache_seqlen "${max_cache_seqlen}" \
    --block_size "${BLOCK_SIZE}" \
    --topk "${TOPK}"
done