#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Run eval on all GPUs simultaneously, with seed = GPU ID

# Get number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Base command (without seed)
BASE_CMD="python scripts/eval_script_vllm.py \
    --model_name Qwen/Qwen3-8B \
    --dataset_name HuggingFaceH4/aime_2024 \
    --strategies baseline \
    --num_queries 30 \
    --batch_size 32 \
    --topk 10 \
    --top_p 0.95 \
    --temperature 0.6 \
    --top_k 10 \
    --store_results \
    --recompute_layers 0 2 7 14 23"

# Launch on each GPU with seed = GPU ID
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching on GPU $GPU_ID with seed=$GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID $BASE_CMD --seed $GPU_ID &
done

echo "All processes launched. Waiting for completion..."
wait
echo "All evaluations complete."
