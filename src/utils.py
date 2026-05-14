# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random

import pandas as pd
import torch
from datasets import load_dataset
from .qadatasets import GSMLongHorizonDataset

num_tokens = 50256
S_TO_MS = 1000
B_TO_GB = 1024 * 1024 * 1024
MATH_REASONING_DATASETS = {
    "HuggingFaceH4/aime_2024",
    "HuggingFaceH4/MATH-500",
    "alesiaivanova/GSM-LongHorizon",
}

def to_gb_str(bytes):
    return f"{bytes / (1024 * 1024 * 1024)} GB"

def print_gpu_mem_stats(loc):
    _, total_mem = torch.cuda.mem_get_info()
    allocated_mem = torch.cuda.memory_allocated()
    reserved_mem = torch.cuda.memory_reserved()
    free_mem = total_mem - allocated_mem
    print(f"GPU mem stats at {loc}: Total: {to_gb_str(total_mem)}, Reserved: {to_gb_str(reserved_mem)}, Allocated: {to_gb_str(allocated_mem)}, Free: {to_gb_str(free_mem)}")

def get_cuda_free_mem(loc):
    free_mem, _ = torch.cuda.mem_get_info()
    print(f"Free memory at {loc}: ", round(free_mem/(1024*1024*1024), 3), "GB")

def get_torch_free_mem(loc):
    torch_reserved = torch.cuda.memory_reserved()
    torch_allocated = torch.cuda.memory_allocated()
    free_torch_mem = torch_reserved - torch_allocated
    print(f"{loc} - ", "Reserved: ", round(torch_reserved/(1024*1024*1024), 3), "GB, ", "Allocated: ", round(torch_allocated/(1024*1024*1024), 3), "GB, ", "Free: ", round(free_torch_mem/(1024*1024*1024), 3), "GB")

def generate_prompts(target_length, tokenizer, num_prompts=5):
    prompts = []
    for _ in range(num_prompts):
        prompt = []
        for _ in range(target_length):
            prompt.append(random.randint(0,num_tokens))
        print(f"Generated prompt length: {len(prompt)}")
        prompts.append(tokenizer.decode(prompt))
    return prompts

def convert_ms_to_s(time_in_ms):
    return time_in_ms / S_TO_MS

def convert_b_to_gb(memory_in_bytes):
    return memory_in_bytes / B_TO_GB

def resolve_dataset_key_for_eval(dataset_name, subset):
    if subset is not None and dataset_name == "THUDM/LongBench":
        return subset
    return dataset_name

def load_eval_dataset(dataset_name, subset, split, cache_dir="/dev/shm"):
    if dataset_name != "alesiaivanova/GSM-LongHorizon":
        return load_dataset(
            dataset_name,
            subset,
            split=split,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

    dataset = load_dataset(
        dataset_name,
        split=split,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if subset is None:
        return dataset
    return GSMLongHorizonDataset.filter_dataset_for_subset(dataset, subset)

def resolve_subset_label_for_eval(dataset_name, subset, default_split):
    if dataset_name == "alesiaivanova/GSM-LongHorizon":
        return GSMLongHorizonDataset.subset_label(subset, default_split)
    return subset if subset is not None else default_split

def should_enable_thinking_tokens_for_dataset(dataset_name):
    return dataset_name in MATH_REASONING_DATASETS

def store_results(experiment_name, results):
    df = pd.DataFrame.from_dict(results, orient='index')
    print("\nStoring Benchmark Results")
    df.to_csv(f"results/{experiment_name}.csv")

def generate_random_kv_cache_for_llama(batch_size, sequence_length, num_layers=32, num_heads=8, head_size=128, dtype=torch.float16, device='cuda'):
    kv_cache = ()
    for _ in range(num_layers):
        kv_cache_layer = ()
        for _ in range(2):
            kv_cache_layer += (torch.randn(batch_size, num_heads, sequence_length, head_size, dtype=dtype, device=device),)
        kv_cache += (kv_cache_layer,)
    return kv_cache
