# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Kascade: Hardware-efficient sparse attention with cross-layer top-k index reuse.

This package provides custom attention kernels and strategies for efficient
long-context inference with large language models.
"""

from .utils import (
    to_gb_str,
    print_gpu_mem_stats,
    get_cuda_free_mem,
    get_torch_free_mem,
    store_results,
)

from .model_utils import (
    get_tokenizer_and_model,
    get_inst_tokens,
    get_eos_token_ids,
)

__version__ = "0.1.0"

__all__ = [
    "to_gb_str",
    "print_gpu_mem_stats", 
    "get_cuda_free_mem",
    "get_torch_free_mem",
    "store_results",
    "get_tokenizer_and_model",
    "get_inst_tokens",
    "get_eos_token_ids",
]
