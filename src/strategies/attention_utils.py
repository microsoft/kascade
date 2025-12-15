# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

def softmax_(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    In-place numerically-stable softmax along dimension `dim`.
    Modifies `x` and also returns it for convenience.
    """
    # 1. subtract max
    # note: .values if using torch.max; keepdim=True to allow broadcasting
    max_vals = x.amax(dim=dim, keepdim=True)
    x.sub_(max_vals)

    # 2. exponentiate in-place
    x.exp_()

    # 3. divide by sum of exps
    sum_vals = x.sum(dim=dim, keepdim=True)
    x.div_(sum_vals)

    return x

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def get_heuristic_config() -> dict:
    # Get CUDA device properties
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    # print(f"CUDA device capability: {sm_version}")
    if sm_version == 89:
        return {
            "block_N": 128,
            "block_H": 64,
            "num_split": 8,
            "num_stages": 0,
            "threads": 128
        }
    else:
        return {
            "block_N": 128,
            "block_H": 64,
            "num_split": 8,
            "num_stages": 2,
            "threads": 128
        }
