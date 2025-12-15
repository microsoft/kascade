# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from logging import config
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial
from ..kernel_utils import *
import math



def get_configs():
    block_M = [128]
    block_M = [16,32]
    block_H = [4]
    block_N = [64,128,256]
    num_stages = [1,2,3]
    threads = [64,128,256]
    threads = [128, 256]
    _configs = list(itertools.product(block_M, block_M, block_H, block_N, num_stages, threads, threads))

    configs = [{
        'block_M': c[0],
        'block_M': c[1],
        'block_H': c[2],
        'block_N': c[3],
        'num_stages': c[4],
        'threads': c[5],
        'threads': c[6]
    } for c in _configs]
    return configs


def flashattn(batch, heads, seq_len, dim, tune=False, groups=1, kernel_type="prefill"):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    kv_group_num = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, groups, dim]
    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_M, block_H, block_N, num_stages, threads):
        scores_shape = [batch, groups, T.ceildiv(seq_len, block_M), seq_len]

        @T.macro
        def MMA0(
            K: T.Tensor(kv_shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by // kv_group_num, :], K_shared)

            for i, j in T.Parallel(block_M, block_N):
                is_valid_causal = bx * block_M + i >= k * block_N + j
                acc_s[i, j] = T.if_then_else(is_valid_causal, 0,
                                                -T.infinity(acc_s.dtype))

            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(kv_shape, dtype),
            V_shared: T.SharedBuffer([block_M, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by // kv_group_num, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax_0(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                row_sum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                row_sum[i] = row_sum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, block_N):
                acc_s_cast[i, j] = acc_s[i, j]

        @T.macro
        def Softmax_1(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                row_sum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                row_sum[i] = row_sum[i] * scores_scale[i] + scores_sum[i]


        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.macro
        def MMA0_f(
            K: T.Tensor(kv_shape, dtype),
            Q_shared: T.SharedBuffer([block_M * block_H, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_N, block_M * block_H], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)

            for i, h, j in T.Parallel(block_M, block_H, block_N):
                is_valid_causal = (bx * block_M + i >= k * block_N + j)
                acc_s[j, i * block_H + h] = T.if_then_else(is_valid_causal, 0, -T.infinity(acc_s.dtype))

            T.gemm(K_shared, Q_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax_and_Aggregate(
                aggregate_scores: T.Tensor(scores_shape, dtype),
                acc_s: T.FragmentBuffer([block_N, block_M * block_H], accum_dtype),
                log_sum_local: T.FragmentBuffer([block_M * block_H], accum_dtype),
                agg_s: T.FragmentBuffer([block_N], accum_dtype),
                k: T.int32,
                bx: T.int32,
                by: T.int32,
                bz: T.int32,
        ):
            for i, h, j in T.Parallel(block_M, block_H, block_N):
                acc_s[j, i * block_H + h] = T.exp2(acc_s[j, i * block_H + h] * scale - log_sum_local[i * block_H + h])
            T.reduce_sum(acc_s, agg_s, dim=1)
            for j in T.Parallel(block_N):
                is_not_valid = (bx * block_M <= k * block_N + j)
                if is_not_valid:
                    agg_s[j] = -T.infinity(agg_s.dtype)
            T.copy(agg_s, aggregate_scores[bz, by, bx, k * block_N:(k + 1) * block_N])

        @T.prim_func
        def aggregate(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                log_sum: T.Tensor([batch, seq_len, heads], accum_dtype),
                aggregate_scores: T.Tensor(scores_shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), groups, batch, threads=threads) as (bx_r, by, bz):
                Q_shared = T.alloc_shared([block_M * block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_N, block_M * block_H], accum_dtype)
                agg_s = T.alloc_fragment([block_N], accum_dtype)
                log_sum_local = T.alloc_fragment([block_M * block_H], accum_dtype)
                bx = T.ceildiv(seq_len, block_M) - 1 - bx_r

                for i, h, j in T.Parallel(block_M, block_H, dim):
                    is_within_bounds = (bx * block_M + i < seq_len)
                    Q_shared[i * block_H + h, j] = T.if_then_else(is_within_bounds,
                                            Q[bz, bx * block_M + i, by * block_H + h, j],
                                            0)
                
                for i, h in T.Parallel(block_M, block_H):
                    is_within_bounds = (bx * block_M + i < seq_len)
                    log_sum_local[i * block_H + h] = T.if_then_else(is_within_bounds, log_sum[bz, bx * block_M + i, by * block_H + h],T.infinity(accum_dtype))              

                loop_range = T.min(T.ceildiv(bx * block_M, block_N), T.ceildiv(seq_len - block_M, block_N))
                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages):
                    MMA0_f(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax_and_Aggregate(aggregate_scores, acc_s, log_sum_local, agg_s, k, bx, by, bz)

        @T.prim_func
        def prefill(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                V: T.Tensor(kv_shape, dtype),
                log_sum: T.Tensor([batch, seq_len, heads], accum_dtype),
                Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx_r, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                row_sum = T.alloc_fragment([block_M], accum_dtype)
                bx = T.ceildiv(seq_len, block_M)  - 1 - bx_r

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(row_sum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                loop_range = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax_0(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, row_sum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= row_sum[i]

                for i in T.Parallel(block_M):
                    row_sum[i] = T.log2(row_sum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_M):
                    is_within_bounds = (bx * block_M + i < seq_len)
                    if is_within_bounds:
                        log_sum[bz, bx * block_M + i, by] = row_sum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        @T.prim_func   
        def compute_scores(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                log_sum: T.Tensor([batch, seq_len, heads], accum_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx_r, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                row_sum = T.alloc_fragment([block_M], accum_dtype)
                bx = T.ceildiv(seq_len, block_M)  - 1 - bx_r

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(row_sum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                loop_range = T.min(T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax_1(acc_s, scores_max, scores_max_prev, scores_scale,
                            scores_sum, row_sum)

                for i in T.Parallel(block_M):
                    row_sum[i] = T.log2(row_sum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_M):
                    is_within_bounds = (bx * block_M + i < seq_len)
                    if is_within_bounds:
                        log_sum[bz, bx * block_M + i, by] = row_sum[i]
         
        if kernel_type == "prefill":
            return prefill
        elif kernel_type == "aggregate":
            return aggregate
        elif kernel_type == "compute_scores":
            return compute_scores

    if tune:

        @autotune(
            configs=get_configs(),
            # keys=["block_M", "block_N", "num_stages", "threads"],
            warmup=10,
            rep=10)
        @tilelang.jit(out_idx=[6])
        def kernel(block_M=None, block_H=None, block_N=None, num_stages=None, threads=None):
            return kernel_func(block_M, block_H, block_N, num_stages, threads)

        return kernel()
    else:

        def kernel(block_M, block_H, block_N, num_stages, threads):
            return kernel_func(block_M, block_H, block_N, num_stages, threads)

        return kernel

def full_kernel(recompute_kernel, aggregate_kernel, reuse_kernel, Q, K, V, topk, tile_size=16, layer=0, topk_percent=10):
    B, L, H = Q.shape[:3]
    G = K.shape[2]
    scores = torch.full((B, G, math.ceil(L / tile_size), L), fill_value=float('-inf'), device=Q.device, dtype=Q.dtype)
    if layer == 0:
        log_sum, o = recompute_kernel(Q, K, V)
    else:
        log_sum = recompute_kernel(Q, K)
    aggregate_kernel(Q, K, log_sum, scores)
    if L <= 131072:
        indices = torch.topk(scores, k=topk, dim=-1).indices.to(torch.int32)
    else:
        indices = torch.empty((B, G, math.ceil(L / tile_size), topk), device=Q.device, dtype=torch.int32)
        chunk_size = 131072 // 2**int(topk_percent // 10)
        num_chunks = math.ceil(L / chunk_size)
        for i in range(num_chunks):
            start = (i * chunk_size) // tile_size
            end = min(((i + 1) * chunk_size ) // tile_size, L)
            indices[:, :, start:end, :] = torch.topk(scores[:, :, start:end, :], k=topk, dim=-1).indices.to(torch.int32)
    if layer != 0:
        o = reuse_kernel(Q, K, V, indices, torch.arange(G, device=Q.device, dtype=torch.int32), topk_percent)
    return o, indices, scores

def ref_program_correct_recompute(query, key, value, topk, tile_size=16, layer=0, rolling=False, topk_percent=10):
    dim = query.shape[-1]
    num_head_groups = query.shape[2] // key.shape[2]
    scale = (1/dim)**0.5

    query = query.transpose(1,2) # [B, H, L, D]
    key = key.transpose(1,2).repeat_interleave(num_head_groups, dim=1) # [B, H, L, D]
    value = value.transpose(1,2).repeat_interleave(num_head_groups, dim=1) # [B, H, L, D]

    attn_weights = torch.matmul(query, key.transpose(2, 3))
    dtype = attn_weights.dtype
    min_dtype = torch.finfo(dtype).min
    B, H, Lq, Lk = attn_weights.shape

    if tile_size > 1 and Lq > tile_size:
        num_tiles = math.ceil(Lq / tile_size)
        pad_len = num_tiles * tile_size - Lq
        a_mask: torch.Tensor = torch.full((B, 1, math.ceil(Lq / tile_size)*tile_size, Lq), fill_value=min_dtype, device="cuda") # [B, 1, seqlen+padding, seqlen]
        a_mask.triu_(1) 
        mask = a_mask.view(B, 1, math.ceil(Lq / tile_size), tile_size, Lq)[:,:,:,-1,:] # [B, 1, num_tiles, seqlen]
        mask = mask.roll(shifts=1, dims=2)
        mask[:, :, 0, :] = min_dtype
        attn_weights.add_(a_mask[:,:,:Lq,:])
        attn_weights = softmax_(attn_weights.to(torch.float32) * scale, dim=-1, div=True)
        attn_weights_per_khead = attn_weights.view(B, H//num_head_groups, num_head_groups, Lq, Lk).sum(dim=2) # [B, g, Lq, Lk]
        if pad_len > 0:
            pad = attn_weights_per_khead.new_zeros(B, H//num_head_groups, pad_len, Lk)
            padded = torch.cat([attn_weights_per_khead, pad], dim=-2)   # [B, H//4, Lq_with_padding, Lk]
        else:
            padded = attn_weights_per_khead
        attn_weights_per_khead = padded.view(B, H//num_head_groups, num_tiles, tile_size, Lk).sum(dim=-2) # [B, H//4, num_tiles, Lk]
        attn_weights_per_khead = attn_weights_per_khead.masked_fill_(mask==min_dtype, min_dtype).to(dtype) # [B, H//4, num_tiles, Lk]
        topk_indices = torch.topk(attn_weights_per_khead, k=topk, dim=-1).indices
        if layer == 0:
            out = torch.matmul(attn_weights.to(dtype), value)  # [B, H, Lq, D]
            out = out.transpose(1,2).contiguous()  # [B, Lq, H, D]
        else:
            out = ref_program_correct_reuse(query, key, value, topk_indices, torch.arange(H//num_head_groups, device=key.device), tile_size, rolling, topk_percent)
        return out, topk_indices, attn_weights_per_khead

def ref_program_correct_reuse(query, key, value, topk_indices, head_mapping, tile_size, rolling, topk_percent):
    """
    Inputs:
    - query (Tensor): [batch, seqlen, heads, dim]
    - key (Tensor): [batch, seqlen, groups, dim]
    - value (Tensor): [batch, seqlen, groups, dim]
    - topk_indices (Tensor): [batch, groups, num_tiles, topk]
    - head_mapping (Tensor): [groups]
    Outputs:
    - output (Tensor): [batch, heads, dim]
    """
    dim = query.shape[-1]
    B = query.shape[0]
    num_head_groups = query.shape[1] // head_mapping.shape[0]
    scale = dim**0.5

    #rolling mask
    if rolling and query.shape[2] > 128:
        num_tiles = math.ceil(query.shape[2] / tile_size)
        tile_indices = torch.arange(0, num_tiles*tile_size, tile_size, device=query.device)
        k_indices = torch.arange(0, topk_indices.shape[-1], device=query.device)
        topk_nums = ((((topk_percent*tile_indices)//100).clamp(min=128)+127)//128)*128
        topk_mask = (k_indices >= topk_nums.unsqueeze(1)).to(query.device) # [num_tiles, topk]
        indices = topk_indices.masked_fill(topk_mask, key.shape[2] - 1)
        del tile_indices, k_indices, topk_nums, topk_mask
        torch.cuda.empty_cache()
    else:
        indices = topk_indices

    #select after attention
    indices = indices[:, head_mapping, :, :]  # [batch, groups, num_tiles, topk]
    indices = indices.repeat_interleave(num_head_groups, dim=1).to(torch.int64)  # [batch, heads, num_tiles, topk]
    indices = indices.repeat_interleave(tile_size, dim=2)[:, :, :query.shape[2], :]  # [batch, heads, seqlen, topk]

    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scale  # [batch_size, heads, seqlen, seqlen]
    a_mask: torch.Tensor = torch.full((B, 1, math.ceil(query.shape[2] / tile_size)*tile_size, query.shape[2]), fill_value=torch.finfo(torch.float16).min, device="cuda") # [B, 1, seqlen+padding, seqlen]
    a_mask.triu_(1) 
    mask = a_mask.view(B, 1, math.ceil(query.shape[2] / tile_size), tile_size, query.shape[2])[:,:,:,-1,:] # [B, 1, num_tiles, seqlen]
    mask = mask.roll(shifts=1, dims=2)
    mask[:, :, 0, :] = torch.finfo(query.dtype).min
    values = attn_weights.gather(3, indices)  # [batch_size, heads, seqlen, topk]
    if tile_size > 1 and query.shape[2] > tile_size:
        mask = mask.repeat_interleave(tile_size, dim=-2)[:, :, :query.shape[2], :]  # [B, 1, seqlen, seqlen]
        a_mask = a_mask[:, :, :query.shape[2], :] # [B, 1, seqlen, seqlen]
        attn_weights.masked_fill_(mask == a_mask, torch.finfo(query.dtype).min)
    else:
        attn_weights.fill_(torch.finfo(query.dtype).min)  # Reset all attention weights
    attn_weights.scatter_(3, indices, values)  # Set the topk attention weights
    attn_weights.add_(a_mask)
    
    attn_weights = torch.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)  # [batch_size, heads, seqlen, seqlen]
    out = torch.matmul(attn_weights, value)  # [batch_size, heads, seqlen, dim]
    out = out.transpose(1,2).contiguous()  # [batch_size, seqlen, heads, dim]

    return out

def ref_program(Q, K, V):
    from flash_attn_interface import flash_attn_func
    output = flash_attn_func(Q, K, V, causal=True)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--seq_len', type=int, default=8192, help='sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--run_type', choices=["correctness", "benchmark"], default="benchmark", help='run type')
    parser.add_argument('--code', action='store_true', help='print code')
    parser.add_argument('--topk', type=float, default=10, help="topk % to use")
    parser.add_argument('--layer', type=int, default=0, help='0 for first layer, 1 for subsequent layers')
    parser.add_argument('--rolling', action='store_true', help='use rolling topk')
    parser.add_argument('--with_ref', action='store_true', help='use reference implementation for benchmarking')
    
    args = parser.parse_args()
    batch, heads, seq_len, dim, groups, topk, layer, rolling = args.batch, args.heads, args.seq_len, args.dim, args.groups, args.topk, args.layer, args.rolling
    tile_size = 32

    if (not args.tune):
        if layer == 0:
            program = flashattn(
                T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, tune=args.tune, groups=groups, kernel_type="prefill")(
                    block_M=128, block_H=(heads // groups), block_N=128, num_stages=2, threads=256)
            recompute_kernel = tilelang.compile(program, out_idx=[3, 4], pass_configs={ "tl.disable_safe_memory_legalize": True})
        else: 
            program = flashattn(
                T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, tune=args.tune, groups=groups, kernel_type="compute_scores")(
                    block_M=128, block_H=(heads // groups), block_N=256, num_stages=2, threads=256)
            recompute_kernel = tilelang.compile(program, out_idx=[2], pass_configs={ "tl.disable_safe_memory_legalize": True})
        
        program = flashattn(
                T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, tune=args.tune, groups=groups, kernel_type="aggregate")(
                    block_M=tile_size, block_H=(heads // groups), block_N=256, num_stages=2, threads=256)
        aggregate_kernel = compile_custom_kernel_from_cu(  
            func=program,  
            cu_file_path=f"./kascade_prefill_cu_kernels/dynamic_alllen_aggregate_cp_async_ts{tile_size}_dim128_gsize4_heads32_groups8.cu",
            so_dir=f"./kascade_prefill_cu_kernels/__socache__/",  
            out_idx=None,  
            execution_backend="cython",  
            pass_configs={ "tl.disable_safe_memory_legalize": True},  
        )
        if args.code:
            print(recompute_kernel.get_kernel_source())
            print(aggregate_kernel.get_kernel_source())
            exit(0)
        
        from reuse_kascade_gqa_prefill import flashattn as flashattn_reuse
        reuse_program = flashattn_reuse(T.symbolic("batch"), heads, T.symbolic("kv_seqlen"), dim, T.symbolic("max_topk_num"), rolling=rolling, tune=args.tune, groups=groups)(
            block_M=tile_size, block_H=(heads // groups), block_N=128, num_stages=2, threads=256)
        reuse_kernel = compile_custom_kernel_from_cu(  
            func=reuse_program,  
            cu_file_path=f"./kascade_prefill_cu_kernels/dynamic_alllen_reuse_cp_async_ts{tile_size}_dim128_gsize4_heads32_groups8{'_rolling' if rolling else ''}.cu",
            so_dir=f"./kascade_prefill_cu_kernels/__socache__/", 
            out_idx=[6],  
            execution_backend="cython",
            pass_configs={ "tl.disable_safe_memory_legalize": True , },  
        )


        if args.run_type == "correctness":
            runs = 100
        else: 
            runs = 10
        
        sum_ref_f_latency = 0.0
        sum_recompute_kernel_latency = 0.0
        sum_aggregate_kernel_latency = 0.0
        sum_full_kernel_latency = 0.0

        cnt_ref_f = 0
        cnt_recompute_kernel = 0
        cnt_aggregate_kernel = 0
        cnt_full_kernel = 0

        for _ in range(runs):
            if args.run_type == "correctness":
                seq_len_sample = torch.randint(128, seq_len + 1, (1,)).item()
                topk_sample = torch.randint(10, 50, size=(1,)).item()
                print(f"seq_len: {seq_len_sample}, topk: {topk_sample}")
            else:
                seq_len_sample = seq_len
                topk_sample = topk

            Q = torch.randn(batch, seq_len_sample, heads, dim, device='cuda', dtype=torch.float16)
            K = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)
            V = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)
            actual_topk_num = min(max(128, int((topk_sample/100)*seq_len_sample)), seq_len_sample)
            rounded_topk_num = min(((actual_topk_num + 127) // 128) * 128, seq_len_sample)  # round to multiple of 128        

            if args.run_type == "correctness":
                o, indices, scores = full_kernel(recompute_kernel, aggregate_kernel, reuse_kernel, Q, K, V, rounded_topk_num, tile_size, layer, topk_sample)
                o_ref, indices_ref, scores_ref = ref_program_correct_recompute(Q, K, V, rounded_topk_num, tile_size, layer, rolling, topk_sample)
                scores.masked_fill_((scores == float('-inf')), 0)

                scores_ref.masked_fill_(scores_ref == torch.finfo(scores_ref.dtype).min, 0)
                eps_c = 1e-2
                eps_s = 1e-5
                assert_ = False
                print_ = True
                assert_similar(scores, scores_ref, name="scores_o_ref", assert_=assert_, print_=print_, eps=eps_s)
                assert_allclose(scores, scores_ref, name="scores_o_ref", assert_=assert_, print_=print_, eps=eps_c)
                assert_similar(indices.sort(dim=-1).values.to(torch.float32), indices_ref.sort(dim=-1).values.to(torch.float32), name="indices_o_ref", assert_=assert_, print_=print_, eps=eps_s)
                assert_equal(indices.sort(dim=-1).values, indices_ref.sort(dim=-1).values, name="indices_o_ref", assert_=assert_, print_=print_)
                assert_similar(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps_s if layer==0 else 1e-3)
                if layer == 0:
                    assert_allclose(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps_c)
            else: 
                if args.with_ref:
                    try:
                        ref_f_latency = do_bench(
                        ref_program, n_warmup=1, n_repeat=5,
                        input_tensors=[Q, K, V]
                        )
                        sum_ref_f_latency += ref_f_latency
                        cnt_ref_f += 1
                    except Exception:
                        pass

                try:
                    inputs = [Q, K]
                    inputs += [V] if layer == 0 else []
                    recompute_kernel_latency = do_bench(
                    recompute_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=inputs,
                    )
                    sum_recompute_kernel_latency += recompute_kernel_latency
                    cnt_recompute_kernel += 1
                except Exception:
                    pass
                
                try:
                    log_sum = torch.full((batch, seq_len_sample, heads), fill_value=1, device="cuda", dtype=torch.float32)
                    agg_scores = torch.full((batch, groups, math.ceil(seq_len_sample / tile_size), seq_len_sample), fill_value=float('-inf'), device="cuda", dtype=torch.float16)
                    aggregate_kernel_latency = do_bench(
                    aggregate_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[Q, K, log_sum, agg_scores],
                    )
                    sum_aggregate_kernel_latency += aggregate_kernel_latency
                    cnt_aggregate_kernel += 1
                    del log_sum, agg_scores
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                try:
                    full_kernel_latency = do_bench(
                    full_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[recompute_kernel, aggregate_kernel, reuse_kernel, Q, K, V, rounded_topk_num, tile_size, layer, topk_sample],
                    )
                    sum_full_kernel_latency += full_kernel_latency
                    cnt_full_kernel += 1
                except Exception:
                    pass

                del Q, K, V
                torch.cuda.empty_cache()
            
        if args.run_type == "benchmark":
            flops_per_causal_matmul = batch * heads * seq_len * seq_len * dim
            flops_per_matmul_topk = 2.0 * batch * heads * seq_len * (rounded_topk_num+tile_size) * dim

            flops_flash = 2.0 * flops_per_causal_matmul
            flops_aggregate = flops_per_matmul_topk
            if rolling:
                flops_reuse = flops_per_matmul_topk
            else:
                flops_reuse = topk * (flops_per_matmul_topk)/100.0 + (100-topk)*(2.0 * flops_per_matmul_topk)/100.0
            if layer == 0:
                flops_recompute = 2.0 * flops_per_causal_matmul
                flops_full = flops_recompute + flops_aggregate
            else:
                flops_recompute = flops_per_causal_matmul
                flops_full = flops_recompute + flops_aggregate + flops_reuse

            if args.with_ref:
                avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
                avg_ref_f_tflops = (flops_flash / avg_ref_f_latency * 1e-9) if avg_ref_f_latency > 0 else 0.0
            avg_recompute_kernel_latency = (sum_recompute_kernel_latency / cnt_recompute_kernel) if cnt_recompute_kernel else 0.0
            avg_recompute_kernel_tflops = (flops_recompute / avg_recompute_kernel_latency * 1e-9) if avg_recompute_kernel_latency > 0 else 0.0
            avg_aggregate_kernel_latency = (sum_aggregate_kernel_latency / cnt_aggregate_kernel) if cnt_aggregate_kernel else 0.0
            avg_aggregate_kernel_tflops = (flops_aggregate / avg_aggregate_kernel_latency * 1e-9) if avg_aggregate_kernel_latency > 0 else 0.0
            avg_full_kernel_latency = (sum_full_kernel_latency / cnt_full_kernel) if cnt_full_kernel else 0.0
            avg_full_kernel_tflops = (flops_full / avg_full_kernel_latency * 1e-9) if avg_full_kernel_latency > 0 else 0.0

            if args.with_ref:
                print(f"{seq_len},{topk},"
                    f"{avg_ref_f_latency:.2f},{avg_ref_f_tflops:.2f},"
                    f"{avg_recompute_kernel_latency:.2f},{avg_recompute_kernel_tflops:.2f},"
                    f"{avg_aggregate_kernel_latency:.2f},{avg_aggregate_kernel_tflops:.2f},"
                    f"{avg_full_kernel_latency:.2f},{avg_full_kernel_tflops:.2f}")
            else:
                print(f"{seq_len},{topk},"
                    f"{avg_recompute_kernel_latency:.2f},{avg_recompute_kernel_tflops:.2f},"
                    f"{avg_aggregate_kernel_latency:.2f},{avg_aggregate_kernel_tflops:.2f},"
                    f"{avg_full_kernel_latency:.2f},{avg_full_kernel_tflops:.2f}")
        else:
            print(f"All {runs} correctness tests passed!")
    else:
        best_result = flashattn(batch, heads, seq_len, dim, tune=args.tune)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
        total_flops = flops_per_matmul
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
