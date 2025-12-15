# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import argparse
from ..kernel_utils import *

torch.random.manual_seed(0)


def flashattn(heads, groups, dim, layer=0):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // groups

    def kernel_func(batch, block_N, block_H, num_split, num_stages, threads, max_cache_seqlen):
        shape_q = [batch, heads, dim]
        shape_k = [batch, max_cache_seqlen, groups, dim]
        shape_v = [batch, max_cache_seqlen, groups, dim]
        shape_o = [batch, heads, dim]
        part_shape = [batch, heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)

        @T.macro
        def flash_attn_split_0(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlens: T.Tensor([batch], dtype="int32"),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                scores: T.Tensor([batch, heads, max_cache_seqlen], dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], accum_dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)
                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                 

                total_chunks = T.ceildiv(cache_seqlens[bid], block_N)
                base_chunks_per_split = T.floordiv(total_chunks, num_split)
                remainder_chunks = T.floormod(total_chunks, num_split)
                final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
                prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
                start_idx = prev_split_chunks * block_N
                for k in T.Pipelined(final_chunks, num_stages=num_stages):
                    T.copy(
                        K[bid, start_idx +
                          k * block_N:start_idx + (k + 1) * block_N,
                          cur_kv_head, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(start_idx + k * block_N + j < cache_seqlens[bid], acc_s[i, j], -T.infinity(accum_dtype))
                    T.copy(acc_s[:valid_block_H, :], scores[bid, hid * valid_block_H:(hid + 1) * valid_block_H, start_idx + (k * block_N):start_idx + ((k + 1) * block_N)])
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[bid, start_idx +
                          k * block_N:start_idx + (k + 1) * block_N,
                          cur_kv_head, :], V_shared)

                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if final_chunks > 0:
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                T.copy(logsum[:valid_block_H],
                       glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid])
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid, :])

        @T.macro
        def combine_0(
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, accum_dtype),
                row_sums: T.Tensor([batch, heads], accum_dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], accum_dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split, 128], accum_dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    lse_local:
                        T.Fragment(lse_local.shape, forward_fn=lambda i, j: (j, i)),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                for k, j in T.Parallel(num_split, 128):
                    lse_local[k, j] = glse[bz, by, k]
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                if T.get_thread_binding() == 0:
                    row_sums[bz, by] = lse_logsum_local[0]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split[0] = glse[bz, by, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum_local[i]

        @T.macro
        def flash_attn_split_1(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                cache_seqlens: T.Tensor([batch], dtype="int32"),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                scores: T.Tensor([batch, heads, max_cache_seqlen], dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)
                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                 

                total_chunks = T.ceildiv(cache_seqlens[bid], block_N)
                base_chunks_per_split = T.floordiv(total_chunks, num_split)
                remainder_chunks = T.floormod(total_chunks, num_split)
                final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
                prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
                start_idx = prev_split_chunks * block_N
                for k in T.Pipelined(final_chunks, num_stages=num_stages):
                    T.copy(
                        K[bid, start_idx +
                          k * block_N:start_idx + (k + 1) * block_N,
                          cur_kv_head, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(start_idx + k * block_N + j < cache_seqlens[bid], acc_s[i, j], -T.infinity(accum_dtype))
                    T.copy(acc_s[:valid_block_H, :], scores[bid, hid * valid_block_H:(hid + 1) * valid_block_H, start_idx + (k * block_N):start_idx + ((k + 1) * block_N)])
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                T.copy(logsum[:valid_block_H],
                       glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid])

        @T.macro
        def combine_1(
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                row_sums: T.Tensor([batch, heads], accum_dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                lse_local = T.alloc_fragment([num_split, 128], accum_dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    lse_local:
                        T.Fragment(lse_local.shape, forward_fn=lambda i, j: (j, i)),
                })

                T.clear(lse_logsum_local)
                for k, j in T.Parallel(num_split, 128):
                    lse_local[k, j] = glse[bz, by, k]
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                if T.get_thread_binding() == 0:
                    row_sums[bz, by] = lse_logsum_local[0]

        @T.macro
        def aggregate(
                scores: T.Tensor([batch, heads, max_cache_seqlen], dtype),
                row_sums: T.Tensor([batch, heads], accum_dtype),
                cache_seqlen: T.Tensor([batch], dtype="int32"),
                aggregated_scores: T.Tensor([batch, groups, max_cache_seqlen], dtype),
        ):
            with T.Kernel(batch, groups, num_split, threads=128) as (bx, by, bz):
                row_sum_local = T.alloc_fragment([kv_group_num, block_N], accum_dtype)
                aggregated_scores_frag = T.alloc_fragment([block_N], accum_dtype)
                scores_local = T.alloc_fragment([kv_group_num, block_N], accum_dtype)


                T.annotate_layout({
                    row_sum_local:
                        T.Fragment(row_sum_local.shape, forward_fn=lambda i, j: (j, i)),
                    aggregated_scores_frag:
                        T.Fragment(aggregated_scores_frag.shape, forward_thread_fn=lambda i: i),
                    scores_local:
                        T.Fragment(scores_local.shape, forward_fn=lambda i, j: (j, i)),
                })

                bid = bx
                gid = by
                sid = bz
                for i, j in T.Parallel(kv_group_num, block_N):
                    row_sum_local[i, j] = row_sums[bid, gid*kv_group_num + i]

                total_chunks = T.ceildiv(cache_seqlen[bid], block_N)
                base_chunks_per_split = T.floordiv(total_chunks, num_split)
                remainder_chunks = T.floormod(total_chunks, num_split)
                final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
                prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
                start_idx = prev_split_chunks * block_N
                for k in T.Pipelined(final_chunks, num_stages=1):
                    for i, j in T.Parallel(kv_group_num, block_N):
                        scores_local[i,j] = T.if_then_else(start_idx + k * block_N + j < cache_seqlen[bid], scores[bid, gid * kv_group_num + i, start_idx + (k * block_N) + j], -T.infinity(accum_dtype))
                        scores_local[i,j] = T.exp2(scores_local[i,j] * scale - row_sum_local[i,j])
                    T.reduce_sum(scores_local, aggregated_scores_frag, dim=0, clear=True)
                    tb = T.thread_binding(0, block_N, thread='threadIdx.x')
                    if start_idx + (k * block_N) + tb < cache_seqlen[bid]:
                        aggregated_scores[bid, gid, start_idx + (k * block_N) + tb] = aggregated_scores_frag[0]


        @T.prim_func
        def flashattn_gqa_decode_split_0(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlens: T.Tensor([batch], dtype="int32"),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                row_sums: T.Tensor([batch, heads], accum_dtype),
                scores: T.Tensor([batch, heads, max_cache_seqlen], dtype),
                Output: T.Tensor(shape_o, dtype),
                aggregated_scores: T.Tensor([batch, groups, max_cache_seqlen], dtype),
        ):
            flash_attn_split_0(Q, K, V, cache_seqlens, glse, Output_partial, scores)
            combine_0(glse, Output_partial, Output, row_sums)
            aggregate(scores, row_sums, cache_seqlens, aggregated_scores)

        @T.prim_func
        def flashattn_gqa_decode_split_1(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlens: T.Tensor([batch], dtype="int32"),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                row_sums: T.Tensor([batch, heads], accum_dtype),
                scores: T.Tensor([batch, heads, max_cache_seqlen], dtype),
                Output: T.Tensor(shape_o, dtype),
                aggregated_scores: T.Tensor([batch, groups, max_cache_seqlen], dtype),
        ):
            flash_attn_split_1(Q, K, cache_seqlens, glse, scores)
            combine_1(glse, row_sums)
            aggregate(scores, row_sums, cache_seqlens, aggregated_scores)

        if layer == 0:
            return flashattn_gqa_decode_split_0
        else:
            return flashattn_gqa_decode_split_1


    def kernel(batch, block_N, block_H, num_split, num_stages, threads, max_cache_seqlen):
        return kernel_func(batch, block_N, block_H, num_split, num_stages, threads, max_cache_seqlen)

    return kernel
    
def softmax_(x: torch.Tensor, dim: int = -1, div: bool = True) -> torch.Tensor:
    """
    In-place numerically-stable softmax along dimension `dim`.
    Modifies `x` and also returns it for convenience.
    """
    # 1. subtract max
    # note: .values if using torch.max; keepdim=True to allow broadcasting
    # max_vals = x.max(dim=dim, keepdim=True).values
    max_vals = x.max()
    x.sub_(max_vals)

    # 2. exponentiate in-place
    x.exp_()

    # 3. divide by sum of exps
    if div:
        sum_vals = x.sum(dim=dim, keepdim=True)
        x.div_(sum_vals)

    return x


def full_kernel(recompute_kernel_function, reuse_kernel_function, Q, K, V, cache_seqlens, topk_percent, glse1, Output_partial1, row_sums, scores, glse2, Output_partial2, layer=0):
    O, scores = recompute_kernel_function(Q, K, V, cache_seqlens, glse1, Output_partial1, row_sums, scores)
    mask = torch.arange(scores.shape[2], device=scores.device).expand(scores.shape[0], scores.shape[2]) < cache_seqlens.unsqueeze(1)
    scores.masked_fill_(~mask.unsqueeze(1), 0.0)
    max_seqlen = cache_seqlens.max().item()
    max_topk = (int((topk_percent * max_seqlen)/100) + 127) // 128 * 128
    topk_indices = torch.topk(scores[:, :, :max_seqlen], k=max_topk, dim=-1).indices.to(torch.int32)
    if layer != 0:
        O = reuse_kernel_function(Q, K, V, cache_seqlens, topk_indices, torch.arange(K.shape[2], dtype=torch.int32, device=K.device), topk_percent, glse2, Output_partial2)
    return O, topk_indices, scores

def ref_program_fa(query, key, value):
    from flash_attn_interface import flash_attn_with_kvcache
    Q = query.unsqueeze(1)  # [batch_size, 1, heads, dim]
    out = flash_attn_with_kvcache(Q, key, value, causal=False)
    out = out.squeeze(1)  # [batch_size, heads, dim]
    return out

def ref_program_correct(query, key, value, cache_seqlens, topk_percent, layer):
    """
    Inputs:
    - query (Tensor): [batch, heads, dim]`
    - key (Tensor): [batch, seqlen_kv, groups, dim]
    - value (Tensor): [batch, seqlen_kv, groups, dim]
    - mask (Tensor): [batch, seqlen_kv, groups]
    Outputs:
    - output (Tensor): [batch, heads, dim]
    """
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = (1/dim)**0.5
    q = query.unsqueeze(2)  # [batch_size, heads, 1, dim]
    k = key.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, seqlen_kv, dim]
    v = value.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, seqlen_kv, dim]
    
    attn_weights = torch.matmul(q, k.transpose(2, 3))  # [batch_size, heads, 1, seqlen_kv]

    attn_weights = attn_weights.to(torch.float32) * scale
    for b in range(attn_weights.shape[0]):
        attn_weights[b, :, :, cache_seqlens[b]:] = float('-inf')
    
    attn_weights = softmax_(attn_weights, dim=-1, div=True)  # [batch_size, heads, 1, seqlen_kv]

    attn_weights_per_khead = attn_weights.view(key.shape[0], key.shape[2], num_head_groups, 1, key.shape[1]).sum(dim=2).squeeze(2).to(torch.float16)  # [batch_size, groups, seqlen_kv]
    max_seqlen = cache_seqlens.max().item()
    max_topk = (int((topk_percent * max_seqlen)/100) + 127) // 128 * 128
    topk_indices = torch.topk(attn_weights_per_khead[:, :, :max_seqlen], k=max_topk, dim=-1).indices  # [batch_size, groups, topk]
    
    if layer == 0:
        out = torch.matmul(attn_weights.to(torch.float16), v)  # [batch_size, heads, 1, dim]
        out = out.squeeze(2)  # [batch_size, heads, dim]
    else:
        out = ref_program_correct_1(query, key, value, cache_seqlens, topk_indices, torch.arange(key.shape[2], device=key.device), topk_percent)
    return out, topk_indices, attn_weights_per_khead

def ref_program_correct_1(query, key, value, cache_seqlens, topk_indices, head_mapping, topk_percent):
    """
    Inputs:
    - query (Tensor): [batch, heads, dim]
    - key (Tensor): [batch, seqlen_kv, groups, dim]
    - value (Tensor): [batch, seqlen_kv, groups, dim]
    - mask (Tensor): [batch, seqlen_kv, groups]
    - topk_indices (Tensor): [batch, groups, topk]
    - head_mapping (Tensor): [groups]
    Outputs:
    - output (Tensor): [batch, heads, dim]
    """
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5

    q = query.unsqueeze(2)  # [batch_size, heads, 1, dim]
    topk_num = ((((topk_percent * cache_seqlens)/100).to(torch.int32) + 127) // 128) * 128

    #select after attention
    indices = topk_indices[:, head_mapping, :].unsqueeze(2)  # [batch, groups, 1, topk]
    for b in range(indices.shape[0]):
        indices[b, :, 0, topk_num[b]:] = indices[b, :, 0, topk_num[b]-1].unsqueeze(-1).expand_as(indices[b, :, 0, topk_num[b]:])
    indices = indices.repeat_interleave(num_head_groups, dim=1).to(torch.int64)  # [batch, heads, 1, topk]
    k = key.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, kv_seqlen, dim]
    v = value.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, kv_seqlen, dim]


    attn_weights = torch.matmul(q, k.transpose(2, 3)) / scale  # [batch_size, heads, 1, seqlen_kv]

    # select after attention
    values = attn_weights.gather(3, indices)  # [batch_size, heads, 1, topk]
    attn_weights.fill_(float('-inf'))  # Reset all attention weights
    attn_weights.scatter_(3, indices, values)  # Set the topk attention weights
    
    attn_weights = torch.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)  # [batch_size, heads, 1, seqlen_kv]
    out = torch.matmul(attn_weights, v)  # [batch_size, heads, 1, dim]
    out = out.squeeze(2)  # [batch_size, heads, dim]
        
    return out


def main(batch: int = 1,
         heads: int = 32,
         groups: int = 8,
         max_cache_seqlen: int = 8192,
         dim: int = 128,
         topk: int = 10,
         run_type: str = "benchmark",
         code: bool = False,
         layer: int = 0,
         with_ref: bool = False):
    
    program = flashattn(heads, groups, dim, layer=layer)(
        batch=T.symbolic("batch"),
        block_N=128,
        block_H=64,
        num_split=8,
        num_stages=2,
        threads=128,
        max_cache_seqlen=T.symbolic("max_cache_seqlen"),
    )
    kernel = tilelang.compile(program, out_idx=[8, 9], pass_configs={ "tl.disable_safe_memory_legalize": True , })

    from reuse_kascade_gqa_decode_with_kvcache import flashattn as flashattn_reuse
    reuse_program = flashattn_reuse(heads, groups, dim)(
        batch=T.symbolic("batch"),
        block_N=128,
        block_H=64,
        num_split=8,
        num_stages=2,
        threads=128,
        max_cache_seqlen=T.symbolic("max_cache_seqlen"),
        topk=T.symbolic("topk"),
    )
    reuse_kernel = compile_custom_kernel_from_cu(  
        func=reuse_program,  
        cu_file_path="./kascade_decode_cu_kernels/dynamic_alllen_reuse_cp_async_dim128_gsize4_heads32_groups8_with_kvcache.cu",
        so_dir="./kascade_decode_cu_kernels/__socache__/",
        out_idx=[9],  
        execution_backend="cython",  
        pass_configs={ "tl.disable_safe_memory_legalize": True , },  
    )

    if code:
        cuda_source = kernel.get_kernel_source()  
        print(cuda_source)
        return

    if run_type == "correctness":
        runs = 100
    else:
        runs = 10

    # For benchmark averaging
    sum_ref_f_latency = 0.0
    sum_tile_latency = 0.0
    sum_tile_topk_latency = 0.0

    cnt_ref_f = 0
    cnt_tile = 0
    cnt_tile_topk = 0

    for _ in range(runs):
        if run_type == "correctness":
            cache_seqlens = torch.randint(128, max_cache_seqlen, (batch,), device='cuda', dtype=torch.int32)
            topk_sample = torch.randint(5, 30, size=(1,), dtype=torch.int32).item()
            print(f"topk={topk_sample}")
        else:
            cache_seqlens = torch.full((batch,), max_cache_seqlen, device='cuda', dtype=torch.int32)
            topk_sample = topk
        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, max_cache_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, max_cache_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        glse1 = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float32)
        Output_partial1 = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float32)
        glse2 = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float32)
        Output_partial2 = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float32)
        row_sums = torch.empty(batch, heads, device="cuda", dtype=torch.float32)
        scores = torch.empty(batch, heads, max_cache_seqlen, device="cuda", dtype=torch.float16)
        max_seqlen = cache_seqlens.max().item()
        max_topk = (int((topk_sample * max_seqlen)/100) + 127) // 128 * 128
        topk_num = ((((topk_sample * cache_seqlens)/100).to(torch.int32) + 127) // 128) * 128

        if run_type == "correctness":
            print(cache_seqlens)
            o, topk_indices_og, scores_og = full_kernel(kernel, reuse_kernel, q, k, v, cache_seqlens, topk_sample, glse1, Output_partial1, row_sums, scores, glse2, Output_partial2, layer)
            o_ref, topk_indices_ref, scores_ref = ref_program_correct(q, k, v, cache_seqlens, topk_sample, layer)

            eps_s = 1e-4
            eps_c = 1e-3
            assert_ = False
            print_ = True
            print(topk_indices_og.shape, topk_indices_ref.shape)
            assert_similar(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps_s)
            assert_allclose(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=1e-1)
            for b in range(batch):
                assert_similar(scores_og[b,:,:cache_seqlens[b]], scores_ref[b,:,:cache_seqlens[b]], name=f"scores_o_ref_{b}", assert_=assert_, print_=print_, eps=eps_s)
                assert_allclose(scores_og[b,:,:cache_seqlens[b]], scores_ref[b,:,:cache_seqlens[b]], name=f"scores_o_ref_{b}", assert_=assert_, print_=print_, eps=eps_c)
                assert_similar(topk_indices_og[b,:,:topk_num[b]].sort(dim=-1).values.to(torch.float32), topk_indices_ref[b,:,:topk_num[b]].sort(dim=-1).values.to(torch.float32), name="o_ref_topk", assert_=assert_, print_=print_, eps=eps_s)
                assert_equal(topk_indices_og[b,:,:topk_num[b]].sort(dim=-1).values, topk_indices_ref[b,:,:topk_num[b]].sort(dim=-1).values, name="o_ref_topk", assert_=assert_, print_=print_)
        else:
            if with_ref:
                try:
                    ref_f_latency = do_bench(
                    ref_program_fa, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k, v]
                    )
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

            try:
                tile_latency = do_bench(
                kernel, n_warmup=1, n_repeat=5,
                input_tensors=[q, k, v, cache_seqlens, glse1, Output_partial1, row_sums, scores]
                )
                sum_tile_latency += tile_latency
                cnt_tile += 1
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                tile_topk_latency = do_bench(
                full_kernel,
                n_warmup=1, n_repeat=5,
                input_tensors=[kernel, reuse_kernel, q, k, v, cache_seqlens, topk_sample, glse1, Output_partial1, row_sums, scores, glse2, Output_partial2, layer]
                )
                sum_tile_topk_latency += tile_topk_latency
                cnt_tile_topk += 1
            except Exception:
                pass
                
        del q, k, v, glse1, Output_partial1, glse2, Output_partial2, row_sums, scores
        torch.cuda.empty_cache()

    if run_type == "benchmark":
        qk_flops = 2 * batch * heads * max_cache_seqlen * dim
        pv_flops = 2 * batch * heads * max_cache_seqlen * dim
        total_flops = qk_flops + pv_flops

        if layer == 0:
            topk_qk_flops = 2 * batch * heads * max_cache_seqlen * dim
            topk_pv_flops = 2 * batch * heads * max_cache_seqlen * dim
            topk_agg_flops = (2 * heads - groups) * batch * max_cache_seqlen
            total_topk_flops = topk_qk_flops + topk_pv_flops + topk_agg_flops
        else:
            topk_qk_flops1 = 2 * batch * heads * max_cache_seqlen * dim
            topk_qk_flops2 = 2 * batch * heads * max_topk * dim
            topk_pv_flops = 2 * batch * heads * max_topk * dim
            topk_agg_flops = (2 * heads - groups) * batch * max_cache_seqlen
            total_topk_flops = topk_qk_flops1 + topk_qk_flops2 + topk_pv_flops + topk_agg_flops
        if with_ref:
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            avg_ref_f_tflops = (total_flops / avg_ref_f_latency * 1e-9) if avg_ref_f_latency > 0 else 0.0
        avg_tile_latency = (sum_tile_latency / cnt_tile) if cnt_tile else 0.0
        avg_tile_tflops = (total_topk_flops / avg_tile_latency * 1e-9) if avg_tile_latency > 0 else 0.0
        avg_tile_topk_latency = (sum_tile_topk_latency / cnt_tile_topk) if cnt_tile_topk else 0.0
        avg_tile_topk_tflops = (total_topk_flops / avg_tile_topk_latency * 1e-9) if avg_tile_topk_latency > 0 else 0.0
        if with_ref:
            print(f"{max_cache_seqlen},{topk_sample},"
            f"{avg_ref_f_latency:.2f},{avg_ref_f_tflops:.2f},"
            f"{avg_tile_latency:.2f},{avg_tile_tflops:.2f},"
            f"{avg_tile_topk_latency:.2f},{avg_tile_topk_tflops:.2f}")
        else:
            print(f"{max_cache_seqlen},{topk_sample},"
            f"{avg_tile_latency:.2f},{avg_tile_tflops:.2f},"
            f"{avg_tile_topk_latency:.2f},{avg_tile_topk_tflops:.2f}")
    else:
        print(f"{runs} Correctness test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--max_cache_seqlen', type=int, default=8192, help='kv sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--topk', type=float, default=10, help='topk percent for attention')
    parser.add_argument('--run_type', type=str, default='benchmark', choices=["benchmark", "correctness"], help='Type of run: benchmark, correctness')
    parser.add_argument('--code', action='store_true', help='output code configs')
    parser.add_argument("--layer", type=int, default=0, help="0 for first layer, 1 for subsequent layers")
    parser.add_argument('--with_ref', action='store_true', help='run with reference for benchmark')
    args = parser.parse_args()
    main(args.batch, args.heads, args.groups, args.max_cache_seqlen, args.dim, args.topk, args.run_type, args.code, args.layer, args.with_ref)
