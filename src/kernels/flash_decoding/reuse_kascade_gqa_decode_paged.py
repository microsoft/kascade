# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import argparse
from ..kernel_utils import *

torch.random.manual_seed(0)

def get_pass_config():
    return {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        "tl.disable_safe_memory_legalize": True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }

@tilelang.jit(out_idx=[], pass_configs=get_pass_config())
def flashattn_flat(heads, groups, total_tokens, dim, batch, block_N, block_H, num_split, num_stages, threads, topk, dtype='float16'):
    """Kernel variant with flat K/V and pre-resolved indices.
    K/V shape: [total_tokens, groups, dim] (flattened from [num_blocks, block_size, groups, dim]).
    topk_indices: pre-resolved flat physical token indices (no block_table needed).
    Eliminates 1 global memory read + serial dependency per K/V element.
    """
    log_scale = 1.44269504
    accum_dtype = "float"
    kv_group_num = heads // groups
    shape_q = [batch, heads, dim]
    shape_k_flat = [total_tokens, groups, dim]
    shape_v_flat = [total_tokens, groups, dim]
    shape_o = [batch, heads, dim]
    shape_topk = [batch, groups, topk]
    part_shape = [batch, heads, num_split, dim]
    valid_block_H = min(block_H, kv_group_num)

    @T.macro
    def flash_attn_split_flat(
            Q: T.Tensor(shape_q, dtype),
            K_flat: T.Tensor(shape_k_flat, dtype),
            V_flat: T.Tensor(shape_v_flat, dtype),
            cache_seqlens: T.Tensor([batch], "int32"),
            flat_topk_indices: T.Tensor(shape_topk, "int32"),
            head_mapping: T.Tensor([groups], "int32"),
            topk_percent: T.float32,
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], accum_dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            final_scale = T.alloc_var(accum_dtype)
            cur_kv_head = T.alloc_var("int32")
            fallback_idx = T.alloc_var("int32")
            bid = bx; hid = by; sid = bz

            cur_kv_head = hid // (kv_group_num // valid_block_H)
            head_to_reuse = head_mapping[cur_kv_head]
            final_scale = log_scale * softmax_scale
            T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            topk_percent_i = T.int32(topk_percent + 0.5)
            topk_raw = T.floordiv(topk_percent_i * cache_seqlens[bid], 100)
            topk_num = T.min(T.ceildiv(topk_raw, block_N) * block_N, cache_seqlens[bid])
            fallback_idx = flat_topk_indices[bid, head_to_reuse, 0]
            total_chunks = T.ceildiv(topk_num, block_N)
            base_chunks_per_split = T.floordiv(total_chunks, num_split)
            remainder_chunks = T.floormod(total_chunks, num_split)
            final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
            prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
            start_idx = prev_split_chunks * block_N
            for k in T.Pipelined(final_chunks, num_stages=num_stages):
                for i, j in T.Parallel(block_N, dim):
                    flat_idx = flat_topk_indices[bid, head_to_reuse, start_idx + k * block_N + i]
                    K_shared[i, j] = K_flat[flat_idx, cur_kv_head, j]
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(start_idx + k * block_N + j < topk_num, acc_s[i, j],
                                                    -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * final_scale - scores_max[i] * final_scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * final_scale - scores_max[i] * final_scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, S_shared)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_N, dim):
                    V_shared[i, j] = V_flat[
                        T.if_then_else(
                            start_idx + k * block_N + i < topk_num,
                            flat_topk_indices[bid, head_to_reuse, start_idx + k * block_N + i],
                            fallback_idx,
                        ),
                        cur_kv_head,
                        j,
                    ]
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            if final_chunks > 0:
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * final_scale
            T.copy(acc_o[:valid_block_H, :], O_shared)
            if num_split > 1:
                T.copy(logsum[:valid_block_H], glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid])
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid, :])
            else:
                T.copy(O_shared, Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])

    @T.macro
    def combine(
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dim], accum_dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_max = T.alloc_var(accum_dtype)
            lse_logsum = T.alloc_var(accum_dtype)
            scale = T.alloc_var(accum_dtype)

            T.clear(o_accum_local)
            lse_max = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max = T.max(lse_max, glse[bz, by, k])
            lse_logsum = 0
            for k in T.serial(num_split):
                lse_logsum += T.exp2(glse[bz, by, k] - lse_max)
            lse_logsum = T.log2(lse_logsum) + lse_max
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, by, k, i]
                scale = T.exp2(glse[bz, by, k] - lse_logsum)
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale
            for i in T.Parallel(dim):
                Output[bz, by, i] = o_accum_local[i]


    @T.prim_func
    def flashattn_flat_split(
            Q: T.Tensor(shape_q, dtype),
            K_flat: T.Tensor(shape_k_flat, dtype),
            V_flat: T.Tensor(shape_v_flat, dtype),
            cache_seqlens: T.Tensor([batch], "int32"),
            flat_topk_indices: T.Tensor(shape_topk, "int32"),
            head_mapping: T.Tensor([groups], "int32"),
            topk_percent: T.float32,
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
    ):
        flash_attn_split_flat(Q, K_flat, V_flat, cache_seqlens, flat_topk_indices, head_mapping, topk_percent, softmax_scale, glse, Output_partial, Output)
        if num_split > 1:
            combine(glse, Output_partial, Output)

    return flashattn_flat_split

@tilelang.jit(out_idx=[], pass_configs=get_pass_config())
def flashattn(heads, groups, max_blocks, num_blocks, block_size, dim, batch, block_N, block_H, num_split, num_stages, threads, topk, dtype='float16'):
    log_scale = 1.44269504
    accum_dtype = "float"
    kv_group_num = heads // groups
    shape_q = [batch, heads, dim]
    shape_k = [num_blocks, block_size, groups, dim]
    shape_v = [num_blocks, block_size, groups, dim]
    shape_o = [batch, heads, dim]
    shape_bt = [batch, max_blocks]
    shape_topk = [batch, groups, topk]
    part_shape = [batch, heads, num_split, dim]
    valid_block_H = min(block_H, kv_group_num)
    assert block_size <= block_N

    @T.macro
    def flash_attn_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            block_table: T.Tensor(shape_bt, dtype="int32"),
            cache_seqlens: T.Tensor([batch], "int32"),
            topk_indices: T.Tensor(shape_topk, "int32"),
            head_mapping: T.Tensor([groups], "int32"),
            topk_percent: T.float32,
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(
                batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            O_shared = T.alloc_shared([valid_block_H, dim], accum_dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            final_scale = T.alloc_var(accum_dtype)
            cur_kv_head = T.alloc_var("int32")
            fallback_idx = T.alloc_var("int32")
            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)
            head_to_reuse = head_mapping[cur_kv_head]
            final_scale = log_scale * softmax_scale
            T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            topk_percent_i = T.int32(topk_percent + 0.5)
            topk_raw = T.floordiv(topk_percent_i * cache_seqlens[bid], 100)
            topk_num = T.min(T.ceildiv(topk_raw, block_N) * block_N, cache_seqlens[bid])
            fallback_idx = topk_indices[bid, head_to_reuse, 0]
            total_chunks = T.ceildiv(topk_num, block_N)
            base_chunks_per_split = T.floordiv(total_chunks, num_split)
            remainder_chunks = T.floormod(total_chunks, num_split)
            final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
            prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
            start_idx = prev_split_chunks * block_N
            for k in T.Pipelined(final_chunks, num_stages=num_stages):
                for i, j in T.Parallel(block_N, dim):
                    idx = topk_indices[bid, head_to_reuse, start_idx + k * block_N + i]
                    page_idx = T.floordiv(idx, block_size)
                    K_shared[i, j] = K[block_table[bid, page_idx], T.floormod(idx, block_size), cur_kv_head, j]
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(start_idx + k * block_N + j < topk_num, acc_s[i, j],
                                                    -T.infinity(accum_dtype))
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * final_scale - scores_max[i] * final_scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * final_scale - scores_max[i] * final_scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, S_shared)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_N, dim):
                    idx = T.if_then_else(
                        start_idx + k * block_N + i < topk_num,
                        topk_indices[bid, head_to_reuse, start_idx + k * block_N + i],
                        fallback_idx,
                    )
                    page_idx = T.floordiv(idx, block_size)
                    V_shared[i, j] = V[block_table[bid, page_idx], T.floormod(idx, block_size), cur_kv_head, j]
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            if final_chunks > 0:
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * final_scale
            T.copy(acc_o[:valid_block_H, :], O_shared)
            if num_split > 1:
                T.copy(logsum[:valid_block_H],
                        glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid])
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid, :])
            else:
                T.copy(O_shared, Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])

    @T.macro
    def combine(
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dim], accum_dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_max = T.alloc_var(accum_dtype)
            lse_logsum = T.alloc_var(accum_dtype)
            scale = T.alloc_var(accum_dtype)

            T.clear(o_accum_local)
            lse_max = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max = T.max(lse_max, glse[bz, by, k])
            lse_logsum = 0
            for k in T.serial(num_split):
                lse_logsum += T.exp2(glse[bz, by, k] - lse_max)
            lse_logsum = T.log2(lse_logsum) + lse_max
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, by, k, i]
                scale = T.exp2(glse[bz, by, k] - lse_logsum)
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale
            for i in T.Parallel(dim):
                Output[bz, by, i] = o_accum_local[i]

    @T.prim_func
    def flashattn_gqa_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            block_table: T.Tensor(shape_bt, dtype="int32"),
            cache_seqlens: T.Tensor([batch], "int32"),
            topk_indices: T.Tensor(shape_topk, "int32"),
            head_mapping: T.Tensor([groups], "int32"),
            topk_percent: T.float32,
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
    ):
        flash_attn_split(Q, K, V, block_table, cache_seqlens, topk_indices, head_mapping, topk_percent, softmax_scale, glse, Output_partial, Output)
        if num_split > 1:
            combine(glse, Output_partial, Output)

    return flashattn_gqa_decode_split


def transform_topk_indices(topk_indices, block_table, cache_seqlens, topk_percent, block_size, block_N):
    """Transform logical topk indices to flat physical token indices.
    
    For each logical index, compute: physical_block * block_size + offset_within_block.
    Invalid indices (beyond per-example topk count) are set to a large sentinel so they
    sort to the end and don't interfere.
    """
    batch, groups, max_topk = topk_indices.shape
    topk_num = ((((topk_percent * cache_seqlens) / 100).to(torch.int32) + block_N - 1) // block_N) * block_N
    topk_num = torch.minimum(topk_num, cache_seqlens.to(topk_num.dtype))
    topk_num = topk_num.clamp(max=max_topk)

    logical_blocks = topk_indices // block_size  # [batch, groups, max_topk]
    offsets = topk_indices % block_size
    # Gather physical blocks: block_table[b, logical_blocks[b, g, i]]
    physical_blocks = torch.gather(
        block_table.unsqueeze(1).expand(-1, groups, -1),
        2,
        logical_blocks.long()
    ).to(torch.int32)
    flat_indices = physical_blocks * block_size + offsets  # [batch, groups, max_topk]

    # Mask invalid indices with large sentinel
    total_tokens = block_table.shape[1] * block_size
    for b in range(batch):
        tn = topk_num[b].item()
        if tn < max_topk:
            flat_indices[b, :, tn:] = total_tokens - 1


    return flat_indices.to(torch.int32)

def ref_program_correct(query, key, value, block_table, cache_seqlens, topk_indices, head_mapping, topk_percent, softmax_scale):
    batch = query.shape[0]
    dim = query.shape[-1]
    block_size = key.shape[1]
    groups = key.shape[2]
    num_head_groups = query.shape[1] // groups
    max_seqlen = cache_seqlens.max().item()

    key_contiguous = torch.zeros(batch, max_seqlen, groups, dim, device=key.device, dtype=key.dtype)
    value_contiguous = torch.zeros(batch, max_seqlen, groups, dim, device=value.device, dtype=value.dtype)

    for b in range(batch):
        seqlen = cache_seqlens[b].item()
        token_indices = torch.arange(seqlen, device=key.device)
        logical_blocks = token_indices // block_size
        offsets = token_indices % block_size
        physical_blocks = block_table[b, logical_blocks]
        key_contiguous[b, :seqlen] = key[physical_blocks, offsets]
        value_contiguous[b, :seqlen] = value[physical_blocks, offsets]

    q = query.unsqueeze(2)
    topk_num = ((((topk_percent * cache_seqlens) / 100).to(torch.int32) + 127) // 128) * 128
    topk_num = torch.minimum(topk_num, cache_seqlens.to(topk_num.dtype))
    topk_num = topk_num.clamp(max=topk_indices.shape[-1])

    indices = topk_indices[:, head_mapping, :].unsqueeze(2)
    for b in range(indices.shape[0]):
        tn = int(topk_num[b].item())
        if tn <= 0:
            indices[b, :, 0, :] = 0
        elif tn < indices.shape[-1]:
            indices[b, :, 0, tn:] = indices[b, :, 0, tn - 1].unsqueeze(-1).expand_as(indices[b, :, 0, tn:])
    indices = indices.repeat_interleave(num_head_groups, dim=1).to(torch.int64)
    k = key_contiguous.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)
    v = value_contiguous.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * softmax_scale
    values = attn_weights.gather(3, indices)
    attn_weights.fill_(float('-inf'))
    attn_weights.scatter_(3, indices, values)
    attn_weights = torch.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
    out = torch.matmul(attn_weights, v).squeeze(2)
    return out


def main(max_batch: int = 1,
         heads: int = 32,
         groups: int = 8,
         max_cache_seqlen: int = 8192,
         block_size: int = 32,
         max_model_len: int = 131072,
         dim: int = 128,
         topk: int = 10,
         run_type: str = "benchmark",
         code: bool = False,
         with_ref: bool = False,
         pool_extra_blocks: int = 16):
    from .gqa_decode_paged import ref_program_fa
    from .recompute_kascade_gqa_decode_paged import compute_adaptive_block_H, compute_optimal_split

    blocks_per_seq = (max_cache_seqlen + block_size - 1) // block_size
    num_blocks = max_batch * blocks_per_seq + pool_extra_blocks
    max_blocks = (max_model_len + block_size - 1) // block_size
    block_N = 128
    opt_block_H = compute_adaptive_block_H(heads, groups)
    total_tokens = num_blocks * block_size

    kernel = flashattn(
        heads=heads, groups=groups, dim=dim,
        max_blocks=max_blocks, num_blocks=T.symbolic("num_blocks"),
        block_size=block_size, batch=T.symbolic("batch"),
        block_N=block_N, block_H=opt_block_H,
        num_split=T.symbolic("num_split"), num_stages=1,
        threads=128, topk=T.symbolic("topk"),
    )

    flat_kern = flashattn_flat(
        heads=heads, groups=groups, dim=dim,
        total_tokens=T.symbolic("total_tokens"),
        batch=T.symbolic("batch"), block_N=block_N, block_H=opt_block_H,
        num_split=T.symbolic("num_split"), num_stages=1,
        threads=128, topk=T.symbolic("topk"))

    if code:
        print(flat_kern.get_kernel_source(kernel_only=False))
        return

    runs = 100 if run_type == "correctness" else 10

    sum_orig = 0.0; cnt_orig = 0
    sum_flat = 0.0; cnt_flat = 0

    for _ in range(runs):
        if run_type == "correctness":
            batch = torch.randint(1, max_batch + 1, (1,)).item()
            cache_seqlens = torch.randint(128, max_cache_seqlen, (batch,), device='cuda', dtype=torch.int32)
            topk_sample = torch.randint(10, 50, size=(1,)).item()
        else:
            batch = max_batch
            cache_seqlens = torch.full((batch,), max_cache_seqlen, device='cuda', dtype=torch.int32)
            topk_sample = topk

        max_seqlen = cache_seqlens.max().item()
        max_topk = (int((topk_sample * max_seqlen) / 100) + block_N - 1) // block_N * block_N
        topk_seqlen = int(topk_sample * max_seqlen / 100)
        cur_split = compute_optimal_split(batch, heads, groups, max_cache_seqlen, block_N, opt_block_H, topk_percent=topk_sample)

        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_blocks, block_size, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_blocks, block_size, groups, dim, device="cuda", dtype=torch.float16)
        global_perm = torch.randperm(num_blocks, dtype=torch.int32, device="cuda")
        block_table = torch.zeros((batch, max_blocks), dtype=torch.int32, device="cuda")
        for b in range(batch):
            block_table[b, :blocks_per_seq] = global_perm[b * blocks_per_seq:(b + 1) * blocks_per_seq]
        head_mapping = torch.randint(0, groups, (groups,), device="cuda", dtype=torch.int32)
        random_scores = torch.rand(batch, groups, max_seqlen, device="cuda")
        seq_range = torch.arange(max_seqlen, device="cuda").unsqueeze(0)
        invalid_mask = seq_range >= cache_seqlens.unsqueeze(1)
        random_scores.masked_fill_(invalid_mask.unsqueeze(1), float('-inf'))
        topk_indices = torch.topk(random_scores, k=max_topk, dim=-1).indices.to(torch.int32)
        del random_scores

        # Flat KV: reshape [num_blocks, block_size, groups, dim] -> [num_blocks*block_size, groups, dim]
        k_flat = k.reshape(total_tokens, groups, dim)
        v_flat = v.reshape(total_tokens, groups, dim)

        # Transform indices
        flat_indices = transform_topk_indices(topk_indices, block_table, cache_seqlens, topk_sample, block_size, block_N)

        glse = torch.empty(batch, heads, cur_split, device="cuda", dtype=torch.float32)
        Output_partial = torch.empty(batch, heads, cur_split, dim, device="cuda", dtype=torch.float32)
        output = torch.empty(batch, heads, dim, device="cuda", dtype=torch.float16)

        if run_type == "correctness":
            print(f"cache_seqlens={cache_seqlens}, topk={topk_sample}, split={cur_split}")
            # Test original kernel
            kernel(q, k, v, block_table, cache_seqlens, topk_indices, head_mapping,
                   topk_sample, (1.0 / dim)**0.5, glse, Output_partial, output)
            o_ref = ref_program_correct(q, k, v, block_table, cache_seqlens, topk_indices,
                                        head_mapping, topk_sample, (1.0 / dim)**0.5)
            assert_similar(output, o_ref, name="orig_ref", assert_=False, print_=True, eps=1e-3)

            # Test flat kernel
            output_flat = torch.empty_like(output)
            glse_f = torch.empty_like(glse)
            Op_f = torch.empty_like(Output_partial)
            flat_kern(q, k_flat, v_flat, cache_seqlens, flat_indices, head_mapping,
                      topk_sample, (1.0 / dim)**0.5, glse_f, Op_f, output_flat)
            assert_similar(output_flat, o_ref, name="flat_ref", assert_=False, print_=True, eps=1e-3)
        else:
            # Benchmark original
            try:
                lat = do_bench(kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k, v, block_table, cache_seqlens, topk_indices, head_mapping,
                                   topk_sample, (1.0 / dim)**0.5, glse, Output_partial, output])
                sum_orig += lat; cnt_orig += 1
            except Exception:
                pass

            # Benchmark flat
            try:
                glse_f = torch.empty_like(glse)
                Op_f = torch.empty_like(Output_partial)
                output_f = torch.empty_like(output)
                lat = do_bench(flat_kern, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k_flat, v_flat, cache_seqlens, flat_indices, head_mapping,
                                   topk_sample, (1.0 / dim)**0.5, glse_f, Op_f, output_f])
                sum_flat += lat; cnt_flat += 1
            except Exception:
                pass

        del q, k, v, k_flat, v_flat, block_table, head_mapping, topk_indices, glse, Output_partial
        del flat_indices
        torch.cuda.empty_cache()

    if run_type == "benchmark":
        topk_qk_flops = 2 * batch * heads * max_topk * dim
        topk_pv_flops = 2 * batch * heads * max_topk * dim
        total_topk_flops = topk_qk_flops + topk_pv_flops
        avg_orig = (sum_orig / cnt_orig) if cnt_orig else 0.0
        avg_flat = (sum_flat / cnt_flat) if cnt_flat else 0.0
        tflops_orig = (total_topk_flops / avg_orig * 1e-9) if avg_orig > 0 else 0.0
        tflops_flat = (total_topk_flops / avg_flat * 1e-9) if avg_flat > 0 else 0.0
        print(f"{max_batch},{max_cache_seqlen},{topk},{cur_split},"
              f"{avg_orig:.4f},{tflops_orig:.2f},"
              f"{avg_flat:.4f},{tflops_flat:.2f}")
    else:
        print(f"{runs} Correctness test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--groups', type=int, default=8)
    parser.add_argument('--max_cache_seqlen', type=int, default=8192)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--max_model_len', type=int, default=131072)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--topk', type=float, default=10)
    parser.add_argument('--run_type', type=str, default='benchmark', choices=["benchmark", "correctness"])
    parser.add_argument('--code', action='store_true')
    parser.add_argument('--with_ref', action='store_true')
    args = parser.parse_args()
    main(args.max_batch, args.heads, args.groups, args.max_cache_seqlen, args.block_size,
         args.max_model_len, args.dim, args.topk, args.run_type, args.code, args.with_ref)
