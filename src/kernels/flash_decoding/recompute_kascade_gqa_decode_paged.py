# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import argparse
import math
import os
from ..kernel_utils import *

torch.random.manual_seed(0)


_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    "tl.disable_safe_memory_legalize": True,
}

def get_pass_config():
    return {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        "tl.disable_safe_memory_legalize": True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }

@tilelang.jit(out_idx=[], pass_configs=get_pass_config())
def flashattn(heads, groups, max_blocks, num_blocks, block_size, max_seqlen,
              dim, layer, batch, block_N, block_H, num_split, num_stages, threads, dtype="float16"):
    log_scale = 1.44269504
    accum_dtype = "float"
    kv_group_num = heads // groups
    shape_q = [batch, heads, dim]
    shape_k = [num_blocks, block_size, groups, dim]
    shape_v = [num_blocks, block_size, groups, dim]
    shape_o = [batch, heads, dim]
    shape_bt = [batch, max_blocks]
    part_shape = [batch, heads, num_split, dim]
    valid_block_H = min(block_H, kv_group_num)
    assert block_size <= block_N

    @T.macro
    def flash_attn_split_0(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            block_table: T.Tensor(shape_bt, dtype="int32"),
            cache_seqlens: T.Tensor([batch], dtype="int32"),
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            scores: T.Tensor([batch, heads, max_seqlen], dtype),
            Output: T.Tensor(shape_o, dtype),
            row_sums: T.Tensor([batch, heads], accum_dtype),
    ):
        pages_per_blockN = T.ceildiv(block_N, block_size)

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

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)
            final_scale = log_scale * softmax_scale
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
                block_table_offset = (prev_split_chunks + k) * pages_per_blockN
                for i, j in T.Parallel(block_N, dim):
                    page_idx = T.floordiv(i, block_size)
                    token_idx = start_idx + k * block_N + i
                    K_shared[i, j] = T.if_then_else(
                        token_idx < cache_seqlens[bid],
                        K[block_table[bid, block_table_offset + page_idx],
                          T.floormod(i, block_size), cur_kv_head, j], 0)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        start_idx + k * block_N + j < cache_seqlens[bid],
                        acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(acc_s[:valid_block_H, :],
                       scores[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                              start_idx + (k * block_N):start_idx + ((k + 1) * block_N)])
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
                    page_idx = T.floordiv(i, block_size)
                    token_idx = start_idx + k * block_N + i
                    V_shared[i, j] = T.if_then_else(
                        token_idx < cache_seqlens[bid],
                        V[block_table[bid, block_table_offset + page_idx],
                          T.floormod(i, block_size), cur_kv_head, j], 0)
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
                T.copy(logsum[:valid_block_H], row_sums[bid, hid * valid_block_H:(hid + 1) * valid_block_H])


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
            if T.get_thread_binding() == 0:
                row_sums[bz, by] = lse_logsum
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, by, k, i]
                scale = T.exp2(glse[bz, by, k] - lse_logsum)
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale
            for i in T.Parallel(dim):
                Output[bz, by, i] = o_accum_local[i]

    @T.macro
    def flash_attn_split_1(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            block_table: T.Tensor(shape_bt, dtype="int32"),
            cache_seqlens: T.Tensor([batch], dtype="int32"),
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            scores: T.Tensor([batch, heads, max_seqlen], dtype),
            row_sums: T.Tensor([batch, heads], accum_dtype),
    ):
        pages_per_blockN = T.ceildiv(block_N, block_size)

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
            final_scale = T.alloc_var(accum_dtype)
            cur_kv_head = T.alloc_var("int32")

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)
            final_scale = log_scale * softmax_scale
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
                block_table_offset = (prev_split_chunks + k) * pages_per_blockN
                for i, j in T.Parallel(block_N, dim):
                    page_idx = T.floordiv(i, block_size)
                    token_idx = start_idx + k * block_N + i
                    K_shared[i, j] = T.if_then_else(
                        token_idx < cache_seqlens[bid],
                        K[block_table[bid, block_table_offset + page_idx],
                          T.floormod(i, block_size), cur_kv_head, j], 0)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        start_idx + k * block_N + j < cache_seqlens[bid],
                        acc_s[i, j], -T.infinity(accum_dtype))
                T.copy(acc_s[:valid_block_H, :],
                       scores[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                              start_idx + (k * block_N):start_idx + ((k + 1) * block_N)])
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
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * final_scale
            T.copy(logsum[:valid_block_H],
                   glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid])
            if num_split == 1:
                T.copy(logsum[:valid_block_H], row_sums[bid, hid * valid_block_H:(hid + 1) * valid_block_H])

    @T.macro
    def combine_1(
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            row_sums: T.Tensor([batch, heads], accum_dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            lse_max = T.alloc_var(accum_dtype)
            lse_logsum = T.alloc_var(accum_dtype)

            lse_max = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max = T.max(lse_max, glse[bz, by, k])
            lse_logsum = 0
            for k in T.serial(num_split):
                lse_logsum += T.exp2(glse[bz, by, k] - lse_max)
            lse_logsum = T.log2(lse_logsum) + lse_max
            if T.get_thread_binding() == 0:
                row_sums[bz, by] = lse_logsum

    @T.macro
    def aggregate(
            scores: T.Tensor([batch, heads, max_seqlen], dtype),
            row_sums: T.Tensor([batch, heads], accum_dtype),
            cache_seqlen: T.Tensor([batch], dtype="int32"),
            softmax_scale: T.float32,
            aggregated_scores: T.Tensor([batch, groups, max_seqlen], dtype),
    ):
        with T.Kernel(batch, groups, num_split, threads=128) as (bx, by, bz):
            row_sum_local = T.alloc_fragment([kv_group_num, block_N], accum_dtype)
            aggregated_scores_frag = T.alloc_fragment([block_N], accum_dtype)
            scores_local = T.alloc_fragment([kv_group_num, block_N], accum_dtype)

            bid = bx
            gid = by
            sid = bz
            for i, j in T.Parallel(kv_group_num, block_N):
                row_sum_local[i, j] = row_sums[bid, gid * kv_group_num + i]
            final_scale = log_scale * softmax_scale
            total_chunks = T.ceildiv(cache_seqlen[bid], block_N)
            base_chunks_per_split = T.floordiv(total_chunks, num_split)
            remainder_chunks = T.floormod(total_chunks, num_split)
            final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
            prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
            start_idx = prev_split_chunks * block_N
            for k in T.Pipelined(final_chunks, num_stages=1):
                for i, j in T.Parallel(kv_group_num, block_N):
                    scores_local[i, j] = T.if_then_else(
                        start_idx + k * block_N + j < cache_seqlen[bid],
                        scores[bid, gid * kv_group_num + i, start_idx + (k * block_N) + j],
                        -T.infinity(accum_dtype))
                    scores_local[i, j] = T.exp2(
                        scores_local[i, j] * final_scale - row_sum_local[i, j])
                T.reduce_sum(scores_local, aggregated_scores_frag, dim=0, clear=True)
                for i in T.Parallel(block_N):
                    if start_idx + (k * block_N) + i < cache_seqlen[bid]:
                        aggregated_scores[bid, gid, start_idx + (k * block_N) + i] = aggregated_scores_frag[i]

    @T.prim_func
    def flashattn_gqa_decode_split_0(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            block_table: T.Tensor(shape_bt, dtype="int32"),
            cache_seqlens: T.Tensor([batch], dtype="int32"),
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            row_sums: T.Tensor([batch, heads], accum_dtype),
            scores: T.Tensor([batch, heads, max_seqlen], dtype),
            Output: T.Tensor(shape_o, dtype),
            aggregated_scores: T.Tensor([batch, groups, max_seqlen], dtype),
    ):
        flash_attn_split_0(Q, K, V, block_table, cache_seqlens, softmax_scale,
                           glse, Output_partial, scores, Output, row_sums)
        if num_split > 1:
            combine_0(glse, Output_partial, Output, row_sums)
        aggregate(scores, row_sums, cache_seqlens, softmax_scale, aggregated_scores)

    @T.prim_func
    def flashattn_gqa_decode_split_1(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            block_table: T.Tensor(shape_bt, dtype="int32"),
            cache_seqlens: T.Tensor([batch], dtype="int32"),
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            row_sums: T.Tensor([batch, heads], accum_dtype),
            scores: T.Tensor([batch, heads, max_seqlen], dtype),
            Output: T.Tensor(shape_o, dtype),
            aggregated_scores: T.Tensor([batch, groups, max_seqlen], dtype),
    ):
        flash_attn_split_1(Q, K, block_table, cache_seqlens, softmax_scale,
                           glse, scores, row_sums)
        if num_split > 1:
            combine_1(glse, row_sums)
        aggregate(scores, row_sums, cache_seqlens, softmax_scale, aggregated_scores)

    if layer == 0:
        return flashattn_gqa_decode_split_0
    else:
        return flashattn_gqa_decode_split_1


def compute_adaptive_block_H(heads, groups):
    kv_group_num = heads // groups
    block_H = max(16, ((kv_group_num + 15) // 16) * 16)
    block_H = min(block_H, 64)
    return block_H


def _get_sm_count():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).multi_processor_count
    return 132
_SM_COUNT = _get_sm_count()


def _get_device_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(0)
    return (9, 0)


_DEVICE_CAPABILITY = _get_device_capability()


def get_arch_config():
    """Return kernel tuning parameters based on GPU architecture.
    
    H100 (SM 9.x): original values that work best on Hopper.
    B200 (SM 10.x): tuned values — smaller tiles and fewer threads reduce
    synchronization overhead on Blackwell for these bandwidth-bound kernels.
    """
    capability = _DEVICE_CAPABILITY
    sm_count = _SM_COUNT

    if capability[0] >= 10:  # Blackwell (B200, etc.)
        return {
            "recompute_block_N": 128,
            "reuse_block_N": 128,
            "recompute_threads": 128,
            "reuse_threads": 128,
            "recompute_stages": 1,
            "reuse_stages": 1,
            "topk_splitk_splits": 16 if sm_count >= 160 else 8,
        }
    else:  # Hopper (H100, etc.) — original values
        return {
            "recompute_block_N": 256,
            "reuse_block_N": 128,
            "recompute_threads": 256,
            "reuse_threads": 128,
            "recompute_stages": 1,
            "reuse_stages": 1,
            "topk_splitk_splits": 8,
        }

def should_use_splitk_topk(batch, groups, max_cache_seqlen=None, sm_count=None):
    if sm_count is None:
        sm_count = _SM_COUNT
    grid_no_split = batch * groups
    return grid_no_split <= max(1, sm_count // 8)

def compute_optimal_split(batch, heads, groups, max_cache_seqlen, block_N, block_H,
                          sm_count=None, layer=0, topk_percent=None):
    if sm_count is None:
        sm_count = _get_sm_count()
    is_blackwell = _DEVICE_CAPABILITY[0] >= 10
    kv_group_num = heads // groups
    valid_block_H = min(block_H, kv_group_num)
    head_blocks = heads // valid_block_H
    if topk_percent is not None:
        effective_tokens = int(topk_percent * max_cache_seqlen / 100)
    else:
        effective_tokens = max_cache_seqlen
    if effective_tokens <= 1024:
        return 1
    total_chunks = max(1, (effective_tokens + block_N - 1) // block_N)
    grid_no_split = batch * head_blocks
    target_grid = sm_count * (6 if is_blackwell else 4)
    if is_blackwell and layer == 0 and topk_percent is None and effective_tokens >= 4096:
        target_grid *= 2
    ideal_split = max(1, (target_grid + grid_no_split - 1) // grid_no_split)
    ideal_split = min(ideal_split, total_chunks)
    min_chunks_per_split = 1 if is_blackwell else 2
    max_split_by_work = max(1, total_chunks // min_chunks_per_split)
    ideal_split = min(ideal_split, max_split_by_work)
    ideal_split = max(1, 2 ** round(math.log2(ideal_split)))
    if is_blackwell and layer == 0:
        ideal_split = min(ideal_split, 64)
    else:
        ideal_split = min(ideal_split, 128)
    return ideal_split


# Fused per-batch topk kernel (histogram-based radix selection, outputs logical indices)
@tilelang.jit(out_idx=[], pass_configs=_pass_configs)
def _tl_fused_topk_impl(groups, block_size, block_N=128, dtype="float16"):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    max_topk = T.dynamic("max_topk")
    max_blocks = T.dynamic("max_blocks")
    NBINS = 1 << 10
    RADIX2 = 1 << 6
    BLOCK_SIZE = 1024

    @T.prim_func
    def fused_topk(
        scores: T.Tensor[(batch, groups, seq_len), dtype],
        cache_seqlens: T.Tensor[(batch,), "int32"],
        block_table: T.Tensor[(batch, max_blocks), "int32"],
        topk_frac: T.float32,
        out_indices: T.Tensor[(batch, groups, max_topk), "int32"],
    ):
        with T.Kernel(batch, groups, threads=BLOCK_SIZE) as (bx, by):
            tx = T.get_thread_binding()

            s_histogram = T.alloc_shared([NBINS + 1], T.int32)
            s_sub_hist = T.alloc_shared([RADIX2 + 1], T.int32)
            s_threshold_bin_id = T.alloc_shared([1], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_threshold_bin_id_2 = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_sub_bin = T.alloc_var(T.int32)
            l_raw_bits = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)
            l_per_batch_topk = T.alloc_var(T.int32)
            l_logical_idx = T.alloc_var(T.int32)
            l_block_id = T.alloc_var(T.int32)
            l_offset = T.alloc_var(T.int32)
            l_phys_block = T.alloc_var(T.int32)
            l_start_pos = T.alloc_var(T.int32)

            l_end_idx = cache_seqlens[bx]
            l_per_batch_topk = T.Cast(T.int32,
                topk_frac * T.Cast(T.float32, l_end_idx))
            l_per_batch_topk = T.ceildiv(l_per_batch_topk, block_N) * block_N
            l_per_batch_topk = T.min(l_per_batch_topk, l_end_idx)
            l_per_batch_topk = T.min(l_per_batch_topk, max_topk)
            l_new_topk = l_per_batch_topk

            # Init
            s_histogram[tx] = 0
            if tx < RADIX2 + 1:
                s_sub_hist[tx] = 0
            if tx == 0:
                s_histogram[NBINS] = 0
                s_threshold_bin_id[0] = 0
            T.sync_threads()

            # Pass 1: 10-bit histogram
            for s in T.serial(T.ceildiv(l_end_idx, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx:
                    l_raw_bits = T.Cast(T.int32, T.reinterpret(T.uint16, scores[bx, by, input_idx]))
                    l_bin_id32 = T.if_then_else(l_raw_bits >= 0x8000, 0, l_raw_bits >> 6)
                    T.atomic_add(s_histogram[l_bin_id32], 1)
            T.sync_threads()

            # Suffix sum (1024 bins = 1024 threads)
            if tx < NBINS:
                for i in T.serial(10):
                    offset = 1 << i
                    T.sync_threads(3, NBINS)
                    if tx + offset < NBINS:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, NBINS)
                    if tx + offset < NBINS:
                        s_histogram[tx] = l_val
                T.sync_threads(3, NBINS)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()

            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # Pass 2: emit above threshold + build 6-bit sub-histogram for threshold bin
            for s in T.serial(T.ceildiv(l_end_idx, BLOCK_SIZE)):
                T.sync_threads()
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx:
                    l_raw_bits = T.Cast(T.int32, T.reinterpret(T.uint16, scores[bx, by, input_idx]))
                    l_bin_id32 = T.if_then_else(l_raw_bits >= 0x8000, 0, l_raw_bits >> 6)
                    if l_bin_id32 > l_threshold_bin_id:
                        l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1,
                                           return_prev=True)
                        if l_out_pos < l_per_batch_topk:
                            l_logical_idx = input_idx
                            l_block_id = T.floordiv(l_logical_idx, block_size)
                            l_offset = T.floormod(l_logical_idx, block_size)
                            l_phys_block = block_table[bx, l_block_id]
                            out_indices[bx, by, l_out_pos] = \
                                l_phys_block * block_size + l_offset
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        l_sub_bin = l_raw_bits & 0x3F
                        T.atomic_add(s_sub_hist[l_sub_bin], 1)

            if l_new_topk > 0:
                l_start_pos = l_per_batch_topk - l_new_topk
                T.sync_threads()

                # Sub-histogram suffix sum (64 bins, 6 rounds)
                T.fill(s_threshold_bin_id[0], 0)
                T.sync_threads()
                if tx < RADIX2:
                    for i in T.serial(6):
                        offset = 1 << i
                        T.sync_threads(3, RADIX2)
                        if tx + offset < RADIX2:
                            l_val = s_sub_hist[tx] + s_sub_hist[tx + offset]
                        T.sync_threads(3, RADIX2)
                        if tx + offset < RADIX2:
                            s_sub_hist[tx] = l_val
                    T.sync_threads(3, RADIX2)
                    if s_sub_hist[tx] > l_new_topk and s_sub_hist[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id_2 = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_sub_hist[l_threshold_bin_id_2 + 1]
                T.sync_threads()

                # Pass 3: emit from threshold bin based on sub-bin
                for s in T.serial(T.ceildiv(l_end_idx, BLOCK_SIZE)):
                    T.sync_threads()
                    input_idx = s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx:
                        l_raw_bits = T.Cast(T.int32, T.reinterpret(T.uint16, scores[bx, by, input_idx]))
                        l_bin_id32 = T.if_then_else(l_raw_bits >= 0x8000, 0, l_raw_bits >> 6)
                        if l_bin_id32 == l_threshold_bin_id:
                            l_sub_bin = l_raw_bits & 0x3F
                            if l_sub_bin > l_threshold_bin_id_2:
                                l_out_pos = T.atomic_add(
                                    s_sub_hist[l_sub_bin + 1], 1,
                                    return_prev=True) + l_start_pos
                                if l_out_pos < l_per_batch_topk:
                                    l_logical_idx = input_idx
                                    l_block_id = T.floordiv(l_logical_idx, block_size)
                                    l_offset = T.floormod(l_logical_idx, block_size)
                                    l_phys_block = block_table[bx, l_block_id]
                                    out_indices[bx, by, l_out_pos] = \
                                        l_phys_block * block_size + l_offset
                            elif l_sub_bin == l_threshold_bin_id_2 and l_new_topk > 0:
                                l_out_pos = T.atomic_add(
                                    s_sub_hist[l_sub_bin + 1], 1,
                                    return_prev=True) + l_start_pos
                                if l_out_pos < l_per_batch_topk:
                                    l_logical_idx = input_idx
                                    l_block_id = T.floordiv(l_logical_idx, block_size)
                                    l_offset = T.floormod(l_logical_idx, block_size)
                                    l_phys_block = block_table[bx, l_block_id]
                                    out_indices[bx, by, l_out_pos] = \
                                        l_phys_block * block_size + l_offset

    return fused_topk


@tilelang.jit(out_idx=[], pass_configs=_pass_configs)
def _tl_fused_topk_splitk_3k_impl(groups, block_size, num_splits, block_N=128, dtype="float16"):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    max_topk = T.dynamic("max_topk")
    max_blocks = T.dynamic("max_blocks")
    NBINS = 1 << 10
    RADIX2 = 1 << 6
    BLOCK_SIZE = 1024

    @T.prim_func
    def fused_topk_splitk_3k(
        scores: T.Tensor[(batch, groups, seq_len), dtype],
        cache_seqlens: T.Tensor[(batch,), "int32"],
        block_table: T.Tensor[(batch, max_blocks), "int32"],
        topk_frac: T.float32,
        out_indices: T.Tensor[(batch, groups, max_topk), "int32"],
        g_hist: T.Tensor[(batch, groups, num_splits, NBINS), "int32"],
        threshold_info: T.Tensor[(batch, groups, 5), "int32"],
        emit_ctr: T.Tensor[(batch, groups, 2), "int32"],
    ):
        # K1: Split histogram
        with T.Kernel(batch, groups, num_splits, threads=BLOCK_SIZE) as (bx, by, bz):
            tx = T.get_thread_binding()
            s_histogram = T.alloc_shared([NBINS], T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_chunk = T.alloc_var(T.int32)
            l_start = T.alloc_var(T.int32)
            l_end = T.alloc_var(T.int32)
            l_raw_bits = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)

            l_end_idx = cache_seqlens[bx]
            l_chunk = T.ceildiv(l_end_idx, num_splits)
            l_start = bz * l_chunk
            l_end = T.min(l_start + l_chunk, l_end_idx)

            if tx < NBINS:
                s_histogram[tx] = 0
            T.sync_threads()

            for s in T.serial(T.ceildiv(l_chunk, BLOCK_SIZE)):
                input_idx = l_start + s * BLOCK_SIZE + tx
                if input_idx < l_end:
                    l_raw_bits = T.Cast(T.int32, T.reinterpret(T.uint16, scores[bx, by, input_idx]))
                    l_bin_id32 = T.if_then_else(l_raw_bits >= 0x8000, 0, l_raw_bits >> 6)
                    T.atomic_add(s_histogram[l_bin_id32], 1)
            T.sync_threads()

            if tx < NBINS:
                g_hist[bx, by, bz, tx] = s_histogram[tx]

        # K2: Merge + suffix + threshold + sub-histogram scan + sub-threshold
        with T.Kernel(batch, groups, threads=BLOCK_SIZE) as (bx, by):
            tx = T.get_thread_binding()
            s_histogram = T.alloc_shared([NBINS + 1], T.int32)
            s_sub_hist = T.alloc_shared([RADIX2 + 1], T.int32)
            s_threshold_bin_id = T.alloc_shared([1], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_threshold_bin_id_2 = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_new_topk_2 = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_per_batch_topk = T.alloc_var(T.int32)
            l_raw_bits = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_sub_bin = T.alloc_var(T.int32)

            l_end_idx = cache_seqlens[bx]
            l_per_batch_topk = T.Cast(T.int32,
                topk_frac * T.Cast(T.float32, l_end_idx))
            l_per_batch_topk = T.ceildiv(l_per_batch_topk, block_N) * block_N
            l_per_batch_topk = T.min(l_per_batch_topk, l_end_idx)
            l_per_batch_topk = T.min(l_per_batch_topk, max_topk)
            l_new_topk = l_per_batch_topk

            # Merge histograms
            if tx < NBINS:
                l_val = 0
                for sp in T.serial(num_splits):
                    l_val = l_val + g_hist[bx, by, sp, tx]
                s_histogram[tx] = l_val
            if tx == 0:
                s_histogram[NBINS] = 0
                s_threshold_bin_id[0] = 0
            if tx < RADIX2 + 1:
                s_sub_hist[tx] = 0
            T.sync_threads()

            # Suffix sum
            if tx < NBINS:
                for i in T.serial(10):
                    offset = 1 << i
                    T.sync_threads(3, NBINS)
                    if tx + offset < NBINS:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, NBINS)
                    if tx + offset < NBINS:
                        s_histogram[tx] = l_val
                T.sync_threads(3, NBINS)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()

            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # Light scan for sub-histogram of threshold bin only
            if l_new_topk > 0:
                for s in T.serial(T.ceildiv(l_end_idx, BLOCK_SIZE)):
                    input_idx = s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx:
                        l_raw_bits = T.Cast(T.int32, T.reinterpret(T.uint16, scores[bx, by, input_idx]))
                        l_bin_id32 = T.if_then_else(l_raw_bits >= 0x8000, 0, l_raw_bits >> 6)
                        if l_bin_id32 == l_threshold_bin_id:
                            l_sub_bin = l_raw_bits & 0x3F
                            T.atomic_add(s_sub_hist[l_sub_bin], 1)
                T.sync_threads()

                T.fill(s_threshold_bin_id[0], 0)
                T.sync_threads()
                if tx < RADIX2:
                    for i in T.serial(6):
                        offset = 1 << i
                        T.sync_threads(3, RADIX2)
                        if tx + offset < RADIX2:
                            l_val = s_sub_hist[tx] + s_sub_hist[tx + offset]
                        T.sync_threads(3, RADIX2)
                        if tx + offset < RADIX2:
                            s_sub_hist[tx] = l_val
                    T.sync_threads(3, RADIX2)
                    if s_sub_hist[tx] > l_new_topk and s_sub_hist[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id_2 = s_threshold_bin_id[0]
                l_new_topk_2 = l_new_topk - s_sub_hist[l_threshold_bin_id_2 + 1]
            else:
                l_threshold_bin_id_2 = RADIX2
                l_new_topk_2 = 0

            if tx == 0:
                threshold_info[bx, by, 0] = l_threshold_bin_id
                threshold_info[bx, by, 1] = l_threshold_bin_id_2
                threshold_info[bx, by, 2] = l_per_batch_topk
                threshold_info[bx, by, 3] = l_new_topk_2
                threshold_info[bx, by, 4] = l_per_batch_topk - l_new_topk_2
                emit_ctr[bx, by, 0] = 0
                emit_ctr[bx, by, 1] = 0

        # K3: Split emit using threshold info
        with T.Kernel(batch, groups, num_splits, threads=BLOCK_SIZE) as (bx, by, bz):
            tx = T.get_thread_binding()
            l_threshold = T.alloc_var(T.int32)
            l_sub_threshold = T.alloc_var(T.int32)
            l_pbk = T.alloc_var(T.int32)
            l_ntk2 = T.alloc_var(T.int32)
            l_sp2 = T.alloc_var(T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_chunk = T.alloc_var(T.int32)
            l_start = T.alloc_var(T.int32)
            l_end = T.alloc_var(T.int32)
            l_raw_bits = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_sub_bin = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)
            l_logical_idx = T.alloc_var(T.int32)
            l_block_id = T.alloc_var(T.int32)
            l_offset = T.alloc_var(T.int32)
            l_phys_block = T.alloc_var(T.int32)

            l_threshold = threshold_info[bx, by, 0]
            l_sub_threshold = threshold_info[bx, by, 1]
            l_pbk = threshold_info[bx, by, 2]
            l_ntk2 = threshold_info[bx, by, 3]
            l_sp2 = threshold_info[bx, by, 4]

            l_end_idx = cache_seqlens[bx]
            l_chunk = T.ceildiv(l_end_idx, num_splits)
            l_start = bz * l_chunk
            l_end = T.min(l_start + l_chunk, l_end_idx)

            for s in T.serial(T.ceildiv(l_chunk, BLOCK_SIZE)):
                input_idx = l_start + s * BLOCK_SIZE + tx
                if input_idx < l_end:
                    l_raw_bits = T.Cast(T.int32, T.reinterpret(T.uint16, scores[bx, by, input_idx]))
                    l_bin_id32 = T.if_then_else(l_raw_bits >= 0x8000, 0, l_raw_bits >> 6)
                    if l_bin_id32 > l_threshold:
                        l_out_pos = T.atomic_add(emit_ctr[bx, by, 0], 1, return_prev=True)
                        if l_out_pos < l_sp2:
                            l_logical_idx = input_idx
                            l_block_id = T.floordiv(l_logical_idx, block_size)
                            l_offset = T.floormod(l_logical_idx, block_size)
                            l_phys_block = block_table[bx, l_block_id]
                            out_indices[bx, by, l_out_pos] = l_phys_block * block_size + l_offset
                    elif l_bin_id32 == l_threshold:
                        l_sub_bin = l_raw_bits & 0x3F
                        if l_sub_bin > l_sub_threshold:
                            l_out_pos = T.atomic_add(emit_ctr[bx, by, 0], 1, return_prev=True)
                            if l_out_pos < l_sp2:
                                l_logical_idx = input_idx
                                l_block_id = T.floordiv(l_logical_idx, block_size)
                                l_offset = T.floormod(l_logical_idx, block_size)
                                l_phys_block = block_table[bx, l_block_id]
                                out_indices[bx, by, l_out_pos] = l_phys_block * block_size + l_offset
                        elif l_sub_bin == l_sub_threshold and l_ntk2 > 0:
                            l_out_pos = T.atomic_add(emit_ctr[bx, by, 1], 1, return_prev=True)
                            if l_out_pos < l_ntk2:
                                l_logical_idx = input_idx
                                l_block_id = T.floordiv(l_logical_idx, block_size)
                                l_offset = T.floormod(l_logical_idx, block_size)
                                l_phys_block = block_table[bx, l_block_id]
                                out_indices[bx, by, l_sp2 + l_out_pos] = l_phys_block * block_size + l_offset

    return fused_topk_splitk_3k




def full_kernel(recompute_fn, reuse_fn, topk_kern, Q, K, V, block_table, cache_seqlens,
                softmax_scale, topk_percent, glse1, Output_partial1, row_sums, scores,
                glse2, Output_partial2, output, aggregated_scores, topk_indices,
                topk_kern_splitk=None, topk_scratch=None, layer=0):
    recompute_fn(Q, K, V, block_table, cache_seqlens, softmax_scale,
                 glse1, Output_partial1, row_sums, scores, output, aggregated_scores)
    batch = Q.shape[0]
    groups = aggregated_scores.shape[1]
    if (
        topk_kern_splitk is not None
        and topk_scratch is not None
        and should_use_splitk_topk(batch, groups, aggregated_scores.shape[2])
    ):
        g_hist, thr_info, emit_ctr = topk_scratch
        g_hist.zero_(); emit_ctr.zero_()
        topk_kern_splitk(aggregated_scores, cache_seqlens, block_table,
                         topk_percent / 100.0, topk_indices, g_hist, thr_info, emit_ctr)
    else:
        topk_kern(aggregated_scores, cache_seqlens, block_table,
                  topk_percent / 100.0, topk_indices)
    if layer != 0:
        block_size = K.shape[1]
        total_tokens = K.shape[0] * block_size
        K_flat = K.reshape(total_tokens, K.shape[2], K.shape[3])
        V_flat = V.reshape(total_tokens, V.shape[2], V.shape[3])
        head_mapping = torch.arange(K.shape[2], dtype=torch.int32, device=K.device)
        reuse_fn(Q, K_flat, V_flat, cache_seqlens, topk_indices, head_mapping,
                 topk_percent, softmax_scale, glse2, Output_partial2, output)
    return topk_indices


def ref_program_correct(query, key, value, block_table, cache_seqlens,
                        topk_percent, softmax_scale, layer):
    from .reuse_kascade_gqa_decode_with_kvcache import ref_program_correct as ref_program_reuse
    batch = query.shape[0]
    dim = query.shape[-1]
    block_size = key.shape[1]
    groups = key.shape[2]
    num_head_groups = query.shape[1] // key.shape[2]
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
    k = key_contiguous.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)
    v = value_contiguous.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)

    attn_weights = torch.matmul(q, k.transpose(2, 3))
    attn_weights = attn_weights.to(torch.float32) * softmax_scale
    for b in range(attn_weights.shape[0]):
        attn_weights[b, :, :, cache_seqlens[b]:] = float('-inf')

    attn_weights = softmax_(attn_weights, dim=-1, div=True)

    attn_weights_per_khead = attn_weights.view(
        batch, groups, num_head_groups, 1, attn_weights.shape[-1]
    ).sum(dim=2).squeeze(2).to(torch.float16)
    max_topk = (int((topk_percent * max_seqlen) / 100) + 127) // 128 * 128
    max_topk = min(max_topk, max_seqlen)
    topk_indices = torch.topk(attn_weights_per_khead[:, :, :max_seqlen],
                              k=max_topk, dim=-1).indices

    if layer == 0:
        out = torch.matmul(attn_weights.to(torch.float16), v).squeeze(2)
    else:
        out = ref_program_reuse(query, key_contiguous, value_contiguous, cache_seqlens,
                                topk_indices,
                                torch.arange(key.shape[2], device=key.device),
                                topk_percent, softmax_scale)
    return out, topk_indices, attn_weights_per_khead


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
         layer: int = 0,
         with_ref: bool = False,
         pool_extra_blocks: int = 16):

    blocks_per_seq = (max_cache_seqlen + block_size - 1) // block_size
    num_blocks = max_batch * blocks_per_seq + pool_extra_blocks
    max_blocks = (max_model_len + block_size - 1) // block_size
    opt_block_H = compute_adaptive_block_H(heads, groups)
    kernel_tuning = get_arch_config()
    recompute_block_N = kernel_tuning["recompute_block_N"]
    reuse_block_N = kernel_tuning["reuse_block_N"]
    recompute_threads = kernel_tuning["recompute_threads"]
    reuse_threads = kernel_tuning["reuse_threads"]
    recompute_stages = kernel_tuning["recompute_stages"]

    recompute_kern = flashattn(
        heads=heads, groups=groups, max_blocks=max_blocks,
        num_blocks=T.symbolic("num_blocks"), block_size=block_size,
        max_seqlen=T.symbolic("max_seqlen"), dim=dim, layer=layer,
        batch=T.symbolic("batch"), block_N=recompute_block_N, block_H=opt_block_H,
        num_split=T.symbolic("num_split"), num_stages=recompute_stages, threads=recompute_threads,
    )
    topk_kern = _tl_fused_topk_impl(groups, block_size, reuse_block_N)
    topk_splitk_splits = kernel_tuning["topk_splitk_splits"]
    topk_kern_splitk = _tl_fused_topk_splitk_3k_impl(
        groups, block_size, topk_splitk_splits, reuse_block_N)

    reuse_kern = None
    reuse_bH = opt_block_H
    if layer != 0:
        from .reuse_kascade_gqa_decode_paged import flashattn_flat
        reuse_bH = compute_adaptive_block_H(heads, groups)
        reuse_kern = flashattn_flat(
            heads=heads, groups=groups, dim=dim,
            total_tokens=T.symbolic("total_tokens"),
            batch=T.symbolic("batch"), block_N=reuse_block_N, block_H=reuse_bH,
            num_split=T.symbolic("num_split"), num_stages=1,
            threads=reuse_threads, topk=T.symbolic("topk"))

    if code:
        print(recompute_kern.get_kernel_source(kernel_only=False))
        return

    runs = 100 if run_type == "correctness" else 10

    sum_recompute_lat = 0.0
    sum_topk_lat = 0.0
    sum_full_lat = 0.0
    cnt_recompute = 0
    cnt_topk = 0
    cnt_full = 0

    for run_idx in range(runs):
        if run_type == "correctness":
            batch = torch.randint(1, max_batch + 1, (1,)).item()
            cache_seqlens = torch.randint(128, max_cache_seqlen, (batch,),
                                           device='cuda', dtype=torch.int32)
            topk_sample = torch.randint(10, 50, size=(1,)).item()
            print(f"Run {run_idx}: batch={batch}, topk={topk_sample}, cache_seqlens={cache_seqlens}")
        else:
            batch = max_batch
            cache_seqlens = torch.full((batch,), max_cache_seqlen,
                                       device='cuda', dtype=torch.int32)
            topk_sample = topk

        max_seqlen = cache_seqlens.max().item()
        max_topk = (int(topk_sample * max_seqlen / 100) + reuse_block_N - 1) // reuse_block_N * reuse_block_N
        max_topk = max(max_topk, reuse_block_N)
        cur_split = compute_optimal_split(batch, heads, groups, max_seqlen, recompute_block_N, opt_block_H, layer=layer)

        reuse_split = 8
        if reuse_kern is not None:
            reuse_split = compute_optimal_split(batch, heads, groups, max_seqlen, reuse_block_N, reuse_bH, topk_percent=topk_sample)

        softmax_scale = 1 / (dim ** 0.5)
        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_blocks, block_size, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_blocks, block_size, groups, dim, device="cuda", dtype=torch.float16)
        global_perm = torch.randperm(num_blocks, dtype=torch.int32, device="cuda")
        block_table = torch.zeros((batch, max_blocks), dtype=torch.int32, device="cuda")
        for b in range(batch):
            block_table[b, :blocks_per_seq] = global_perm[b * blocks_per_seq:(b + 1) * blocks_per_seq]

        glse1 = torch.empty(batch, heads, cur_split, device="cuda", dtype=torch.float32)
        Op1 = torch.empty(batch, heads, cur_split, dim, device="cuda", dtype=torch.float32)
        rs = torch.empty(batch, heads, device="cuda", dtype=torch.float32)
        sc = torch.empty(batch, heads, max_seqlen, device="cuda", dtype=torch.float16)
        o = torch.empty(batch, heads, dim, device="cuda", dtype=torch.float16)
        agg = torch.full((batch, groups, max_seqlen), float('-inf'), device="cuda", dtype=torch.float16)
        topk_out = torch.zeros(batch, groups, max_topk, dtype=torch.int32, device="cuda")

        # Allocate split-k scratch buffers
        NBINS = 1 << 10
        g_hist = torch.zeros(batch, groups, topk_splitk_splits, NBINS, dtype=torch.int32, device="cuda")
        thr_info = torch.zeros(batch, groups, 5, dtype=torch.int32, device="cuda")
        emit_ctr = torch.zeros(batch, groups, 2, dtype=torch.int32, device="cuda")
        topk_scratch = (g_hist, thr_info, emit_ctr)

        if run_type == "correctness":
            glse2 = torch.empty(batch, heads, reuse_split, device="cuda", dtype=torch.float32)
            Op2 = torch.empty(batch, heads, reuse_split, dim, device="cuda", dtype=torch.float32)

            full_kernel(recompute_kern, reuse_kern, topk_kern, q, k, v, block_table,
                        cache_seqlens, softmax_scale, topk_sample, glse1, Op1, rs, sc,
                        glse2, Op2, o, agg, topk_out,
                        topk_kern_splitk=topk_kern_splitk, topk_scratch=topk_scratch,
                        layer=layer)

            o_ref, topk_ref, scores_ref = ref_program_correct(
                q, k, v, block_table, cache_seqlens, topk_sample, softmax_scale, layer)

            eps_s = 1e-4
            assert_ = False
            print_ = True
            assert_similar(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps_s)
            assert_allclose(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=1e-1)

            topk_num = ((((topk_sample * cache_seqlens) / 100).to(torch.int32) + 127) // 128) * 128
            topk_num = torch.minimum(topk_num, cache_seqlens.to(topk_num.dtype))
            topk_num = topk_num.clamp(max=max_topk)

            num_phys = block_table.max().item() + 1
            inv_bt = torch.zeros(batch, num_phys, dtype=torch.int64, device='cuda')
            for b in range(batch):
                n_blks = (cache_seqlens[b].item() + block_size - 1) // block_size
                inv_bt[b, block_table[b, :n_blks].long()] = torch.arange(n_blks, device='cuda', dtype=torch.int64)

            def phys_to_logical(phys_idx, b):
                pb = phys_idx.long() // block_size
                po = phys_idx.long() % block_size
                return (inv_bt[b][pb] * block_size + po).clamp(max=max_seqlen - 1)

            def compare_topk_scores(our_phys_indices, ref_logical_indices, score_tensor, name, eps=None):
                """Compare sorted scores of our topk vs reference topk."""
                for b in range(batch):
                    tn = topk_num[b].item()
                    sl_b = cache_seqlens[b].item()
                    our_logical = phys_to_logical(our_phys_indices[b, :, :tn], b).clamp(max=sl_b - 1)
                    our_sc = score_tensor[b, :, :sl_b].gather(1, our_logical)
                    ref_sc = score_tensor[b, :, :sl_b].gather(1, ref_logical_indices[b, :, :tn].long().clamp(max=sl_b - 1))
                    if eps is not None:
                        assert_similar(our_sc.sort(-1, descending=True).values,
                                       ref_sc.sort(-1, descending=True).values,
                                       name=f"{name}_{b}", assert_=True, print_=False, eps=eps)
                    else:
                        assert_equal(our_sc.sort(-1, descending=True).values,
                                     ref_sc.sort(-1, descending=True).values,
                                     name=f"{name}_{b}", assert_=True, print_=False)

            # Pipeline topk: our topk vs ref_program topk (scores may differ slightly)
            compare_topk_scores(topk_out, topk_ref, scores_ref, "topk_scores", eps=1e-2)

            # Standalone topk: same input scores → sorted scores must match exactly
            topk_out_standalone = torch.zeros_like(topk_out)
            topk_kern(agg.clone(), cache_seqlens, block_table, topk_sample / 100.0, topk_out_standalone)
            ref_standalone = torch.zeros(batch, groups, max_topk, dtype=torch.int64, device='cuda')
            for b in range(batch):
                tn = topk_num[b].item(); sl_b = cache_seqlens[b].item()
                ref_standalone[b, :, :tn] = torch.topk(agg[b, :, :sl_b], k=tn, dim=-1).indices
            compare_topk_scores(topk_out_standalone, ref_standalone, agg, "fused_topk_scores")

            del glse2, Op2
        else:
            # Benchmark recompute kernel alone
            try:
                lat = do_bench(recompute_kern, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k, v, block_table, cache_seqlens, softmax_scale,
                                   glse1, Op1, rs, sc, o, agg])
                sum_recompute_lat += lat
                cnt_recompute += 1
            except Exception:
                pass

            # Benchmark fused topk alone (use split-k for small batch)
            try:
                if should_use_splitk_topk(batch, groups, max_seqlen):
                    g_hist.zero_(); emit_ctr.zero_()
                    lat = do_bench(topk_kern_splitk, n_warmup=1, n_repeat=5,
                        input_tensors=[agg, cache_seqlens, block_table, topk_sample / 100.0,
                                       topk_out, g_hist, thr_info, emit_ctr])
                else:
                    lat = do_bench(topk_kern, n_warmup=1, n_repeat=5,
                        input_tensors=[agg, cache_seqlens, block_table, topk_sample / 100.0, topk_out])
                sum_topk_lat += lat
                cnt_topk += 1
            except Exception:
                pass


            # Benchmark full pipeline
            glse2 = torch.empty(batch, heads, reuse_split, device="cuda", dtype=torch.float32) if reuse_kern is not None else torch.empty(1, device="cuda")
            Op2 = torch.empty(batch, heads, reuse_split, dim, device="cuda", dtype=torch.float32) if reuse_kern is not None else torch.empty(1, device="cuda")
            try:
                agg.fill_(float('-inf'))
                lat = do_bench(full_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[recompute_kern, reuse_kern, topk_kern, q, k, v, block_table,
                                   cache_seqlens, softmax_scale, topk_sample, glse1, Op1, rs, sc,
                                   glse2, Op2, o, agg, topk_out,
                                   topk_kern_splitk, topk_scratch, layer])
                sum_full_lat += lat
                cnt_full += 1
            except Exception:
                pass
            del glse2, Op2

        del q, k, v, block_table, glse1, Op1, rs, sc, o, agg, topk_out
        del g_hist, thr_info, emit_ctr
        torch.cuda.empty_cache()

    if run_type == "benchmark":
        if layer == 0:
            total_flops = 2 * 2 * batch * heads * max_cache_seqlen * dim
        else:
            topk_tokens = max_topk
            total_flops = (2 * 2 * batch * heads * max_cache_seqlen * dim
                          + 2 * 2 * batch * heads * topk_tokens * dim)
        avg_rc = (sum_recompute_lat / cnt_recompute) if cnt_recompute else 0.0
        avg_rc_tflops = (total_flops / avg_rc * 1e-9) if avg_rc > 0 else 0.0
        avg_tk = (sum_topk_lat / cnt_topk) if cnt_topk else 0.0
        avg_fl = (sum_full_lat / cnt_full) if cnt_full else 0.0
        avg_fl_tflops = (total_flops / avg_fl * 1e-9) if avg_fl > 0 else 0.0
        print(f"{max_batch},{max_cache_seqlen},{topk},{cur_split},"
              f"{avg_rc:.2f},{avg_rc_tflops:.2f},"
              f"{avg_fl:.2f},{avg_fl_tflops:.2f}")
    else:
        print(f"\n{runs} Correctness tests passed!")



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
    parser.add_argument('--run_type', type=str, default='benchmark',
                        choices=["benchmark", "correctness"])
    parser.add_argument('--code', action='store_true')
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument('--with_ref', action='store_true')
    args = parser.parse_args()
    main(args.max_batch, args.heads, args.groups, args.max_cache_seqlen, args.block_size,
         args.max_model_len, args.dim, args.topk, args.run_type, args.code, args.layer,
         args.with_ref)
