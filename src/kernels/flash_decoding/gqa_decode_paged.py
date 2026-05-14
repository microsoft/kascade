# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import argparse
from ..kernel_utils import *
from .recompute_kascade_gqa_decode_paged import compute_adaptive_block_H, compute_optimal_split, _get_device_capability

torch.random.manual_seed(0)


def get_pass_config():
    return { "tl.disable_safe_memory_legalize": True, "tl.disable_tma_lower": True}

@tilelang.jit(out_idx=[], execution_backend="tvm_ffi", pass_configs=get_pass_config())
def flashattn(heads, groups, max_blocks, num_blocks, block_size, dim, batch, block_N, block_H, num_split, num_stages, threads, dtype='float16'):
    log_scale = 1.44269504  # log2(e)
    accum_dtype = "float"
    kv_group_num = heads // groups
    shape_q = [batch, heads, dim]
    shape_k = [num_blocks, block_size, groups, dim]
    shape_v = [num_blocks, block_size, groups, dim]
    shape_o = [batch, heads, dim]
    shape_bt = [batch, max_blocks]
    part_shape = [batch, heads, num_split, dim]
    valid_block_H = min(block_H, kv_group_num)
    assert block_size <= block_N, "block_size must be less than or equal to block_N"

    @T.macro
    def flash_attn_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_v, dtype),
            block_table: T.Tensor(shape_bt, dtype="int32"),
            cache_seqlen: T.Tensor([batch], dtype="int32"),
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
            O_shared = T.alloc_shared([valid_block_H, dim], accum_dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            final_scale = T.alloc_var(accum_dtype)
            cur_kv_head = T.alloc_var("int32")
            pages_per_blockN = T.alloc_var("int32")
            # block_table_offset = T.alloc_var("int32")

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)
            final_scale = log_scale * softmax_scale
            pages_per_blockN = T.ceildiv(block_N, block_size)

            T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            total_chunks = T.ceildiv(cache_seqlen[bid], block_N)
            base_chunks_per_split = T.floordiv(total_chunks, num_split)
            remainder_chunks = T.floormod(total_chunks, num_split)
            final_chunks = base_chunks_per_split + T.if_then_else(sid < remainder_chunks, 1, 0)
            prev_split_chunks = base_chunks_per_split * sid + T.min(sid, remainder_chunks)
            start_idx = prev_split_chunks * block_N
            for k in T.Pipelined(final_chunks, num_stages=num_stages):
                block_table_offset = (prev_split_chunks + k) * pages_per_blockN
                for i,j in T.Parallel(block_N, dim):
                    page_idx = T.floordiv(i, block_size)
                    token_idx = start_idx + k * block_N + i
                    K_shared[i,j] = T.if_then_else(token_idx < cache_seqlen[bid], K[block_table[bid, block_table_offset + page_idx], T.floormod(i, block_size), cur_kv_head, j], 0)
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(start_idx + k * block_N + j < cache_seqlen[bid], acc_s[i, j],
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
                for i,j in T.Parallel(block_N, dim):
                    page_idx = T.floordiv(i, block_size)
                    token_idx = start_idx + k * block_N + i
                    V_shared[i,j] = T.if_then_else(token_idx < cache_seqlen[bid], V[block_table[bid, block_table_offset + page_idx], T.floormod(i, block_size), cur_kv_head, j], 0)
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
        with T.Kernel(heads, batch, threads=256) as (by, bz):
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
            cache_seqlen: T.Tensor([batch], dtype="int32"),
            softmax_scale: T.float32,
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            Output: T.Tensor(shape_o, dtype),
    ):
        flash_attn_split(Q, K, V, block_table, cache_seqlen, softmax_scale, glse, Output_partial, Output)
        if num_split > 1:
            combine(glse, Output_partial, Output)

    return flashattn_gqa_decode_split

class FlashAttnGQADecodeKVCache(torch.nn.Module):
    def __init__(self, heads: int, groups: int, dim: int, max_blocks: int, num_blocks: int, block_size: int):
        super().__init__()
        self.heads = heads
        self.groups = groups
        self.dim = dim
        self.max_blocks = max_blocks
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.block_N = 192
        self.block_H = compute_adaptive_block_H(heads, groups)
        self.kernel = flashattn(
            heads=heads,
            groups=groups,
            dim=dim,
            max_blocks=max_blocks,
            num_blocks=T.symbolic("num_blocks"),       
            block_size=block_size,
            batch=T.symbolic("batch"),
            block_N=self.block_N,
            block_H=self.block_H,
            num_split=T.symbolic("num_split"),
            num_stages=1,
            threads=128)

        # print(self.kernel.get_kernel_source(kernel_only=False))

    def forward(self, query, key, value, block_table, cache_seqlen, softmax_scale):
        batch_size = query.shape[0]
        max_seqlen = cache_seqlen.max().item()
        num_split = compute_optimal_split(batch_size, self.heads, self.groups, max_seqlen, self.block_N, self.block_H)
        glse = torch.empty(batch_size, self.heads, num_split, device=query.device, dtype=torch.float32)
        output_partial = torch.empty(batch_size, self.heads, num_split, self.dim, device=query.device, dtype=torch.float32)
        output = torch.empty(batch_size, self.heads, self.dim, device=query.device, dtype=query.dtype)
        self.kernel(query, key, value, block_table, cache_seqlen, softmax_scale, glse, output_partial, output)
        return output

def ref_program_fa(query, key, value, block_table, cache_seqlens, softmax_scale, max_seqlen_k, cu_seqlens_q):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    is_blackwell = (_get_device_capability()[0] >= 10)
    is_hopper = (_get_device_capability()[0] >= 9)
    fa_version = 2
    if is_blackwell:
        fa_version = 4
    elif is_hopper:
        fa_version = 3
    out = flash_attn_varlen_func(query, key, value, 1, cu_seqlens_q, max_seqlen_k, seqused_k=cache_seqlens, causal=False, softmax_scale=softmax_scale, block_table=block_table, fa_version=fa_version)
    return out


def main(max_batch: int = 1,
         heads: int = 32,
         groups: int = 8,
         max_cache_seqlen: int = 8192,
         block_size: int = 32,
         max_model_len: int = 131072,
         dim: int = 128,
         run_type: str = "benchmark",  # "benchmark", "correctness"
         pool_extra_blocks: int = 16,  # extra unused blocks in KV cache pool
         ):
    qk_flops = 2 * max_batch * heads * max_cache_seqlen * dim
    pv_flops = 2 * max_batch * heads * max_cache_seqlen * dim
    total_flops = qk_flops + pv_flops
    blocks_per_seq = (max_cache_seqlen + block_size - 1) // block_size
    num_blocks = max_batch * blocks_per_seq + pool_extra_blocks
    max_blocks = (max_model_len + block_size-1) // block_size
    kernel = FlashAttnGQADecodeKVCache(heads, groups, dim, max_blocks, num_blocks, block_size)

    runs = 100 if run_type == "correctness" else 10

    lat_f_sum = lat_og_sum = 0.0
    cnt_f = cnt_og = 0

    for _ in range(runs):
        if run_type == "correctness":
            batch = torch.randint(1, max_batch + 1, (1,)).item()
            cache_seqlens = torch.randint(80, max_cache_seqlen, (batch,), device='cuda', dtype=torch.int32)
        else:
            batch = max_batch
            cache_seqlens = torch.full((batch,), max_cache_seqlen, device='cuda', dtype=torch.int32)
        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(num_blocks, block_size, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(num_blocks, block_size, groups, dim, device="cuda", dtype=torch.float16)
        # Create block_table with valid block IDs in [0, num_blocks), randomly scattered
        # Global permutation simulates a fragmented block allocator (no spatial locality bias)
        global_perm = torch.randperm(num_blocks, dtype=torch.int32, device="cuda")
        block_table = torch.zeros((batch, max_blocks), dtype=torch.int32, device="cuda")
        for b in range(batch):
            block_table[b, :blocks_per_seq] = global_perm[b * blocks_per_seq : (b + 1) * blocks_per_seq]
        max_seqlen_k = cache_seqlens.max().item()
        cu_seqlens_q = torch.arange(batch + 1, device='cuda', dtype=torch.int32)
        

        if run_type == "correctness":
            o_ref = ref_program_fa(q, k, v, block_table, cache_seqlens, (1.0 / dim)**0.5, max_seqlen_k, cu_seqlens_q)
            o_og = kernel(q, k, v, block_table, cache_seqlens, (1.0 / dim)**0.5)
            assert_similar(o_ref, o_og, name="ref_og", assert_=True, print_=True)

        if run_type == "benchmark":
            num_split = compute_optimal_split(batch, heads, groups, max_cache_seqlen, kernel.block_N, kernel.block_H)
            glse = torch.empty(batch, heads, num_split, device='cuda', dtype=torch.float32)
            output_partial = torch.empty(batch, heads, num_split, dim, device='cuda', dtype=torch.float32)
            output = torch.empty(batch, heads, dim, device='cuda', dtype=torch.float16)
            try:
                latency_f = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v, block_table, cache_seqlens, (1.0 / dim)**0.5, max_seqlen_k, cu_seqlens_q])
                lat_f_sum += latency_f
                cnt_f += 1
            except Exception:
                pass
            try:
                latency_og = do_bench(kernel.kernel, n_warmup=1, n_repeat=5, input_tensors=[q, k, v, block_table, cache_seqlens, (1.0 / dim)**0.5, glse, output_partial, output])
                lat_og_sum += latency_og
                cnt_og += 1
            except Exception:
                pass
        del q, k, v, cache_seqlens
        torch.cuda.empty_cache()

    if run_type == "benchmark":
        avg_f = lat_f_sum / cnt_f if cnt_f > 0 else 0.0
        avg_og = lat_og_sum / cnt_og if cnt_og > 0 else 0.0

        results = [f"{max_batch},{max_cache_seqlen}"]
        tflops_f = total_flops / avg_f * 1e-9 if avg_f > 0 else 0
        tflops_og = total_flops / avg_og * 1e-9 if avg_og > 0 else 0
        results.extend([f"{avg_f:.2f}", f"{tflops_f:.2f}"])
        results.extend([f"{avg_og:.2f}", f"{tflops_og:.2f}"])
        print(",".join(results))
    else:
        print(f"\n{runs} Correctness tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_batch', type=int, default=4, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--max_cache_seqlen', type=int, default=8192, help='max cache sequence length')
    parser.add_argument('--block_size', type=int, default=32, help='block size <= 128')
    parser.add_argument('--max_model_len', type=int, default=131072, help='max model length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--run_type', type=str, default="benchmark", choices=["benchmark", "correctness"], help='Type of run: benchmark, correctness')
    args = parser.parse_args()
    main(args.max_batch, args.heads, args.groups, args.max_cache_seqlen, args.block_size, args.max_model_len, args.dim, args.run_type)
