# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from tilelang.engine.param import KernelParam 
import argparse
import itertools
from tvm import tir, ir
from ..kernel_utils import *

torch.random.manual_seed(0)


def get_configs():
    block_N = [64, 128]
    block_H = [64]
    num_split = [2, 4, 8]
    num_stages = [1, 2, 3]
    threads = [128]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

    configs = [{
        'block_N': c[0],
        'block_H': c[1],
        'num_split': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def flashattn(heads, groups, dim):
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
        def flash_attn(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlen: T.Tensor([batch], dtype="int32"),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
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
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((cache_seqlen[bid] // num_split), block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    # for i, j in T.Parallel(block_H, block_N):
                    #     acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                    #                                  -T.infinity(accum_dtype))
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
                    T.copy(V[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])

        @T.macro
        def flash_attn_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlen: T.Tensor([batch], dtype="int32"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
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

                total_chunks = T.ceildiv(cache_seqlen[bid], block_N)
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
                        acc_s[i, j] = T.if_then_else(start_idx + k * block_N + j < cache_seqlen[bid], acc_s[i, j],
                                                     -T.infinity(accum_dtype))
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
        def combine(
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split, 128], dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    # lse_local: (local_id, thread_id)
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
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split[0] = glse[bz, by, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def flashattn_gqa_decode_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlen: T.Tensor([batch], dtype="int32"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, cache_seqlen, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def flashattn_gqa_decode_no_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlen: T.Tensor([batch], dtype="int32"),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn(Q, K, V, cache_seqlen, Output)

        if num_split > 1:
            return flashattn_gqa_decode_split
        else:
            return flashattn_gqa_decode_no_split

    def kernel(batch, block_N, block_H, num_split, num_stages, threads, max_cache_seqlen):
        return kernel_func(batch, block_N, block_H, num_split, num_stages, threads, max_cache_seqlen)

    return kernel

class FlashAttnGQADecodeKVCache(torch.nn.Module):
    def __init__(self, heads: int, groups: int, dim: int):
        super().__init__()
        self.heads = heads
        self.groups = groups
        self.dim = dim
        self.block_N = 128
        self.block_H = 64
        self.num_split = 8
        program = flashattn(
            heads=heads,
            groups=groups,
            dim=dim)(
                batch=T.symbolic("batch"),
                block_N=self.block_N,
                block_H=self.block_H,
                num_split=self.num_split,
                num_stages=2,
                threads=128,
                max_cache_seqlen=T.symbolic("max_cache_seqlen"))

        self.kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")

    def forward(self, query, key, value, cache_seqlen):
        batch_size = query.size(0)
        glse = torch.empty(batch_size, self.heads, 8, device=query.device, dtype=query.dtype)
        output_partial = torch.empty(batch_size, self.heads, 8, self.dim, device=query.device, dtype=query.dtype)
        output = self.kernel(query, key, value, cache_seqlen, glse, output_partial)
        return output

def ref_program_fa(query, key, value, cache_seqlens):
    from flash_attn_interface import flash_attn_with_kvcache
    Q = query.unsqueeze(1)  # [batch_size, 1, heads, dim]
    out = flash_attn_with_kvcache(Q, key, value, cache_seqlens=cache_seqlens, causal=False)
    out = out.squeeze(1)  # [batch_size, heads, dim]
    return out

def ref_program_correct(query, key, value, cache_seqlens):
    """
    Inputs:
    - query (Tensor): [batch, heads, dim]
    - key (Tensor): [batch, seqlen_kv, groups, dim]
    - value (Tensor): [batch, seqlen_kv, groups, dim]
    Outputs:
    - output (Tensor): [batch, heads, dim]
    """
    dim = query.shape[-1]
    num_head_groups = query.shape[1] // key.shape[2]
    scale = dim**0.5
    q = query.unsqueeze(2)  # [batch_size, heads, 1, dim]
    k = key.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, seqlen_kv, dim]
    v = value.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, seqlen_kv, dim]

    attn_weights = torch.matmul(q, k.transpose(2, 3)) / scale  # [batch_size, heads, 1, seqlen_kv]
    for b in range(attn_weights.shape[0]):
        attn_weights[b, :, :, cache_seqlens[b]:] = float('-inf')
    attn_weights = torch.softmax(attn_weights, dim=-1)  # [batch_size, heads, 1, seqlen_kv]
    out = torch.matmul(attn_weights, v)  # [batch_size, heads, 1, dim]
    out = out.squeeze(2)  # [batch_size, heads, dim]
    return out

def main(batch: int = 1,
         heads: int = 32,
         groups: int = 8,
         max_cache_seqlen: int = 8192,
         dim: int = 128,
         run_type: str = "benchmark"  # "benchmark", "correctness"
         ):
    qk_flops = 2 * batch * heads * max_cache_seqlen * dim
    pv_flops = 2 * batch * heads * max_cache_seqlen * dim
    total_flops = qk_flops + pv_flops
    kernel = FlashAttnGQADecodeKVCache(heads, groups, dim)
        
    # cuda_source = kernel.get_kernel_source()  
    # print(cuda_source)

    if (run_type == "correctness"):
        num_runs = 100
    else:
        num_runs = 10

    lat_f_sum = lat_og_sum = 0.0
    cnt_f = cnt_og = 0

    for _ in range(num_runs):
        if run_type == "correctness":
            cache_seqlens = torch.randint(128, max_cache_seqlen, (batch,), device='cuda', dtype=torch.int32)
        else:
            cache_seqlens = torch.full((batch,), max_cache_seqlen, device='cuda', dtype=torch.int32)
        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, max_cache_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, max_cache_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        

        if run_type == "correctness":
            o_ref = ref_program_correct(q, k, v, cache_seqlens)
            o_og = kernel(q, k, v, cache_seqlens)
            assert_similar(o_ref, o_og, name="ref_og", assert_=True, print_=False)

        if run_type == "benchmark":
            # try:
            # latency_f = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v, cache_seqlens])
            # lat_f_sum += latency_f
            # cnt_f += 1
            # except Exception:
            #     pass
            # try:
            latency_og = do_bench(kernel, n_warmup=1, n_repeat=5, input_tensors=[q, k, v, cache_seqlens])
            lat_og_sum += latency_og
            cnt_og += 1
            # except Exception:
            #     pass
        del q, k, v, cache_seqlens
        torch.cuda.empty_cache()

    if run_type == "benchmark":
        avg_f = lat_f_sum / cnt_f if cnt_f > 0 else 0.0
        avg_og = lat_og_sum / cnt_og if cnt_og > 0 else 0.0

        results = [f"{max_cache_seqlen}"]
        tflops_f = total_flops / avg_f * 1e-9 if avg_f > 0 else 0
        tflops_og = total_flops / avg_og * 1e-9 if avg_og > 0 else 0
        results.extend([f"{avg_f:.2f}", f"{tflops_f:.2f}"])
        results.extend([f"{avg_og:.2f}", f"{tflops_og:.2f}"])
        print(",".join(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--max_cache_seqlen', type=int, default=8192, help='max cache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--run_type', type=str, default="benchmark", choices=["benchmark", "correctness"], help='Type of run: benchmark, correctness')
    args = parser.parse_args()
    main(args.batch, args.heads, args.groups, args.max_cache_seqlen, args.dim, args.run_type)
