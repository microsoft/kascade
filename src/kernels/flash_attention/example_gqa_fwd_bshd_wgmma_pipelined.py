# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

"""
This logic is largely taken from the examples in tile-ai/tilelang and modified
"""

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial
from ..kernel_utils import *


def get_configs():
    block_M = [64]
    block_N = [64,128,256]
    num_stages = [1,2,3]
    threads = [64,128,256]
    _configs = list(itertools.product(block_M, block_N, num_stages, threads))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'num_stages': c[2],
        'threads': c[3]
    } for c in _configs]
    return configs


def flashattn(batch, heads, seq_len, dim, is_causal, tune=False, groups=1):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    kv_group_num = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, groups, dim]
    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_M, block_N, num_stages, threads):

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
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    is_valid_causal = bx * block_M + i >= k * block_N + j
                    # is_within_bounds = (k * block_N + j < seq_len)
                    acc_s[i, j] = T.if_then_else(is_valid_causal, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
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
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
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
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            # T.copy(acc_s, acc_s_cast)
            for i, j in T.Parallel(block_M, block_N):
                acc_s_cast[i, j] = acc_s[i, j]

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                V: T.Tensor(kv_shape, dtype),
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
                logsum = T.alloc_fragment([block_M], accum_dtype)
                bx = T.ceildiv(seq_len, block_M)  - 1 - bx_r

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages):
                        # order=[-1, 0, 3, 1, -1, 2],
                        # stage=[-1, 0, 0, 1, -1, 1],
                        # group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main

    if tune:

        @autotune(
            configs=get_configs(),
            # keys=["block_M", "block_N", "num_stages", "threads"],
            warmup=10,
            rep=10)
        @tilelang.jit(out_idx=[3])
        def kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return kernel_func(block_M, block_N, num_stages, threads)

        return kernel()
    else:
        def kernel(block_M, block_N, num_stages, threads):
            return kernel_func(block_M, block_N, num_stages, threads)

        return kernel


def ref_program(Q, K, V, is_causal):
    from flash_attn_interface import flash_attn_func
    # print(Q.size(), K.size(), V.size())
    output = flash_attn_func(Q, K, V, causal=is_causal)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--seq_len', type=int, default=8192, help='sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--run_type', choices=["correctness", "benchmark"], default="benchmark", help='run type')
    parser.add_argument('--code', action='store_true', help='print code')
    
    args = parser.parse_args()
    batch, heads, seq_len, dim, is_causal, groups = args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups

    if (not args.tune):
        program = flashattn(
            T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, is_causal, tune=args.tune, groups=groups)(
                block_M=128, block_N=128, num_stages=2, threads=256)
        kernel = tilelang.compile(program, out_idx=[3], pass_configs={ "tl.disable_safe_memory_legalize":True , "tl.disable_warp_specialized": False})

        if args.code:
            print(kernel.get_kernel_source())
            exit(0)

        if args.run_type == "correctness":
            runs = 100
        else: 
            runs = 10
        
        sum_ref_f_latency = 0.0
        sum_tile_latency = 0.0

        cnt_ref_f = 0
        cnt_tile = 0

        for _ in range(runs):
            if args.run_type == "correctness":
                seq_len_sample = torch.randint(64, seq_len + 1, (1,)).item()
                print(f"seq_len: {seq_len_sample}")
            else:
                seq_len_sample = seq_len

            Q = torch.randn(batch, seq_len_sample, heads, dim, device='cuda', dtype=torch.float16)
            K = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)
            V = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)

            if args.run_type == "correctness":
                o = kernel(Q, K, V)
                o_ref = ref_program(Q, K, V, is_causal)
                eps_c = 1e-3
                eps_s = 1e-5
                assert_ = False
                print_ = True
                assert_similar(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps_s)
                assert_allclose(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps_c)
            else: 
                flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
                total_flops = 2 * flops_per_matmul
                if is_causal:
                    total_flops *= 0.5
                
                try:
                    ref_f_latency = do_bench(
                    ref_program, n_warmup=1, n_repeat=5,
                    input_tensors=[Q, K, V, is_causal]
                    )
                    ref_f_tflops = total_flops / ref_f_latency * 1e-9
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

                try:
                    tile_latency = do_bench(
                    kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[Q, K, V]
                    )
                    tile_tflops = total_flops / tile_latency * 1e-9
                    sum_tile_latency += tile_latency
                    cnt_tile += 1
                except Exception:
                    pass

                del Q, K, V
                torch.cuda.empty_cache()
            
        if args.run_type == "benchmark":
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            avg_ref_f_tflops = (flops_per_matmul / avg_ref_f_latency * 1e-9) if avg_ref_f_latency > 0 else 0.0
            avg_tile_latency = (sum_tile_latency / cnt_tile) if cnt_tile else 0.0
            avg_tile_tflops = (flops_per_matmul / avg_tile_latency * 1e-9) if avg_tile_latency > 0 else 0.0

            print(f"{seq_len},"
                f"{avg_ref_f_latency:.2f},{avg_ref_f_tflops:.2f},"
                f"{avg_tile_latency:.2f},{avg_tile_tflops:.2f}")
        else:
            print(f"{runs} Correctness test passed!")
    else:
        best_result = flashattn(batch, heads, seq_len, dim, is_causal, tune=args.tune)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
        total_flops = 2 * flops_per_matmul
        if is_causal:
            total_flops *= 0.5
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
