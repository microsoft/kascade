# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial
from ..kernel_utils import *
import tilelang.language.tir.op as tir_op
import math

def get_configs():
    block_M = [16]
    block_H = [4]
    block_N = [64,128,256]
    num_stages = [1,2,3]
    threads = [64,128,256]
    _configs = list(itertools.product(block_M, block_H, block_N, num_stages, threads))

    configs = [{
        'block_M': c[0],
        'block_H': c[1],
        'block_N': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def flashattn(batch, heads, seq_len, dim, max_topk_num, rolling=False, tune=False, groups=1):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, groups, dim]
    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_M, block_H, block_N, num_stages, threads):
        shape_topk = [batch, groups, T.ceildiv(seq_len, block_M), max_topk_num]

        @T.macro
        def MMA0(
            K: T.Tensor(kv_shape, dtype),
            topk_indices: T.Tensor(shape_topk, "int32"),
            Q_shared: T.SharedBuffer([block_M * block_H, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M * block_H, block_N], accum_dtype),
            head_to_reuse: T.int32,
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
            topk_num: T.int32,
        ):
            if bx * block_M <= topk_num:
                T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
            elif (k + 1) * block_N <= topk_num:
                for i, j in T.Parallel(block_N, dim):
                    idx = topk_indices[bz, head_to_reuse, bx, k * block_N + i]
                    with T.attr(T.int32(0), "async_scope", 1): 
                        K_shared[i, j] = K[bz, idx, by, j]
            else:
                T.copy(K[bz, bx * block_M: bx * block_M + block_N, by, :], K_shared)

            if bx * block_M <= topk_num:
                for i, h, j in T.Parallel(block_M, block_H, block_N):
                    is_valid_causal = (bx * block_M + i >= k * block_N + j)
                    acc_s[i * block_H + h, j] = T.if_then_else(is_valid_causal, 0, -T.infinity(acc_s.dtype))
            elif (k + 1) * block_N <= topk_num:
                T.fill(acc_s, 0)
            else:
                for i, h, j in T.Parallel(block_M, block_H, block_N):
                    is_valid_causal = (i >= j)
                    acc_s[i * block_H + h, j] = T.if_then_else(is_valid_causal, 0, -T.infinity(acc_s.dtype))

            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(kv_shape, dtype),
            topk_indices: T.Tensor(shape_topk, "int32"),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M * block_H, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M * block_H, dim], accum_dtype),
            head_to_reuse: T.int32,
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
            topk_num: T.int32,
        ):
            if bx * block_M <= topk_num:
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            elif (k + 1) * block_N <= topk_num:
                for i, j in T.Parallel(block_N, dim):
                    idx = topk_indices[bz, head_to_reuse, bx, k * block_N + i]
                    with T.attr(T.int32(0), "async_scope", 1): 
                        V_shared[i, j] = V[bz, idx, by, j]
            else:
                T.copy(V[bz, bx * block_M: bx * block_M + block_N, by, :], V_shared)

            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M * block_H, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M * block_H, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M * block_H], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M * block_H], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M * block_H], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M * block_H], accum_dtype),
                logsum: T.FragmentBuffer([block_M * block_H], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i, j in T.Parallel(block_M, block_H):
            #     scores_max[i * block_H + j] = T.if_then_else(scores_max[i * block_H + j] == -T.infinity(accum_dtype), 0, scores_max[i * block_H + j])
            for i, j in T.Parallel(block_M, block_H):
                scores_scale[i * block_H + j] = T.exp2(scores_max_prev[i * block_H + j] * scale - scores_max[i * block_H + j] * scale)
            for i, h, j in T.Parallel(block_M, block_H, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i * block_H + h, j] = T.exp2(acc_s[i * block_H + h, j] * scale - scores_max[i * block_H + h] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i, j in T.Parallel(block_M, block_H):
                logsum[i * block_H + j] = logsum[i * block_H + j] * scores_scale[i * block_H + j] + scores_sum[i * block_H + j]
            for i, h, j in T.Parallel(block_M, block_H, block_N):
                acc_s_cast[i * block_H + h, j] = acc_s[i * block_H + h, j]

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M * block_H, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M * block_H], accum_dtype),
        ):
            for i, h, j in T.Parallel(block_M, block_H, dim):
                acc_o[i * block_H + h, j] *= scores_scale[i * block_H + h]

        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                V: T.Tensor(kv_shape, dtype),
                topk_indices: T.Tensor(shape_topk, "int32"),
                head_mapping: T.Tensor([groups], "int32"),
                topk_percent: T.float32,
                Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), groups, batch, threads=threads) as (bx_r, by, bz):
                Q_shared = T.alloc_shared([block_M * block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M * block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_M * block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M * block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_M * block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M * block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M * block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_M * block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_M * block_H], accum_dtype)
                logsum = T.alloc_fragment([block_M * block_H], accum_dtype)
                head_to_reuse = head_mapping[by]
                bx = T.ceildiv(seq_len, block_M) - 1 - bx_r
                topk_num = T.if_then_else(rolling and seq_len > block_N, T.ceildiv(T.max(T.int32((topk_percent / 100) * (bx * block_M)), block_N), block_N)*128, max_topk_num)
                
                for i, h, j in T.Parallel(block_M, block_H, dim):
                    is_within_bounds = (bx * block_M + i < seq_len)
                    Q_shared[i * block_H + h, j] = T.if_then_else(is_within_bounds,
                                            Q[bz, bx * block_M + i, by * block_H + h, j],
                                            0)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                loop_range = T.if_then_else(bx * block_M <= topk_num, T.ceildiv((bx + 1) * block_M, block_N), T.ceildiv(topk_num, block_N)+1)

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages):
                    MMA0(K, topk_indices, Q_shared, K_shared, acc_s, head_to_reuse, k, bx, by, bz, topk_num)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, topk_indices, V_shared, acc_s_cast, acc_o, head_to_reuse, k, bx, by, bz, topk_num)
                for i, h, j in T.Parallel(block_M, block_H, dim):
                    acc_o[i * block_H + h, j] /= logsum[i * block_H + h]
                T.copy(acc_o, O_shared)
                for i, h, j in T.Parallel(block_M, block_H, dim):
                    is_within_bounds = (bx * block_M + i < seq_len)
                    if is_within_bounds:
                        Output[bz, bx * block_M + i, by * block_H + h, j] = O_shared[i * block_H + h, j]

        return main

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


def ref_program(Q, K, V):
    from flash_attn_interface import flash_attn_func
    output = flash_attn_func(Q, K, V, causal=True)
    return output


def ref_program_correct(query, key, value, topk_indices, head_mapping, tile_size, rolling, topk_percent):
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
    num_head_groups = query.shape[2] // key.shape[2]
    scale = dim**0.5
    min_dtype = torch.finfo(query.dtype).min

    query = query.transpose(1, 2)  # [batch_size, heads, seqlen, dim]

    #rolling mask
    if rolling and query.shape[2] > 128:
        num_tiles = math.ceil(query.shape[2] / tile_size)
        tile_indices = torch.arange(0, num_tiles*tile_size, tile_size, device=query.device)
        k_indices = torch.arange(0, topk_indices.shape[-1], device=query.device)
        topk_nums = ((topk_percent*tile_indices)//100).clamp(min=128)
        topk_mask = (k_indices >= topk_nums.unsqueeze(1)).to(query.device) # [num_tiles, topk]
        indices = topk_indices.masked_fill(topk_mask, key.shape[1] - 1)
        del tile_indices, k_indices, topk_nums, topk_mask
        torch.cuda.empty_cache()
    else:
        indices = topk_indices

    #select after attention
    indices = indices[:, head_mapping, :, :]  # [batch, groups, num_tiles, topk]
    indices = indices.repeat_interleave(num_head_groups, dim=1).to(torch.int64)  # [batch, heads, num_tiles, topk]
    indices = indices.repeat_interleave(tile_size, dim=2)[:, :, :query.shape[2], :]  # [batch, heads, seqlen, topk]
    key = key.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, seqlen, dim]
    value = value.transpose(1, 2).repeat_interleave(num_head_groups, dim=1)  # [batch_size, heads, seqlen, dim]

    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scale  # [batch_size, heads, seqlen, seqlen]
    a_mask: torch.Tensor = torch.full((B, 1, math.ceil(query.shape[2] / tile_size)*tile_size, query.shape[2]), fill_value=min_dtype, device="cuda") # [B, 1, seqlen+padding, seqlen]
    a_mask.triu_(1) 
    mask = a_mask.view(B, 1, math.ceil(query.shape[2] / tile_size), tile_size, query.shape[2])[:,:,:,-1,:] # [B, 1, num_tiles, seqlen]
    mask = mask.roll(shifts=1, dims=2)
    mask[:, :, 0, :] = torch.finfo(query.dtype).min

    # select after attention
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
    parser.add_argument('--rolling', action='store_true', help='use rolling topk')
    parser.add_argument("--with_ref", action="store_true", help="use reference flash attention for benchmark")
    
    args = parser.parse_args()
    batch, heads, seq_len, dim, groups, topk, rolling = args.batch, args.heads, args.seq_len, args.dim, args.groups, args.topk, args.rolling
    tile_size = 32

    if (not args.tune):
        program = flashattn(
            T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, T.symbolic("max_topk_num"), rolling=rolling, tune=args.tune, groups=groups)(
                block_M=tile_size, block_H=(heads // groups), block_N=128, num_stages=2, threads=256)
        kernel = tilelang.compile(program, out_idx=[6], pass_configs={ "tl.disable_safe_memory_legalize": True})

        modified_kernel = compile_custom_kernel_from_cu(  
            func=program,  
            cu_file_path=f"./kascade_prefill_cu_kernels/dynamic_alllen_reuse_cp_async_ts{tile_size}_dim128_gsize4_heads32_groups8{'_rolling' if rolling else ''}.cu",
            so_dir="./kascade_prefill_cu_kernels/__socache__/",
            out_idx=[6],  
            execution_backend="cython",
            pass_configs={ "tl.disable_safe_memory_legalize": True},  
        )  

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
                seq_len_sample = torch.randint(128, seq_len + 1, (1,)).item()
                topk_sample = torch.randint(10, 50, size=(1,)).item()
                print(f"seq_len: {seq_len_sample}, topk: {topk_sample}")
            else:
                seq_len_sample = seq_len
                topk_sample = topk
            actual_topk_num = min(max(128, int((topk_sample/100)*seq_len_sample)), seq_len_sample)
            rounded_topk_num = min(((actual_topk_num + 127) // 128) * 128, seq_len_sample)  # round to multiple of 128
            head_mapping = torch.randint(0, groups, (groups,), device="cuda", dtype=torch.int32)
            random_scores = torch.rand(batch, groups, math.ceil(seq_len_sample / tile_size), seq_len_sample, device="cuda", dtype=torch.float16)
            mask = make_tile_causal_mask(batch, seq_len_sample, tile_size, dtype=random_scores.dtype, device="cuda")
            random_scores = random_scores.add_(mask)
            del mask
            torch.cuda.empty_cache()
            if seq_len_sample <= 131072:
                topk_indices = torch.topk(random_scores, k=rounded_topk_num, dim=-1).indices.to(torch.int32)
            else:
                topk_indices = torch.empty((batch, groups, math.ceil(seq_len_sample / tile_size), rounded_topk_num), device="cuda", dtype=torch.int32)
                chunk_size = 131072 // 2**int(topk_sample // 10)
                num_chunks = math.ceil(seq_len_sample / chunk_size)
                for i in range(num_chunks):
                    start = (i * chunk_size) // tile_size
                    end = min(((i + 1) * chunk_size ) // tile_size, seq_len_sample)
                    topk_indices[:, :, start:end, :] = torch.topk(random_scores[:, :, start:end, :], k=rounded_topk_num, dim=-1).indices.to(torch.int32)

            del random_scores
            torch.cuda.empty_cache()

            Q = torch.randn(batch, seq_len_sample, heads, dim, device='cuda', dtype=torch.float16)
            K = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)
            V = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)

            if args.run_type == "correctness":
                o = modified_kernel(Q, K, V, topk_indices, head_mapping, topk_sample)
                o_ref = ref_program_correct(Q, K, V, topk_indices, head_mapping, tile_size, rolling, topk_sample)
                eps_c = 1e-2
                eps_s = 1e-5
                assert_ = False
                print_ = True
                assert_similar(o, o_ref, name="o_ref", assert_=assert_, print_=print_, eps=eps_s)
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
                    tile_latency = do_bench(
                    modified_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[Q, K, V, topk_indices, head_mapping, topk_sample]
                    )
                    sum_tile_latency += tile_latency
                    cnt_tile += 1
                except Exception:
                    pass

                del Q, K, V
                torch.cuda.empty_cache()
            
        if args.run_type == "benchmark":
            flops_per_causal_matmul = batch * heads * seq_len * seq_len * dim
            flops_per_matmul_topk = 2.0 * batch * heads * seq_len * (rounded_topk_num+tile_size) * dim

            flops_flash = 2.0 * flops_per_causal_matmul
            if args.rolling:
                flops_reuse = flops_per_matmul_topk
            else:
                flops_reuse = topk * (flops_per_matmul_topk)/100.0 + (100-topk)*(2.0 * flops_per_matmul_topk)/100.0
            if args.with_ref:
                avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
                avg_ref_f_tflops = (flops_flash / avg_ref_f_latency * 1e-9) if avg_ref_f_latency > 0 else 0.0
            avg_tile_latency = (sum_tile_latency / cnt_tile) if cnt_tile else 0.0
            avg_tile_tflops = (flops_per_matmul_topk / avg_tile_latency * 1e-9) if avg_tile_latency > 0 else 0.0
            if args.with_ref:
                print(f"{seq_len},{topk},"
                    f"{avg_ref_f_latency:.2f},{avg_ref_f_tflops:.2f},"
                    f"{avg_tile_latency:.2f},{avg_tile_tflops:.2f}")
            else:
                print(f"{seq_len},{topk},"
                    f"{avg_tile_latency:.2f},{avg_tile_tflops:.2f}")
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
