#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unified prefill kernel benchmarking script.

Usage:
    # Run specific kernel with custom config
    python scripts/benchmark_prefill.py --kernel tilelang_fa --seq_len 8192 --mode benchmark
    python scripts/benchmark_prefill.py --kernel recompute --seq_len 8192 --topk 10 --layer 0 --mode benchmark
    python scripts/benchmark_prefill.py --kernel reuse --seq_len 8192 --topk 10 --rolling --mode benchmark
    
    # Run full benchmark suite (equivalent to old run.sh)
    python scripts/benchmark_prefill.py --all
    
    # Run correctness tests
    python scripts/benchmark_prefill.py --kernel recompute --mode correctness
"""

import argparse
import csv
import math
import os
from datetime import datetime
import torch
import tilelang
import tilelang.language as T


from kascade.kernels.kernel_utils import (
    do_bench, assert_similar, assert_allclose, assert_equal,
    compile_custom_kernel_from_cu, make_tile_causal_mask
)
from kascade.kernels.flash_attention.example_gqa_fwd_bshd_wgmma_pipelined import (
    flashattn as tilelang_fa_kernel, ref_program as ref_program_fa
)
from kascade.kernels.flash_attention.recompute_kascade_gqa_prefill import (
    flashattn as recompute_kernel_fn, ref_program as ref_program_recompute,
    ref_program_correct_recompute, full_kernel as recompute_full_kernel
)
from kascade.kernels.flash_attention.reuse_kascade_gqa_prefill import (
    flashattn as reuse_kernel_fn, ref_program_correct as ref_program_correct_reuse
)


# Global results storage
BENCHMARK_RESULTS = []


def run_ref_fa(batch, heads, seq_len, dim, groups, is_causal):
    """Run reference FlashAttention kernel benchmark (for --all mode)."""
    runs = 10
    sum_ref_latency = 0.0
    cnt_ref = 0

    for _ in range(runs):
        Q = torch.randn(batch, seq_len, heads, dim, device='cuda', dtype=torch.float16)
        K = torch.randn(batch, seq_len, groups, dim, device='cuda', dtype=torch.float16)
        V = torch.randn(batch, seq_len, groups, dim, device='cuda', dtype=torch.float16)

        try:
            ref_latency = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[Q, K, V, is_causal])
            sum_ref_latency += ref_latency
            cnt_ref += 1
        except Exception:
            pass

        del Q, K, V
        torch.cuda.empty_cache()

    avg_ref_latency = (sum_ref_latency / cnt_ref) if cnt_ref else 0.0
    
    result = {
        'kernel': 'ref_fa',
        'seq_len': seq_len,
        'topk': None,
        'layer': None,
        'latency_ms': avg_ref_latency,
    }
    BENCHMARK_RESULTS.append(result)
    print(f"{seq_len},{avg_ref_latency:.2f}")


def get_results_dir():
    """Get the results directory, creating it if necessary."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def get_kernel_dir():
    """Get the directory containing the kernel CUDA files."""
    return os.path.join(os.path.dirname(__file__), '..', 'src', 'kernels', 'flash_attention')


def run_tilelang_fa(batch, heads, seq_len, dim, groups, is_causal, mode, with_ref, code):
    """Run TileLang FlashAttention kernel benchmark."""
    program = tilelang_fa_kernel(
        T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, is_causal, tune=False, groups=groups
    )(block_M=128, block_N=128, num_stages=2, threads=256)
    kernel = tilelang.compile(program, out_idx=[3], pass_configs={"tl.disable_safe_memory_legalize": True, "tl.disable_warp_specialized": False})

    if code:
        print(kernel.get_kernel_source())
        return

    runs = 100 if mode == "correctness" else 10
    sum_ref_f_latency = sum_tile_latency = 0.0
    cnt_ref_f = cnt_tile = 0

    for _ in range(runs):
        seq_len_sample = torch.randint(64, seq_len + 1, (1,)).item() if mode == "correctness" else seq_len

        Q = torch.randn(batch, seq_len_sample, heads, dim, device='cuda', dtype=torch.float16)
        K = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)
        V = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)

        if mode == "correctness":
            print(f"seq_len: {seq_len_sample}")
            o = kernel(Q, K, V)
            o_ref = ref_program_fa(Q, K, V, is_causal)
            assert_similar(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-5)
            assert_allclose(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-3)
        else:
            if with_ref:
                try:
                    ref_f_latency = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[Q, K, V, is_causal])
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

            try:
                tile_latency = do_bench(kernel, n_warmup=1, n_repeat=5, input_tensors=[Q, K, V])
                sum_tile_latency += tile_latency
                cnt_tile += 1
            except Exception:
                pass

        del Q, K, V
        torch.cuda.empty_cache()

    if mode == "benchmark":
        avg_tile_latency = (sum_tile_latency / cnt_tile) if cnt_tile else 0.0
        
        result = {
            'kernel': 'tilelang_fa',
            'seq_len': seq_len,
            'topk': None,
            'layer': None,
            'latency_ms': avg_tile_latency,
        }
        BENCHMARK_RESULTS.append(result)
        
        if with_ref:
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            print(f"{seq_len},{avg_ref_f_latency:.2f},{avg_tile_latency:.2f}")
        else:
            print(f"{seq_len},{avg_tile_latency:.2f}")
    else:
        print(f"{runs} Correctness test passed!")


def run_recompute(batch, heads, seq_len, dim, groups, topk, layer, rolling, mode, with_ref, code):
    """Run recompute Kascade kernel benchmark."""
    import os
    kernel_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'kernels', 'flash_attention')
    tile_size = 32

    if layer == 0:
        program = recompute_kernel_fn(
            T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, tune=False, groups=groups, kernel_type="prefill"
        )(block_M=128, block_H=(heads // groups), block_N=128, num_stages=2, threads=256)
        recompute_kernel = tilelang.compile(program, out_idx=[3, 4], pass_configs={"tl.disable_safe_memory_legalize": True})
    else:
        program = recompute_kernel_fn(
            T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, tune=False, groups=groups, kernel_type="compute_scores"
        )(block_M=128, block_H=(heads // groups), block_N=256, num_stages=2, threads=256)
        recompute_kernel = tilelang.compile(program, out_idx=[2], pass_configs={"tl.disable_safe_memory_legalize": True})

    program = recompute_kernel_fn(
        T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, tune=False, groups=groups, kernel_type="aggregate"
    )(block_M=tile_size, block_H=(heads // groups), block_N=256, num_stages=2, threads=256)
    aggregate_kernel = compile_custom_kernel_from_cu(
        func=program,
        cu_file_path=os.path.join(kernel_dir, f"kascade_prefill_cu_kernels/dynamic_alllen_aggregate_cp_async_ts{tile_size}_dim128_gsize4_heads32_groups8.cu"),
        so_dir=os.path.join(kernel_dir, "kascade_prefill_cu_kernels/__socache__/"),
        out_idx=None,
        execution_backend="cython",
        pass_configs={"tl.disable_safe_memory_legalize": True},
    )

    reuse_program = reuse_kernel_fn(
        T.symbolic("batch"), heads, T.symbolic("kv_seqlen"), dim, T.symbolic("max_topk_num"), rolling=rolling, tune=False, groups=groups
    )(block_M=tile_size, block_H=(heads // groups), block_N=128, num_stages=2, threads=256)
    reuse_kernel = compile_custom_kernel_from_cu(
        func=reuse_program,
        cu_file_path=os.path.join(kernel_dir, f"kascade_prefill_cu_kernels/dynamic_alllen_reuse_cp_async_ts{tile_size}_dim128_gsize4_heads32_groups8{'_rolling' if rolling else ''}.cu"),
        so_dir=os.path.join(kernel_dir, "kascade_prefill_cu_kernels/__socache__/"),
        out_idx=[6],
        execution_backend="cython",
        pass_configs={"tl.disable_safe_memory_legalize": True},
    )

    if code:
        print(recompute_kernel.get_kernel_source())
        print(aggregate_kernel.get_kernel_source())
        return

    runs = 100 if mode == "correctness" else 10
    sum_ref_f_latency = sum_full_kernel_latency = 0.0
    cnt_ref_f = cnt_full_kernel = 0

    for _ in range(runs):
        if mode == "correctness":
            seq_len_sample = torch.randint(128, seq_len + 1, (1,)).item()
            topk_sample = torch.randint(10, 50, size=(1,)).item()
            print(f"seq_len: {seq_len_sample}, topk: {topk_sample}")
        else:
            seq_len_sample = seq_len
            topk_sample = topk

        Q = torch.randn(batch, seq_len_sample, heads, dim, device='cuda', dtype=torch.float16)
        K = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)
        V = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)
        actual_topk_num = min(max(128, int((topk_sample / 100) * seq_len_sample)), seq_len_sample)
        rounded_topk_num = min(((actual_topk_num + 127) // 128) * 128, seq_len_sample)

        if mode == "correctness":
            o, indices, scores = recompute_full_kernel(recompute_kernel, aggregate_kernel, reuse_kernel, Q, K, V, rounded_topk_num, tile_size, layer, topk_sample)
            o_ref, indices_ref, scores_ref = ref_program_correct_recompute(Q, K, V, rounded_topk_num, tile_size, layer, rolling, topk_sample)
            scores.masked_fill_((scores == float('-inf')), 0)
            scores_ref.masked_fill_(scores_ref == torch.finfo(scores_ref.dtype).min, 0)
            
            assert_similar(scores, scores_ref, name="scores_o_ref", assert_=False, print_=True, eps=1e-5)
            assert_similar(indices.sort(dim=-1).values.to(torch.float32), indices_ref.sort(dim=-1).values.to(torch.float32), name="indices_o_ref", assert_=False, print_=True, eps=1e-5)
            assert_similar(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-5 if layer == 0 else 1e-3)
        else:
            if with_ref:
                try:
                    ref_f_latency = do_bench(ref_program_recompute, n_warmup=1, n_repeat=5, input_tensors=[Q, K, V])
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass
            
            try:
                full_kernel_latency = do_bench(
                    recompute_full_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[recompute_kernel, aggregate_kernel, reuse_kernel, Q, K, V, rounded_topk_num, tile_size, layer, topk_sample]
                )
                sum_full_kernel_latency += full_kernel_latency
                cnt_full_kernel += 1
            except Exception:
                pass

        del Q, K, V
        torch.cuda.empty_cache()

    if mode == "benchmark":
        avg_full_kernel_latency = (sum_full_kernel_latency / cnt_full_kernel) if cnt_full_kernel else 0.0
        
        result = {
            'kernel': 'recompute',
            'seq_len': seq_len,
            'topk': topk,
            'layer': layer,
            'latency_ms': avg_full_kernel_latency,
        }
        BENCHMARK_RESULTS.append(result)
        
        if with_ref:
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            print(f"{seq_len},{topk},{avg_ref_f_latency:.2f},{avg_full_kernel_latency:.2f}")
        else:
            print(f"{seq_len},{topk},{avg_full_kernel_latency:.2f}")
    else:
        print(f"All {runs} correctness tests passed!")


def run_reuse(batch, heads, seq_len, dim, groups, topk, rolling, mode, with_ref, code):
    """Run reuse Kascade kernel benchmark."""
    import os
    kernel_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'kernels', 'flash_attention')
    tile_size = 32

    program = reuse_kernel_fn(
        T.symbolic("batch"), heads, T.symbolic("seq_len"), dim, T.symbolic("max_topk_num"), rolling=rolling, tune=False, groups=groups
    )(block_M=tile_size, block_H=(heads // groups), block_N=128, num_stages=2, threads=256)
    
    modified_kernel = compile_custom_kernel_from_cu(
        func=program,
        cu_file_path=os.path.join(kernel_dir, f"kascade_prefill_cu_kernels/dynamic_alllen_reuse_cp_async_ts{tile_size}_dim128_gsize4_heads32_groups8{'_rolling' if rolling else ''}.cu"),
        so_dir=os.path.join(kernel_dir, "kascade_prefill_cu_kernels/__socache__/"),
        out_idx=[6],
        execution_backend="cython",
        pass_configs={"tl.disable_safe_memory_legalize": True},
    )

    if code:
        print(modified_kernel.get_kernel_source())
        return

    runs = 100 if mode == "correctness" else 10
    sum_ref_f_latency = sum_tile_latency = 0.0
    cnt_ref_f = cnt_tile = 0

    for _ in range(runs):
        if mode == "correctness":
            seq_len_sample = torch.randint(128, seq_len + 1, (1,)).item()
            topk_sample = torch.randint(10, 50, size=(1,)).item()
            print(f"seq_len: {seq_len_sample}, topk: {topk_sample}")
        else:
            seq_len_sample = seq_len
            topk_sample = topk

        actual_topk_num = min(max(128, int((topk_sample / 100) * seq_len_sample)), seq_len_sample)
        rounded_topk_num = min(((actual_topk_num + 127) // 128) * 128, seq_len_sample)
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
            chunk_size = 131072 // 2 ** int(topk_sample // 10)
            num_chunks = math.ceil(seq_len_sample / chunk_size)
            for i in range(num_chunks):
                start = (i * chunk_size) // tile_size
                end = min(((i + 1) * chunk_size) // tile_size, seq_len_sample)
                topk_indices[:, :, start:end, :] = torch.topk(random_scores[:, :, start:end, :], k=rounded_topk_num, dim=-1).indices.to(torch.int32)

        del random_scores
        torch.cuda.empty_cache()

        Q = torch.randn(batch, seq_len_sample, heads, dim, device='cuda', dtype=torch.float16)
        K = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)
        V = torch.randn(batch, seq_len_sample, groups, dim, device='cuda', dtype=torch.float16)

        if mode == "correctness":
            o = modified_kernel(Q, K, V, topk_indices, head_mapping, topk_sample)
            o_ref = ref_program_correct_reuse(Q, K, V, topk_indices, head_mapping, tile_size, rolling, topk_sample)
            assert_similar(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-5)
            assert_allclose(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-2)
        else:
            if with_ref:
                from kascade.kernels.flash_attention.reuse_kascade_gqa_prefill import ref_program
                try:
                    ref_f_latency = do_bench(ref_program, n_warmup=1, n_repeat=5, input_tensors=[Q, K, V])
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

            try:
                tile_latency = do_bench(modified_kernel, n_warmup=1, n_repeat=5, input_tensors=[Q, K, V, topk_indices, head_mapping, topk_sample])
                sum_tile_latency += tile_latency
                cnt_tile += 1
            except Exception:
                pass

        del Q, K, V
        torch.cuda.empty_cache()

    if mode == "benchmark":
        avg_tile_latency = (sum_tile_latency / cnt_tile) if cnt_tile else 0.0

        result = {
            'kernel': 'reuse',
            'seq_len': seq_len,
            'topk': topk,
            'layer': None,
            'latency_ms': avg_tile_latency,
        }
        BENCHMARK_RESULTS.append(result)

        if with_ref:
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            print(f"{seq_len},{topk},{avg_ref_f_latency:.2f},{avg_tile_latency:.2f}")
        else:
            print(f"{seq_len},{topk},{avg_tile_latency:.2f}")
    else:
        print(f"All {runs} correctness tests passed!")


def run_all_benchmarks(batch, heads, dim, groups):
    """Run full benchmark suite (equivalent to old run.sh)."""
    seq_lens = [8192, 16384, 32768, 65536, 131072, 262144]
    topk_values = [10, 20, 30]

    print("=== Kascade Prefill Kernel Benchmarks ===\n")

    print("ref_fa prefill")
    for seq_len in seq_lens:
        run_ref_fa(batch, heads, seq_len, dim, groups, is_causal=True)

    print("\ntilelang_fa prefill")
    for seq_len in seq_lens:
        run_tilelang_fa(batch, heads, seq_len, dim, groups, is_causal=True, mode="benchmark", with_ref=False, code=False)

    print("\nrecompute layer 0 prefill")
    for topk in topk_values:
        for seq_len in seq_lens:
            run_recompute(batch, heads, seq_len, dim, groups, topk, layer=0, rolling=True, mode="benchmark", with_ref=False, code=False)

    print("\nrecompute layer 1 prefill")
    for topk in topk_values:
        for seq_len in seq_lens:
            run_recompute(batch, heads, seq_len, dim, groups, topk, layer=1, rolling=True, mode="benchmark", with_ref=False, code=False)

    print("\nreuse kernel prefill")
    for topk in topk_values:
        for seq_len in seq_lens:
            run_reuse(batch, heads, seq_len, dim, groups, topk, rolling=True, mode="benchmark", with_ref=False, code=False)

    print("\n=== Prefill benchmarks complete ===")

    # Save results to CSV
    if BENCHMARK_RESULTS:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(get_results_dir(), f'benchmark_prefill_{timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['kernel', 'seq_len', 'topk', 'layer', 'latency_ms'])
            writer.writeheader()
            writer.writerows(BENCHMARK_RESULTS)
        print(f"\nResults saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified prefill kernel benchmarking")
    parser.add_argument('--kernel', type=str, choices=['tilelang_fa', 'recompute', 'reuse'], help='Kernel to benchmark')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--groups', type=int, default=8, help='Number of KV groups (for GQA)')
    parser.add_argument('--seq_len', type=int, default=8192, help='Sequence length')
    parser.add_argument('--dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--topk', type=float, default=10, help='Top-k percentage for Kascade kernels')
    parser.add_argument('--layer', type=int, default=0, choices=[0, 1], help='Layer type: 0 for first layer, 1 for subsequent')
    parser.add_argument('--rolling', action='store_true', help='Use rolling top-k')
    parser.add_argument('--is_causal', action='store_true', help='Use causal masking (for tilelang_fa)')
    parser.add_argument('--mode', type=str, default='benchmark', choices=['benchmark', 'correctness'], help='Run mode')
    parser.add_argument('--with_ref', action='store_true', help='Include reference FA benchmark')
    parser.add_argument('--code', action='store_true', help='Print kernel CUDA source code')
    parser.add_argument('--all', action='store_true', help='Run full benchmark suite')
    
    args = parser.parse_args()

    if args.all:
        run_all_benchmarks(args.batch, args.heads, args.dim, args.groups)
    elif args.kernel == 'tilelang_fa':
        run_tilelang_fa(args.batch, args.heads, args.seq_len, args.dim, args.groups, args.is_causal, args.mode, args.with_ref, args.code)
    elif args.kernel == 'recompute':
        run_recompute(args.batch, args.heads, args.seq_len, args.dim, args.groups, args.topk, args.layer, args.rolling, args.mode, args.with_ref, args.code)
    elif args.kernel == 'reuse':
        run_reuse(args.batch, args.heads, args.seq_len, args.dim, args.groups, args.topk, args.rolling, args.mode, args.with_ref, args.code)
    else:
        parser.print_help()
        print("\nError: Please specify --kernel or --all")


if __name__ == "__main__":
    main()
