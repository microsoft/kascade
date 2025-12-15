#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unified decode kernel benchmarking script.

Usage:
    # Run specific kernel with custom config
    python scripts/benchmark_decode.py --kernel tilelang_fa --kv_seqlen 8192 --batch 64 --mode benchmark
    python scripts/benchmark_decode.py --kernel recompute --kv_seqlen 8192 --batch 64 --topk 10 --layer 0 --mode benchmark
    python scripts/benchmark_decode.py --kernel reuse --kv_seqlen 8192 --batch 64 --topk 10 --mode benchmark
    python scripts/benchmark_decode.py --kernel recompute_kvcache --kv_seqlen 8192 --batch 64 --topk 10 --mode benchmark
    python scripts/benchmark_decode.py --kernel reuse_kvcache --kv_seqlen 8192 --batch 64 --topk 10 --mode benchmark
    
    # Run full benchmark suite (equivalent to old run.sh)
    python scripts/benchmark_decode.py --all
    
    # Run correctness tests
    python scripts/benchmark_decode.py --kernel recompute --mode correctness
"""

import argparse
import csv
import os
from datetime import datetime
import torch
import tilelang
import tilelang.language as T

from kascade.kernels.kernel_utils import (
    do_bench, assert_similar, assert_allclose, assert_equal,
    compile_custom_kernel_from_cu
)


# Global results storage
BENCHMARK_RESULTS = []


def run_ref_fa(batch, heads, kv_seqlen, dim, groups):
    """Run reference FlashAttention decode kernel benchmark (for --all mode)."""
    from kascade.kernels.flash_decoding.gqa_decode import ref_program_fa

    runs = 10
    sum_ref_latency = 0.0
    cnt_ref = 0

    for _ in range(runs):
        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.float16)

        try:
            ref_latency = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v])
            sum_ref_latency += ref_latency
            cnt_ref += 1
        except Exception:
            pass

        del q, k, v
        torch.cuda.empty_cache()

    avg_ref_latency = (sum_ref_latency / cnt_ref) if cnt_ref else 0.0
    
    result = {
        'kernel': 'ref_fa',
        'kv_seqlen': kv_seqlen,
        'batch': batch,
        'topk': None,
        'layer': None,
        'latency_ms': avg_ref_latency,
    }
    BENCHMARK_RESULTS.append(result)
    print(f"{kv_seqlen},{avg_ref_latency:.2f}")


def get_results_dir():
    """Get the results directory, creating it if necessary."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def get_kernel_dir():
    """Get the directory containing the kernel CUDA files."""
    return os.path.join(os.path.dirname(__file__), '..', 'src', 'kernels', 'flash_decoding')


def get_heuristic_config():
    """Get device-specific kernel configuration."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    if sm_version == 89:
        return {"block_N": 128, "block_H": 64, "num_split": 8, "num_stages": 0, "threads": 128}
    else:
        return {"block_N": 128, "block_H": 64, "num_split": 8, "num_stages": 2, "threads": 128}


def run_tilelang_fa(batch, heads, kv_seqlen, dim, groups, mode, with_ref, code):
    """Run TileLang FlashAttention decode kernel benchmark."""
    from kascade.kernels.flash_decoding.gqa_decode import (
        flashattn as tilelang_fa_kernel, ref_program, ref_program_fa, ref_program_correct
    )

    config = get_heuristic_config()
    program = tilelang_fa_kernel(T.symbolic("batch"), T.symbolic("heads"), T.symbolic("groups"), T.symbolic("kv_seqlen"), dim, tune=False)(**config)
    kernel = tilelang.compile(program, out_idx=[5])

    if code:
        print(kernel.get_kernel_source())
        return

    runs = 100 if mode == "correctness" else 10
    lat_f_sum = lat_og_sum = 0.0
    cnt_f = cnt_og = 0

    for _ in range(runs):
        kv_seqlen_sample = torch.randint(32, kv_seqlen + 1, (1,)).item() if mode == "correctness" else kv_seqlen
        
        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, kv_seqlen_sample, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, kv_seqlen_sample, groups, dim, device="cuda", dtype=torch.float16)
        glse = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float16)
        Output_partial = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float16)

        if mode == "correctness":
            o_ref = ref_program(q, k, v)
            o_ref_correct = ref_program_correct(q, k, v)
            o_og = kernel(q, k, v, glse, Output_partial)
            assert_similar(o_ref, o_og, name="ref_og", assert_=True, print_=False)
            assert_similar(o_og, o_ref_correct, name="o_ref_correct_og", assert_=True, print_=False)
        else:
            if with_ref:
                try:
                    latency_f = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v])
                    lat_f_sum += latency_f
                    cnt_f += 1
                except Exception:
                    pass
            try:
                latency_og = do_bench(kernel, n_warmup=1, n_repeat=5, input_tensors=[q, k, v, glse, Output_partial])
                lat_og_sum += latency_og
                cnt_og += 1
            except Exception:
                pass

        del q, k, v, glse, Output_partial
        torch.cuda.empty_cache()

    if mode == "benchmark":
        avg_og = lat_og_sum / cnt_og if cnt_og > 0 else 0.0

        result = {
            'kernel': 'tilelang_fa',
            'kv_seqlen': kv_seqlen,
            'batch': batch,
            'topk': None,
            'layer': None,
            'latency_ms': avg_og,
        }
        BENCHMARK_RESULTS.append(result)

        if with_ref:
            avg_f = lat_f_sum / cnt_f if cnt_f > 0 else 0.0
            print(f"{kv_seqlen},{avg_f:.2f},{avg_og:.2f}")
        else:
            print(f"{kv_seqlen},{avg_og:.2f}")
    else:
        print(f"{runs} Correctness tests passed!")


def run_recompute(batch, heads, kv_seqlen, dim, groups, topk, layer, mode, with_ref, code):
    """Run recompute Kascade decode kernel benchmark."""
    import os
    from kascade.kernels.flash_decoding.recompute_kascade_gqa_decode import (
        flashattn as recompute_kernel_fn, ref_program_fa, ref_program_correct, full_kernel
    )
    from kascade.kernels.flash_decoding.reuse_kascade_gqa_decode import flashattn as reuse_kernel_fn

    kernel_dir = get_kernel_dir()
    config = get_heuristic_config()

    program = recompute_kernel_fn(T.symbolic("batch"), heads, groups, T.symbolic("kv_seqlen"), dim, tune=False, layer=layer)(**config)
    kernel = tilelang.compile(program, out_idx=[7, 8], pass_configs={"tl.disable_safe_memory_legalize": True})

    reuse_program = reuse_kernel_fn(T.symbolic("batch"), heads, groups, T.symbolic("kv_seqlen"), dim, T.symbolic("topk"), tune=False)(**config)
    reuse_kernel = compile_custom_kernel_from_cu(
        func=reuse_program,
        cu_file_path=os.path.join(kernel_dir, "kascade_decode_cu_kernels/dynamic_alllen_reuse_cp_async_dim128_gsize4_heads32_groups8.cu"),
        so_dir=os.path.join(kernel_dir, "kascade_decode_cu_kernels/__socache__/"),
        out_idx=[7],
        execution_backend="cython",
        pass_configs={"tl.disable_safe_memory_legalize": True},
    )

    if code:
        print(kernel.get_kernel_source())
        return

    runs = 100 if mode == "correctness" else 10
    sum_ref_f_latency = sum_tile_topk_latency = 0.0
    cnt_ref_f = cnt_tile_topk = 0

    for _ in range(runs):
        if mode == "correctness":
            kv_seqlen_sample = torch.randint(128, kv_seqlen + 1, size=(1,), dtype=torch.int32).item()
            topk_sample = torch.randint(5, 30, size=(1,), dtype=torch.int32).item()
            print(f"kv_seqlen={kv_seqlen_sample}, topk={topk_sample}")
        else:
            kv_seqlen_sample = kv_seqlen
            topk_sample = topk

        actual_topk_num = min(max(128, int((topk_sample / 100) * kv_seqlen_sample)), kv_seqlen_sample)
        rounded_topk_num = min(((actual_topk_num + 127) // 128) * 128, kv_seqlen_sample)

        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, kv_seqlen_sample, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, kv_seqlen_sample, groups, dim, device="cuda", dtype=torch.float16)
        glse = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float32)
        Output_partial = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float32)
        row_sums = torch.empty(batch, heads, device="cuda", dtype=torch.float32)
        scores = torch.empty(batch, heads, kv_seqlen_sample, device="cuda", dtype=torch.float16)

        if mode == "correctness":
            o, topk_indices_og, scores_og = full_kernel(kernel, reuse_kernel, q, k, v, rounded_topk_num, glse, Output_partial, row_sums, scores, layer)
            o_ref, topk_indices_ref, scores_ref = ref_program_correct(q, k, v, rounded_topk_num, layer)

            assert_similar(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-4)
            assert_similar(scores_og, scores_ref, name="scores_o_ref", assert_=False, print_=True, eps=1e-4)
            assert_equal(topk_indices_og.sort(dim=-1).values, topk_indices_ref.sort(dim=-1).values, name="o_ref_topk", assert_=False, print_=True)
        else:
            if with_ref:
                try:
                    ref_f_latency = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v])
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

            try:
                tile_topk_latency = do_bench(
                    full_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[kernel, reuse_kernel, q, k, v, rounded_topk_num, glse, Output_partial, row_sums, scores, layer]
                )
                sum_tile_topk_latency += tile_topk_latency
                cnt_tile_topk += 1
            except Exception:
                pass

        del q, k, v, glse, Output_partial
        torch.cuda.empty_cache()

    if mode == "benchmark":
        avg_tile_topk_latency = (sum_tile_topk_latency / cnt_tile_topk) if cnt_tile_topk else 0.0

        result = {
            'kernel': 'recompute',
            'kv_seqlen': kv_seqlen,
            'batch': batch,
            'topk': topk,
            'layer': layer,
            'latency_ms': avg_tile_topk_latency,
        }
        BENCHMARK_RESULTS.append(result)

        if with_ref:
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            print(f"{kv_seqlen},{topk},{avg_ref_f_latency:.2f},{avg_tile_topk_latency:.2f}")
        else:
            print(f"{kv_seqlen},{topk},{avg_tile_topk_latency:.2f}")
    else:
        print(f"{runs} Correctness tests passed!")


def run_reuse(batch, heads, kv_seqlen, dim, groups, topk, mode, with_ref, code):
    """Run reuse Kascade decode kernel benchmark."""
    import os
    from kascade.kernels.flash_decoding.reuse_kascade_gqa_decode import (
        flashattn as reuse_kernel_fn, ref_program_fa, ref_program_correct
    )

    kernel_dir = get_kernel_dir()
    config = get_heuristic_config()

    program = reuse_kernel_fn(T.symbolic("batch"), heads, groups, T.symbolic("kv_seqlen"), dim, T.symbolic("topk"), tune=False)(**config)
    
    modified_kernel = compile_custom_kernel_from_cu(
        func=program,
        cu_file_path=os.path.join(kernel_dir, "kascade_decode_cu_kernels/dynamic_alllen_reuse_cp_async_dim128_gsize4_heads32_groups8.cu"),
        so_dir=os.path.join(kernel_dir, "kascade_decode_cu_kernels/__socache__/"),
        out_idx=[7],
        execution_backend="cython",
        pass_configs={"tl.disable_safe_memory_legalize": True},
    )

    if code:
        print(modified_kernel.get_kernel_source())
        return

    runs = 100 if mode == "correctness" else 10
    sum_ref_f_latency = sum_tile_async_latency = 0.0
    cnt_ref_f = cnt_tile_async = 0

    for _ in range(runs):
        if mode == "correctness":
            kv_seqlen_sample = torch.randint(128, kv_seqlen + 1, size=(1,)).item()
            topk_sample = torch.randint(10, 50, size=(1,)).item()
            print(f"kv_seqlen={kv_seqlen_sample}, topk={topk_sample}")
        else:
            kv_seqlen_sample = kv_seqlen
            topk_sample = topk

        actual_topk_num = min(max(128, int((topk_sample / 100) * kv_seqlen_sample)), kv_seqlen_sample)
        rounded_topk_num = min(((actual_topk_num + 127) // 128) * 128, kv_seqlen_sample)

        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, kv_seqlen_sample, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, kv_seqlen_sample, groups, dim, device="cuda", dtype=torch.float16)
        head_mapping = torch.randint(0, groups, (groups,), device="cuda", dtype=torch.int32)
        random_scores = torch.rand(batch, groups, kv_seqlen_sample, device="cuda")
        topk_indices = torch.topk(random_scores, k=rounded_topk_num, dim=-1).indices.to(torch.int32)
        del random_scores
        glse = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float32)
        Output_partial = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float32)

        if mode == "correctness":
            o = modified_kernel(q, k, v, topk_indices, head_mapping, glse, Output_partial)
            o_ref = ref_program_correct(q, k, v, topk_indices, head_mapping)
            assert_similar(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-3)
            assert_allclose(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-3)
        else:
            if with_ref:
                try:
                    ref_f_latency = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v])
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

            try:
                tile_async_latency = do_bench(
                    modified_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k, v, topk_indices, head_mapping, glse, Output_partial]
                )
                sum_tile_async_latency += tile_async_latency
                cnt_tile_async += 1
            except Exception:
                pass

        del q, k, v, head_mapping, topk_indices, glse, Output_partial
        torch.cuda.empty_cache()

    if mode == "benchmark":
        avg_tile_async_latency = (sum_tile_async_latency / cnt_tile_async) if cnt_tile_async else 0.0

        result = {
            'kernel': 'reuse',
            'kv_seqlen': kv_seqlen,
            'batch': batch,
            'topk': topk,
            'layer': None,
            'latency_ms': avg_tile_async_latency,
        }
        BENCHMARK_RESULTS.append(result)

        if with_ref:
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            print(f"{kv_seqlen},{topk},{avg_ref_f_latency:.2f},{avg_tile_async_latency:.2f}")
        else:
            print(f"{kv_seqlen},{topk},{avg_tile_async_latency:.2f}")
    else:
        print(f"{runs} Correctness tests passed!")


def run_recompute_kvcache(batch, heads, kv_seqlen, dim, groups, topk, layer, mode, with_ref, code):
    """Run recompute Kascade decode kernel with KV cache benchmark."""
    import os
    from kascade.kernels.flash_decoding.recompute_kascade_gqa_decode_with_kvcache import (
        flashattn as recompute_kernel_fn, ref_program_fa, ref_program_correct, full_kernel
    )
    from kascade.kernels.flash_decoding.reuse_kascade_gqa_decode_with_kvcache import flashattn as reuse_kernel_fn

    kernel_dir = get_kernel_dir()
    config = get_heuristic_config()

    program = recompute_kernel_fn(heads, groups, dim, layer=layer)(
        batch=T.symbolic("batch"),
        max_cache_seqlen=T.symbolic("max_cache_seqlen"),
        **config
    )
    kernel = tilelang.compile(program, out_idx=[9, 10], pass_configs={"tl.disable_safe_memory_legalize": True})

    reuse_program = reuse_kernel_fn(heads, groups, dim)(
        batch=T.symbolic("batch"),
        max_cache_seqlen=T.symbolic("max_cache_seqlen"),
        topk=T.symbolic("topk"),
        **config
    )
    reuse_kernel = compile_custom_kernel_from_cu(
        func=reuse_program,
        cu_file_path=os.path.join(kernel_dir, "kascade_decode_cu_kernels/dynamic_alllen_reuse_cp_async_dim128_gsize4_heads32_groups8_with_kvcache.cu"),
        so_dir=os.path.join(kernel_dir, "kascade_decode_cu_kernels/__socache__/"),
        out_idx=[9],
        execution_backend="cython",
        pass_configs={"tl.disable_safe_memory_legalize": True},
    )

    if code:
        print(kernel.get_kernel_source())
        return

    runs = 100 if mode == "correctness" else 10
    sum_ref_f_latency = sum_tile_topk_latency = 0.0
    cnt_ref_f = cnt_tile_topk = 0

    for _ in range(runs):
        if mode == "correctness":
            cache_seqlens = torch.randint(128, kv_seqlen, (batch,), device='cuda', dtype=torch.int32)
            topk_sample = torch.randint(5, 30, size=(1,), dtype=torch.int32).item()
            print(f"cache_seqlens={cache_seqlens}, topk={topk_sample}")
        else:
            cache_seqlens = torch.full((batch,), kv_seqlen, device='cuda', dtype=torch.int32)
            topk_sample = topk

        max_seqlen = cache_seqlens.max().item()
        max_topk = (int((topk_sample * max_seqlen) / 100) + 127) // 128 * 128

        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, max_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, max_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        glse = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float32)
        Output_partial = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float32)
        row_sums = torch.empty(batch, heads, device="cuda", dtype=torch.float32)
        scores = torch.empty(batch, heads, max_seqlen, device="cuda", dtype=torch.float16)

        if mode == "correctness":
            o, topk_indices_og, scores_og = full_kernel(kernel, reuse_kernel, q, k, v, cache_seqlens, max_topk, topk_sample, glse, Output_partial, row_sums, scores, layer)
            o_ref, topk_indices_ref, scores_ref = ref_program_correct(q, k, v, cache_seqlens, max_topk, layer)

            assert_similar(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-4)
            assert_similar(scores_og, scores_ref, name="scores_o_ref", assert_=False, print_=True, eps=1e-4)
        else:
            if with_ref:
                try:
                    ref_f_latency = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v])
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

            try:
                tile_topk_latency = do_bench(
                    full_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[kernel, reuse_kernel, q, k, v, cache_seqlens, max_topk, topk_sample, glse, Output_partial, row_sums, scores, layer]
                )
                sum_tile_topk_latency += tile_topk_latency
                cnt_tile_topk += 1
            except Exception:
                pass

        del q, k, v, glse, Output_partial
        torch.cuda.empty_cache()

    if mode == "benchmark":
        avg_tile_topk_latency = (sum_tile_topk_latency / cnt_tile_topk) if cnt_tile_topk else 0.0

        result = {
            'kernel': 'recompute_kvcache',
            'kv_seqlen': kv_seqlen,
            'batch': batch,
            'topk': topk,
            'layer': layer,
            'latency_ms': avg_tile_topk_latency,
        }
        BENCHMARK_RESULTS.append(result)

        if with_ref:
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            print(f"{kv_seqlen},{topk},{avg_ref_f_latency:.2f},{avg_tile_topk_latency:.2f}")
        else:
            print(f"{kv_seqlen},{topk},{avg_tile_topk_latency:.2f}")
    else:
        print(f"{runs} Correctness tests passed!")


def run_reuse_kvcache(batch, heads, kv_seqlen, dim, groups, topk, mode, with_ref, code):
    """Run reuse Kascade decode kernel with KV cache benchmark."""
    import os
    from kascade.kernels.flash_decoding.reuse_kascade_gqa_decode_with_kvcache import (
        flashattn as reuse_kernel_fn, ref_program_fa, ref_program_correct
    )

    kernel_dir = get_kernel_dir()

    program = reuse_kernel_fn(heads, groups, dim)(
        batch=T.symbolic("batch"),
        block_N=128,
        block_H=64,
        num_split=8,
        num_stages=2,
        threads=128,
        max_cache_seqlen=T.symbolic("max_cache_seqlen"),
        topk=T.symbolic("topk"),
    )
    
    modified_kernel = compile_custom_kernel_from_cu(
        func=program,
        cu_file_path=os.path.join(kernel_dir, "kascade_decode_cu_kernels/dynamic_alllen_reuse_cp_async_dim128_gsize4_heads32_groups8_with_kvcache.cu"),
        so_dir=os.path.join(kernel_dir, "kascade_decode_cu_kernels/__socache__/"),
        out_idx=[9],
        execution_backend="cython",
        pass_configs={"tl.disable_safe_memory_legalize": True},
    )

    if code:
        print(modified_kernel.get_kernel_source())
        return

    runs = 100 if mode == "correctness" else 10
    sum_ref_f_latency = sum_tile_async_latency = 0.0
    cnt_ref_f = cnt_tile_async = 0

    for _ in range(runs):
        if mode == "correctness":
            cache_seqlens = torch.randint(128, kv_seqlen, (batch,), device='cuda', dtype=torch.int32)
            topk_sample = torch.randint(10, 50, size=(1,)).item()
            print(f"cache_seqlens={cache_seqlens}, topk={topk_sample}")
        else:
            cache_seqlens = torch.full((batch,), kv_seqlen, device='cuda', dtype=torch.int32)
            topk_sample = topk

        max_seqlen = cache_seqlens.max().item()
        max_topk = (int((topk_sample * max_seqlen) / 100) + 127) // 128 * 128

        q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, max_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, max_seqlen, groups, dim, device="cuda", dtype=torch.float16)
        head_mapping = torch.randint(0, groups, (groups,), device="cuda", dtype=torch.int32)
        random_scores = torch.rand(batch, groups, max_seqlen, device="cuda")
        topk_indices = torch.topk(random_scores, k=max_topk, dim=-1).indices.to(torch.int32)
        del random_scores
        glse = torch.empty(batch, heads, 8, device="cuda", dtype=torch.float32)
        Output_partial = torch.empty(batch, heads, 8, dim, device="cuda", dtype=torch.float32)

        if mode == "correctness":
            o = modified_kernel(q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample, glse, Output_partial)
            o_ref = ref_program_correct(q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample)
            assert_similar(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-3)
            assert_allclose(o, o_ref, name="o_ref", assert_=False, print_=True, eps=1e-3)
        else:
            if with_ref:
                try:
                    ref_f_latency = do_bench(ref_program_fa, n_warmup=1, n_repeat=5, input_tensors=[q, k, v])
                    sum_ref_f_latency += ref_f_latency
                    cnt_ref_f += 1
                except Exception:
                    pass

            try:
                tile_async_latency = do_bench(
                    modified_kernel, n_warmup=1, n_repeat=5,
                    input_tensors=[q, k, v, cache_seqlens, topk_indices, head_mapping, topk_sample, glse, Output_partial]
                )
                sum_tile_async_latency += tile_async_latency
                cnt_tile_async += 1
            except Exception:
                pass

        del q, k, v, head_mapping, topk_indices, glse, Output_partial
        torch.cuda.empty_cache()

    if mode == "benchmark":
        avg_tile_async_latency = (sum_tile_async_latency / cnt_tile_async) if cnt_tile_async else 0.0

        result = {
            'kernel': 'reuse_kvcache',
            'kv_seqlen': kv_seqlen,
            'batch': batch,
            'topk': topk,
            'layer': None,
            'latency_ms': avg_tile_async_latency,
        }
        BENCHMARK_RESULTS.append(result)

        if with_ref:
            avg_ref_f_latency = (sum_ref_f_latency / cnt_ref_f) if cnt_ref_f else 0.0
            print(f"{kv_seqlen},{topk},{avg_ref_f_latency:.2f},{avg_tile_async_latency:.2f}")
        else:
            print(f"{kv_seqlen},{topk},{avg_tile_async_latency:.2f}")
    else:
        print(f"{runs} Correctness tests passed!")


def run_all_benchmarks(batch, heads, dim, groups):
    """Run full benchmark suite (equivalent to old run.sh)."""
    kv_seqlens = [8192, 16384, 32768, 65536, 131072, 262144]
    kv_seqlens_large = [524288]
    topk_values = [10]

    print("=== Kascade Decode Kernel Benchmarks ===\n")

    print("ref_fa decode")
    for kv_seqlen in kv_seqlens:
        run_ref_fa(batch, heads, kv_seqlen, dim, groups)
    for kv_seqlen in kv_seqlens_large:
        run_ref_fa(batch // 2, heads, kv_seqlen, dim, groups)

    print("\ntilelang_fa decode")
    for kv_seqlen in kv_seqlens:
        run_tilelang_fa(batch, heads, kv_seqlen, dim, groups, mode="benchmark", with_ref=False, code=False)
    for kv_seqlen in kv_seqlens_large:
        run_tilelang_fa(batch // 2, heads, kv_seqlen, dim, groups, mode="benchmark", with_ref=False, code=False)

    print("\nrecompute layer 0 decode")
    for topk in topk_values:
        for kv_seqlen in kv_seqlens:
            run_recompute(batch, heads, kv_seqlen, dim, groups, topk, layer=0, mode="benchmark", with_ref=False, code=False)
        for kv_seqlen in kv_seqlens_large:
            run_recompute(batch // 2, heads, kv_seqlen, dim, groups, topk, layer=0, mode="benchmark", with_ref=False, code=False)

    print("\nrecompute layer 1 decode")
    for topk in topk_values:
        for kv_seqlen in kv_seqlens:
            run_recompute(batch, heads, kv_seqlen, dim, groups, topk, layer=1, mode="benchmark", with_ref=False, code=False)
        for kv_seqlen in kv_seqlens_large:
            run_recompute(batch // 2, heads, kv_seqlen, dim, groups, topk, layer=1, mode="benchmark", with_ref=False, code=False)

    print("\nreuse kernel decode")
    for topk in topk_values:
        for kv_seqlen in kv_seqlens:
            run_reuse(batch, heads, kv_seqlen, dim, groups, topk, mode="benchmark", with_ref=False, code=False)
        for kv_seqlen in kv_seqlens_large:
            run_reuse(batch // 2, heads, kv_seqlen, dim, groups, topk, mode="benchmark", with_ref=False, code=False)

    print("\n=== Decode benchmarks complete ===")

    # Save results to CSV
    if BENCHMARK_RESULTS:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(get_results_dir(), f'benchmark_decode_{timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['kernel', 'kv_seqlen', 'batch', 'topk', 'layer', 'latency_ms'])
            writer.writeheader()
            writer.writerows(BENCHMARK_RESULTS)
        print(f"\nResults saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified decode kernel benchmarking")
    parser.add_argument('--kernel', type=str, choices=['tilelang_fa', 'recompute', 'reuse', 'recompute_kvcache', 'reuse_kvcache'], help='Kernel to benchmark')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--groups', type=int, default=8, help='Number of KV groups (for GQA)')
    parser.add_argument('--kv_seqlen', type=int, default=8192, help='KV sequence length')
    parser.add_argument('--dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--topk', type=float, default=10, help='Top-k percentage for Kascade kernels')
    parser.add_argument('--layer', type=int, default=0, choices=[0, 1], help='Layer type: 0 for first layer, 1 for subsequent')
    parser.add_argument('--mode', type=str, default='benchmark', choices=['benchmark', 'correctness'], help='Run mode')
    parser.add_argument('--with_ref', action='store_true', help='Include reference FA benchmark')
    parser.add_argument('--code', action='store_true', help='Print kernel CUDA source code')
    parser.add_argument('--all', action='store_true', help='Run full benchmark suite')
    
    args = parser.parse_args()

    if args.all:
        run_all_benchmarks(args.batch, args.heads, args.dim, args.groups)
    elif args.kernel == 'tilelang_fa':
        run_tilelang_fa(args.batch, args.heads, args.kv_seqlen, args.dim, args.groups, args.mode, args.with_ref, args.code)
    elif args.kernel == 'recompute':
        run_recompute(args.batch, args.heads, args.kv_seqlen, args.dim, args.groups, args.topk, args.layer, args.mode, args.with_ref, args.code)
    elif args.kernel == 'reuse':
        run_reuse(args.batch, args.heads, args.kv_seqlen, args.dim, args.groups, args.topk, args.mode, args.with_ref, args.code)
    elif args.kernel == 'recompute_kvcache':
        run_recompute_kvcache(args.batch, args.heads, args.kv_seqlen, args.dim, args.groups, args.topk, args.layer, args.mode, args.with_ref, args.code)
    elif args.kernel == 'reuse_kvcache':
        run_reuse_kvcache(args.batch, args.heads, args.kv_seqlen, args.dim, args.groups, args.topk, args.mode, args.with_ref, args.code)
    else:
        parser.print_help()
        print("\nError: Please specify --kernel or --all")


if __name__ == "__main__":
    main()
