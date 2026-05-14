# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Benchmark script for measuring e2e latency of prefill and decode in vLLM.

Usage:
    # Prefill benchmark
    python scripts/benchmark_vllm.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
                                     --batch_size 8 \
                                     --seq_len 4096 \
                                     --type prefill
    
    # Decode benchmark (uses DecodeBenchConnector to fill KV cache)
    python scripts/benchmark_vllm.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
                                     --batch_size 64 \
                                     --seq_len 2048 \
                                     --type decode

    # Multi-decode benchmark (start_len prompt, fixed decode steps)
    python scripts/benchmark_vllm.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
                                     --batch_size 64 \
                                     --start_len 2048 \
                                     --end_len 128 \
                                     --type decode_multi \
                                     --decode_repeats 3
"""

import os
import argparse
import csv
# Disable multiprocessing to ensure timing happens in the same process as execution
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time
import uuid
import gc
import re
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple, Set, Any

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    model_name: str
    batch_size: int
    seq_len: int  # Sequence length: prefill input length or decode KV cache length
    benchmark_type: str  # "prefill", "decode", or "decode_multi"
    warmup_iters: int
    num_iters: int
    tensor_parallel_size: int
    dtype: str
    gpu_memory_utilization: float
    enforce_eager: bool = False
    max_num_batched_tokens: int = 8192
    num_layers: Optional[int] = None  # Limit model to first N layersomde
    backend: str = "kascade"
    sweep_style: str = "fixed_batch"
    block_size: int = 64
    profile: bool = False
    start_len: Optional[int] = None  # Prompt length for decode_multi
    end_len: Optional[int] = None  # Decode steps for decode_multi


@dataclass
class BenchmarkResult:
    latency_ms: float
    std_ms: float
    throughput_tokens_per_sec: float
    batch_size: int
    seq_len: int
    benchmark_type: str
    gpu_ms: Optional[float] = None
    gpu_std_ms: Optional[float] = None


def get_sweep_results_csv_path(config: BenchmarkConfig) -> str:
    """Return CSV output path under results/perf/<model_basename>/ for sweep benchmarks."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    perf_dir = os.path.join(project_root, "results", "perf")
    model_basename = config.model_name.split("/")[-1].strip().replace(" ", "_")
    model_dir = os.path.join(perf_dir, model_basename)
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{config.sweep_style}_{config.benchmark_type}_{config.backend}.csv"
    return os.path.join(model_dir, filename)


def save_sweep_results_to_csv(config: BenchmarkConfig, results: List[BenchmarkResult]) -> str:
    """Save sweep benchmark results to CSV and return the saved path."""
    output_path = get_sweep_results_csv_path(config)
    with open(output_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["batch_size", "seq_len", "block_size", "latency(ms)", "std_dev", "throughput(tokens/s)"])
        for result in results:
            writer.writerow([
                result.batch_size,
                result.seq_len,
                config.block_size,
                result.latency_ms,
                result.std_ms,
                result.throughput_tokens_per_sec,
            ])

    return output_path


def configure_attention_backend(backend: str) -> dict:
    """Configure attention backend before importing vLLM runtime modules."""
    from kascade.vllm_backends import configure_attention_backend as _configure
    return _configure(backend)


def monkey_patch_model_loader():
    """
    Monkey-patch DefaultModelLoader.load_weights to filter out layers that exceed
    the model's num_hidden_layers config. This is model-agnostic and works for any
    architecture that uses 'layers.N.' naming convention (Llama, Mistral, Qwen, etc.).
    """
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except ImportError:
        print("Could not import DefaultModelLoader for patching. Skipping.")
        return

    if hasattr(DefaultModelLoader, "_original_load_weights"):
        return

    print("Patching DefaultModelLoader.load_weights to handle layer limiting...")
    DefaultModelLoader._original_load_weights = DefaultModelLoader.load_weights

    def patched_load_weights(self, model, model_config):
        num_layers = getattr(model_config.hf_config, 'num_hidden_layers', None)
        
        if num_layers is None:
            return DefaultModelLoader._original_load_weights(self, model, model_config)
        
        # Wrap model.load_weights to filter weights
        original_model_load_weights = model.load_weights
        
        def filtered_load_weights(weights_iter, *args, **kwargs):
            def filter_weights(weights):
                for name, tensor in weights:
                    # Match layer indices in weight names like "layers.N." or "model.layers.N."
                    match = re.search(r"(?:^|[./])layers\.(\d+)\.", name)
                    if match:
                        layer_idx = int(match.group(1))
                        if layer_idx >= num_layers:
                            # Skip weights for layers beyond num_hidden_layers
                            continue
                    yield name, tensor
            
            return original_model_load_weights(filter_weights(weights_iter), *args, **kwargs)
        
        # Temporarily replace model.load_weights
        model.load_weights = filtered_load_weights
        try:
            result = DefaultModelLoader._original_load_weights(self, model, model_config)
        finally:
            # Restore original method
            model.load_weights = original_model_load_weights
        
        return result

    DefaultModelLoader.load_weights = patched_load_weights


def monkey_patch_decode_bench_connector():
    """
    Monkey-patch DecodeBenchConnectorWorker._fill_blocks to use in-place operations.
    This prevents OOM by avoiding the creation of a massive temporary tensor when
    filling the KV cache.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import DecodeBenchConnectorWorker
    except ImportError:
        print("Could not import DecodeBenchConnectorWorker for patching. Skipping.")
        return

    print("Patching DecodeBenchConnectorWorker._fill_blocks for memory efficiency...")

    def optimized_fill_blocks(self, group_idx: int, block_ids: list[int], num_tokens: int):
        if not block_ids:
            return

        assert self.kv_caches is not None
        assert self.group_to_layers is not None

        # Get the layers that belong to this group
        layer_names = self.group_to_layers.get(group_idx, [])

        for layer_name in layer_names:
            if layer_name not in self.kv_caches:
                continue

            kv_cache = self.kv_caches[layer_name]

            # Convert block_ids to tensor on device
            block_ids_tensor = torch.tensor(
                block_ids, dtype=torch.long, device=kv_cache.device
            )

            # Filter invalid block IDs
            valid_mask = block_ids_tensor < kv_cache.shape[0]
            valid_block_ids = block_ids_tensor[valid_mask]

            if len(valid_block_ids) == 0:
                continue

            with torch.no_grad():
                if self.fill_std == 0:
                    # In-place constant fill - no temporary allocation
                    kv_cache.index_fill_(0, valid_block_ids, self.fill_mean)
                else:
                    # In-place random fill - iterate over blocks
                    # kv_cache[block_id] returns a view, .normal_() fills in-place
                    for block_id in valid_block_ids:
                        kv_cache[block_id].normal_(mean=self.fill_mean, std=self.fill_std)

    DecodeBenchConnectorWorker._fill_blocks = optimized_fill_blocks


def get_context_lengths(max_model_len: int, max_tokens: int, start_len: int = 1024) -> List[int]:
    """
    Generate context lengths starting from start_len, doubling each time,
    until we reach or exceed max_model_len - max_tokens.
    The last context length should be the largest power of 2 that fits,
    or max_model_len - max_tokens - buffer if that's smaller.
    """
    # We need room for max_tokens generation plus some buffer
    buffer = 100
    max_ctx = max_model_len - max_tokens - buffer
    
    context_lengths = []
    ctx = start_len
    
    while ctx < max_ctx:
        context_lengths.append(ctx)
        ctx *= 2
    
    # Add the last context length that fits (if different from last added)
    if context_lengths and context_lengths[-1] < max_ctx:
        # Check if max_ctx is significantly larger than last power of 2
        last_power_of_2 = context_lengths[-1]
        if max_ctx > last_power_of_2:
            # Add max_ctx as the final context length
            context_lengths.append(max_ctx)
    
    return context_lengths


def get_fixed_token_sweep(
    seq_lengths: List[int],
    start_batch_size: int = 256,
    min_batch_size: int = 2,
) -> List[Tuple[int, int]]:
    """
    Build (batch_size, seq_len) pairs for fixed-token all-eval style.
    Batch size starts at start_batch_size for the first seq_len and halves each step,
    floored at min_batch_size. This naturally includes final capped seq_len values.
    """
    sweep = []
    batch_size = start_batch_size
    for seq_len in seq_lengths:
        sweep.append((batch_size, seq_len))
        batch_size = max(min_batch_size, batch_size // 2)
    return sweep


def generate_dummy_prompt_token_ids(batch_size: int, ctx_length: int) -> List[dict]:
    """
    Generate dummy prompts as token IDs directly (like vLLM's latency.py).
    Uses random token IDs to avoid any tokenization overhead.
    """
    dummy_prompt_token_ids = np.random.randint(10, 10000, size=(batch_size, ctx_length))
    return [{"prompt_token_ids": batch.tolist()} for batch in dummy_prompt_token_ids]


def _get_generated_lengths(request_outputs: Iterable[Any]) -> List[int]:
    lengths: List[int] = []
    for request_output in request_outputs:
        if request_output.outputs:
            lengths.append(len(request_output.outputs[0].token_ids))
        else:
            lengths.append(0)
    return lengths


def benchmark_prefill(
    llm: Any,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """
    Benchmark prefill latency using llm_engine.step().
    """
    from vllm import SamplingParams

    dummy_prompts = generate_dummy_prompt_token_ids(config.batch_size, config.seq_len)
    
    # For prefill, we just want to process the prompt.
    # max_tokens=1 ensures we stop after 1 decode step, but we only measure the first step (prefill).
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=True,
    )
    
    e2e_samples = []
    gpu_samples = []
    
    print(f"Warming up with {config.warmup_iters} iterations...")
    for i in tqdm(range(config.warmup_iters), desc="Warmup"):
        # Add requests to the engine
        for j, prompt in enumerate(dummy_prompts):
            req_id = f"warmup_{i}_{j}"
            llm.llm_engine.add_request(
                req_id,
                prompt,
                sampling_params,
            )
        
        # Step until all requests are done
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()
            
    print(f"Running {config.num_iters} benchmark iterations...")
    print(f"  Batch size: {config.batch_size}, Sequence length: {config.seq_len}")
    
    for i in tqdm(range(config.num_iters), desc="Benchmark"):
        for j, prompt in enumerate(dummy_prompts):
            req_id = f"bench_{i}_{j}"
            llm.llm_engine.add_request(
                req_id,
                prompt,
                sampling_params,
            )
            
        # Measure the first step (prefill)
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        wall_t0 = time.perf_counter()
        
        # This single step should process the prefill for all requests 
        # (assuming batch size fits in max_num_batched_tokens)
        llm.llm_engine.step()
        
        end_evt.record()
        torch.cuda.synchronize()
        wall_t1 = time.perf_counter()
        
        e2e_samples.append((wall_t1 - wall_t0) * 1000.0)
        gpu_samples.append(float(start_evt.elapsed_time(end_evt)))
        
        # Clean up any remaining steps (decode step)
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()
    
    e2e = np.array(e2e_samples)
    gpu = np.array(gpu_samples)
    
    print(f"\\nPrefill latency (batch_size={config.batch_size}, seq_len={config.seq_len}):")
    print(f"  E2E: mean={e2e.mean():.3f}ms  std={e2e.std():.3f}ms  p50={np.percentile(e2e, 50):.3f}ms  p90={np.percentile(e2e, 90):.3f}ms  p99={np.percentile(e2e, 99):.3f}ms")
    print(f"  GPU: mean={gpu.mean():.3f}ms  std={gpu.std():.3f}ms  p50={np.percentile(gpu, 50):.3f}ms  p90={np.percentile(gpu, 90):.3f}ms  p99={np.percentile(gpu, 99):.3f}ms")
    
    total_tokens = config.batch_size * config.seq_len
    throughput = total_tokens / (e2e.mean() / 1000.0)
    
    return BenchmarkResult(
        latency_ms=e2e.mean(),
        std_ms=e2e.std(),
        throughput_tokens_per_sec=throughput,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        benchmark_type="prefill",
        gpu_ms=gpu.mean(),
        gpu_std_ms=gpu.std(),
    )


def benchmark_decode(
    llm: Any,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """
    Benchmark decode latency using llm_engine.step().
    
    Strategy:
    1. Warmup: Run warmup_iters decode cycles to warm up CUDA graphs
    2. Measurement: For each iteration:
       - Add requests (KV cache gets filled by DecodeBenchConnector)
       - Step 1: prefill step (all requests enter decode phase)
       - Step 2: actual decode step (measure this)
       - Abort and drain
    3. Average the measurements
    """
    from vllm import SamplingParams

    
    prompt_len = config.seq_len
    
    # Need more than 2 tokens to avoid "final step" overhead in measurement
    # Prefill generates token 1, decode generates token 2, then we abort
    max_tokens = 3
    
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=max_tokens, ignore_eos=True
    )
    
    latencies = []
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Running benchmark for batch_size={config.batch_size}, seq_len={config.seq_len}...")
    
    # ========== WARMUP PHASE ==========
    if config.warmup_iters > 0:
        print(f"  Warming up with {config.warmup_iters} cycles...")
        for warmup_i in range(config.warmup_iters):
            warmup_prompts = generate_dummy_prompt_token_ids(config.batch_size, prompt_len)
            warmup_req_ids = []
            for j, prompt in enumerate(warmup_prompts):
                req_id = f"warmup_{warmup_i}_{j}"
                warmup_req_ids.append(req_id)
                llm.llm_engine.add_request(req_id, prompt=prompt, params=sampling_params)
            
            # Run until we get one decode step
            steps_done = 0
            while llm.llm_engine.has_unfinished_requests():
                request_outputs = llm.llm_engine.step()
                if len(request_outputs) == config.batch_size:
                    steps_done += 1
                    if steps_done >= 2:  # prefill + one decode
                        break
            
            # Abort and drain
            for req_id in warmup_req_ids:
                try:
                    llm.llm_engine.abort_request(req_id)
                except:
                    pass
            while llm.llm_engine.has_unfinished_requests():
                llm.llm_engine.step()
    
    # ========== MEASUREMENT PHASE ==========
    print(f"  Measuring {config.num_iters} iterations...")
    if config.profile:
        llm.start_profile()
    for iter_i in range(config.num_iters):
        # Add fresh requests
        dummy_prompts = generate_dummy_prompt_token_ids(config.batch_size, prompt_len)
        req_ids = []
        for j, prompt in enumerate(dummy_prompts):
            req_id = f"bench_{iter_i}_{j}"
            req_ids.append(req_id)
            llm.llm_engine.add_request(req_id, prompt=prompt, params=sampling_params)
        
        # Step 1: prefill (skip timing)
        while True:
            request_outputs = llm.llm_engine.step()
            if len(request_outputs) == config.batch_size:
                break
        
        # Step 2: decode (measure this)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        request_outputs = llm.llm_engine.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        if len(request_outputs) == config.batch_size:
            latencies.append((t1 - t0) * 1000.0)
        
        # Abort and drain
        for req_id in req_ids:
            try:
                llm.llm_engine.abort_request(req_id)
            except:
                pass
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()

    if config.profile:
        llm.stop_profile()
    e2e = np.array(latencies)
    
    # Print results
    print(f"\nDecode step latency (batch_size={config.batch_size}, seq_len={config.seq_len}):")
    print(f"  Iterations: {len(latencies)}")
    if len(e2e) > 0:
        print(f"  E2E: mean={e2e.mean():.3f}ms  std={e2e.std():.3f}ms  p50={np.percentile(e2e, 50):.3f}ms  p99={np.percentile(e2e, 99):.3f}ms")
        throughput = config.batch_size / (e2e.mean() / 1000.0)
    else:
        print("  No valid decode steps recorded.")
        throughput = 0.0

    return BenchmarkResult(
        latency_ms=e2e.mean() if len(e2e) > 0 else 0.0,
        std_ms=e2e.std() if len(e2e) > 0 else 0.0,
        throughput_tokens_per_sec=throughput,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        benchmark_type="decode",
        gpu_ms=None,
        gpu_std_ms=None,
    )


def benchmark_decode_multi(
    llm: Any,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """
    Benchmark end-to-end batch latency for multiple decode steps.

    Each iteration:
      1. Add batch requests with prompt length `start_len`.
      2. Decode exactly `end_len` steps per request (`ignore_eos=True`).
      3. Measure total wall time of the full request batch.
    """
    from vllm import SamplingParams

    if config.start_len is None or config.end_len is None:
        raise ValueError("decode_multi requires both start_len and end_len in BenchmarkConfig")
    if config.end_len < 1:
        raise ValueError("decode_multi requires end_len >= 1")

    prompt_len = config.start_len
    decode_steps = config.end_len
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=decode_steps,
        ignore_eos=True,
    )

    latencies = []

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"Running multi-decode benchmark for batch_size={config.batch_size}, "
        f"start_len={prompt_len}, decode_steps={decode_steps}..."
    )

    if config.warmup_iters > 0:
        print(f"  Warming up with {config.warmup_iters} iterations...")
        for warmup_i in tqdm(range(config.warmup_iters), desc="Warmup"):
            warmup_prompts = generate_dummy_prompt_token_ids(config.batch_size, prompt_len)
            warmup_outputs = llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
            if len(warmup_outputs) != config.batch_size:
                raise RuntimeError(
                    f"Warmup iteration {warmup_i}: expected {config.batch_size} outputs, got {len(warmup_outputs)}"
                )
            warmup_lengths = _get_generated_lengths(warmup_outputs)
            if any(length != decode_steps for length in warmup_lengths):
                raise RuntimeError(
                    f"Warmup iteration {warmup_i}: expected {decode_steps} decode steps per request, "
                    f"got lengths={warmup_lengths[:8]}"
                )

    print(f"  Measuring {config.num_iters} iterations...")
    if config.profile:
        llm.start_profile()

    for iter_i in tqdm(range(config.num_iters), desc="Benchmark"):
        prompts = generate_dummy_prompt_token_ids(config.batch_size, prompt_len)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        request_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if len(request_outputs) != config.batch_size:
            raise RuntimeError(
                f"Iteration {iter_i}: expected {config.batch_size} outputs, got {len(request_outputs)}"
            )

        generated_lengths = _get_generated_lengths(request_outputs)
        if any(length != decode_steps for length in generated_lengths):
            bad = [(idx, length) for idx, length in enumerate(generated_lengths) if length != decode_steps][:8]
            raise RuntimeError(
                f"Iteration {iter_i}: some requests did not decode exactly {decode_steps} steps. "
                f"examples={bad}"
            )

        latencies.append((t1 - t0) * 1000.0)

    if config.profile:
        llm.stop_profile()

    e2e = np.array(latencies)
    print(
        f"\nMulti-decode latency (batch_size={config.batch_size}, "
        f"start_len={prompt_len}, decode_steps={decode_steps}):"
    )
    print(f"  Iterations: {len(latencies)}")
    if len(e2e) > 0:
        print(
            f"  E2E: mean={e2e.mean():.3f}ms  std={e2e.std():.3f}ms  "
            f"p50={np.percentile(e2e, 50):.3f}ms  p99={np.percentile(e2e, 99):.3f}ms"
        )
        throughput = (config.batch_size * decode_steps) / (e2e.mean() / 1000.0)
    else:
        print("  No valid multi-decode iterations recorded.")
        throughput = 0.0

    return BenchmarkResult(
        latency_ms=e2e.mean() if len(e2e) > 0 else 0.0,
        std_ms=e2e.std() if len(e2e) > 0 else 0.0,
        throughput_tokens_per_sec=throughput,
        batch_size=config.batch_size,
        seq_len=prompt_len,
        benchmark_type="decode_multi",
        gpu_ms=None,
        gpu_std_ms=None,
    )


def run_single_benchmark(llm: Any, config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark with the given configuration."""
    print(f"\n{'-'*60}")
    if config.benchmark_type == "decode_multi":
        print(
            "Running benchmark for "
            f"start_len={config.start_len}, decode_steps={config.end_len}"
        )
    else:
        print(f"Running benchmark for seq_len={config.seq_len}")
    print(f"{'-'*60}")
    
    # Run the appropriate benchmark
    if config.benchmark_type == "prefill":
        result = benchmark_prefill(llm, config)
    elif config.benchmark_type == "decode":
        result = benchmark_decode(llm, config)
    else:
        result = benchmark_decode_multi(llm, config)
    
    # Print results
    if config.benchmark_type == "decode_multi":
        print(f"\nResults for start_len={config.start_len}, decode_steps={config.end_len}:")
    else:
        print(f"\nResults for seq_len={config.seq_len}:")
    print(f"  Average Latency: {result.latency_ms:.2f} ms ± {result.std_ms:.2f} ms")
    if result.gpu_ms is not None:
        print(f"  GPU Latency: {result.gpu_ms:.2f} ms ± {result.gpu_std_ms:.2f} ms")
    print(f"  Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
    
    return result


def run_benchmark(config: BenchmarkConfig, mode: str = "single", attn_kwargs: dict = None) -> List[BenchmarkResult]:
    """Run the benchmark with the given configuration."""
    from vllm import LLM
    from vllm.config import KVTransferConfig, ProfilerConfig

    print(f"\n{'='*60}")
    print(f"Benchmark Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Mode: {mode}")
    if mode == "single":
        if config.benchmark_type == "decode_multi":
            print(f"  Start Length: {config.start_len}")
            print(f"  Decode Steps: {config.end_len}")
        else:
            print(f"  Sequence Length: {config.seq_len}")
    print(f"  Type: {config.benchmark_type}")
    print(f"  Warmup Iterations: {config.warmup_iters}")
    print(f"  Benchmark Iterations: {config.num_iters}")
    print(f"  Tensor Parallel Size: {config.tensor_parallel_size}")
    print(f"  Dtype: {config.dtype}")
    print(f"  Backend: {config.backend}")
    print(f"  Block Size: {config.block_size}")
    if mode == "sweep":
        print(f"  Sweep Style: {config.sweep_style}")
    print(f"  Enforce Eager: {config.enforce_eager}")
    if config.num_layers is not None:
        print(f"  Num Layers: {config.num_layers}")
    print(f"{'='*60}\n")
    
    # Apply monkey patch if needed (always apply it to be safe if num_layers is used)
    if config.num_layers is not None:
        monkey_patch_model_loader()

    # Apply DecodeBenchConnector patch for memory efficiency (use in-place ops)
    if config.benchmark_type == "decode":
        monkey_patch_decode_bench_connector()

    if config.benchmark_type == "decode_multi" and mode == "sweep":
        raise ValueError("decode_multi is only supported with --mode single")

    # Prepare arguments for LLM
    hf_overrides = {}
    if config.num_layers is not None:
        hf_overrides["num_hidden_layers"] = config.num_layers
    
    kv_transfer_config = None
    if config.benchmark_type == "decode":
        # Use fill_std=0 for constant fill (uses in-place index_fill_, no extra memory)
        kv_transfer_config = KVTransferConfig(
            kv_connector="DecodeBenchConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"fill_mean": 0.015, "fill_std": 0.1},
        )

    # Determine sequence lengths / sweep pairs
    if mode == "sweep":
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(config.model_name)
        max_model_len = getattr(model_config, 'max_position_embeddings', None)
        if max_model_len is None:
            max_model_len = getattr(model_config, 'n_positions', None)
        if max_model_len is None:
            max_model_len = getattr(model_config, 'seq_length', 131072)
        
        print(f"Model max length: {max_model_len}")
        seq_lengths = get_context_lengths(max_model_len, 128)
        print(f"Sequence lengths to benchmark: {seq_lengths}")
        if config.sweep_style == "fixed_tokens":
            sweep_pairs = get_fixed_token_sweep(seq_lengths)
            print(f"Fixed-token sweep (batch_size, seq_len): {sweep_pairs}")
        else:
            sweep_pairs = [(config.batch_size, seq_len) for seq_len in seq_lengths]
        max_seq_needed = max(seq_lengths) + 128
    else:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(config.model_name)
        max_model_len = getattr(model_config, 'max_position_embeddings', None)
        if max_model_len is None:
            max_model_len = getattr(model_config, 'n_positions', None)
        if max_model_len is None:
            max_model_len = getattr(model_config, 'seq_length', None)
        if config.benchmark_type == "decode_multi":
            if config.start_len is None or config.end_len is None:
                raise ValueError("decode_multi requires start_len and end_len")
            seq_lengths = [config.start_len]
            sweep_pairs = [(config.batch_size, config.start_len)]
            if max_model_len is not None and (config.start_len + config.end_len) > max_model_len:
                raise ValueError(
                    f"start_len + end_len ({config.start_len + config.end_len}) exceeds model max length ({max_model_len})."
                )
        else:
            seq_lengths = [config.seq_len]
            sweep_pairs = [(config.batch_size, config.seq_len)]
        max_seq_needed = max_model_len

    max_batch_in_sweep = max(batch_size for batch_size, _ in sweep_pairs)
    
    # Calculate max_num_batched_tokens
    max_num_batched_tokens = config.max_num_batched_tokens
    if config.benchmark_type == "prefill":
            max_tokens_across_sweep = max(batch_size * seq_len for batch_size, seq_len in sweep_pairs)
            max_num_batched_tokens = max(max_num_batched_tokens, max_tokens_across_sweep)

    print("Loading vLLM model...")
    print(f"  max_num_batched_tokens: {max_num_batched_tokens}")
    print(f"  max_num_seqs: {max_batch_in_sweep * 2}")
    
    # For decode benchmark with few layers, increase gpu_memory_utilization
    # to have enough KV cache blocks for all requests.
    # The OOM issue was from max_num_batched_tokens, not gpu_memory_utilization.
    gpu_mem_util = config.gpu_memory_utilization
    from vllm.config import AttentionConfig
    
    llm = LLM(
        model=config.model_name,
        tensor_parallel_size=config.tensor_parallel_size,
        dtype=config.dtype,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_seq_needed,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=False,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_batch_in_sweep,
        kv_transfer_config=kv_transfer_config,
        hf_overrides=hf_overrides if hf_overrides else None,
        block_size=config.block_size,
        profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir="./vllm_profile", torch_profiler_with_stack=True, torch_profiler_record_shapes=True) if config.profile else None,
        **(attn_kwargs or {}),
    )
    
    try:
        cache_config = llm.llm_engine.vllm_config.cache_config
        print(f"  KV cache block_size: {cache_config.block_size}")
        print(f"  num_gpu_blocks: {cache_config.num_gpu_blocks}")
        tokens_capacity = cache_config.num_gpu_blocks * cache_config.block_size
        if config.benchmark_type == "decode_multi":
            max_tokens_needed = max(
                batch_size * (seq_len + (config.end_len or 0))
                for batch_size, seq_len in sweep_pairs
            )
            effective_max_seq_len = max(seq_lengths) + (config.end_len or 0)
        else:
            max_tokens_needed = max(batch_size * seq_len for batch_size, seq_len in sweep_pairs)
            effective_max_seq_len = max(seq_lengths)
        blocks_needed = max_tokens_needed // cache_config.block_size + max_batch_in_sweep
        print(f"  Total token capacity: {tokens_capacity:,} tokens")
        print(f"  Tokens needed for full batch: {max_tokens_needed:,} tokens")
        print(f"  Blocks needed: {blocks_needed:,} blocks")
        print(f"  Blocks available: {cache_config.num_gpu_blocks:,} blocks")
        if cache_config.num_gpu_blocks < blocks_needed:
            print(f"  WARNING: Not enough blocks! Need {blocks_needed - cache_config.num_gpu_blocks:,} more blocks")
            print(
                f"  Max requests that can run: "
                f"{cache_config.num_gpu_blocks // (effective_max_seq_len // cache_config.block_size + 1)}"
            )
    except Exception as e:
        import traceback
        print(f"  Could not get KV cache info: {e}")
        traceback.print_exc()
    
    results = []
    
    for batch_size, seq_len in sweep_pairs:
        current_config = BenchmarkConfig(
            model_name=config.model_name,
            batch_size=batch_size,
            seq_len=seq_len,
            benchmark_type=config.benchmark_type,
            warmup_iters=config.warmup_iters,
            num_iters=config.num_iters,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_num_batched_tokens=config.max_num_batched_tokens,
            num_layers=config.num_layers,
            backend=config.backend,
            sweep_style=config.sweep_style,
            block_size=config.block_size,
            profile=config.profile,
            start_len=config.start_len,
            end_len=config.end_len,
        )
        
        result = run_single_benchmark(llm, current_config)
        results.append(result)
    
    if mode == "sweep":
        print(f"\n{'='*60}")
        print(f"Summary of All Benchmarks ({config.benchmark_type}):")
        print(f"Backend={config.backend}, sweep_style={config.sweep_style}, block_size={config.block_size}")
        print(f"{'='*60}")
        print(f"{'Batch':>8} | {'Seq Length':>12} | {'Latency (ms)':>15} | {'Std (ms)':>10} | {'Throughput (tok/s)':>18}")
        print(f"{'-'*60}")
        for result in results:
            print(f"{result.batch_size:>8} | {result.seq_len:>12} | {result.latency_ms:>15.2f} | {result.std_ms:>10.2f} | {result.throughput_tokens_per_sec:>18.2f}")
        print(f"{'='*60}\n")

        csv_path = save_sweep_results_to_csv(config, results)
        print(f"Sweep CSV saved to: {csv_path}")
    else:
        result = results[0]
        print(f"\n{'='*60}")
        print(f"Benchmark Results ({config.benchmark_type}):")
        print(f"  Average Latency: {result.latency_ms:.2f} ms ± {result.std_ms:.2f} ms")
        if result.gpu_ms is not None:
            print(f"  GPU Latency: {result.gpu_ms:.2f} ms ± {result.gpu_std_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
        print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark e2e latency of prefill and decode in vLLM"
    )    
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Sequence length: input length for prefill, KV cache length for decode",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["prefill", "decode", "decode_multi"],
        default="prefill",
        help="Type of benchmark: prefill, decode, or decode_multi",
    )
    parser.add_argument(
        "--start_len",
        type=int,
        default=None,
        help="For decode_multi: prompt length to start from. Defaults to --seq_len if omitted.",
    )
    parser.add_argument(
        "--end_len",
        type=int,
        default=None,
        help="For decode_multi: number of decode steps each request must generate.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "sweep"],
        default="single",
        help="Mode: 'single' runs one seq_len, 'sweep' runs from 1024 doubling to model max len with fixed batch or fixed tokens",
    )
    parser.add_argument(
        "--sweep_style",
        type=str,
        choices=["fixed_batch", "fixed_tokens"],
        default="fixed_batch",
        help="For mode=sweep: fixed_batch keeps batch fixed, fixed_tokens halves batch as seq_len doubles",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=3,
        help="Number of warmup decode steps before measurement (for CUDA graph capture)",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=5,
        help="Number of benchmark iterations (decode steps to measure)",
    )
    parser.add_argument(
        "--decode_repeats",
        type=int,
        default=3,
        help="For decode_multi: number of repeated end-to-end runs to measure.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=8192,
        help="Max batched tokens (default 8192 for decode, auto-computed for prefill)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for distributed inference",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Data type for model weights",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "tilelang", "kascade"],
        default="kascade",
        help="Attention backend to use",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        choices=[16, 32, 64, 128],
        default=64,
        help="KV cache block size",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graphs (use eager mode). Enable this to compare with vllm bench latency.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Limit model to first N layers (useful for testing large batch/seq on smaller GPU)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch profiler and save traces to ./vllm_profile/",
    )
    
    args = parser.parse_args()

    start_len = None
    end_len = None
    num_iters = args.num_iters
    seq_len = args.seq_len

    if args.type == "decode_multi":
        start_len = args.start_len if args.start_len is not None else args.seq_len
        end_len = args.end_len
        if start_len < 1:
            parser.error("--start_len must be >= 1 for --type decode_multi")
        if end_len is None:
            parser.error("--end_len is required for --type decode_multi")
        if end_len < 1:
            parser.error("--end_len must be >= 1 for --type decode_multi")
        seq_len = start_len
        num_iters = args.decode_repeats
     
    config = BenchmarkConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        seq_len=seq_len,
        benchmark_type=args.type,
        warmup_iters=args.warmup_iters,
        num_iters=num_iters,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_layers=args.num_layers,
        backend=args.backend,
        sweep_style=args.sweep_style,
        block_size=args.block_size,
        profile=args.profile,
        start_len=start_len,
        end_len=end_len,
    )

    attn_kwargs = configure_attention_backend(args.backend)
    
    results = run_benchmark(config, mode=args.mode, attn_kwargs=attn_kwargs)
    
    return results


if __name__ == "__main__":
    main()
