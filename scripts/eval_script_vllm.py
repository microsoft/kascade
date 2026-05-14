# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""Evaluation script using vLLM as the inference backend.

Supports three attention backends:
  - default   : vLLM default (FA-3 / FlashAttention)  → maps to 'baseline' strategy
  - tilelang  : TileLang paged decode kernel           → maps to 'baseline' strategy
  - kascade   : Kascade sparse-attention backend       → maps to 'decode_only_kascade' strategy

Usage examples
--------------
# Full-attention baseline (default FA-3)
python scripts/eval_script_vllm.py \\
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --dataset_name hotpotqa/hotpot_qa \\
    --strategies baseline --num_queries 100

# TileLang kernel
python scripts/eval_script_vllm.py \\
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --dataset_name hotpotqa/hotpot_qa \\
    --strategies baseline --vllm_backend tilelang --num_queries 100

# Kascade sparse attention
python scripts/eval_script_vllm.py \\
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --dataset_name hotpotqa/hotpot_qa \\
    --strategies decode_only_kascade \\
    --topk 10 --recompute_layers 0 2 8 13 14 \\
    --num_queries 100
"""

import argparse
import os

# Keep vLLM in-process for deterministic timing.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from transformers import AutoTokenizer, GenerationConfig, set_seed

from kascade.dataset2config import (
    dataset2class,
    dataset2max_batch_size,
    dataset2max_new_tokens,
    dataset2metric,
    dataset2prompt_format,
    dataset2split,
)
from kascade.model_utils import get_inst_tokens, get_eos_token_ids
from kascade.runners import RunConfig, VllmMetricsRunner, VLLM_SUPPORTED_STRATEGIES, resolve_vllm_backend
from kascade.utils import (
    load_eval_dataset,
    resolve_dataset_key_for_eval,
    resolve_subset_label_for_eval,
    should_enable_thinking_tokens_for_dataset,
)


def configure_attention_backend(backend: str) -> dict:
    """Configure attention backend before importing the vLLM runtime."""
    from kascade.vllm_backends import configure_attention_backend as _configure
    return _configure(backend)


def build_sampling_params(generation_config, SamplingParams):
    """Convert a HF GenerationConfig into vLLM SamplingParams."""
    temperature = getattr(generation_config, "temperature", None) or 0.0
    top_p = getattr(generation_config, "top_p", None) or 1.0
    top_k = getattr(generation_config, "top_k", None) or -1
    max_tokens = getattr(generation_config, "max_new_tokens", 32) or 32

    eos_ids = getattr(generation_config, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    elif eos_ids is None:
        eos_ids = []

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop_token_ids=eos_ids,
    )


def apply_request_multiplier(dataset, request_multiplier: int):
    """Repeat the selected dataset deterministically multiplier times."""
    if request_multiplier == 1 or len(dataset) == 0:
        return dataset
    selected_indices = list(range(len(dataset)))
    expanded_indices = selected_indices * request_multiplier
    return dataset.select(expanded_indices)


def main():
    parser = argparse.ArgumentParser(
        "eval_script_vllm.py",
        description="Evaluate QA datasets using vLLM as the inference backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Strategy / backend ──────────────────────────────────────────────────
    parser.add_argument(
        "--strategies",
        nargs="+",
        required=True,
        help=(
            "Attention strategy to evaluate. "
            f"Supported with vLLM: {list(VLLM_SUPPORTED_STRATEGIES.keys())}. "
            "Unsupported strategy names print a message and are skipped."
        ),
    )
    parser.add_argument(
        "--vllm_backend",
        choices=["default", "tilelang", "kascade"],
        default=None,
        help=(
            "Override the vLLM attention backend. "
            "If omitted, the default backend for the chosen strategy is used."
        ),
    )

    # ── Model / dataset ─────────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--subsets", type=str, nargs="+", default=None)
    parser.add_argument("--num_queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_format", type=int, default=-1,
                        help="Prompt format index (-1 uses the dataset default).")

    # ── Generation ──────────────────────────────────────────────────────────
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--stop_strings", type=str, nargs="+", default=[])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--batch_mode",
        type=str,
        choices=["fixed", "all_queries"],
        default="fixed",
        help=(
            "fixed: use --batch_size capped by dataset defaults. "
            "all_queries: submit the full selected query set in one generate call."
        ),
    )
    parser.add_argument(
        "--request_multiplier",
        type=int,
        default=1,
        help="Repeat the selected query set this many times (effective requests = selected_queries * multiplier).",
    )

    # ── Kascade-specific ─────────────────────────────────────────────────────
    parser.add_argument("--topk", type=float, default=10,
                        help="Top-k%% KV positions to attend to (kascade backend only).")
    parser.add_argument("--recompute_layers", type=int, nargs="+", default=None,
                        help="Layer indices that recompute full attention (kascade only).")

    # ── vLLM engine ─────────────────────────────────────────────────────────
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument(
        "--max_llm_batch_size",
        type=int,
        default=256,
        help="Maximum in-flight request batch size inside vLLM (mapped to max_num_seqs).",
    )
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graphs (required for some custom kernels).")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32", "auto"],
                        help="Model weight dtype (default: float16).")

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument("--store_results", action="store_true")

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.request_multiplier < 1:
        parser.error("--request_multiplier must be >= 1")
    if args.max_llm_batch_size < 1:
        parser.error("--max_llm_batch_size must be >= 1")

    # Single-process accelerator — vLLM handles multi-GPU via tensor_parallel_size.
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200))],
    )

    default_split = dataset2split.get(args.dataset_name, "test")
    subsets_to_run = args.subsets if args.subsets else [None]

    for strategy_name in args.strategies:
        set_seed(args.seed + 8)

        # Validate / auto-select backend for this strategy.
        try:
            vllm_backend = resolve_vllm_backend(strategy_name, args.vllm_backend)
        except RuntimeError as e:
            print(f"[UNSUPPORTED] {e}")
            continue

        # Register the vLLM attention backend before importing vLLM runtime.
        attn_kwargs = configure_attention_backend(vllm_backend)

        # Build additional_config for backends that need it.
        additional_config: dict = {}
        if vllm_backend == "kascade":
            additional_config = {
                "topk": args.topk,
                "recompute_layers": args.recompute_layers or [0, 2, 8, 13, 14],
            }

        # Import vLLM here (after backend registration).
        from vllm import LLM, SamplingParams

        # Create one LLM per strategy and reuse it across all subsets.
        llm = None
        try:
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_llm_batch_size,
                enforce_eager=args.enforce_eager,
                dtype=args.dtype,
                enable_prefix_caching=False,
                additional_config=additional_config,
                **attn_kwargs,
            )

            for subset in subsets_to_run:
                dataset_key = resolve_dataset_key_for_eval(args.dataset_name, subset)
                subset_label = resolve_subset_label_for_eval(args.dataset_name, subset, default_split)

                dataset = load_eval_dataset(args.dataset_name, subset, default_split)
                dataset = dataset.shuffle(seed=args.seed + 8)
                selected_queries = min(args.num_queries, len(dataset))
                dataset = dataset.select(range(selected_queries))
                dataset = apply_request_multiplier(dataset, args.request_multiplier)
                effective_num_queries = len(dataset)

                if args.batch_mode == "all_queries":
                    effective_batch_size = max(1, effective_num_queries)
                else:
                    effective_batch_size = min(
                        args.batch_size,
                        dataset2max_batch_size.get(args.dataset_name, args.batch_size),
                    )

                print(
                    f"\nRunning: model={args.model_name}, subset={subset_label}, "
                    f"strategy={strategy_name}, vllm_backend={vllm_backend}, "
                    f"queries={effective_num_queries}, request_multiplier={args.request_multiplier}, "
                    f"batch_mode={args.batch_mode}, batch_size={effective_batch_size}, "
                    f"max_llm_batch_size={args.max_llm_batch_size}"
                )

                # Build generation config and tokenizer for stop-token handling.
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
                generation_config = GenerationConfig()
                generation_config.max_new_tokens = dataset2max_new_tokens[dataset_key]
                generation_config.temperature = args.temperature
                generation_config.top_p = args.top_p
                generation_config.top_k = args.top_k
                extra_eos = get_eos_token_ids(args.stop_strings, tokenizer)
                eos_id = tokenizer.eos_token_id
                base_eos = [eos_id] if isinstance(eos_id, int) else (eos_id or [])
                generation_config.eos_token_id = base_eos + extra_eos

                prompt_format = (
                    dataset2prompt_format[dataset_key]
                    if args.prompt_format == -1
                    else args.prompt_format
                )

                config = RunConfig(
                    max_length=None,
                    batch_size=effective_batch_size,
                    dataset_class=dataset2class[args.dataset_name],
                    prompt_format=prompt_format,
                    num_queries=effective_num_queries,
                    metric=dataset2metric[dataset_key],
                    inst_tokens=get_inst_tokens(
                        args.model_name,
                        enable_thinking=should_enable_thinking_tokens_for_dataset(args.dataset_name),
                    ),
                    dataset_name=args.dataset_name,
                    subset=subset_label,
                    seed=args.seed + 8,
                    run_type="evaluate",
                    store_results=args.store_results,
                    generation_config=generation_config,
                    model_name=args.model_name,
                )

                sampling_params = build_sampling_params(generation_config, SamplingParams)

                # Label used in CSV filenames, e.g. "vllm_kascade_decode_only_kascade".
                strategy_label = f"vllm_{vllm_backend}_{strategy_name}"
                runner = VllmMetricsRunner(
                    llm=llm,
                    sampling_params=sampling_params,
                    strategy_label=strategy_label,
                    strategy=None,
                    model=None,
                    dataset=dataset,
                    config=config,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                )
                runner.run()
                del runner

        finally:
            # Release GPU memory before moving to the next strategy.
            if "runner" in locals():
                del runner
            if llm is not None:
                llm.sleep(level=2)
                engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
                shutdown = getattr(engine_core, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                del llm
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()



if __name__ == "__main__":
    main()
