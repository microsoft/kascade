# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from accelerate import Accelerator
from kascade.model_utils import get_tokenizer_and_model, get_inst_tokens, get_eos_token_ids
from kascade.dataset2config import *
from kascade.strategies import *
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from kascade.runners import MetricsRunner, StatsRunner, RunConfig
from datasets import load_dataset
from transformers import set_seed
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available
import torch

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser("eval_script.py")
    strategy_choices = [
        "baseline",
        "sinked_sliding_window",
        "oracle_topk",
        "oracle_topk_layer0_global",
        "pre_softmax_gqa_pooled_oracle_topk",
        "post_softmax_gqa_pooled_oracle_topk",
        "post_softmax_all_heads_pooled_oracle_topk",
        "pre_softmax_pooled_prefill_topk",
        "post_softmax_pooled_prefill_topk",
        "post_softmax_all_heads_pooled_prefill_topk",
        "kascade",
        "pooled_kascade",
        "decode_only_kascade",
        "efficient_kascade",
        "no_remap_kascade",
        "quest",
        "omni_kv",
        "less_is_more",
    ]
    
    parser.add_argument("--model_name", type=str, required=True, help="model to evaluate")
    parser.add_argument("--strategies", choices=strategy_choices, nargs="+", required=True, help="List of Attention strategies to evaluate")
    parser.add_argument("--subsets", type=str, nargs="+", required=False, default=None, help="List of dataset subsets to evaluate on (optional, uses default split if not provided)")
    parser.add_argument(
        "--run_type",
        type=str,
        choices=["evaluate", "plot_similarity", "plot_sparsity", "select_layers"],
        default="evaluate",
        help="Type of run to execute",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="The dataset to use")
    parser.add_argument("--num_queries", type=int, default=1000, help="The number of queries to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use for the random number generator")
    parser.add_argument("--prompt_format", type=int, default=-1, help="The prompt format to use, -1 uses the default format for the dataset")
    parser.add_argument("--sliding_window", type=int, default=30, help="The sliding window size as percent of ctx length if using SinkedSlidingWindowStrategy")
    parser.add_argument("--num_sink_tokens", type=int, default=4, help="The number of sink tokens if using SinkedSlidingWindowStrategy")
    parser.add_argument("--topk", type=float, default=10, help="The number of top-k tokens to use in TopKStrategy")
    parser.add_argument("--recompute_layers", type=int, nargs="+", default=None, help="A list of layer indices to recompute topk in TopKStrategy, ablations and KascadeStrategy")
    parser.add_argument("--recompute_layers_l", type=int, nargs="+", default=None, help="A list of layer indices to recompute topk in LessIsMoreStrategy")
    parser.add_argument("--recompute_layers_o", type=int, nargs="+", default=None, help="A list of layer indices to recompute topk in OmniKVStrategy")
    parser.add_argument("--tile_size", type=int, default=1, help="The tile size to use in TopKStrategy, ablations and KascadeStrategy")
    parser.add_argument("--tile_size_q", type=int, default=1, help="The tile size to use in QuestStrategy")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling in the generation")
    parser.add_argument("--temperature", type=float, default=None, help="The temperature to use in the generation")
    parser.add_argument("--top_p", type=float, default=None, help="The top_p to use in the generation")
    parser.add_argument("--top_k", type=int, default=None, help="The top_k to use in the generation")
    parser.add_argument("--stop_strings", type=str, nargs="+", default=[], help="A list of strings to stop generation on")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size to use")
    parser.add_argument("--rolling_prefill", action="store_true", help="Whether to use rolling prefill in ReuseTopKHeadsStrategy")
    parser.add_argument("--lim_ratio_factor", type=float, default=0.25, help="The ratio factor for LLM in LessIsMoreStrategy")
    parser.add_argument("--debug", action="store_true", help="Raise generation errors instead of skipping them")
    parser.add_argument("--store_results", action="store_true", help="Store results")
    parser.add_argument("--use_precomputed_stats", action="store_true", help="Use precomputed statistics")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="no", kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200))])

    default_attn_impl = "flash_attention_3" if is_flash_attn_3_available() else ("flash_attention_2" if is_flash_attn_2_available() else "sdpa")
    # Load model and tokenizer once per strategy
    if args.use_precomputed_stats:
        model, tokenizer = None, None
    else:
        model, tokenizer = get_tokenizer_and_model(
            args.model_name,
            default_attn_impl,
            accelerator.device,
        )

    # Get default split for this dataset
    default_split = dataset2split.get(args.dataset_name, "test")
    
    # If no subsets provided, use a placeholder (will load default split)
    subsets_to_run = args.subsets if args.subsets else [None]

    # Loop: models -> strategies -> subsets
    for strategy_name in args.strategies:
        if strategy_name == "efficient_kascade" and model.config.torch_dtype != torch.float16:
            raise ValueError("Efficient Kascade strategy requires model to be in float16 precision. Please go to line 17 in src/model_utils.py and change torch_dtype=torch.float16 when loading the model for running with efficient_kascade.")

        set_seed(args.seed) # Ensure reproducibility per run
        
        # Create strategy
        strategy2class = {
            "baseline": lambda: BaselineStrategy(),
            "sinked_sliding_window": lambda: SinkedSlidingWindowStrategy(sliding_window=args.sliding_window, num_sink_tokens=args.num_sink_tokens),
            "oracle_topk": lambda: OracleTopkStrategy(k=args.topk),
            "oracle_topk_layer0_global": lambda: OracleTopkLayer0GlobalStrategy(k=args.topk),
            "pre_softmax_gqa_pooled_oracle_topk": lambda: PreSoftmaxGQAPooledOracleTopKStrategy(k=args.topk),
            "post_softmax_gqa_pooled_oracle_topk": lambda: PostSoftmaxGQAPooledOracleTopKStrategy(k=args.topk),
            "post_softmax_all_heads_pooled_oracle_topk": lambda: PostSoftmaxAllHeadsPooledOracleTopKStrategy(k=args.topk),
            "pre_softmax_pooled_prefill_topk": lambda: PreSoftmaxPooledPrefillTopkStrategy(k=args.topk, tile_size=args.tile_size, rolling_prefill=args.rolling_prefill),
            "post_softmax_pooled_prefill_topk": lambda: PostSoftmaxPooledPrefillTopkStrategy(k=args.topk, tile_size=args.tile_size, rolling_prefill=args.rolling_prefill),
            "post_softmax_all_heads_pooled_prefill_topk": lambda: PostSoftmaxAllHeadsPooledPrefillTopkStrategy(k=args.topk, tile_size=args.tile_size, rolling_prefill=args.rolling_prefill),
            "kascade": lambda: KascadeStrategy(recompute_layers=args.recompute_layers, model_name=args.model_name, k=args.topk, tile_size=args.tile_size, rolling_prefill=args.rolling_prefill),
            "pooled_kascade": lambda: PooledKascadeStrategy(recompute_layers=args.recompute_layers, k=args.topk, tile_size=args.tile_size, rolling_prefill=args.rolling_prefill),
            "decode_only_kascade": lambda: DecodeOnlyKascadeStrategy(recompute_layers=args.recompute_layers, model_name=args.model_name, k=args.topk),
            "no_remap_kascade": lambda: NoRemapKascadeStrategy(recompute_layers=args.recompute_layers, k=args.topk, tile_size=args.tile_size, rolling_prefill=args.rolling_prefill),
            "efficient_kascade": lambda: EfficientKascadeStrategy(recompute_layers=args.recompute_layers, model_name=args.model_name, k=args.topk, tile_size=args.tile_size, rolling_prefill=args.rolling_prefill),
            "quest": lambda: QuestStrategy(tile_size=args.tile_size_q, k=args.topk),
            "omni_kv": lambda: OmniKVStrategy(k=args.topk, recompute_layers=args.recompute_layers_o),
            "less_is_more": lambda: LessIsMoreStrategy(k=args.topk, recompute_layers=args.recompute_layers_l, lim_ratio_factor=args.lim_ratio_factor, num_sink_tokens=args.num_sink_tokens),
        }
        
        strategy: Strategy = strategy2class[strategy_name]()
        if strategy_name != "baseline":
            model.config._attn_implementation = strategy.name

        
        for subset in subsets_to_run:
            # Determine dataset key for config lookups
            if subset is not None:
                dataset_key = subset if args.dataset_name == "THUDM/LongBench" else args.dataset_name
            else:
                dataset_key = args.dataset_name
            
            dataset = load_dataset(args.dataset_name, subset, split=default_split, trust_remote_code=True, cache_dir="/dev/shm")
            
            dataset = dataset.shuffle(seed=args.seed)
            num_queries = min(args.num_queries, len(dataset))
            dataset = dataset.select(range(num_queries))

            accelerator.print(f"\nRunning: model={args.model_name}, subset={subset if subset else 'default'}, strategy={strategy_name}")

            # Setup generation config
            if args.use_precomputed_stats:
                generation_config = None
            else:
                generation_config = model.generation_config
                generation_config.do_sample = args.do_sample
                existing = ([generation_config.eos_token_id] 
                            if isinstance(generation_config.eos_token_id, int) 
                            else (generation_config.eos_token_id or []))
                new_eos = get_eos_token_ids(args.stop_strings, tokenizer)
                generation_config.eos_token_id = existing + new_eos
                generation_config.pad_token_id = tokenizer.pad_token_id
                generation_config.max_new_tokens = dataset2max_new_tokens[dataset_key]
                generation_config.top_p = args.top_p
                generation_config.top_k = args.top_k
                generation_config.temperature = args.temperature
            
            # Create run config
            config = RunConfig(
                max_length=None,
                batch_size=min(args.batch_size, dataset2max_batch_size.get(args.dataset_name, args.batch_size)),
                dataset_class=dataset2class[args.dataset_name],
                prompt_format=dataset2prompt_format[dataset_key] if args.prompt_format == -1 else args.prompt_format,
                num_queries=num_queries,
                metric=dataset2metric[dataset_key],
                inst_tokens=get_inst_tokens(args.model_name, enable_thinking=True if any(d in dataset_key.lower() for d in ["aime", "math"]) else False),
                dataset_name=args.dataset_name,
                subset=subset if subset else default_split,
                seed=args.seed,
                run_type=args.run_type,
                debug=args.debug,
                store_results=args.store_results,
                generation_config=generation_config,
                use_precomputed_stats=args.use_precomputed_stats,
                model_name=args.model_name
            )
            
            # Create and run runner
            run_type2runner = {
                "evaluate": MetricsRunner,
                "plot_similarity": StatsRunner,
                "plot_sparsity": StatsRunner,
                "select_layers": StatsRunner,
            }
            
            runner_cls = run_type2runner[config.run_type]
            runner = runner_cls(
                strategy,
                model,
                dataset,
                config,
                tokenizer,
                accelerator,
            )
            
            if config.run_type != "evaluate":
                if not hasattr(strategy, "attach_stats_runner"):
                    raise ValueError(f"Strategy {strategy.name} does not support run_type '{config.run_type}'.")
                strategy.attach_stats_runner(runner)
            
            runner.run()

    accelerator.end_training()

if __name__ == "__main__":
    main()
