#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
import csv
from collections import defaultdict

# AIME configuration
DATASET_NAME = "HuggingFaceH4/aime_2024"
NUM_RUNS = 8  # Number of runs to average over (AIME can have variance)
NUM_QUERIES = 30  # All AIME questions
TOP_P = 0.95
TEMPERATURE = 0.6
TOPK = 10

# Model configurations with model-specific strategy parameters
MODELS = [
    {
        "name": "Qwen/Qwen3-8B",
        "strategies": [
            {"name": "baseline", "args": []},
            {"name": "sinked_sliding_window", "args": []},
            {"name": "less_is_more", "args": [
                    "--recompute_layers_l", "2", "12",
                    "--lim_ratio_factor", "0.25",
                ]
            },
            {"name": "quest", "args": [
                "--tile_size_q", "16",
            ]},
            {"name": "kascade", "args": [
                "--tile_size", "32",
                "--rolling_prefill",
                "--recompute_layers", "0", "2", "7", "14", "23",
            ]},
            {"name": "decode_only_kascade", "args": [
                "--recompute_layers", "0", "2", "7", "14", "23",
            ]},  
            {"name": "pooled_kascade", "args": [
                "--tile_size", "32",
                "--rolling_prefill",
                "--recompute_layers", "0", "2", "7", "14", "23",
            ]},     
        ]
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "strategies": [
            {"name": "baseline", "args": []},
            {"name": "sinked_sliding_window", "args": []},
            {"name": "less_is_more", "args": [
                    "--recompute_layers_l", "2", "13",
                    "--lim_ratio_factor", "0.25",
                ]
            },
            {"name": "omni_kv", "args": [
                    "--recompute_layers_o", "2", "8", "18",
                ]
            },
            {"name": "quest", "args": [
                "--tile_size_q", "16",
            ]},
            {"name": "kascade", "args": [
                "--tile_size", "32",
                "--rolling_prefill",
                "--recompute_layers", "0", "2", "8", "13", "14",
            ]},
            {"name": "decode_only_kascade", "args": [
                "--recompute_layers", "0", "2", "8", "13", "14",
            ]},
            {"name": "pooled_kascade", "args": [
                "--tile_size", "32",
                "--rolling_prefill",
                "--recompute_layers", "0", "2", "8", "13", "14",
            ]},
        ]
    },
    
]


def run_evaluation(model_config, run_idx):
    """Run evaluation for a model across all strategies for one run"""
    model_name = model_config["name"]
    strategies = model_config["strategies"]
    
    # Build base command
    base_cmd = ["accelerate", "launch", "./scripts/eval_script.py"]
    
    # Add model
    base_cmd.extend(["--model_name", model_name])
    
    # Add dataset
    base_cmd.extend(["--dataset_name", DATASET_NAME])
    
    # Add all strategy names
    strategy_names = [s["name"] for s in strategies]
    base_cmd.extend(["--strategies"] + strategy_names)
    
    # Add num_queries
    base_cmd.extend(["--num_queries", str(NUM_QUERIES)])
    base_cmd.extend(["--topk", str(TOPK)])
    base_cmd.extend(["--seed", str(run_idx)])
    base_cmd.extend(["--do_sample"])
    base_cmd.extend(["--top_p", str(TOP_P)])
    base_cmd.extend(["--temperature", str(TEMPERATURE)])
    if model_name == "Qwen/Qwen3-8B":
        base_cmd.extend(["--top_k", "20"])
    
    # Enable result storage
    base_cmd.append("--store_results")
    
    # Add strategy-specific arguments (collects all unique args)
    all_args = []
    for strategy in strategies:
        all_args.extend(strategy["args"])
    
    base_cmd.extend(all_args)
    
    print(f"\n{'='*80}")
    print(f"Run {run_idx + 1}/{NUM_RUNS} for model: {model_name}")
    print(f"Command: {' '.join(base_cmd)}")
    print(f"{'='*80}\n")
    
    subprocess.run(base_cmd)


def calculate_run_averages(model_name):
    """Calculate averages across runs from stored CSV results"""
    
    # For each strategy, collect results across all runs
    strategy_results = defaultdict(lambda: {'score': [], 'decode_len': []})
    
    # Find all strategy CSV files
    results_dir = f"./results/evals/{model_name.split('/')[-1]}/{DATASET_NAME.split('/')[-1]}"
    
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return
    
    for csv_file in os.listdir(results_dir):
        if not csv_file.endswith(".csv"):
            continue
        
        strategy_name = csv_file.replace(".csv", "")
        csv_path = os.path.join(results_dir, csv_file)
        
        # Read all rows from the CSV (no header, positional indices)
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 4:
                        continue
                    
                    # Last 4 entries are: score, prefill_len, decode_len, runtime
                    score = float(row[-4])*100
                    decode_len = float(row[-2])
                    
                    strategy_results[strategy_name]['score'].append(score)
                    strategy_results[strategy_name]['decode_len'].append(decode_len)
                    
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
    
    # Calculate averages for each strategy
    summary_results = {}
    for strategy_name, metrics in strategy_results.items():
        summary_results[strategy_name] = {
            'score': sum(metrics['score']) / len(metrics['score']) if metrics['score'] else 0.0,
            'decode_len': sum(metrics['decode_len']) / len(metrics['decode_len']) if metrics['decode_len'] else 0.0,
        }
    
    # Write summary to CSV
    summary_dir = f"./results/summary/{model_name.split('/')[-1]}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "AIME24.csv")
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["strategy", "score", "decode_len"])
        
        # Write each strategy's averages
        for strategy_name in sorted(summary_results.keys()):
            metrics = summary_results[strategy_name]
            writer.writerow([
                strategy_name,
                f"{metrics['score']:.4f}",
                f"{metrics['decode_len']:.2f}",
            ])
    
    print(f"\nSummary saved to: {summary_path}")

def main():
    """Main evaluation loop"""
    # Run evaluations for all models across multiple runs
    for model_config in MODELS:
        for run_idx in range(NUM_RUNS):
            run_evaluation(model_config, run_idx)
    
    # Calculate and store run-averaged results for each model
    for model_config in MODELS:
        calculate_run_averages(model_config["name"])


if __name__ == "__main__":
    main()
