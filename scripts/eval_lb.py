#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
import csv
from collections import defaultdict

TOPK = 10  # Fixed TopK for LongBench evaluations

# LongBench datasets
longbench_datasets = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
    "hotpotqa", "2wikimqa", "musique", "dureader", 
    "gov_report", "qmsum", "multi_news", "vcsum", 
    "trec", "triviaqa", "samsum", "lsht", 
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", 
    "lcc", "repobench-p"
]

# Class mappings for averaging
dataset_classes = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5]
classes = ["Single-Doc QA", "Multi-Doc QA", "Summarization", "Fewshot", "Synthetic", "Code"]

# Dataset -> num_queries mapping
dataset_num_queries = {
    "lcc": 500,
    "repobench-p": 500,
    "multifieldqa_en": 150,
}

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
                "--recompute_layers", "0", "2", "7", "13", "17", "23"
            ]},
            {"name": "decode_only_kascade", "args": [
                "--recompute_layers", "0", "2", "7", "13", "17", "23"
            ]},
            {"name": "pooled_kascade", "args": [
                "--tile_size", "32",
                "--rolling_prefill",
                "--recompute_layers", "0", "2", "7", "13", "17", "23"
            ]},
        ]
    },
    {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
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
                "--recompute_layers", "0", "2", "4", "8", "13", "16"
            ]},
            {"name": "decode_only_kascade", "args": [
                "--recompute_layers", "0", "2", "4", "8", "13", "16"
            ]},
            {"name": "pooled_kascade", "args": [
                "--tile_size", "32",
                "--rolling_prefill",
                "--recompute_layers", "0", "2", "4", "8", "13", "16"
            ]},
        ]
    },
]


def run_evaluation(model_config):
    """Run evaluation for a model across all LongBench subsets and strategies"""
    model_name = model_config["name"]
    strategies = model_config["strategies"]
    
    # Build base command
    base_cmd = ["accelerate", "launch", "./scripts/eval_script.py"]
    
    # Add model
    base_cmd.extend(["--model_name", model_name])
    
    # Add dataset
    base_cmd.extend(["--dataset_name", "THUDM/LongBench"])
    
    # Add all subsets
    base_cmd.extend(["--subsets"] + longbench_datasets)
    
    # Add all strategy names
    strategy_names = [s["name"] for s in strategies]
    base_cmd.extend(["--strategies"] + strategy_names)
    
    # Calculate num_queries (use max for simplicity)
    max_num_queries = max(dataset_num_queries.get(d, 200) for d in longbench_datasets)
    base_cmd.extend(["--num_queries", str(max_num_queries)])
    base_cmd.extend(["--topk", str(TOPK)])
    
    # Enable result storage
    base_cmd.append("--store_results")
    
    # Add strategy-specific arguments (assumes they apply globally for this model)
    # Note: This collects all unique args from all strategies
    # If strategies have conflicting args, you may need to run separately
    all_args = []
    for strategy in strategies:
        all_args.extend(strategy["args"])
    
    base_cmd.extend(all_args)
    
    print(f"Running evaluation for model: {model_name}")
    print(f"Command: {' '.join(base_cmd)}")
    
    subprocess.run(base_cmd)


def calculate_class_averages(model_name):
    """Calculate per-class averages from stored CSV results"""
    
    # For each strategy, collect results across all datasets
    # strategy_results[strategy_name][class_idx] = [list of scores]
    strategy_results = defaultdict(lambda: defaultdict(list))
    
    # Find all strategy CSV files
    results_dir = f"./results/evals/{model_name.split('/')[-1]}/LongBench"
    
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return
    
    for csv_file in os.listdir(results_dir):
        if not csv_file.endswith(".csv"):
            continue
        
        strategy_name = csv_file.replace(".csv", "")
        csv_path = os.path.join(results_dir, csv_file)
        
        # Read the CSV (no header, positional indices)
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 4:
                        continue
                    
                    # First entry is subset name
                    subset = row[0]
                    
                    # Fourth last entry is the metric score
                    metric_score = float(row[-4])*100
                    
                    # Find the class for this subset
                    if subset in longbench_datasets:
                        dataset_idx = longbench_datasets.index(subset)
                        dataset_class = dataset_classes[dataset_idx]
                        
                        # Add score to this class for this strategy
                        strategy_results[strategy_name][dataset_class].append(metric_score)
                    
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
    
    # Calculate averages per class for each strategy
    summary_results = {}
    for strategy_name, class_scores in strategy_results.items():
        summary_results[strategy_name] = {}
        
        for class_idx in range(len(classes)):
            if class_idx in class_scores and class_scores[class_idx]:
                avg_score = sum(class_scores[class_idx]) / len(class_scores[class_idx])
                summary_results[strategy_name][classes[class_idx]] = avg_score
            else:
                summary_results[strategy_name][classes[class_idx]] = 0.0
    
    # Write summary to CSV
    summary_dir = f"./results/summary/{model_name.split('/')[-1]}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "LongBench.csv")
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["strategy"] + classes)
        
        # Write each strategy's class averages
        for strategy_name in sorted(summary_results.keys()):
            row = [strategy_name]
            for class_name in classes:
                row.append(f"{summary_results[strategy_name][class_name]:.4f}")
            writer.writerow(row)
    
    print(f"Summary saved to: {summary_path}")


def main():
    """Main evaluation loop"""
    # Run evaluations for all models
    for model_config in MODELS:
        run_evaluation(model_config)
    
    # Calculate and store class-averaged results for each model
    for model_config in MODELS:
        calculate_class_averages(model_config["name"])


if __name__ == "__main__":
    main()
