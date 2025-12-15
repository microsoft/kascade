#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
import csv
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') # Ignore the broken VS Code/SSH screen connection
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Configuration parameters
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME_MUSIQUE = "bdsaglam/musique"
DATASET_NAME_2WIKI = "framolfese/2WikiMultihopQA"
SUBSET_MUSIQUE = "answerable"
SUBSET_2WIKI = "default"
NUM_QUERIES = 1000

# Ablation 1: Oracle TopK comparison
ABLATION1_TOPK_VALUES = [0.25, 0.5, 1, 2.5, 5, 7.5, 10]
ABLATION1_STRATEGIES = ["oracle_topk", "oracle_topk_layer0_global"]

# Ablation 2: Tile size variation
ABLATION2_TILE_SIZES = [1, 16, 32, 64]
ABLATION2_TOPK = 10
ABLATION2_STRATEGIES = ["pre_softmax_pooled_prefill_topk", "post_softmax_pooled_prefill_topk"]

# Ablation 3: Head mapping variation
ABLATION3_TOPK_VALUES = [2.5, 5, 7.5, 10]
ABLATION3_TILE_SIZE = 32
ABLATION3_STRATEGIES = ["kascade", "pooled_kascade", "no_remap_kascade"]

# Color and marker configurations
strategy_colors = {
    "oracle_topk": "blue",
    "oracle_topk_layer0_global": "orange",
    "pre_softmax_pooled_prefill_topk": "blue",
    "post_softmax_pooled_prefill_topk": "green",
    "kascade": "green",
    "pooled_kascade": "red", 
    "no_remap_kascade": "blue",
}

strategy_labels = {
    "oracle_topk": "Oracle TopK",
    "oracle_topk_layer0_global": "Oracle TopK (Full Attn in Layer 0)",
    "pre_softmax_pooled_prefill_topk": "Pre-Softmax",
    "post_softmax_pooled_prefill_topk": "Post-Softmax",
    "kascade": "Kascade",
    "pooled_kascade": "Kascade (All Heads Pooled)",
    "no_remap_kascade": "Kascade (No Head Remapping)",
}

strategy_markers = {
    "oracle_topk": "o",
    "oracle_topk_layer0_global": "s",
    "pre_softmax_pooled_prefill_topk": "o",
    "post_softmax_pooled_prefill_topk": "s",
    "kascade": "s",
    "pooled_kascade": "^",
    "no_remap_kascade": "o",
}


def run_evaluation(model, dataset, subset, num_queries, strategies, topk=None, tile_size=None):
    """Run evaluation with specified parameters"""
    
    # Build command
    cmd = ["accelerate", "launch", "./scripts/eval_script.py"]
    
    # Add strategies
    cmd.extend(["--strategies"] + strategies)
    
    # Add model
    cmd.extend(["--model_name", model])
    
    # Add dataset and subset
    cmd.extend(["--dataset_name", dataset])
    cmd.extend(["--subsets", subset])
    
    # Add num_queries
    cmd.extend(["--num_queries", str(num_queries)])
    
    # Add topk value if specified
    if topk is not None:
        cmd.extend(["--topk", str(topk)])
    
    # Add tile_size if specified
    if tile_size is not None:
        cmd.extend(["--tile_size", str(tile_size)])
        cmd.append("--rolling_prefill")
        cmd.extend(["--recompute_layers", "0", "2", "8", "11", "13", "14"])
    
    # Enable result storage
    cmd.append("--store_results")
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {strategies}, topk: {topk}, tile_size: {tile_size}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    subprocess.run(cmd)


def parse_results(model, dataset, strategies):
    """Parse CSV results for specified strategies"""
    
    # Determine results directory
    model_short = model.split('/')[-1]
    dataset_short = dataset.split('/')[-1] if '/' in dataset else dataset
    results_dir = f"./results/evals/{model_short}/{dataset_short}"
    
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return {}
    
    # Dictionary to store results: {strategy: {(topk, tile_size): score}}
    strategy_results = {strategy: {} for strategy in strategies}
    
    # Read CSV files for each strategy
    for strategy in strategies:
        csv_path = os.path.join(results_dir, f"{strategy}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            continue
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 4:
                        continue
                    
                    # First entry is subset name
                    subset_name = row[0]
                    
                    # Third entry is topk percentage
                    topk = float(row[2])
                    
                    # Fourth entry is tile_size (if present)
                    tile_size = int(row[3]) if len(row) > 3 and row[3].isdigit() else None
                    
                    # Fourth-last entry is the score
                    score = float(row[-4]) * 100
                    
                    # Store result with key as (topk, tile_size)
                    key = (topk, tile_size)
                    strategy_results[strategy][key] = score
                    
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
    
    return strategy_results


def plot_ablation1_oracle_topk(strategy_results, dataset_name):
    """Plot F1 score vs TopK for oracle strategies"""
    
    # Ensure plots directory exists
    os.makedirs("./results/plots", exist_ok=True)

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Baseline score (adjust as needed)
    baseline = [50.23] * len(ABLATION1_TOPK_VALUES)
    plt.plot(ABLATION1_TOPK_VALUES, baseline, linestyle='--', color='gray', label='Baseline', linewidth=2)
    
    # Plot each strategy
    for strategy in ABLATION1_STRATEGIES:
        if strategy not in strategy_results:
            continue
        
        topk_values = []
        scores = []
        
        # Extract data for plotting
        for (topk, tile_size), score in strategy_results[strategy].items():
            if tile_size is None:  # Oracle strategies don't use tile_size
                topk_values.append(topk)
                scores.append(score)
        
        if topk_values:
            # Sort by topk for proper line plotting
            sorted_data = sorted(zip(topk_values, scores))
            topk_values, scores = zip(*sorted_data)
            
            plt.plot(
                topk_values,
                scores,
                marker=strategy_markers[strategy],
                color=strategy_colors[strategy],
                label=strategy_labels[strategy],
                linewidth=2,
                markersize=8
            )
    
    # Configure plot
    max_score = max([max(strategy_results[s].values()) for s in ABLATION1_STRATEGIES if s in strategy_results and strategy_results[s]], default=60)
    plt.yticks(np.arange(0, max_score + 10, 5))
    plt.xlabel('TopK Percentage', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    dataset_short = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
    output_path = f"./results/plots/ablation1_oracle_topk_{dataset_short}"
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight', format='png')
    print(f"\nAblation 1 plot saved to: {output_path}")
    
    plt.close()


def plot_ablation2_tile_size(strategy_results):
    """Plot F1 score vs Tile Size for pooled strategies"""
    
    # Ensure plots directory exists
    os.makedirs("./results/plots", exist_ok=True)

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot each strategy
    for strategy in ABLATION2_STRATEGIES:
        if strategy not in strategy_results:
            continue
        
        tile_sizes = []
        scores = []
        
        # Extract data for the fixed topk
        for (topk, tile_size), score in strategy_results[strategy].items():
            if topk == ABLATION2_TOPK and tile_size is not None:
                tile_sizes.append(tile_size * 4)  # Multiply by 4 as requested
                scores.append(score)
        
        if tile_sizes:
            # Sort by tile_size for proper line plotting
            sorted_data = sorted(zip(tile_sizes, scores))
            tile_sizes, scores = zip(*sorted_data)
            
            plt.plot(
                tile_sizes,
                scores,
                marker=strategy_markers[strategy],
                color=strategy_colors[strategy],
                label=strategy_labels[strategy],
                linewidth=2,
                markersize=8
            )
    
    # Configure plot
    max_score = max([max(strategy_results[s].values()) for s in ABLATION2_STRATEGIES if s in strategy_results and strategy_results[s]], default=40)
    plt.yticks(np.arange(0, max_score + 5, 5))
    plt.xlabel('Tile Size', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = "./results/plots/ablation2_tile_size"
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight', format='png')
    print(f"\nAblation 2 plot saved to: {output_path}")
    
    plt.close()


def plot_ablation3_head_mapping(strategy_results):
    """Plot F1 score vs TopK for head mapping strategies"""
    
    # Ensure plots directory exists
    os.makedirs("./results/plots", exist_ok=True)

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot each strategy
    for strategy in ABLATION3_STRATEGIES:
        if strategy not in strategy_results:
            continue
        
        topk_values = []
        scores = []
        
        # Extract data for the fixed tile size
        for (topk, tile_size), score in strategy_results[strategy].items():
            if tile_size == ABLATION3_TILE_SIZE:
                topk_values.append(topk)
                scores.append(score)
        
        if topk_values:
            # Sort by topk for proper line plotting
            sorted_data = sorted(zip(topk_values, scores))
            topk_values, scores = zip(*sorted_data)
            
            plt.plot(
                topk_values,
                scores,
                marker=strategy_markers[strategy],
                color=strategy_colors[strategy],
                label=strategy_labels[strategy],
                linewidth=2,
                markersize=8,
                alpha=0.7
            )
    
    # Configure plot
    max_score = max([max(strategy_results[s].values()) for s in ABLATION3_STRATEGIES if s in strategy_results and strategy_results[s]], default=40)
    plt.yticks(np.arange(0, max_score + 5, 5))
    plt.xlabel('TopK Percentage', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.legend(fontsize=14, loc='lower right', ncol=1)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = "./results/plots/ablation3_head_mapping"
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight', format='png')
    print(f"\nAblation 3 plot saved to: {output_path}")
    
    plt.close()


def run_ablation1():
    """Run ablation 1: Oracle TopK comparison on 2wiki"""
    print(f"Starting Ablation 1: Oracle TopK comparison")
    print(f"Dataset: {DATASET_NAME_2WIKI}")
    print(f"TopK Values: {ABLATION1_TOPK_VALUES}")
    print(f"Strategies: {ABLATION1_STRATEGIES}")
    
    # Run evaluations for all topk values
    for topk in ABLATION1_TOPK_VALUES:
        run_evaluation(MODEL, DATASET_NAME_2WIKI, SUBSET_2WIKI, NUM_QUERIES, ABLATION1_STRATEGIES, topk=topk)
    
    # Parse and plot results
    strategy_results = parse_results(MODEL, DATASET_NAME_2WIKI, ABLATION1_STRATEGIES)
    if any(strategy_results.values()):
        plot_ablation1_oracle_topk(strategy_results, DATASET_NAME_2WIKI)
        print("Ablation 1 complete!")
    else:
        print("Warning: No results found for Ablation 1.")


def run_ablation2():
    """Run ablation 2: Tile size variation on musique"""
    print(f"Starting Ablation 2: Tile size variation")
    print(f"Dataset: {DATASET_NAME_MUSIQUE}")
    print(f"Tile Sizes: {ABLATION2_TILE_SIZES}")
    print(f"TopK: {ABLATION2_TOPK}")
    print(f"Strategies: {ABLATION2_STRATEGIES}")
    
    # Run evaluations for all tile sizes
    for tile_size in ABLATION2_TILE_SIZES:
        run_evaluation(MODEL, DATASET_NAME_MUSIQUE, SUBSET_MUSIQUE, NUM_QUERIES, ABLATION2_STRATEGIES, 
                      topk=ABLATION2_TOPK, tile_size=tile_size)
    
    # Parse and plot results
    strategy_results = parse_results(MODEL, DATASET_NAME_MUSIQUE, ABLATION2_STRATEGIES)
    if any(strategy_results.values()):
        plot_ablation2_tile_size(strategy_results)
        print("Ablation 2 complete!")
    else:
        print("Warning: No results found for Ablation 2.")


def run_ablation3():
    """Run ablation 3: Head mapping variation on musique"""
    print(f"Starting Ablation 3: Head mapping variation")
    print(f"Dataset: {DATASET_NAME_MUSIQUE}")
    print(f"TopK Values: {ABLATION3_TOPK_VALUES}")
    print(f"Tile Size: {ABLATION3_TILE_SIZE}")
    print(f"Strategies: {ABLATION3_STRATEGIES}")
    
    # Run evaluations for all combinations
    for topk in ABLATION3_TOPK_VALUES:
        run_evaluation(MODEL, DATASET_NAME_MUSIQUE, SUBSET_MUSIQUE, NUM_QUERIES, ABLATION3_STRATEGIES,
                      topk=topk, tile_size=ABLATION3_TILE_SIZE)
    
    # Parse and plot results
    strategy_results = parse_results(MODEL, DATASET_NAME_MUSIQUE, ABLATION3_STRATEGIES)
    if any(strategy_results.values()):
        plot_ablation3_head_mapping(strategy_results)
        print("Ablation 3 complete!")
    else:
        print("Warning: No results found for Ablation 3.")


def plot_only_ablation1():
    """Only plot ablation 1 results"""
    strategy_results = parse_results(MODEL, DATASET_NAME_2WIKI, ABLATION1_STRATEGIES)
    if any(strategy_results.values()):
        plot_ablation1_oracle_topk(strategy_results, DATASET_NAME_2WIKI)
        print("Ablation 1 plotting complete!")
    else:
        print("Warning: No results found for Ablation 1.")


def plot_only_ablation2():
    """Only plot ablation 2 results"""
    strategy_results = parse_results(MODEL, DATASET_NAME_MUSIQUE, ABLATION2_STRATEGIES)
    if any(strategy_results.values()):
        plot_ablation2_tile_size(strategy_results)
        print("Ablation 2 plotting complete!")
    else:
        print("Warning: No results found for Ablation 2.")


def plot_only_ablation3():
    """Only plot ablation 3 results"""
    strategy_results = parse_results(MODEL, DATASET_NAME_MUSIQUE, ABLATION3_STRATEGIES)
    if any(strategy_results.values()):
        plot_ablation3_head_mapping(strategy_results)
        print("Ablation 3 plotting complete!")
    else:
        print("Warning: No results found for Ablation 3.")


def main():
    parser = argparse.ArgumentParser(description='Run unified ablation studies')
    parser.add_argument('--ablation', type=int, choices=[1, 2, 3], 
                       help='Which ablation to run (1: Oracle TopK, 2: Tile Size, 3: Head Mapping)')
    parser.add_argument('--all', action='store_true',
                       help='Run all ablations')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot results without running evaluations')
    
    args = parser.parse_args()
    
    if args.plot_only:
        if args.all or args.ablation is None:
            print("Running all plots...")
            plot_only_ablation1()
            plot_only_ablation2()
            plot_only_ablation3()
        elif args.ablation == 1:
            plot_only_ablation1()
        elif args.ablation == 2:
            plot_only_ablation2()
        elif args.ablation == 3:
            plot_only_ablation3()
    else:
        if args.all or args.ablation is None:
            print("Running all ablations...")
            # run_ablation1()
            run_ablation2()
            run_ablation3()
        elif args.ablation == 1:
            run_ablation1()
        elif args.ablation == 2:
            run_ablation2()
        elif args.ablation == 3:
            run_ablation3()


if __name__ == "__main__":
    main()