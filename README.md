<p align="center">
  <picture>
    <img alt="Kascade" src="assets/logo_kascade.png" height="20%" width="20%">
  </picture>
</p>

<h3 align="center">
 A Practical Sparse Attention Method for Long-Context LLM Inference
</h3>

<p align="center">
| <a href="#"><b>Paper</b></a> |
</p>

## Abstract

Attention is the dominant source of latency during long-context LLM inference, an increasingly popular workload with reasoning models and RAG. We propose Kascade, a training-free sparse attention method that leverages known observations such as 1) post-softmax attention is intrinsically sparse, and 2) the identity of high-weight keys is stable across nearby layers. Kascade computes exact Top-*k* indices in a small set of **anchor** layers, then reuses those indices in intermediate **reuse** layers. The **anchor** layers are selected algorithmically, via a dynamic-programming objective that maximizes cross-layer similarity over a development set, allowing easy deployment across models. The method incorporates efficient implementation constraints (e.g. tile-level operations), across both prefill and decode attention. The Top-*k* selection and reuse in Kascade is _head_-aware and we show in our experiments that this is critical for high accuracy. Kascade achieves up to $4.1 \times$ speedup in decode attention and $2.2 \times$ speedup in prefill attention over FlashAttention-3 baseline on H100 GPUs while closely matching dense attention accuracy on long-context benchmarks such as LongBench and AIME-24.

Below are speedup results for Top-*k* set to 10%:
<p align="center">
  <picture>
    <img alt="results_perf" src="assets/perf.png" width="98%" height="98%">
  </picture>
</p>

## Quick Start

### Prerequisites

- NVIDIA GPU (For running efficient Kascade kernels, need H100 or hopper architecture GPU. For running only accuracy evals using unoptimized PyTorch code other GPUs should be fine)
- CUDA 12.8+
- Conda

### Installation

```bash
# 1. Create conda environment
conda create -n kascade python=3.12.11
conda activate kascade

# 2. One-command install (builds all dependencies)
./install.sh
```
If you want to use models on HuggingFace that have gated access (Llama models) then create a token by following instructions at https://huggingface.co/docs/hub/security-tokens.
A token with read scope should be enough just for using the models. After running the below command paste the token.

```bash
huggingface-cli login
```

## Running Evaluations and Performance Benchmarks

Before running Kascade evals on a particulal model, one needs to choose the number of anchor layers and generate the best set of anchor layers and head mappings for that model.

```bash
python scripts/eval_script.py --model_name [MODEL_NAME] --dataset_name bdsaglam/musique --subsets answerable --num_queries 1000 --strategies post_softmax_pooled_prefill_topk --tile_size 32 --run_type select_layers 
```

This will print the best anchor layers for no. of anchor layers from 3 to 8. The head_mappings will be stored in `results/head_mappings` folder in `.npy` format. Each file is labeled as `<model_name>_<anchor layers separated by _>.npy`.
You can choose which mapping to use based on how many anchor layers you want. To use the mapping, pass the set of anchor layers to the evaluation scripts (see below). 

For `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct` and `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` which were used for evaluations in the paper, the best head mappings are already stored at `./results/head_mappings` in `.npy` format. 
The name of the file also gives the set of layers corresponding to the head_mapping.

Below is the template for running evaluation on one of the supported datasets using a given model and a set of strategies. In addition to Kascade, we have provided implementations of various other sparse attention strategies, used for accuracy comparisons.
```bash
python scripts/eval_script.py --model_name [MODEL_NAME] --dataset_name [DATASET_NAME] --subsets [SUBSET_IN_DATASET] --num_queries [NUM_QUERIES] --strategies [strategy-name-1] [strategy-name-2] ... [strategy-name-n] --strategy-specific-args --store_results{OPTIONAL}
```
Example for Kascade:
```bash
python scripts/eval_script.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name bdsaglam/musique --subsets answerable --strategies efficient_kascade --tile_size 32 --recompute_layers 0 2 8 13 14 --rolling_prefill
```

**NOTE:** Currently for `efficient_kascade` which uses our efficient kernels only tile_size 32 is supported.

**NOTE**: To use multiple GPUs you can run `scripts/eval_script.py` with `accelerate launch` and take advantage of DDP for faster processing of queries. If you run into errors, fallback to single gpu runs.

The above template command will print the results in the format below, with one row per specified strategy:
```bash
# model_name, dataset_name, subset, seed, num_queries, prompt_template, strategy-args(like name, tile_size, recompute_layers, etc), accuracy metric, avg. prefill tokens, avg. decode tokens, wall clock time
meta-llama/Meta-Llama-3.1-8B-Instruct,bdsaglam/musique,answerable,0,100,0,efficient_kascade,10,32,False,[0, 2, 8, 13, 14],0.37323,2304.06,6.96,21.101
```

 If `--store_results` is given then the result will also be stored in `./results/evals/model_name/strategy_name` folder in a `.csv` file.  

### [Longbench](https://huggingface.co/datasets/zai-org/LongBench) Evaluations
<p align="center">
  <picture>
    <img alt="results_longbench" src="assets/longbench_results.png" width="98%" height="98%">
  </picture>
</p>

To generate accuracy results for Longbench, from Table 1 in the paper, run

```bash
python eval_lb.py
```

This runs a given set of strategies and models on all Longbench datasets, averages the results across the different types of subsets in Longbench, and stores them. 

### [AIME-24](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) Evaluations
<p align="center">
  <picture>
    <img alt="results_laime" src="assets/results_aime.png" width="60%" height="60%">
  </picture>
</p>

To generate accuracy results for AIME-24, from Table 2 in the paper, run

```bash
python eval_aime.py
```

This runs a given set of strategies and models on AIME-24 dataset. For every model and strategy pair, it does `NUM_RUNS`(default 8) independent runs and averages the scores. The sampling params are set to values recommended by the official huggingface model pages.

To run different models and strategies, update the `MODELS` list. The averaged results are stored at `./results/summary`, and individual results are stored at `./results/evals`.

### Kernel Benchmarks (H100 or hopper architecture needed)

```bash
# Full benchmark suites
python scripts/benchmark_prefill.py --all
python scripts/benchmark_decode.py --all

# Individual kernel benchmarks
python scripts/benchmark_prefill.py --kernel recompute --seq_len 8192 --topk 10 --layer 0 --mode benchmark

python scripts/benchmark_decode.py --kernel recompute --kv_seqlen 8192 --topk 10 --layer 0 --mode benchmark

# Correctness tests
python scripts/benchmark_prefill.py --kernel recompute --mode correctness
python scripts/benchmark_decode.py --kernel reuse --mode correctness
```
**NOTE:** Currently only certain configs (like tile_size=32) are supported for the kernels. More flexibility may be added in future. Support for paged and varlen kernels will be added soon!

## Reproduciblity

Commands below can be used to reproduce the various figures and ablations in the paper.

### Figure 1: Attention Sparsity per Layer per Head
```bash
python scripts/eval_script.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name bdsaglam/musique --subsets answerable --num_queries 1000 --strategies oracle_topk --run_type plot_sparsity
```

### Figure 3: Cross-Layer Similarity
```bash
python scripts/eval_script.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name bdsaglam/musique --subsets answerable --num_queries 1000 --strategies post_softmax_all_heads_pooled_oracle_topk  --run_type plot_similarity
```

### Figure 4: Layer-Importance
```bash
python scripts/eval_script.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name bdsaglam/musique --subsets answerable --num_queries 1000 --strategies post_softmax_pooled_prefill_topk --tile_size 32 --run_type select_layers
```

### Ablations
To run all ablations
```bash
python ablations.py --all
```

To run a specific ablation 1 (Figure 2), 2 (Figure 5) or 3 (Figure 6)
```bash
python ablations.py --ablation 1
```

Generated plots are stored at `./results/plots`. Individual results are stored at `./results/evals`.

### Python API

```python
import kascade

# Load model with Kascade attention
from kascade.model_utils import get_tokenizer_and_model
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model, tokenizer = get_tokenizer_and_model(model_name, "sdpa", "cuda")

# Use strategies
from kascade.strategies import EfficientKascadeStrategy
strategy = EfficientKascadeStrategy(recompute_layers=[0,2,8,13,14], model_name=model_name, k=10, tile_size=32, rolling_prefill=True)
output = strategy.generate(prompt, context)
```

## Project Structure

```
kascade/
├── install.sh              # One-command installation
├── pyproject.toml          # Package configuration
├── assets/                 # Images used in README.md
├── results/                # For storing head_mappings and other results generated during experiments
    ├── head_mappings/      # Head mappings for kascade used in experiments
├── scripts/                # CLI scripts and benchmarks
│   ├── eval_script.py      # Main evaluation script
│   ├── eval_lb.py          # LongBench evaluation
│   ├── eval_aime.py        # AIME evaluation
│   ├── benchmark_prefill.py # Unified prefill benchmark
│   └── benchmark_decode.py  # Unified decode benchmark
├── src/                    # Main package (import kascade)
│   ├── kernels/            # CUDA attention kernels
│   │   ├── flash_attention/
│   │   └── flash_decoding/
│   ├── metrics/
│   ├── qadatasets/
│   ├── runners/
│   ├── strategies/
│   ├── model_utils.py
│   └── utils.py
└── third_party/            # Built dependencies (git submodules)
```

## Citation

If you find Kascade useful, please cite our work:

```bibtex
@article{kascade2024,
  title={Kascade: Efficient Sparse Attention for Long-Context RAG},
  author={},
  journal={},
  year={2024}
}
```

## Acknowledgments

Kascade uses excellent open-source projects:

- [Transformers](https://github.com/huggingface/transformers) - State-of-the-art pretrained models for inference and training
- [TileLang](https://github.com/tile-ai/tilelang) - Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels

Some of metrics and evaluation related code is borrowed from the following open-source projects:
- [Longbench](https://github.com/THUDM/LongBench/tree/main/LongBench) - A Bilingual, Multitask Benchmark for Long Context Understanding
- [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness) - A simple toolkit for benchmarking LLMs on mathematical reasoning tasks. 

We also use FlashAttention-3 code for benchmarking
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Fast and memory-efficient attention


### Trademark Notice

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
