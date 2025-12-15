# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from typing import Tuple
import os

import torch
from accelerate.utils import reduce, gather

from .base import BaseGenerationRunner, GenerationStep


def _score_example(metric_fn, prediction: str, ground_truths, classes, prompt_format: int) -> Tuple[int, int, float]:
    if prompt_format in [10, 11, 12, 13]:
        prediction = prediction.lstrip("\n").split("\n")[0]

    if not isinstance(ground_truths, (list, tuple)):
        ground_truths = [ground_truths]

    best_score = 0.0
    for ground in ground_truths:
        best_score = max(best_score, metric_fn(prediction, ground, all_classes=classes))
    return best_score


class MetricsRunner(BaseGenerationRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._metric_fn = self.config.metric
        self._prompt_format = self.config.prompt_format
        self._totals = torch.zeros(3, dtype=torch.float64, device=self.accelerator.device)
        self._start_time: float | None = None

    def run(self):
        self._start_time = time.time()
        return self._run_inference()

    def collect(self, step: GenerationStep) -> None:
        dataset_metric = _score_example(
            self._metric_fn,
            step.generated_text,
            step.reference_answers,
            step.all_classes,
            self._prompt_format,
        )
        decode_len = self._decode_length(step.generated_ids)
        additions = torch.tensor(
            [dataset_metric, step.prefill_len, decode_len],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        self._totals += additions

    def _decode_length(self, ids: torch.Tensor) -> int:
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            matches = (ids == eos_id).nonzero(as_tuple=False)
            if matches.numel() > 0:
                return matches[0].item() + 1
        return ids.numel() if isinstance(ids, torch.Tensor) else len(ids)

    def finalize(self):
        totals_with_queries = torch.cat(
            [
                torch.tensor([self._num_queries], dtype=torch.float64, device=self.accelerator.device),
                self._totals,
            ]
        )
        totals_with_queries = reduce(totals_with_queries, reduction="sum")
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
        elapsed = torch.tensor([elapsed], dtype=torch.float32, device=self.accelerator.device)
        elapsed = gather(elapsed).max().item()
        self._num_queries = int(totals_with_queries[0].item())
        if self._num_queries == 0:
            self.accelerator.print("No queries processed. Returning zero metrics.")
            return 0, 0, 0, 0, 0, 0

        skipped = max(int(self.config.num_queries - self._num_queries), 0)
        avg_tensor = totals_with_queries[1:] / self._num_queries

        avg_metric = avg_tensor[0].item()
        avg_prefill = avg_tensor[1].item()
        avg_decode = avg_tensor[2].item()

        if skipped > 0:
            self.accelerator.print(f"Skipped {skipped} queries due to generation errors.")

        console_summary = self._get_results_summary(
            metrics=(
                round(avg_metric, 5),
                avg_prefill,
                avg_decode,
                round(elapsed, 3),
            ),
            for_csv=False,
        )
        self.accelerator.print(console_summary)

        if self.config.store_results and self.accelerator.is_main_process:
            csv_summary = self._get_results_summary(
                metrics=(
                    round(avg_metric, 5),
                    avg_prefill,
                    avg_decode,
                    round(elapsed, 3),
                ),
                for_csv=True,
            )
            self._store_to_csv(csv_summary)

        return self._num_queries, avg_metric, avg_prefill, avg_decode

    def _store_to_csv(self, summary_line: str):
        """Append results to CSV file organized by model/dataset/strategy."""
        model_name = self.config.model_name.split('/')[-1]
        strategy_name = self.strategy.name
        dataset_name = self.config.dataset_name.split('/')[-1] if '/' in self.config.dataset_name else self.config.dataset_name
        
        results_dir = f"./results/evals/{model_name}/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        csv_path = f"{results_dir}/{strategy_name}.csv"
        
        with open(csv_path, 'a') as f:
            f.write(summary_line + '\n')

    def _get_results_summary(self, metrics, for_csv=False):
        """
        Print all instance variables of the strategy object with experiment config and metrics results.
        Returns a comma-separated string suitable for CSV output or console display.
        
        Args:
            metrics: Tuple of metric values
            for_csv: If True, omit model/dataset/strategy names (stored in file path).
                    If False, include everything for console output.
        """
        if for_csv:
            # For CSV: Only subset, strategy config, and metrics
            # (model, dataset, strategy names are in the file path)
            result = f"{self.config.subset},{self._num_queries},"
            
            # Add strategy config variables (excluding name and private vars)
            all_vars = vars(self.strategy)
            for key, value in all_vars.items():
                if not key.startswith('_') and key != 'name':
                    if isinstance(value, list):
                        strlist = f"'{value}'"
                        result += f"{strlist},"
                    else:
                        result += f"{value},"
            
            # Add metrics
            for value in metrics:
                result += f"{value},"
            
            # Remove trailing comma
            return result.rstrip(',')
        else:
            # For console: Everything (full context)
            result = f"{self.model.config._name_or_path},{self.config.dataset_name},{self.config.subset},{self.config.seed},{self._num_queries},{self._prompt_format},"
            
            # Add strategy name and config
            all_vars = vars(self.strategy)
            result += f"{self.strategy.name}"  # Always include name first for strategy values
            for key, value in all_vars.items():
                if not key.startswith('_') and key != 'name':
                    result += f",{value}"
            
            # Add metrics
            for value in metrics:
                result += f",{value}"
            
            return result