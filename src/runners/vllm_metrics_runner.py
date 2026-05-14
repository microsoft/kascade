# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""vLLM inference backend for MetricsRunner.

Swaps in vLLM's LLM.generate for the default HF model.generate, reusing
all prompt-building, scoring, and CSV output logic from MetricsRunner.

The script (eval_script_vllm.py) is responsible for:
  - Registering the vLLM attention backend
  - Creating the LLM and SamplingParams objects
  - Passing them here

This runner knows nothing about backends, topk, or recompute_layers.
"""

import torch

from .metrics_runner import MetricsRunner


# Strategies that have a supported vLLM backend counterpart.
# Format: strategy_name -> [default_backend, *allowed_overrides]
VLLM_SUPPORTED_STRATEGIES: dict[str, list[str]] = {
    "baseline":            ["default", "tilelang"],
    "decode_only_kascade": ["kascade"],
}


def resolve_vllm_backend(strategy_name: str, vllm_backend: str | None) -> str:
    """Return the vLLM backend to use, or raise if the combination is invalid.

    Raises RuntimeError (not ValueError) for unsupported strategies so callers
    can catch it and skip gracefully without crashing.
    """
    if strategy_name not in VLLM_SUPPORTED_STRATEGIES:
        raise RuntimeError(
            f"Strategy '{strategy_name}' is not supported with the vLLM runner. "
            f"Supported strategies: {list(VLLM_SUPPORTED_STRATEGIES.keys())}. Skipping."
        )
    allowed = VLLM_SUPPORTED_STRATEGIES[strategy_name]
    if vllm_backend is None:
        return allowed[0]
    if vllm_backend not in allowed:
        raise ValueError(
            f"Backend '{vllm_backend}' is not valid for strategy '{strategy_name}'. "
            f"Allowed backends: {allowed}."
        )
    return vllm_backend


class VllmMetricsRunner(MetricsRunner):
    """MetricsRunner that uses a pre-built vLLM LLM instance for generation.

    Parameters
    ----------
    llm:
        A ``vllm.LLM`` instance, fully configured and ready to call ``.generate()``.
    sampling_params:
        A ``vllm.SamplingParams`` instance.
    strategy_label:
        String used as the strategy identifier in CSV filenames and console
        output (e.g. ``"vllm_kascade_decode_only_kascade"``).
    *args, **kwargs:
        Forwarded to ``MetricsRunner.__init__``. Pass ``strategy=None`` and
        ``model=None`` since generation is handled by ``llm``.
    """

    def __init__(self, llm, sampling_params, strategy_label: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm = llm
        self._sampling_params = sampling_params
        self._strategy_label_str = strategy_label

    # ------------------------------------------------------------------
    # Override: generation backend
    # ------------------------------------------------------------------

    def _generate_batch(self, prompts):
        """Call vLLM LLM.generate instead of HF model.generate."""
        outputs = self._llm.generate(prompts, self._sampling_params, use_tqdm=False)
        generated_ids = [
            torch.tensor(list(o.outputs[0].token_ids)) for o in outputs
        ]
        generated_texts = [o.outputs[0].text for o in outputs]
        prefill_lens = [len(o.prompt_token_ids) for o in outputs]
        return generated_ids, generated_texts, prefill_lens, None  # no hidden states

    # ------------------------------------------------------------------
    # Override: strategy label (no Strategy object present)
    # ------------------------------------------------------------------

    def _strategy_label(self) -> str:
        return self._strategy_label_str

