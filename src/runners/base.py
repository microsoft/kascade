# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Type, Callable

import torch
from accelerate import Accelerator
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, GenerationConfig, PreTrainedModel

from kascade.prompt_templates import get_prompt_template
from kascade.qadatasets import QADataset
from kascade.strategies import Strategy

@dataclass
class RunConfig:
    max_length: Optional[int] = None
    batch_size: int = 1
    dataset_class: Type[QADataset] = None
    prompt_format: int = 0
    num_queries: int = 1000
    metric: Callable[..., Any] = None
    inst_tokens: List[str] = field(default_factory=list)
    generation_config: GenerationConfig = None
    dataset_name: Optional[str] = None
    subset: Optional[str] = None
    seed: int = 0
    run_type: str = "evaluate"
    debug: bool = False
    store_results: bool = False
    use_precomputed_stats: bool = False
    model_name: Optional[str] = None

@dataclass
class GenerationStep:
    """Container for information produced per generated example."""

    index: int
    prefill_len: int
    generated_ids: torch.Tensor
    generated_text: str
    reference_answers: Any
    all_classes: Any
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None


class BaseGenerationRunner:
    """Shared generation loop used by metrics and stats runners."""

    def __init__(
        self,
        strategy: Strategy,
        model: PreTrainedModel,
        dataset: Dataset,
        config: RunConfig,
        tokenizer: PreTrainedTokenizer,
        accelerator: Accelerator,
    ) -> None:
        self.strategy = strategy
        self.model = model
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self._num_queries: int = 0
        self._failed_queries: int = 0
        self._output_hidden_states: bool = False
        self._debug: bool = bool(getattr(config, "debug", False))

    def run(self) -> Any:
        raise NotImplementedError

    def _infer_batches(self) -> Iterable[List[GenerationStep]]:
        """Yield batches of generation steps with resilient error handling."""

        self._num_queries = 0

        with self.accelerator.split_between_processes(self.dataset, apply_padding=False) as split_dataset:
            with torch.no_grad():
                qa_dataset: QADataset = self.config.dataset_class(split_dataset)
                use_inst_tokens = False if self.config.prompt_format in [10, 11, 12, 13, 17, 18] else True
                header, query_prompt = get_prompt_template(
                    self.config.prompt_format,
                    self.config.inst_tokens,
                    use_inst_tokens,
                )
                self.model.eval()

                batch_size = self.config.batch_size
                for batch_idx in tqdm(
                    range(0, len(qa_dataset), batch_size),
                    desc=f"Inference Progress GPU {self.accelerator.process_index}",
                    position=self.accelerator.process_index,
                    leave=False,
                ):
                    batch_indices = list(range(batch_idx, min(batch_idx + batch_size, len(qa_dataset))))
                    contexts = qa_dataset.get_context(batch_indices)
                    queries = qa_dataset.get_question(batch_indices)
                    prompts = [header + context + query_prompt.format(question) for context, question in zip(contexts, queries)]

                    answers = qa_dataset.get_answer(batch_indices)
                    if isinstance(answers, tuple):
                        reference_answers, all_classes = answers
                        reference_answers = list(reference_answers)
                        all_classes = list(all_classes)
                    else:
                        reference_answers = list(answers)
                        all_classes = [None] * len(queries)

                    original_block_size = getattr(self.strategy, "_block_size", None)
                    attempt_overrides: List[int | None] = [None]
                    if original_block_size is not None and original_block_size > 256:
                        attempt_overrides.append(256)

                    steps: List[GenerationStep] = []
                    processed = 0

                    for override in attempt_overrides:
                        try:
                            if override is not None and original_block_size is not None:
                                setattr(self.strategy, "_block_size", override)

                            batch = self.tokenizer(
                                prompts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.config.max_length,
                                padding_side="left",
                            )
                            outputs = self.model.generate(
                                batch["input_ids"].to(self.model.device),
                                attention_mask = batch["attention_mask"].to(self.model.device),
                                generation_config = self.config.generation_config,
                                output_hidden_states = self._output_hidden_states,
                                return_dict_in_generate = True,
                                synced_gpus = False,
                            )
                            prefill_lens = batch["attention_mask"].sum(dim=1).tolist()
                            input_length = batch["input_ids"].shape[1]
                            generated_ids = [ids[input_length:] for ids in outputs.sequences]
                            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                            steps = [
                                GenerationStep(
                                    index=batch_indices[offset],
                                    prefill_len=prefill_lens[offset],
                                    generated_ids=generated_ids[offset],
                                    generated_text=generated_texts[offset],
                                    reference_answers=reference_answers[offset],
                                    all_classes=all_classes[offset],
                                    hidden_states=outputs.hidden_states,
                                )
                                for offset in range(len(batch_indices))
                            ]
                            processed = len(steps)
                            break
                        except Exception:
                            if self._debug:
                                raise
                            torch.cuda.empty_cache()
                            continue
                        finally:
                            if override is not None and original_block_size is not None:
                                setattr(self.strategy, "_block_size", original_block_size)

                    if processed == 0:
                        continue

                    self._num_queries += processed
                    yield steps

        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

    def collect(self, step: GenerationStep) -> None:
        raise NotImplementedError

    def finalize(self) -> Any:
        raise NotImplementedError

    def _run_inference(self) -> None:
        for steps in self._infer_batches():
            for step in steps:
                self.collect(step)
        return self.finalize()
