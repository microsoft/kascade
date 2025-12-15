# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .Strategy import Strategy
import torch
from typing import Optional,  Callable
from transformers.masking_utils import sliding_window_overlay, and_masks, causal_mask_function, or_masks, sdpa_mask, ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
class SinkedSlidingWindowStrategy(Strategy):
    def __init__(self, name: str = "sinked_sliding_window", sliding_window: int = 30, num_sink_tokens: int = 4):
        super().__init__(name)
        assert sliding_window >= 0 and sliding_window <= 100, f"Invalid sliding_window: {sliding_window}. It should be between 0 and 100%."
        self.sliding_window = sliding_window
        self.num_sink_tokens = num_sink_tokens

    def custom_attention_mask(
        self,
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function: Callable = causal_mask_function,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        def sink_mask_overlay(num_sink_tokens: int, padding_lens: Optional[torch.Tensor]) -> Callable:
                def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
                    return kv_idx - padding_lens[batch_idx] < num_sink_tokens
                return inner_mask
        def mask_function(sliding_window: int, padding_lens: Optional[torch.Tensor]) -> Callable:
            return and_masks(or_masks(sink_mask_overlay(self.num_sink_tokens, padding_lens), sliding_window_overlay(sliding_window - self.num_sink_tokens)), causal_mask_function)

        sliding_window = max(128, int((self.sliding_window/100)*kv_length))
        padding_lens = (attention_mask.shape[1] - attention_mask.sum(dim=1)) if attention_mask is not None else [0] * batch_size
        _ = kwargs.pop("allow_is_causal_skip", None)
        mask = sdpa_mask(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=mask_function(sliding_window, padding_lens),
            attention_mask=attention_mask,
            allow_is_causal_skip=False,
            allow_torch_fix=False,
            **kwargs,
        )
        return mask

    def register_attention(self):
        ALL_MASK_ATTENTION_FUNCTIONS.register(self.name, self.custom_attention_mask)
        ALL_ATTENTION_FUNCTIONS.register(self.name, ALL_ATTENTION_FUNCTIONS["sdpa"])