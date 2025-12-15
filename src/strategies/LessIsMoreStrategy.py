# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .Strategy import Strategy
from .attention_utils import repeat_kv, softmax_
import torch
from torch import nn
from typing import List, Optional
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class LessIsMoreStrategy(Strategy):
    def __init__(self, recompute_layers: Optional[List[int]], name='less_is_more', k=1, lim_ratio_factor=1, num_sink_tokens=4):
        super().__init__(name=name)
        if recompute_layers is None: raise(f"recompute_layers cannot be None for LessIsMoreStrategy")
        self.k = k
        self.recompute_layers = recompute_layers
        self.lim_ratio_factor = lim_ratio_factor
        self.num_sink_tokens = num_sink_tokens

    def attention_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        B, H, Lq, D = query.shape
        _, Hk, Lk, _ = key.shape
        token_budget = min(max(128, int((self.k/100)*Lk)), Lk)

        # prefill fallback
        if Lq > 1 or module.layer_idx < self.recompute_layers[0]:
            return ALL_ATTENTION_FUNCTIONS["sdpa"](
                module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
            )

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        # generation with sparse attention
        attn_weights = (
            torch.matmul(query, key_states.transpose(2, 3)) * scaling
        )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # decoding
        if (
            module.layer_idx in self.recompute_layers
        ) and token_budget == Lk:  # if there is no need to do sparse attention
            self._pos_mask = torch.ones_like(attn_weights)
        elif module.layer_idx in self.recompute_layers:
            # Split token budget between top-k, attention sink, and most recent tokens
            middle_budget = int(token_budget * (1 - self.lim_ratio_factor))  # top-k
            most_recent_amount = token_budget - middle_budget  # window attention

            if most_recent_amount < self.num_sink_tokens:
                self.num_sink_tokens = 0
            else:
                most_recent_amount -= self.num_sink_tokens

            assert middle_budget + self.num_sink_tokens + most_recent_amount == token_budget

            # get sink token indices
            sink_indices = torch.arange(self.num_sink_tokens, device=attn_weights.device)
            sink_indices = sink_indices.expand(
                attn_weights.shape[:-1] + (self.num_sink_tokens,)
            )

            # do top-k selection from the middle tokens
            recent_start = Lk - most_recent_amount
            middle_scores = attn_weights[..., self.num_sink_tokens:recent_start]
            _, middle_indices = torch.topk(middle_scores, k=middle_budget, dim=-1)
            middle_indices = middle_indices + self.num_sink_tokens

            # Union operation capped by token_budget
            union_tensor = middle_indices.transpose(1, 3).contiguous().view(B, -1)
            union_list = list(dict.fromkeys(union_tensor[0].tolist()))
            if len(union_list) > middle_budget:
                union_list = union_list[:middle_budget]

            # Reshape back to proper dimensions
            middle_indices = torch.tensor(
                union_list, dtype=middle_indices.dtype, device=middle_indices.device
            )
            middle_indices = middle_indices.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            middle_indices = middle_indices.expand(B, H, Lq, -1)

            # get most recent tokens
            recent_indices = torch.arange(
                recent_start, Lk, device=attn_weights.device
            )
            recent_indices = recent_indices.expand(
                attn_weights.shape[:-1] + (most_recent_amount,)
            )

            # combine indices
            top_k_indices = torch.cat(
                [sink_indices, middle_indices, recent_indices], dim=-1
            )

            top_k_mask = torch.zeros_like(attn_weights).scatter_(-1, top_k_indices, 1.0)
            self._pos_mask = top_k_mask  # store top_k mask
        else:
            # apply top_k mask
            if not hasattr(self, "_pos_mask") or self._pos_mask is None:
                raise ValueError("pos mask should be set up in sparse attn layers")
            min_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(
                self._pos_mask.to(attn_weights.device) == 0, min_value
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, None

    def register_attention(self):
        _attention_forward = lambda module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs: self.attention_forward(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(self.name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        ALL_ATTENTION_FUNCTIONS.register(self.name, _attention_forward)