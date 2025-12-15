# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .Strategy import Strategy
from .attention_utils import repeat_kv, softmax_
import torch
from torch import nn
from typing import List, Optional
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class OmniKVStrategy(Strategy):
    def __init__(self, recompute_layers: List[int], name='omni_kv', k=1):
        super().__init__(name=name)
        if recompute_layers is None: raise(f"recompute_layers cannot be None for OmniKVStrategy")
        self.k = k
        self.recompute_layers = recompute_layers
        self._topk_to_reuse = None

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
        k = min(max(128, int((self.k/100)*Lk)), Lk)
        # prefill / small‐K fallback
        if Lq > 1 or Lk <= 128 or module.layer_idx in [0, 1] or (self.recompute_layers is not None and ((module.layer_idx - 1) in self.recompute_layers)):
            return ALL_ATTENTION_FUNCTIONS["sdpa"](
                module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
            )

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        if self.recompute_layers is not None and (module.layer_idx in self.recompute_layers):
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            # if attention_mask is not None:
            #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            #     attn_weights = attn_weights.multiply_(causal_mask)
            softmax_(attn_weights, dim=-1)
            attn_weights_max_pooled = attn_weights.amax(dim=1, keepdim=True)  # [B, 1, Lq, Lk]
            self._topk_to_reuse = torch.topk(attn_weights_max_pooled, k, dim=-1).indices  # [B, 1, Lq, k]
            attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()

            return attn_output, attn_weights

        attention_mask = torch.zeros(
            B, 1, Lq, Lk, dtype=torch.bool, device=query.device
        )
        if self._topk_to_reuse is not None:
            attention_mask.scatter_(-1, self._topk_to_reuse.to(attention_mask.device), True)

        return ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )

    def register_attention(self):
        _attention_forward = lambda module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs: self.attention_forward(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(self.name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        ALL_ATTENTION_FUNCTIONS.register(self.name, _attention_forward)