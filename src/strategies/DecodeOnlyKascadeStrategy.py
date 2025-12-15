# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .KascadeStrategy import KascadeStrategy
import torch
from torch import nn
from typing import List, Optional
import math
from .attention_utils import repeat_kv, softmax_
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class DecodeOnlyKascadeStrategy(KascadeStrategy): # Does not support stats collection since never used for stats collection
    def __init__(self, recompute_layers: List[int], model_name:str, k=1):
        super().__init__(name="decode_only_kascade", recompute_layers=recompute_layers, model_name=model_name, k=k, tile_size=1, rolling_prefill=False, block_size=12288)
        
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
        # query: [B, H, Lq, D]
        k = min(max(128, int((self.k/100)*key.shape[-2])), key.shape[-2])
        B, H, Lq, _ = query.shape
        _, Hk, Lk, _ = key.shape
        if Lq > 1:
            return ALL_ATTENTION_FUNCTIONS["sdpa"](
                module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
            )
        
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) #* scaling
        dtype = attn_weights.dtype
        min_dtype = torch.finfo(dtype).min
        B, H, Lq, Lk = attn_weights.shape
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            if not (module.layer_idx in self.recompute_layers
                or module.layer_idx == 0):
                indices = self._topk_to_reuse[:, self._head_mapping[module.layer_idx], :, :].to(torch.int64)
                indices = indices.repeat_interleave(module.num_key_value_groups, dim=1)  # [B, H, Lq, k]
                values = attn_weights.gather(dim=3, index=indices)
                attn_weights.fill_(min_dtype)
                attn_weights.scatter_(3, indices, values)
            attn_weights = attn_weights.add_(causal_mask)

        attn_weights = softmax_(attn_weights.to(torch.float32) * scaling, dim=-1).to(query.dtype)

        if (module.layer_idx in self.recompute_layers
            or module.layer_idx == 0):
            # update mask_to_reuse
            attn_weights_per_khead = attn_weights.view(B, H//module.num_key_value_groups, module.num_key_value_groups, Lq, Lk).sum(dim=2)  # [B, H//4, Lq, Lk]
            indices = torch.topk(attn_weights_per_khead, k, dim=-1).indices # [B, H//4, Lq, k]
            self._topk_to_reuse = indices.to(torch.int32)

            indices = indices.repeat_interleave(module.num_key_value_groups, dim=1)  # [B, H, Lq, k]
            values = attn_weights.gather(dim=3, index=indices)
            if module.layer_idx > 0: # do global in layer 0
                attn_weights.fill_(0)
                attn_weights.scatter_(3, indices, values)
                attn_weights.div_(attn_weights.sum(dim=-1, keepdim=True))
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights
