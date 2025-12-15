# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .BaselineStrategy import Strategy
import torch
from torch import nn
from typing import Optional
from .attention_utils import repeat_kv, softmax_
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

class OracleTopkStrategy(Strategy):
    def __init__(self, name='oracle_topk', k=1, block_size=12288):
        super().__init__(name)
        assert k >= 0 and k <= 100, f"Invalid k: {k}. It should be between 0 and 100%."
        self.k = k
        self._block_size = block_size
        self._stats_runner = None
        self._stats_initialized = False

    def attach_stats_runner(self, stats_runner) -> None:
        self._stats_runner = stats_runner
        self._stats_initialized = False

    def _attention_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        start: int = 0,
        end: int = None,
        k: int = 1,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        is_decode = (attention_mask.shape[2] == 1)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights.add_(causal_mask)

        # Apply softmax, select top-k, and sparsify attention in one step
        attn_weights = self._softmax_topk_sparsify_(attn_weights, attention_mask, module, k, is_decode, start, end)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        return attn_output, attn_weights

    def _softmax_topk_sparsify_(self, attn_weights, attention_mask, module, k, is_decode, start, end):
        """
        Apply softmax, select top-k, and sparsify attention weights.
        
        This function takes raw attention scores and returns the final sparsified 
        softmax attention matrix. Subclasses can override this to customize the 
        entire attention processing pipeline for different ablations.
        
        Args:
            attn_weights: Raw attention scores after matmul [B, H, Lq, Lk] 
                         (modified in-place)
            module: The attention module
            k: Number of top elements to select
            is_decode: Whether this is decode step
            
        Returns:
            Final sparsified softmax attention weights [B, H, Lq, Lk]
        """
        softmax_(attn_weights, dim=-1)
        B, H, Lq, Lk = attn_weights.shape
        values, indices = torch.topk(attn_weights, k, dim=-1)

        ### Stats collection
        if self._stats_runner is not None:
            self._stats_runner.update(is_decode, module.layer_idx, indices, attn_weights)
        
        attn_weights.fill_(0)
        attn_weights.scatter_(3, indices, values)
        attn_weights.div_(attn_weights.sum(dim=-1, keepdim=True))
        
        return attn_weights
    
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
        ### Stats collection initialization
        if self._stats_runner is not None and not self._stats_initialized:
            head_count = self._get_head_count_for_stats(module)
            self._stats_runner.prepare(module.config.num_hidden_layers, head_count)
            self._stats_initialized = True

        # query: [B, H, Lq, D]
        if kwargs.get("k", None) is not None:
            k = kwargs.get("k")
            kwargs.pop("k")
        else:
            k = min(max(5, int((self.k/100)*key.shape[-2])), key.shape[-2]) if self._stats_runner is None else 256
        B, _, Lq, _ = query.shape
        is_decode = (Lq == 1)
        outputs = []
        for start in range(0, Lq, self._block_size):
            end = min(start + self._block_size, Lq)
            query_block = query[:, :, start:end, :]
            if attention_mask is not None:
                mask_block = attention_mask[:, :, start:end, :]
            else:
                mask_block = None
            out, attn_w = self._attention_forward(
                module,
                query_block,
                key,
                value,
                mask_block,
                scaling,
                dropout=dropout,
                start=start,
                end=end,
                k=k,
                **kwargs,
            )
            outputs.append(out)
            attn_weights = attn_w
        attn_output = torch.cat(outputs, dim=1)

        ### Stats collection of hidden states after attention
        if self._stats_runner is not None:
            hidden_states = attn_output.reshape(B, Lq, -1).contiguous() # [B, Lq, D]
            hidden_states = module.o_proj(hidden_states) # [B, Lq, D]
            self._stats_runner.update_hidden_states(is_decode, module.layer_idx, hidden_states=hidden_states)
        return attn_output, attn_weights

    def register_attention(self):
        _attention_forward = lambda module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs: self.attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(self.name, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
        ALL_ATTENTION_FUNCTIONS.register(self.name, _attention_forward)

    def _get_head_count_for_stats(self, module):
        """
        Hook method to determine the number of heads for statistics tracking.
        Default implementation uses attention heads. GQA strategies override this.
        
        Args:
            module: The attention module
            
        Returns:
            Number of heads to track in statistics
        """
        return module.config.num_attention_heads