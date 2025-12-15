# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .Strategy import Strategy
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .attention_utils import repeat_kv
import math

class QuestStrategy(Strategy):
    def __init__(self, name: str = "quest", tile_size: int = 16, k: float = 10):
        super().__init__(name)
        self.tile_size = tile_size
        self.k = k

    def local_heavy_hitter_mask(self, attn_weights, token_budget, tile_size):
        # attn_weights (BS, head, query, keys)

        # expend attn_weights to be divisible by tile_size
        seq_length = attn_weights.shape[-1]
        padding_length = self.tile_size - ((seq_length - 1) % self.tile_size + 1)
        attn_weights = torch.cat(
            [
                attn_weights,
                torch.ones(
                    (
                        attn_weights.shape[0],
                        attn_weights.shape[1],
                        attn_weights.shape[2],
                        padding_length,
                    ),
                    device=attn_weights.device,
                )
                * torch.tensor(torch.finfo(attn_weights.dtype).min),
            ],
            dim=-1,
        )

        # chunk attn_weights into self.tile_size tokens
        chunk_attn_weights = attn_weights.reshape(
            attn_weights.shape[0],
            attn_weights.shape[1],
            attn_weights.shape[2],
            attn_weights.shape[3] // self.tile_size,
            self.tile_size,
        ).amax(dim=-1)

        _, topk = chunk_attn_weights.topk(
            k=min(max(3, token_budget // self.tile_size), chunk_attn_weights.size(-1)), dim=-1
        )
        # repeat topk self.tile_size times and recover the original indexes (* self.tile_size + arange(self.tile_size))
        topk = topk.unsqueeze(-1).repeat(
            1, 1, 1, 1, self.tile_size
        ) * self.tile_size + torch.arange(self.tile_size, device=topk.device)
        topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom.scatter_(-1, topk, True)
        # remove the padding
        mask_bottom = mask_bottom[:, :, :, :seq_length]
        return mask_bottom


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
        bsz, H, q_len, D = query.shape
        _, Hk, kv_seq_len, _ = key.shape

        # prefill / small‐K fallback
        if q_len > 1 or kv_seq_len <= 8 * self.tile_size or module.layer_idx in [0, 1]:
            return ALL_ATTENTION_FUNCTIONS["sdpa"](
                module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
            )

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # [B, H, Lq, Lk]

        sign = (query > 0) + (~(query > 0)) * -1
        max_key = key_states * sign
        postive_query = query * sign

        # expend max_key to be divisible by self.tile_size
        seq_length = max_key.shape[-2]
        padding_length = self.tile_size - ((seq_length - 1) % self.tile_size + 1)
        max_key = torch.cat(
            [
                max_key,
                torch.ones(
                    (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                    device=max_key.device,
                )
                * torch.tensor(torch.finfo(max_key.dtype).min),
            ],
            dim=-2,
        )

        # chunk max_key into self.tile_size tokens
        chunk_max_key = max_key.reshape(
            max_key.shape[0],
            max_key.shape[1],
            max_key.shape[2] // self.tile_size,
            self.tile_size,
            max_key.shape[3],
        ).amax(dim=-2)

        # duplicate chunk_max_key self.tile_size times
        chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.tile_size, 1)
        # reshape chunk_max_key to the original shape
        chunk_max_key = chunk_max_key.reshape(
            chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
        )[:, :, :seq_length, :]

        quantized_weight = torch.matmul(
            postive_query.float(),
            chunk_max_key.transpose(2, 3),
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            quantized_weight = quantized_weight + attention_mask
            quantized_weight = torch.max(
                quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
            )

        token_budget = min(max(128, int((self.k/100)*kv_seq_len)), kv_seq_len)

        attn_weights_for_selection = quantized_weight

        if token_budget > 0:
            mask_bottom = self.local_heavy_hitter_mask(
                attn_weights_for_selection, token_budget, self.tile_size
            )  # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

        mask_bottom = torch.tril(mask_bottom, diagonal=kv_seq_len + 1)
        attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights
    
    def register_attention(self):
        _attention_forward = lambda module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs: self.attention_forward(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(self.name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        ALL_ATTENTION_FUNCTIONS.register(self.name, _attention_forward)