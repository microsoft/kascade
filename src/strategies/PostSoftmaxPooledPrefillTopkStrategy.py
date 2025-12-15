# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .PostSoftmaxGQAPooledOracleTopKStrategy import PostSoftmaxGQAPooledOracleTopKStrategy
import torch
from torch import nn
from typing import Optional
import math
from .attention_utils import softmax_

class PostSoftmaxPooledPrefillTopkStrategy(PostSoftmaxGQAPooledOracleTopKStrategy):
    def __init__(self, name='post_softmax_pooled_prefill_topk', k=1, tile_size=1, rolling_prefill=False, block_size=12288):
        super().__init__(name=name, k=k, block_size=block_size)
        self.tile_size = tile_size
        self.rolling_prefill = rolling_prefill

    def _softmax_topk_sparsify_(self, attn_weights, attention_mask, module, k, is_decode, start, end):
        """
        Post-softmax GQA+prefill pooling: Apply softmax first, then pool across q heads of a group and tile_size queries and select top-k.
        """
        softmax_(attn_weights, dim=-1)
        dtype = attn_weights.dtype
        min_dtype = torch.finfo(dtype).min
        B, H, Lq, Lk = attn_weights.shape
        attn_weights_per_khead = attn_weights.view(B, H//module.num_key_value_groups, module.num_key_value_groups, Lq, Lk).sum(dim=2)  # [B, G, Lq, Lk]
        num_tiles = math.ceil(Lq / self.tile_size)
        if self.tile_size > 1 and Lq > self.tile_size:
            pad_len = num_tiles * self.tile_size - Lq
            if pad_len > 0:
                pad = attn_weights_per_khead.new_zeros(B, H//module.num_key_value_groups, pad_len, Lk)
                padded = torch.cat([attn_weights_per_khead, pad], dim=-2)   # [B, G, Lq_with_padding, Lk]
                mask_pad = attention_mask.new_full((B, 1, pad_len, Lk), min_dtype)
                exclude_last_tile_mask = torch.cat([attention_mask, mask_pad], dim=-2) # [B, 1, Lq_with_padding, Lk]
            else:
                padded = attn_weights_per_khead
                exclude_last_tile_mask = attention_mask
            exclude_last_tile_mask = exclude_last_tile_mask.view(B, 1, num_tiles, self.tile_size, Lk)[:,:,:,-1,:] # [B, 1, num_tiles, Lk]
            # each tile only attends to keys before that tile
            exclude_last_tile_mask = torch.roll(exclude_last_tile_mask, shifts=1, dims=2) # Shift the mask to the right for causal masking
            exclude_last_tile_mask[:, :, 0, :] = exclude_last_tile_mask[:, :, 1, :] # Set the first tile to min_dtype for causal masking
            exclude_last_tile_mask[:, :, 0, start:start+self.tile_size] = min_dtype # correct the first tile mask to allow attention to only previous tiles
            attn_weights_per_khead = padded.view(B, H//module.num_key_value_groups, num_tiles, self.tile_size, Lk).sum(dim=-2) # [B, G, Lq // tile_size, Lk]
            attn_weights_per_khead.multiply_((exclude_last_tile_mask==0).to(attn_weights_per_khead.dtype))  # [B, G, Lq // tile_size, Lk]
            indices = torch.topk(attn_weights_per_khead, k, dim=-1).indices
            if self.rolling_prefill and self._topk_mask is not None:
                start_tile = start // self.tile_size
                end_tile = math.ceil(end / self.tile_size)
                topk_mask = self._topk_mask[start_tile:end_tile, :]
                indices.masked_fill_(topk_mask, Lk-1)
            indices_to_store = indices
            indices = indices.repeat_interleave(self.tile_size, dim=-2)   # [B, G, Lq_with_padding, Lk]
            indices = indices[:, :, :Lq, :]  # [B, G, Lq, Lk]
        else:
            indices = torch.topk(attn_weights_per_khead, k, dim=-1).indices  # [B, G, Lq, Lk]
            if self.rolling_prefill and self._topk_mask is not None:
                start_tile = start // self.tile_size
                end_tile = math.ceil(end / self.tile_size)
                topk_mask = self._topk_mask[start_tile:end_tile, :]
                indices.masked_fill_(topk_mask, Lk-1)
            indices_to_store = indices
    
        indices = indices.repeat_interleave(module.num_key_value_groups, dim=1)  # [B, H, Lq, k]
        values = attn_weights.gather(dim=3, index=indices)

        ### Stats collection
        if self._stats_runner is not None:
            last_tile_attn = torch.zeros(B, module.config.num_key_value_heads, num_tiles, device=attn_weights.device, dtype=attn_weights.dtype)

        if module.layer_idx > 0 or self._stats_runner is not None: # do global in layer 0 or collect stats
            if self.tile_size > 1 and Lq > self.tile_size:
                exclude_last_tile_mask = exclude_last_tile_mask.repeat_interleave(self.tile_size, dim=-2)  # [B, 1, num_tiles, Lk]
                exclude_last_tile_mask = exclude_last_tile_mask[:, :, :Lq, :]  # [B, 1, Lq, Lk]
                exclude_last_tile_mask = (exclude_last_tile_mask == attention_mask).to(attention_mask.device)  # [B, 1, Lq, Lk] 
                attn_weights.masked_fill_(exclude_last_tile_mask, 0)
                if self._stats_runner is not None:
                    lta = attn_weights.view(B, module.config.num_key_value_heads, module.num_key_value_groups, Lq, Lk).sum(dim=2).sum(dim=-1)  # [B, G, Lq]
                    pad = lta.new_zeros(B, module.config.num_key_value_heads, pad_len)
                    lta = torch.cat([lta, pad], dim=-1)   # [B, G, Lq_with_padding]
                    last_tile_attn = lta.view(B, module.config.num_key_value_heads, num_tiles, self.tile_size).sum(dim=-1)  # [B, G, num_tiles]
            else:
                attn_weights.fill_(0)
            attn_weights.scatter_(3, indices, values)
            attn_weights.div_(attn_weights.sum(dim=-1, keepdim=True))
        
        if self._stats_runner is not None:
            self._stats_runner.update(is_decode, module.layer_idx, indices_to_store, attn_weights_per_khead.to(torch.float32), last_tile_attn=last_tile_attn)
            
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
        # query: [B, H, Lq, D]
        if self._stats_runner is not None:
            if self._stats_runner.config.run_type == "select_layers":
                k = 64
            else:
                k = 256
        else:
            k = min(max(128, int((self.k/100)*key.shape[-2])), key.shape[-2])
        if self.tile_size > 1:
            k -= self.tile_size
        B, _, Lq, _ = query.shape
        H = key.shape[1]
        if module.layer_idx == 0:
            if self.rolling_prefill:
                num_tiles = math.ceil(Lq / self.tile_size)
                tile_indices = torch.arange(0, num_tiles*self.tile_size, self.tile_size, device=query.device)
                k_indices = torch.arange(0, k, device=query.device)
                topk_nums = ((self.k*tile_indices)//100).clamp_(min=128)
                self._topk_mask = (k_indices >= topk_nums.unsqueeze(1)).to(query.device)
                del tile_indices, k_indices, topk_nums
            else:
                self._topk_mask = None
        return super().attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            k=k,
            **kwargs,
        )