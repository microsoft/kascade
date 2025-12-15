# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .PostSoftmaxPooledPrefillTopkStrategy import PostSoftmaxPooledPrefillTopkStrategy
import torch
from torch import nn
from typing import List, Optional
import math
from .attention_utils import repeat_kv, softmax_

class PooledKascadeStrategy(PostSoftmaxPooledPrefillTopkStrategy): # Does not support stats collection since never used for stats collection
    def __init__(self, recompute_layers: List[int], k=1, tile_size=1, rolling_prefill=False, block_size=12288):
        super().__init__(name="pooled_kascade", k=k, tile_size=tile_size, rolling_prefill=rolling_prefill, block_size=block_size)
        if recompute_layers is None: raise(f"recompute_layers cannot be None for KascadeStrategy")
        self.recompute_layers = recompute_layers

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

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) #* scaling
        dtype = attn_weights.dtype
        min_dtype = torch.finfo(dtype).min
        B, H, Lq, Lk = attn_weights.shape
        if self.tile_size > 1 and Lq > self.tile_size:
            num_tiles = math.ceil(Lq / self.tile_size)
            pad_len = num_tiles * self.tile_size - Lq
        if attention_mask is not None:
            if self.tile_size > 1 and Lq > self.tile_size:
                if pad_len > 0:
                    mask_pad = attention_mask.new_full((B, 1, pad_len, key_states.shape[-2]), min_dtype)
                    exclude_last_tile_mask = torch.cat([attention_mask, mask_pad], dim=-2) # [B, 1, Lq_with_padding, Lk]
                else:
                    exclude_last_tile_mask = attention_mask
                exclude_last_tile_mask = exclude_last_tile_mask.view(B, 1, num_tiles, self.tile_size, key_states.shape[-2])[:,:,:,-1,:] # [B, 1, num_tiles, Lk]
                # each tile only attends to keys before that tile
                exclude_last_tile_mask = torch.roll(exclude_last_tile_mask, shifts=1, dims=2) # Shift the mask to the right for causal masking
                exclude_last_tile_mask[:, :, 0, :] = exclude_last_tile_mask[:, :, 1, :] # Set the first tile to min_dtype for causal masking
                exclude_last_tile_mask[:, :, 0, start:start+self.tile_size] = min_dtype # correct the first tile mask to allow attention to only previous tiles
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            if not (module.layer_idx in self.recompute_layers
                or module.layer_idx == 0):
                indices = self._topk_to_reuse[:, :1, start:end, :].to(torch.int64)
                indices = indices.repeat_interleave(module.config.num_attention_heads, dim=1)  # [B, H, Lq, k]
                values = attn_weights.gather(dim=3, index=indices)
                if self.tile_size > 1 and Lq > self.tile_size:
                    exclude_last_tile_mask = exclude_last_tile_mask.repeat_interleave(self.tile_size, dim=-2)  # [B, 1, num_tiles, Lk]
                    exclude_last_tile_mask = exclude_last_tile_mask[:, :, :Lq, :]  # [B, 1, Lq, Lk]
                    attn_weights.masked_fill_(exclude_last_tile_mask == attention_mask, min_dtype)
                else:
                    attn_weights.fill_(min_dtype)
                attn_weights.scatter_(3, indices, values)
            attn_weights = attn_weights.add_(causal_mask)

        attn_weights = softmax_(attn_weights.to(torch.float32) * scaling, dim=-1).to(query.dtype)

        if (module.layer_idx in self.recompute_layers
            or module.layer_idx == 0):
            # update mask_to_reuse
            attn_weights_per_khead = attn_weights.sum(dim=1, keepdim=True)  # [B, 1, Lq, Lk]
            if self.tile_size > 1 and Lq > self.tile_size:
                if pad_len > 0:
                    pad = attn_weights_per_khead.new_zeros(B, 1, pad_len, Lk)
                    padded = torch.cat([attn_weights_per_khead, pad], dim=-2)   # [B, 1, Lq_with_padding, Lk]
                else:
                    padded = attn_weights_per_khead
                attn_weights_per_khead = padded.view(B, 1, num_tiles, self.tile_size, Lk).sum(dim=-2) # [B, 1, Lq // tile_size, Lk]
                attn_weights_per_khead.multiply_((exclude_last_tile_mask==0).to(attn_weights_per_khead.dtype))  # [B, 1, Lq // tile_size, Lk]
                indices = torch.topk(attn_weights_per_khead, k, dim=-1).indices
                if self.rolling_prefill and self._topk_mask is not None:
                    start_tile = start // self.tile_size
                    end_tile = math.ceil(end / self.tile_size)
                    topk_mask = self._topk_mask[start_tile:end_tile, :]
                    indices.masked_fill_(topk_mask, Lk-1)
                indices = indices.repeat_interleave(self.tile_size, dim=-2)   # [B, 1, Lq_with_padding, Lk]
                indices = indices[:, :, :Lq, :]  # [B, H, Lq, Lk]
            else:
                indices = torch.topk(attn_weights_per_khead, k, dim=-1).indices # [B, 1, Lq, Lk]
                
            self._topk_to_reuse[:, :, start:end, :] = indices.to(torch.int32)
            indices = indices.repeat_interleave(module.config.num_attention_heads, dim=1)  # [B, H, Lq, k]
            values = attn_weights.gather(dim=3, index=indices)
            if module.layer_idx > 0: # do global in layer 0
                if self.tile_size > 1 and Lq > self.tile_size:
                    exclude_last_tile_mask = exclude_last_tile_mask.repeat_interleave(self.tile_size, dim=-2)  # [B, 1, num_tiles, Lk]
                    exclude_last_tile_mask = exclude_last_tile_mask[:, :, :Lq, :]  # [B, 1, Lq, Lk]
                    exclude_last_tile_mask = (exclude_last_tile_mask == attention_mask).to(query.device)  # [B, 1, Lq, Lk] 
                    attn_weights.masked_fill_(exclude_last_tile_mask, 0)
                else:
                    attn_weights.fill_(0)
                attn_weights.scatter_(3, indices, values)
                attn_weights.div_(attn_weights.sum(dim=-1, keepdim=True))
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

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
        if self.tile_size > 1:
            k -= self.tile_size
        B, _, Lq, _ = query.shape
        G = key.shape[1]
        if module.layer_idx == 0:
            self._topk_to_reuse = torch.zeros((B, 1, Lq, k), device=query.device, dtype=torch.int32)
            if Lq > 128 and self.rolling_prefill:
                num_tiles = math.ceil(Lq / self.tile_size)
                tile_indices = torch.arange(0, num_tiles*self.tile_size, self.tile_size, device=query.device)
                k_indices = torch.arange(0, k, device=query.device)
                topk_nums = ((self.k*tile_indices)//100).clamp_(min=128)
                self._topk_mask = (k_indices >= topk_nums.unsqueeze(1)).to(query.device)
                # print(self._topk_mask.shape, Lq)
                # print(key.shape[-2] - self._topk_mask.sum(-1))
                # print(torch.equal(key.shape[-2] - self._topk_mask.sum(-1), topk_nums))
                # print(tile_indices)
                del tile_indices, k_indices, topk_nums
            else:
                self._topk_mask = None
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
        return attn_output, attn_weights