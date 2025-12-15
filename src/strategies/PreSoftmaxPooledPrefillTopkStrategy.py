# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .PostSoftmaxPooledPrefillTopkStrategy import PostSoftmaxPooledPrefillTopkStrategy
import torch
import math
from .attention_utils import softmax_

class PreSoftmaxPooledPrefillTopkStrategy(PostSoftmaxPooledPrefillTopkStrategy): # Does not support stats collection since never used for stats collection
    def __init__(self, name='pre_softmax_pooled_prefill_topk', k=1, tile_size=1, rolling_prefill=False, block_size=12288):
        super().__init__(name=name, k=k, tile_size=tile_size, rolling_prefill=rolling_prefill, block_size=block_size)
        self.tile_size = tile_size
        self.rolling_prefill = rolling_prefill 

    def _softmax_topk_sparsify_(self, attn_weights, attention_mask, module, k, is_decode, start, end):
        """
        Post-softmax GQA+prefill pooling: Pool across q heads of a group and tile_size queries and select top-k, then apply softmax.
        """
        dtype = attn_weights.dtype
        min_dtype = torch.finfo(dtype).min
        B, H, Lq, Lk = attn_weights.shape
        attn_weights_per_khead = attn_weights.view(B, H//module.num_key_value_groups, module.num_key_value_groups, Lq, Lk).sum(dim=2)  # [B, G, Lq, Lk]
        if self.tile_size > 1 and Lq > self.tile_size:
            num_tiles = math.ceil(Lq / self.tile_size)
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
            indices = torch.topk(attn_weights_per_khead, k, dim=-1).indices
            if self.rolling_prefill and self._topk_mask is not None:
                start_tile = start // self.tile_size
                end_tile = math.ceil(end / self.tile_size)
                topk_mask = self._topk_mask[start_tile:end_tile, :]
                indices.masked_fill_(topk_mask, Lk-1)
            indices = indices.repeat_interleave(self.tile_size, dim=-2)   # [B, G, Lq_with_padding, Lk]
            indices = indices[:, :, :Lq, :]  # [B, H, Lq, Lk]
        else:
            indices = torch.topk(attn_weights_per_khead, k, dim=-1).indices # [B, G, Lq, Lk]

        indices = indices.repeat_interleave(module.num_key_value_groups, dim=1)  # [B, H, Lq, k]
        softmax_(attn_weights, dim=-1)
        values = attn_weights.gather(dim=3, index=indices)
        if module.layer_idx > 0: # do global in layer 0
            if self.tile_size > 1 and Lq > self.tile_size:
                exclude_last_tile_mask = exclude_last_tile_mask.repeat_interleave(self.tile_size, dim=-2)  # [B, 1, num_tiles, Lk]
                exclude_last_tile_mask = exclude_last_tile_mask[:, :, :Lq, :]  # [B, 1, Lq, Lk]
                exclude_last_tile_mask = (exclude_last_tile_mask == attention_mask).to(attention_mask.device)  # [B, 1, Lq, Lk] 
                attn_weights.masked_fill_(exclude_last_tile_mask, 0)
            else:
                attn_weights.fill_(0)
            attn_weights.scatter_(3, indices, values)
            attn_weights.div_(attn_weights.sum(dim=-1, keepdim=True))

        return attn_weights