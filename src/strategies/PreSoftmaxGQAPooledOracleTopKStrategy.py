# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .OracleTopkLayer0GlobalStrategy import OracleTopkLayer0GlobalStrategy
import torch
from .attention_utils import softmax_

class PreSoftmaxGQAPooledOracleTopKStrategy(OracleTopkLayer0GlobalStrategy): # Does not support stats collection since never used for stats collection
    def __init__(self, name='pre_softmax_gqa_pooled_oracle_topk', k=1, block_size=12288):
        super().__init__(name=name, k=k, block_size=block_size)

    def _softmax_topk_sparsify_(self, attn_weights, attention_mask, module, k, is_decode, start, end):
        """
        Pre-softmax GQA pooling: Pool raw scores first, then select top-k, then apply softmax.
        """
        B, H, Lq, Lk = attn_weights.shape
        attn_weights_per_khead = attn_weights.view(B, module.config.num_key_value_heads, module.num_key_value_groups, Lq, Lk).sum(dim=2)  # [B, G, Lq, Lk]
        indices = torch.topk(attn_weights_per_khead, k, dim=-1).indices  # [B, G, Lq, k]
        indices = indices.repeat_interleave(module.num_key_value_groups, dim=1)  # [B, H, Lq, k]
        softmax_(attn_weights, dim=-1)
        values = attn_weights.gather(dim=3, index=indices)
        if module.layer_idx > 0: # Layer 0: No modification - keeps original softmax weights
            attn_weights.fill_(0)
            attn_weights.scatter_(3, indices, values)
            attn_weights.div_(attn_weights.sum(dim=-1, keepdim=True))
        
        return attn_weights