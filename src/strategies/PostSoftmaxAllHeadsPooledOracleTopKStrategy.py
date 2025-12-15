# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .PostSoftmaxGQAPooledOracleTopKStrategy import PostSoftmaxGQAPooledOracleTopKStrategy
import torch
from .attention_utils import softmax_

class PostSoftmaxAllHeadsPooledOracleTopKStrategy(PostSoftmaxGQAPooledOracleTopKStrategy):
    def __init__(self, name='post_softmax_all_heads_pooled_oracle_topk', k=1, block_size=12288):
        super().__init__(name=name, k=k, block_size=block_size)

    def _get_head_count_for_stats(self, module):
        """Use key-value heads for statistics tracking in GQA."""
        return 1

    def _softmax_topk_sparsify_(self, attn_weights, attention_mask, module, k, is_decode, start, end):
        """
        Post-softmax All Heads pooling: Apply softmax then pool attention weights across all heads, then compute top-k.
        """
        softmax_(attn_weights, dim=-1)
        B, H, Lq, Lk = attn_weights.shape
        attn_weights_summed = attn_weights.sum(dim=1, keepdim=True)  # [B, 1, Lq, Lk]
        indices = torch.topk(attn_weights_summed, k, dim=-1).indices  # [B, 1, Lq, k]

        ### Stats collection
        if self._stats_runner is not None:
            self._stats_runner.update(is_decode, module.layer_idx, indices, attn_weights_summed.to(torch.float32))
            return attn_weights # No modification in layer 0 when collecting stats
        
        indices = indices.expand(-1, module.config.num_attention_heads, -1, -1)  # [B, H, Lq, k]
        values = attn_weights.gather(dim=3, index=indices)
        if module.layer_idx > 0: # Layer 0: No modification - keeps original softmax weights
            attn_weights.fill_(0)
            attn_weights.scatter_(3, indices, values)
            attn_weights.div_(attn_weights.sum(dim=-1, keepdim=True))
        
        return attn_weights