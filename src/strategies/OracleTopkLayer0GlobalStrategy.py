# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .OracleTopkStrategy import OracleTopkStrategy
from .attention_utils import softmax_
import torch

class OracleTopkLayer0GlobalStrategy(OracleTopkStrategy): #Does not support stats collection since nothing different compared to OracleTopkStrategy in terms of stats
    def __init__(self, name='oracle_topk_layer0_global', k=1, block_size=12288):
        super().__init__(name=name, k=k, block_size=block_size)

    def _softmax_topk_sparsify_(self, attn_weights, attention_mask, module, k, is_decode, start, end):
        """
        Override to only apply top-k sparsification for layers after layer 0.
        Layer 0 keeps the original full attention weights.
        """
        softmax_(attn_weights, dim=-1)
        values, indices = torch.topk(attn_weights, k, dim=-1)
        if module.layer_idx > 0: # Layer 0: No modification - keeps original softmax weights
            attn_weights.fill_(0)
            attn_weights.scatter_(3, indices, values)
            attn_weights.div_(attn_weights.sum(dim=-1, keepdim=True))
        
        return attn_weights

