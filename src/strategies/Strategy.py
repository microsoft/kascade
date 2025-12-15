# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch import nn
import torch

class Strategy:
    def __init__(self, name: str):
        self.name = name
        self.register_attention()
    
    def register_attention(self):
        pass
    
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
        pass
