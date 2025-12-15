# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .KascadeStrategy import KascadeStrategy
from kascade.kernels import decode_reuse_kernel, decode_recompute_kernel, prefill_reuse_kernel, prefill_recompute_kernel, compile_custom_kernel_from_cu
from typing import List, Optional
import tilelang
import tilelang.language as T
from .attention_utils import get_heuristic_config
import torch
from torch import nn
import math
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

class EfficientKascadeStrategy(KascadeStrategy):
    def __init__(self, recompute_layers: List[int], model_name: str, k=1, tile_size=1, rolling_prefill=False, block_size=12288):
        super().__init__(name="efficient_kascade", recompute_layers=recompute_layers, model_name=model_name, k=k, tile_size=tile_size, rolling_prefill=rolling_prefill, block_size=block_size)
        self._heads = 32
        self._dim = 128
        self._groups = 8
        config = get_heuristic_config()

        decode_recomp0_program = decode_recompute_kernel(T.symbolic("batch"), self._heads, self._groups, T.symbolic("kv_seqlen"), self._dim, tune=False, layer=0)(**config)
        self._decode_recomp0_kernel = tilelang.compile(decode_recomp0_program, out_idx=[7, 8], pass_configs={ "tl.disable_safe_memory_legalize": True , })

        decode_recomp1_program = decode_recompute_kernel(T.symbolic("batch"), self._heads, self._groups, T.symbolic("kv_seqlen"), self._dim, tune=False, layer=1)(**config)
        self._decode_recomp1_kernel = tilelang.compile(decode_recomp1_program, out_idx=[7, 8], pass_configs={ "tl.disable_safe_memory_legalize": True , })

        decode_reuse_program = decode_reuse_kernel(T.symbolic("batch"), self._heads, self._groups, T.symbolic("kv_seqlen"), self._dim, T.symbolic("topk"), tune=False)(**config)
        self._decode_reuse_kernel = compile_custom_kernel_from_cu(
            func=decode_reuse_program,
            cu_file_path="./src/kernels/flash_decoding/kascade_decode_cu_kernels/dynamic_alllen_reuse_cp_async_dim128_gsize4_heads32_groups8.cu",  # Path to your compiled CUDA source
            so_dir="./src/kernels/flash_decoding/kascade_decode_cu_kernels/__socache__/",  # Path to your compiled library   
            out_idx=[7],  
            execution_backend="cython",  # or "ctypes"  
            pass_configs={ "tl.disable_safe_memory_legalize": True , },  
        )

        prefill_recomp0_program = prefill_recompute_kernel(
            T.symbolic("batch"), self._heads, T.symbolic("seq_len"), self._dim, tune=False, groups=self._groups, kernel_type="prefill")(
                block_M=128, block_H=(self._heads // self._groups), block_N=128, num_stages=2, threads=256)
        self._prefill_recomp0_kernel = tilelang.compile(prefill_recomp0_program, out_idx=[3, 4], pass_configs={ "tl.disable_safe_memory_legalize": True})
        prefill_recomp1_program = prefill_recompute_kernel(
            T.symbolic("batch"), self._heads, T.symbolic("seq_len"), self._dim, tune=False, groups=self._groups, kernel_type="compute_scores")(
                block_M=128, block_H=(self._heads // self._groups), block_N=256, num_stages=2, threads=256)
        self._prefill_recomp1_kernel = tilelang.compile(prefill_recomp1_program, out_idx=[2], pass_configs={ "tl.disable_safe_memory_legalize": True})

        aggregate_program = prefill_recompute_kernel(
                T.symbolic("batch"), self._heads, T.symbolic("seq_len"), self._dim, tune=False, groups=self._groups, kernel_type="aggregate")(
                    block_M=tile_size, block_H=(self._heads // self._groups), block_N=256, num_stages=2, threads=256)
        self._aggregate_kernel = compile_custom_kernel_from_cu(
            func=aggregate_program,
            cu_file_path=f"./src/kernels/flash_attention/kascade_prefill_cu_kernels/dynamic_alllen_aggregate_cp_async_ts{tile_size}_dim128_gsize4_heads32_groups8.cu",  # Path to your compiled CUDA source
            so_dir=f"./src/kernels/flash_attention/kascade_prefill_cu_kernels/__socache__/",  # Path to your compiled library
            out_idx=None,
            execution_backend="cython",  # or "ctypes"
            pass_configs={ "tl.disable_safe_memory_legalize": True},
        )

        prefill_reuse_program = prefill_reuse_kernel(T.symbolic("batch"), self._heads, T.symbolic("kv_seqlen"), self._dim, T.symbolic("max_topk_num"), rolling=self.rolling_prefill, tune=False, groups=self._groups)(
            block_M=tile_size, block_H=(self._heads // self._groups), block_N=128, num_stages=2, threads=256)
        self._prefill_reuse_kernel = compile_custom_kernel_from_cu(
            func=prefill_reuse_program,
            cu_file_path=f"./src/kernels/flash_attention/kascade_prefill_cu_kernels/dynamic_alllen_reuse_cp_async_ts{tile_size}_dim128_gsize4_heads32_groups8{'_rolling' if self.rolling_prefill else ''}.cu",  # Path to your compiled CUDA source
            so_dir=f"./src/kernels/flash_attention/kascade_prefill_cu_kernels/__socache__/",  # Path to your compiled library
            out_idx=[6],
            execution_backend="cython",  # or "ctypes"
            pass_configs={ "tl.disable_safe_memory_legalize": True , },
        )

    def _prefill_recomp_forward(self, Q, K, V, topk, tile_size, layer, topk_percent):
        B, L, H = Q.shape[:3]
        G = K.shape[2]
        scores = torch.full((B, G, math.ceil(L / tile_size), L), fill_value=float('-inf'), device=Q.device, dtype=Q.dtype)
        if layer == 0:
            log_sum, o = self._prefill_recomp0_kernel(Q, K, V)
        else:
            log_sum = self._prefill_recomp1_kernel(Q, K)
        self._aggregate_kernel(Q, K, log_sum, scores)
        if L <= 131072:
            self._topk_to_reuse = torch.topk(scores, k=topk, dim=-1).indices.to(torch.int32)
        else:
            self._topk_to_reuse = torch.empty((B, G, math.ceil(L / tile_size), topk), device=Q.device, dtype=torch.int32)
            chunk_size = 131072 // 2**int(topk_percent // 10)
            num_chunks = math.ceil(L / chunk_size)
            for i in range(num_chunks):
                start = (i * chunk_size) // tile_size
                end = min(((i + 1) * chunk_size ) // tile_size, L)
                self._topk_to_reuse[:, :, start:end, :] = torch.topk(scores[:, :, start:end, :], k=topk, dim=-1).indices.to(torch.int32)
        if layer != 0:
            o = self._prefill_reuse_kernel(Q, K, V, self._topk_to_reuse, torch.arange(G, device=Q.device, dtype=torch.int32), topk_percent)
        return o

    def _prefill_reuse_forward(self, Q, K, V, topk_percent, layer):
        head_mapping = torch.tensor(self._head_mapping[layer], device=Q.device, dtype=torch.int32)
        o = self._prefill_reuse_kernel(Q, K, V, self._topk_to_reuse, head_mapping, topk_percent)
        return o

    def _decode_recomp_forward(self, Q, K, V, topk, layer):
        B, L = K.shape[:2]
        H = Q.shape[1]
        glse = torch.empty(B, H, 8, device="cuda", dtype=torch.float32)
        Output_partial = torch.empty(B, H, 8, self._dim, device="cuda", dtype=torch.float32)
        row_sums = torch.empty(B, H, device="cuda", dtype=torch.float32)
        scores = torch.empty(B, H, L, device="cuda", dtype=torch.float16)
        if layer == 0:          
            o, scores = self._decode_recomp0_kernel(Q, K, V, glse, Output_partial, row_sums, scores)
        else:
            o, scores = self._decode_recomp1_kernel(Q, K, V, glse, Output_partial, row_sums, scores)
        self._topk_to_reuse = torch.topk(scores, k=topk, dim=-1).indices.to(torch.int32)
        if layer != 0:
            glse.zero_()
            o = self._decode_reuse_kernel(Q, K, V, self._topk_to_reuse, torch.arange(K.shape[2], dtype=torch.int32, device=K.device), glse, Output_partial)
        return o
    
    def _decode_reuse_forward(self, Q, K, V, layer):
        B, H = Q.shape[:2]
        glse = torch.empty(B, H, 8, device="cuda", dtype=torch.float32)
        Output_partial = torch.empty(B, H, 8, self._dim, device="cuda", dtype=torch.float32)
        head_mapping = torch.tensor(self._head_mapping[layer], device=Q.device, dtype=torch.int32)
        o = self._decode_reuse_kernel(Q, K, V, self._topk_to_reuse, head_mapping, glse, Output_partial)
        return o
    
    
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
        if key.shape[-2] <= 128:
            return ALL_ATTENTION_FUNCTIONS["sdpa"](
                module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
            )
        is_decode = (query.shape[2] == 1)
        k = min(max(128, int((self.k/100)*key.shape[-2])), key.shape[-2])
        rounded_k = min(((k + 127) // 128) * 128, key.shape[-2])  # round to multiple of 128
        query = query.transpose(1, 2) # B, L, H, D
        key = key.transpose(1, 2) # B, L, H, D
        value = value.transpose(1, 2) # B, L, H, D
        if is_decode:
            query = query.squeeze(1) # B, H, D
            key = key.contiguous()
            value = value.contiguous()
            if module.layer_idx in self.recompute_layers:
                o = self._decode_recomp_forward(query, key, value, rounded_k, module.layer_idx)
            else:
                o = self._decode_reuse_forward(query, key, value, module.layer_idx)
            o = o.unsqueeze(1) # B, 1, H, D
        else:
            if module.layer_idx in self.recompute_layers:
                o = self._prefill_recomp_forward(query, key, value, rounded_k, self.tile_size, module.layer_idx, self.k)
            else:
                o = self._prefill_reuse_forward(query, key, value, self.k, module.layer_idx)
        return o, None

    def register_attention(self):
        _attention_forward = lambda module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs: self.attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(self.name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        ALL_ATTENTION_FUNCTIONS.register(self.name, _attention_forward)