"""
This logic is largely taken from the flash_attn backend file in vllm and modified
"""

"""Kascade Attention Backend"""

from dataclasses import dataclass
from typing import ClassVar

import torch
import tilelang.language as T

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.model_executor.models.utils import extract_layer_index

from kascade.kernels.flash_decoding.recompute_kascade_gqa_decode_paged import (
    flashattn as decode_recompute_paged_kernel,
    compute_adaptive_block_H,
    compute_optimal_split,
    get_arch_config,
    should_use_splitk_topk,
    _tl_fused_topk_impl,
    _tl_fused_topk_splitk_3k_impl,
)
from kascade.kernels.flash_decoding.reuse_kascade_gqa_decode_paged import (
    flashattn_flat as decode_reuse_flat_kernel,
)

logger = init_logger(__name__)


class KascadeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16] # TODO: support bf16


    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "KASCADE_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """KascadeAttention supports all attention types."""
        # DD: Current support only for decoding so decoder only models prefill is FA-3
        return attn_type in (
            AttentionType.DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["KascadeAttentionImpl"]:
        return KascadeAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["KascadeAttentionMetadataBuilder"]:
        return KascadeAttentionMetadataBuilder
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    # DD: logical and physical shapes match stride order not needed

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        # no fp8
        return kv_cache_dtype in ["auto"]

    @classmethod
    def supports_sink(cls) -> bool:
        # no sink support
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


@dataclass
class KascadeAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    #kascade specific metadata
    k: int
    recompute_layers: list[int]
    head_mapping: torch.Tensor
    topk_indices: torch.Tensor | None
    recompute_glse: torch.Tensor | None = None
    recompute_output_partial: torch.Tensor | None = None
    reuse_glse: torch.Tensor | None = None
    reuse_output_partial: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    agg_scores: torch.Tensor | None = None
    row_sums: torch.Tensor | None = None
    topk_histograms: torch.Tensor | None = None
    topk_threshold_info: torch.Tensor | None = None
    topk_emit_ctr: torch.Tensor | None = None

    causal: bool = True


class KascadeAttentionMetadataBuilder(AttentionMetadataBuilder[KascadeAttentionMetadata]):
    # Current suport only for single token decode with cudagraphs since no unified kernel
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        
        kascade_cfg = vllm_config.additional_config or {}
        self.k = kascade_cfg.get("topk", 10)
        self.recompute_layers = kascade_cfg.get("recompute_layers", [0, 2, 8, 13, 14])
        model_name = vllm_config.model_config.model
        self.head_mapping = self.get_head_mapping(model_name).to(self.device).to(torch.int32)

    def get_head_mapping(self, model_name: str):
        try:
            import numpy as np
            model_basename = model_name.split('/')[-1]
            head_mapping = torch.from_numpy(np.load(
                f"./results/head_mappings/{model_basename}_layers_{"_".join(map(str, self.recompute_layers))}.npy"
            ))
        except Exception:
            raise ValueError(f"Head mapping file not found for model {model_name} with layers {self.recompute_layers}.")
        return head_mapping

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KascadeAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        # scores = torch.empty(num_decodes, self.num_heads_q, max_seq_len, device=self.device, dtype=self.model_config.dtype)
        # agg_scores = torch.full((num_decodes, self.num_heads_kv, max_seq_len), float('-inf'), device=self.device, dtype=self.model_config.dtype)
        # row_sums = torch.empty(num_decodes, self.num_heads_q, device=self.device, dtype=torch.float32)

        attn_metadata = KascadeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            causal=causal,
            k=self.k,
            recompute_layers=self.recompute_layers,
            head_mapping=self.head_mapping,
            # scores=scores,
            # agg_scores=agg_scores,
            # row_sums=row_sums,
            topk_indices=None,
        )
        return attn_metadata


class KascadeAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = vllm_is_batch_invariant()
        kernel_tuning = get_arch_config()
        self.recompute_block_N = kernel_tuning["recompute_block_N"]
        self.reuse_block_N = kernel_tuning["reuse_block_N"]
        self.block_H = compute_adaptive_block_H(num_heads, num_kv_heads)
        self.topk_splitk_splits = kernel_tuning["topk_splitk_splits"]
        recompute_threads = kernel_tuning["recompute_threads"]
        reuse_threads = kernel_tuning["reuse_threads"]
        self.topk_hist_bins = 1 << 10
        vllm_config = get_current_vllm_config()
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_blocks = cdiv(self.max_model_len, self.block_size)
        self.recompute_kernel_0 = decode_recompute_paged_kernel(
                                    self.num_heads, 
                                    self.num_kv_heads, 
                                    self.max_blocks,
                                    T.symbolic("num_blocks"),
                                    self.block_size,
                                    T.symbolic("max_seqlen"),
                                    self.head_size, 
                                    0,
                                    batch=T.symbolic("batch"),
                                    block_N=self.recompute_block_N,
                                    block_H=self.block_H,
                                    num_split=T.symbolic("num_split"),
                                    num_stages=1,
                                    threads=recompute_threads,
                                )
        self.recompute_kernel_1 = decode_recompute_paged_kernel(
                                    self.num_heads, 
                                    self.num_kv_heads, 
                                    self.max_blocks,
                                    T.symbolic("num_blocks"),
                                    self.block_size,
                                    T.symbolic("max_seqlen"),
                                    self.head_size, 
                                    1,
                                    batch=T.symbolic("batch"),
                                    block_N=self.recompute_block_N,
                                    block_H=self.block_H,
                                    num_split=T.symbolic("num_split"),
                                    num_stages=1,
                                    threads=recompute_threads,
                                )
        self.reuse_kernel = decode_reuse_flat_kernel(
                                    heads=self.num_heads, 
                                    groups=self.num_kv_heads, 
                                    dim=self.head_size,
                                    total_tokens=T.symbolic("total_tokens"),
                                    batch=T.symbolic("batch"),
                                    block_N=self.reuse_block_N,
                                    block_H=self.block_H,
                                    num_split=T.symbolic("num_split"),
                                    num_stages=1,
                                    threads=reuse_threads,
                                    topk=T.symbolic("topk"),
                                )
        self.topk_kern = _tl_fused_topk_impl(self.num_kv_heads, self.block_size, self.reuse_block_N)
        self.topk_kern_splitk = _tl_fused_topk_splitk_3k_impl(
            self.num_kv_heads,
            self.block_size,
            self.topk_splitk_splits,
            self.reuse_block_N,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KascadeAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Kascade Attention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for TiellangAttentionImpl."
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)
        layer_idx = extract_layer_index(layer.layer_name)


        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        if attn_metadata.num_decodes > 0:
            query = query.contiguous()
            batch = attn_metadata.num_decodes
            max_seq = attn_metadata.max_seq_len
            recompute_splits = compute_optimal_split(batch, self.num_heads, self.num_kv_heads, self.max_model_len, self.recompute_block_N, self.block_H, layer=0 if layer_idx == 0 else 1)
            reuse_splits = compute_optimal_split(batch, self.num_heads, self.num_kv_heads, self.max_model_len, self.reuse_block_N, self.block_H, topk_percent=attn_metadata.k)
            if layer_idx in attn_metadata.recompute_layers:
                if attn_metadata.recompute_glse is None:
                    attn_metadata.recompute_glse = torch.empty(batch, self.num_heads, recompute_splits, device=query.device, dtype=torch.float32)
                    attn_metadata.recompute_output_partial = torch.empty(batch, self.num_heads, recompute_splits, self.head_size, device=query.device, dtype=torch.float32)
                if layer_idx != 0 and attn_metadata.reuse_glse is None:
                    attn_metadata.reuse_glse = torch.empty(batch, self.num_heads, reuse_splits, device=query.device, dtype=torch.float32)
                    attn_metadata.reuse_output_partial = torch.empty(batch, self.num_heads, reuse_splits, self.head_size, device=query.device, dtype=torch.float32)
                self.recompute_forward(attn_metadata, query, key_cache, value_cache, block_table, seqused_k, layer_idx, reuse_splits, attn_metadata.recompute_glse, attn_metadata.recompute_output_partial, output)
            else:
                if attn_metadata.reuse_glse is None:
                    attn_metadata.reuse_glse = torch.empty(batch, self.num_heads, reuse_splits, device=query.device, dtype=torch.float32)
                    attn_metadata.reuse_output_partial = torch.empty(batch, self.num_heads, reuse_splits, self.head_size, device=query.device, dtype=torch.float32)
                self.reuse_forward(attn_metadata, query, key_cache, value_cache, block_table, seqused_k, layer_idx, attn_metadata.reuse_glse, attn_metadata.reuse_output_partial, output)


        if attn_metadata.num_prefills > 0:
            flash_attn_varlen_func(
                q=query[attn_metadata.num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[attn_metadata.num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q[attn_metadata.num_decodes:] - attn_metadata.num_decode_tokens,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k[attn_metadata.num_decodes:],
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                block_table=block_table[attn_metadata.num_decodes:],
                fa_version=self.vllm_flash_attn_version,
            )

        return output

    def recompute_forward(
        self,
        attn_metadata: KascadeAttentionMetadata,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seqused_k: torch.Tensor,
        layer_idx: int,
        reuse_splits: int,
        glse: torch.Tensor,
        output_partial: torch.Tensor,
        output: torch.Tensor,
    ):
        batch = attn_metadata.num_decodes
        max_seq_len = attn_metadata.max_seq_len
        if attn_metadata.scores is None:
            attn_metadata.scores = torch.empty(batch, self.num_heads, max_seq_len, device=query.device, dtype=query.dtype)
            attn_metadata.agg_scores = torch.empty(batch, self.num_kv_heads, max_seq_len, device=query.device, dtype=query.dtype)
            attn_metadata.row_sums = torch.empty(batch, self.num_heads, device=query.device, dtype=torch.float32)
        scores = attn_metadata.scores
        agg_scores = attn_metadata.agg_scores
        agg_scores.fill_(float('-inf'))
        row_sums = attn_metadata.row_sums
        if layer_idx == 0:
            self.recompute_kernel_0(
                query[:attn_metadata.num_decode_tokens],
                key_cache, value_cache,
                block_table[:attn_metadata.num_decodes],
                seqused_k[:attn_metadata.num_decodes],
                self.scale,
                glse, output_partial, row_sums, scores,
                output[:attn_metadata.num_decode_tokens], agg_scores,
            )
        else:
            self.recompute_kernel_1(
                query[:attn_metadata.num_decode_tokens],
                key_cache, value_cache,
                block_table[:attn_metadata.num_decodes],
                seqused_k[:attn_metadata.num_decodes],
                self.scale,
                glse, output_partial, row_sums, scores,
                output[:attn_metadata.num_decode_tokens], agg_scores,
            )
        max_topk = ((int((attn_metadata.k * max_seq_len) / 100) + self.reuse_block_N - 1)
                    // self.reuse_block_N) * self.reuse_block_N
        max_topk = max(max_topk, self.reuse_block_N)
        if attn_metadata.topk_indices is None:
            attn_metadata.topk_indices = torch.zeros(batch, self.num_kv_heads, max_topk, dtype=torch.int32, device=query.device)

        use_splitk_topk = should_use_splitk_topk(batch, self.num_kv_heads)
        if use_splitk_topk:
            if attn_metadata.topk_histograms is None:
                attn_metadata.topk_histograms = torch.empty(
                    batch,
                    self.num_kv_heads,
                    self.topk_splitk_splits,
                    self.topk_hist_bins,
                    dtype=torch.int32,
                    device=query.device,
                )
                attn_metadata.topk_threshold_info = torch.empty(
                    batch, self.num_kv_heads, 5, dtype=torch.int32, device=query.device
                )
                attn_metadata.topk_emit_ctr = torch.empty(
                    batch, self.num_kv_heads, 2, dtype=torch.int32, device=query.device
                )
            self.topk_kern_splitk(
                agg_scores,
                seqused_k[:batch],
                block_table[:batch],
                attn_metadata.k / 100.0,
                attn_metadata.topk_indices,
                attn_metadata.topk_histograms,
                attn_metadata.topk_threshold_info,
                attn_metadata.topk_emit_ctr,
            )
        else:
            self.topk_kern(
                agg_scores,
                seqused_k[:batch],
                block_table[:batch],
                attn_metadata.k / 100.0,
                attn_metadata.topk_indices,
            )
        if layer_idx != 0:
            self.reuse_forward(attn_metadata, query, key_cache, value_cache, block_table, seqused_k, layer_idx, attn_metadata.reuse_glse, attn_metadata.reuse_output_partial, output)
        return output

    def reuse_forward(
        self,
        attn_metadata: KascadeAttentionMetadata,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seqused_k: torch.Tensor,
        layer_idx: int,
        glse: torch.Tensor,
        output_partial: torch.Tensor,
        output: torch.Tensor,
    ):
        total_tokens = key_cache.shape[0] * key_cache.shape[1]
        k_flat = key_cache.reshape(total_tokens, self.num_kv_heads, self.head_size)
        v_flat = value_cache.reshape(total_tokens, self.num_kv_heads, self.head_size)
        self.reuse_kernel(
            query[:attn_metadata.num_decode_tokens],
            k_flat, v_flat,
            seqused_k[:attn_metadata.num_decodes],
            attn_metadata.topk_indices,
            attn_metadata.head_mapping[layer_idx],
            attn_metadata.k,
            self.scale,
            glse, output_partial,
            output[:attn_metadata.num_decode_tokens],
        )
        return output
