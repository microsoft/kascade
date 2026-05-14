"""
This logic is largely taken from the flash_attn backend file in vllm and modified
"""

"""ilelang Flash Attention Backend"""

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

from kascade.kernels.flash_decoding.gqa_decode_paged import flashattn
from kascade.kernels.flash_decoding.recompute_kascade_gqa_decode_paged import (
    compute_adaptive_block_H,
    compute_optimal_split,
)

logger = init_logger(__name__)


class TilelangAttentionBackend(AttentionBackend):
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
        return "TILELANG_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TilelangAttention supports all attention types."""
        # DD: Current support only for decoding so decoder only models prefill is FA-3
        return attn_type in (
            AttentionType.DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["TilelangAttentionImpl"]:
        return TilelangAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TilelangAttentionMetadataBuilder"]:
        return TilelangAttentionMetadataBuilder
    
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
class TilelangAttentionMetadata:
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

    causal: bool = True

    glse: torch.Tensor | None = None
    output_partial: torch.Tensor | None = None


class TilelangAttentionMetadataBuilder(AttentionMetadataBuilder[TilelangAttentionMetadata]):
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


    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TilelangAttentionMetadata:
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

        attn_metadata = TilelangAttentionMetadata(
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
        )
        return attn_metadata


class TilelangAttentionImpl(AttentionImpl):
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
        self.block_N = 192
        self.block_H = compute_adaptive_block_H(num_heads, num_kv_heads)
        vllm_config = get_current_vllm_config()
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_blocks = cdiv(self.max_model_len, self.block_size)
        self.kernel = flashattn(
            heads=self.num_heads,
            groups=self.num_kv_heads,
            dim=self.head_size,
            max_blocks=self.max_blocks,
            num_blocks=T.symbolic("num_blocks"),
            block_size=self.block_size,
            batch=T.symbolic("batch"),
            block_N=self.block_N,
            block_H=self.block_H,
            num_split=T.symbolic("num_split"),
            num_stages=1,
            threads=128)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TilelangAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Tilelang Attention.

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
            num_splits = compute_optimal_split(batch, self.num_heads, self.num_kv_heads, max_seq, self.block_N, self.block_H)
            if attn_metadata.glse is None:
                attn_metadata.glse = torch.empty(batch, self.num_heads, num_splits, device=query.device, dtype=torch.float32)
                attn_metadata.output_partial = torch.empty(batch, self.num_heads, num_splits, self.head_size, device=query.device, dtype=torch.float32)
            self.kernel(query[:attn_metadata.num_decode_tokens], key_cache, value_cache, block_table[:batch], seqused_k[:batch], self.scale, attn_metadata.glse, attn_metadata.output_partial, output[:attn_metadata.num_decode_tokens]) 


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
