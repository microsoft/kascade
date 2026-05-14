# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def __getattr__(name: str):
    if name == "TilelangAttentionBackend":
        from .tilelang_attn import TilelangAttentionBackend
        return TilelangAttentionBackend
    if name == "KascadeAttentionBackend":
        from .kascade_attn import KascadeAttentionBackend
        return KascadeAttentionBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def configure_attention_backend(backend: str) -> dict:
    """Configure vLLM attention backend before importing vLLM runtime modules.

    Args:
        backend: One of ``"default"``, ``"tilelang"``, or ``"kascade"``.

    Returns:
        Dict of kwargs to pass to LLM() for attention_config.
    """
    if backend in {"tilelang", "kascade"}:
        # Bind FlashInfer to the real CUDA runtime before TileLang backends load.
        import torch  # noqa: F401
        import flashinfer.comm.cuda_ipc  # noqa: F401

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
    )

    if backend == "default":
        # Auto-select FlashAttention version based on GPU architecture
        import torch
        from vllm.config import AttentionConfig
        major, _ = torch.cuda.get_device_capability()
        if major >= 10:  # Blackwell (B200) or newer
            return {"attention_config": AttentionConfig(backend="FLASH_ATTN", flash_attn_version=4)}
        else:  # Hopper (H100) or older - explicit FA3
            return {"attention_config": AttentionConfig(backend="FLASH_ATTN", flash_attn_version=3)}

    backend_map = {
        "tilelang": ("TILELANG_ATTN", "kascade.vllm_backends.TilelangAttentionBackend"),
        "kascade": ("KASCADE_ATTN", "kascade.vllm_backends.KascadeAttentionBackend"),
    }
    if backend not in backend_map:
        raise ValueError(f"Unknown vllm attention backend: {backend!r}. Choose 'default', 'tilelang', or 'kascade'.")

    enum_name, class_path = backend_map[backend]

    # Register custom backend as a new enum member
    member = object.__new__(AttentionBackendEnum)
    member._name_ = enum_name
    member._value_ = class_path
    AttentionBackendEnum._member_map_[enum_name] = member
    if enum_name not in AttentionBackendEnum._member_names_:
        AttentionBackendEnum._member_names_.append(enum_name)
    AttentionBackendEnum._value2member_map_[class_path] = member

    from vllm.config import AttentionConfig
    return {"attention_config": AttentionConfig(backend=enum_name)}


__all__ = ["TilelangAttentionBackend", "KascadeAttentionBackend", "configure_attention_backend"]
