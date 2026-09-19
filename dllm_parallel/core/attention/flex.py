# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Runtime contract and compiled primitives for BDLM FlexAttention."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import torch

from dllm_parallel.core.attention.fa4 import (
    fa4_backward_from_state,
    fa4_dense_backward_from_state,
)
_compiled_backward_from_state: dict[tuple[Any, ...], Callable[..., Any]] = {}


@dataclass(frozen=True)
class FlexAttentionMetadata:
    backend: str
    fa4_version: str
    native_pack_gqa_backward: bool
    torch_version: str
    torch_cuda_version: str | None
    device_capability: tuple[int, int] | None

    def to_log_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_native_pack_gqa_backward(interface: Any) -> None:
    """Reject FA4 runtimes that silently disable the fused GQA backward."""

    if not callable(
        getattr(interface, "_validate_pack_gqa_backward_capability", None)
    ):
        raise RuntimeError(
            "BDLM fused CP/BP requires an FA4 runtime with native Pack-GQA "
            "backward support; reinstall the vendored flash-attn-4 package"
        )


def verify_flex_attention_runtime() -> FlexAttentionMetadata:
    try:
        from torch.nn.attention.flex_attention import (
            AuxRequest,
            _apply_kernel_options,
            _identity,
            create_block_mask,
            flex_attention,
        )
        from torch._higher_order_ops.flex_attention import (
            create_fw_bw_graph,
            flex_attention_backward,
        )
        from torch._inductor.compile_fx import compile_fx_inner
        from torch.fx.experimental.proxy_tensor import make_fx
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "BDLM training requires the Torch 2.10 FlexAttention training runtime"
        ) from error
    required = (
        flex_attention,
        create_block_mask,
        _apply_kernel_options,
        _identity,
        create_fw_bw_graph,
        flex_attention_backward,
        compile_fx_inner,
        make_fx,
    )
    if not all(callable(item) for item in required):
        raise RuntimeError("Torch FlexAttention operators are not callable")
    if AuxRequest(lse=True).lse is not True:
        raise RuntimeError("Torch FlexAttention does not expose LSE auxiliary output")
    if not callable(getattr(torch, "compile", None)):
        raise RuntimeError("BDLM FlexAttention requires torch.compile")
    try:
        from flash_attn.cute import interface as fa4_interface
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "BDLM training requires the vendored flash-attn-4 runtime"
        ) from error
    _require_native_pack_gqa_backward(fa4_interface)
    try:
        fa4_version = version("flash-attn-4")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "BDLM training requires an installed flash-attn-4 distribution"
        ) from error
    capability = (
        tuple(int(value) for value in torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else None
    )
    return FlexAttentionMetadata(
        backend="torch_compiled_flex_attention",
        fa4_version=fa4_version,
        native_pack_gqa_backward=True,
        torch_version=str(torch.__version__),
        torch_cuda_version=(
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        device_capability=capability,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(int(size) for size in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


def _compile_backward_from_state(
    *,
    inputs: tuple[torch.Tensor, ...],
    block_mask: tuple[Any, ...],
    mask_mod: Callable[..., torch.Tensor],
    mask_buffer_count: int,
    scale: float,
    kernel_options: dict[str, Any] | None,
) -> Callable[..., Any]:
    from torch._higher_order_ops.flex_attention import (
        create_fw_bw_graph,
        flex_attention_backward,
    )
    from torch._inductor.compile_fx import compile_fx_inner
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.nn.attention.flex_attention import _apply_kernel_options, _identity

    query, key, value = inputs[:3]
    options = _apply_kernel_options(
        query,
        key,
        value,
        True,
        kernel_options,
    )
    score_examples = (
        query.new_zeros((), requires_grad=True),
        query.new_zeros((), dtype=torch.int),
        query.new_zeros((), dtype=torch.int),
        query.new_zeros((), dtype=torch.int),
        query.new_zeros((), dtype=torch.int),
    )
    score_graph, joint_graph = create_fw_bw_graph(
        _identity,
        score_examples,
        (),
    )
    query_length, key_length = block_mask[:2]
    query_block_size, key_block_size = block_mask[10:12]

    def invoke(
        query_arg: torch.Tensor,
        key_arg: torch.Tensor,
        value_arg: torch.Tensor,
        output_arg: torch.Tensor,
        lse_arg: torch.Tensor,
        grad_output_arg: torch.Tensor,
        grad_lse_arg: torch.Tensor,
        kv_num_blocks: torch.Tensor,
        kv_indices: torch.Tensor,
        full_kv_num_blocks: torch.Tensor,
        full_kv_indices: torch.Tensor,
        q_num_blocks: torch.Tensor,
        q_indices: torch.Tensor,
        full_q_num_blocks: torch.Tensor,
        full_q_indices: torch.Tensor,
        *mask_buffers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_query, grad_key, grad_value, _ = flex_attention_backward(
            query_arg,
            key_arg,
            value_arg,
            output_arg,
            lse_arg,
            grad_output_arg,
            grad_lse_arg,
            score_graph,
            joint_graph,
            (
                query_length,
                key_length,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                q_num_blocks,
                q_indices,
                full_q_num_blocks,
                full_q_indices,
                query_block_size,
                key_block_size,
                mask_mod,
            ),
            float(scale),
            options,
            (),
            tuple(mask_buffers[:mask_buffer_count]),
        )
        return grad_query, grad_key, grad_value

    graph = make_fx(invoke, tracing_mode="fake")(*inputs)
    return compile_fx_inner(
        graph,
        inputs,
        cudagraphs=False,
        is_backward=True,
        is_inference=False,
    )


def flex_attention_backward_from_state(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    block_mask: Any,
    mask_mod: Callable[..., torch.Tensor],
    mask_buffers: tuple[torch.Tensor, ...],
    scale: float,
    kernel_options: dict[str, Any] | None,
    packed_gqa: bool = False,
    force_torch: bool = False,
    grad_query: torch.Tensor | None = None,
    grad_key: torch.Tensor | None = None,
    grad_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate FlexAttention from an already-merged output and LSE."""

    if not query.is_cuda:
        raise RuntimeError("FA4 FlexAttention backward requires CUDA tensors")
    block_mask_tuple = block_mask.as_tuple()
    block_tensors = tuple(block_mask_tuple[2:10])
    if not all(torch.is_tensor(item) for item in block_tensors):
        raise RuntimeError("FlexAttention block metadata is incomplete")
    if bool(force_torch):
        gradients = _torch_reference_backward_from_state(
            query=query,
            key=key,
            value=value,
            output=output.to(dtype=query.dtype),
            lse=lse,
            grad_output=grad_output.to(dtype=query.dtype),
            grad_lse=grad_lse,
            block_mask=block_mask,
            mask_mod=mask_mod,
            mask_buffers=mask_buffers,
            scale=float(scale),
            kernel_options=kernel_options,
        )
        copied = []
        for actual, supplied in zip(
            gradients,
            (grad_query, grad_key, grad_value),
        ):
            if supplied is None:
                copied.append(actual)
            else:
                supplied.copy_(actual)
                copied.append(supplied)
        return copied[0], copied[1], copied[2]
    return fa4_backward_from_state(
        query=query,
        key=key,
        value=value,
        output=output.to(dtype=query.dtype),
        lse=lse,
        grad_output=grad_output.to(dtype=query.dtype),
        block_mask=block_mask_tuple,
        mask_buffers=mask_buffers,
        scale=float(scale),
        pack_gqa=bool(packed_gqa),
        grad_lse=grad_lse,
        grad_query=grad_query,
        grad_key=grad_key,
        grad_value=grad_value,
    )


def dense_attention_backward_from_state(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate unmasked attention from an already-merged output and LSE."""

    return fa4_dense_backward_from_state(
        query=query,
        key=key,
        value=value,
        output=output,
        lse=lse,
        grad_output=grad_output,
        grad_lse=grad_lse,
        scale=float(scale),
    )


def _torch_reference_backward_from_state(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    grad_lse: torch.Tensor | None,
    block_mask: Any,
    mask_mod: Callable[..., torch.Tensor],
    mask_buffers: tuple[torch.Tensor, ...],
    scale: float,
    kernel_options: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch compiler reference used to validate the production FA4 path."""

    if not query.is_cuda:
        raise RuntimeError("compiled FlexAttention backward requires CUDA tensors")
    block_mask_tuple = block_mask.as_tuple()
    block_tensors = tuple(block_mask_tuple[2:10])
    if not all(torch.is_tensor(item) for item in block_tensors):
        raise RuntimeError("FlexAttention block metadata is incomplete")
    lse_log2 = lse * (1.0 / math.log(2.0))
    grad_lse_tensor = torch.zeros_like(lse) if grad_lse is None else grad_lse
    inputs = (
        query,
        key,
        value,
        output,
        lse_log2,
        grad_output,
        grad_lse_tensor,
        *block_tensors,
        *mask_buffers,
    )
    cache_key = (
        tuple(_tensor_signature(tensor) for tensor in inputs),
        int(block_mask_tuple[0]),
        int(block_mask_tuple[1]),
        int(block_mask_tuple[10]),
        int(block_mask_tuple[11]),
        float(scale),
        tuple(sorted((kernel_options or {}).items())),
        mask_mod.__module__,
        mask_mod.__qualname__,
    )
    compiled = _compiled_backward_from_state.get(cache_key)
    if compiled is None:
        compiled = _compile_backward_from_state(
            inputs=inputs,
            block_mask=block_mask_tuple,
            mask_mod=mask_mod,
            mask_buffer_count=len(mask_buffers),
            scale=float(scale),
            kernel_options=kernel_options,
        )
        _compiled_backward_from_state[cache_key] = compiled
    grad_query, grad_key, grad_value = compiled(list(inputs))
    return grad_query, grad_key, grad_value


__all__ = [
    "FlexAttentionMetadata",
    "dense_attention_backward_from_state",
    "flex_attention_backward_from_state",
    "verify_flex_attention_runtime",
]
