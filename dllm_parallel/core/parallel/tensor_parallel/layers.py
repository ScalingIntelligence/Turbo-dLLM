# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tensor-parallel nn.Module layers (column/row/vocab parallel)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm_parallel.core.kernels.tiled_linear_cross_entropy import tiled_linear_cross_entropy
from dllm_parallel.core.parallel.tensor_parallel._common import (
    _all_reduce_sum_,
    _ensure_group,
    _runtime_tp_group,
)
from dllm_parallel.core.parallel.tensor_parallel.cross_entropy import _VocabParallelCrossEntropy
from dllm_parallel.core.parallel.tensor_parallel.mappings import (
    _copy_with_tp_backward,
    _gather_from_tp,
    _reduce_from_tp,
    column_parallel_linear,
)
from dllm_parallel.core.parallel.tensor_parallel.random import (
    _kaiming_uniform_partition_,
    _uniform_bias_,
    partition_bounds,
    partition_sizes,
)
from dllm_parallel.core.parallel.tensor_parallel.sequence_parallel import (
    reduce_scatter_to_sequence_parallel_region,
)


class ColumnParallelLinear(nn.Module):
    """Linear layer with output features sharded over the TP group."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        gather_output: bool = False,
        local_out_features: int | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.tensor_parallel_rank = int(tensor_parallel_rank)
        if local_out_features is None:
            start, stop = partition_bounds(
                self.out_features,
                self.tensor_parallel_size,
                self.tensor_parallel_rank,
            )
            self.output_start = start
            self.output_stop = stop
            local_out_features = stop - start
        else:
            self.output_start = -1
            self.output_stop = -1
        self.local_out_features = int(local_out_features)
        self.gather_output = bool(gather_output)
        self.tensor_parallel_group = None
        self.weight = nn.Parameter(torch.empty(self.local_out_features, self.in_features))
        self.bias = (
            nn.Parameter(torch.empty(self.local_out_features)) if bias else None
        )
        self.weight._dllm_tensor_parallel_sharded = self.tensor_parallel_size > 1
        if self.bias is not None:
            self.bias._dllm_tensor_parallel_sharded = self.tensor_parallel_size > 1
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _kaiming_uniform_partition_(self.weight, self.tensor_parallel_rank)
        if self.bias is not None:
            _uniform_bias_(self.bias, self.in_features, self.tensor_parallel_rank)

    def set_tensor_parallel_runtime(self, runtime: Any | None) -> None:
        self.tensor_parallel_group = _runtime_tp_group(runtime)

    def copy_input(self, x: torch.Tensor) -> torch.Tensor:
        return _copy_with_tp_backward(
            x,
            self.tensor_parallel_group,
            self.tensor_parallel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = column_parallel_linear(
            x,
            self.weight,
            self.bias,
            group=self.tensor_parallel_group,
            world_size=self.tensor_parallel_size,
            allreduce_dgrad=True,
        )
        if self.gather_output:
            out = _gather_from_tp(
                out,
                group=self.tensor_parallel_group,
                world_size=self.tensor_parallel_size,
                rank=self.tensor_parallel_rank,
                sizes=partition_sizes(self.out_features, self.tensor_parallel_size),
            )
        return out


class RowParallelLinear(nn.Module):
    """Linear layer with input features sharded over the TP group."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        local_in_features: int | None = None,
        sequence_parallel_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.tensor_parallel_rank = int(tensor_parallel_rank)
        if local_in_features is None:
            start, stop = partition_bounds(
                self.in_features,
                self.tensor_parallel_size,
                self.tensor_parallel_rank,
            )
            self.input_start = start
            self.input_stop = stop
            local_in_features = stop - start
        else:
            self.input_start = -1
            self.input_stop = -1
        self.local_in_features = int(local_in_features)
        self.sequence_parallel_output = bool(sequence_parallel_output)
        self.tensor_parallel_group = None
        self.weight = nn.Parameter(torch.empty(self.out_features, self.local_in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.weight._dllm_tensor_parallel_sharded = self.tensor_parallel_size > 1
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _kaiming_uniform_partition_(self.weight, self.tensor_parallel_rank)
        if self.bias is not None:
            _uniform_bias_(self.bias, self.in_features, self.tensor_parallel_rank)

    def set_tensor_parallel_runtime(self, runtime: Any | None) -> None:
        self.tensor_parallel_group = _runtime_tp_group(runtime)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, None)
        if self.sequence_parallel_output:
            out = reduce_scatter_to_sequence_parallel_region(
                out,
                group=self.tensor_parallel_group,
                world_size=self.tensor_parallel_size,
            )
        else:
            out = _reduce_from_tp(
                out,
                self.tensor_parallel_group,
                self.tensor_parallel_size,
            )
        if self.bias is not None:
            out = out + self.bias
        return out


class VocabParallelEmbedding(nn.Module):
    """Embedding table sharded along vocabulary rows."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        embedding_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.tensor_parallel_rank = int(tensor_parallel_rank)
        self.embedding_scale = float(embedding_scale)
        self.vocab_start, self.vocab_stop = partition_bounds(
            self.num_embeddings,
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
        )
        self.tensor_parallel_group = None
        self.weight = nn.Parameter(
            torch.empty(self.vocab_stop - self.vocab_start, self.embedding_dim)
        )
        self.weight._dllm_tensor_parallel_sharded = True
        self.weight._dllm_vocab_embedding_shard = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _kaiming_uniform_partition_(self.weight, self.tensor_parallel_rank)

    def set_tensor_parallel_runtime(self, runtime: Any | None) -> None:
        self.tensor_parallel_group = _runtime_tp_group(runtime)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel_size <= 1:
            output = F.embedding(x, self.weight)
            return output if self.embedding_scale == 1.0 else output * self.embedding_scale
        return _VocabParallelEmbeddingFunction.apply(
            x,
            self.weight,
            int(self.vocab_start),
            int(self.vocab_stop),
            _ensure_group(self.tensor_parallel_group, self.tensor_parallel_size),
            int(self.tensor_parallel_size),
            self.embedding_scale,
        )


class _VocabParallelEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        vocab_start: int,
        vocab_stop: int,
        group: Any,
        world_size: int,
        embedding_scale: float = 1.0,
    ) -> torch.Tensor:
        mask = (x < int(vocab_start)) | (x >= int(vocab_stop))
        local_x = (x - int(vocab_start)).masked_fill(mask, 0)
        out = F.embedding(local_x, weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        if embedding_scale != 1.0:
            out.mul_(embedding_scale)
        if int(world_size) > 1:
            out = _all_reduce_sum_(out.contiguous(), group)
        ctx.save_for_backward(local_x, ~mask)
        ctx.weight_shape = tuple(weight.shape)
        ctx.embedding_scale = float(embedding_scale)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_x, valid = ctx.saved_tensors
        grad_weight = grad_output.new_zeros(ctx.weight_shape)
        grad_weight.index_add_(
            0,
            local_x.masked_select(valid).reshape(-1),
            grad_output[valid].reshape(-1, ctx.weight_shape[1]),
            alpha=ctx.embedding_scale,
        )
        return (None, grad_weight, None, None, None, None, None)[:len(ctx.needs_input_grad)]


class VocabParallelLinear(nn.Module):
    """Output projection sharded along vocabulary rows."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        gather_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.tensor_parallel_rank = int(tensor_parallel_rank)
        self.vocab_start, self.vocab_stop = partition_bounds(
            self.out_features,
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
        )
        self.local_out_features = self.vocab_stop - self.vocab_start
        self.gather_output = bool(gather_output)
        self.tensor_parallel_group = None
        self.weight = nn.Parameter(torch.empty(self.local_out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.local_out_features)) if bias else None
        self.weight._dllm_tensor_parallel_sharded = True
        if self.bias is not None:
            self.bias._dllm_tensor_parallel_sharded = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _kaiming_uniform_partition_(self.weight, self.tensor_parallel_rank)
        if self.bias is not None:
            _uniform_bias_(self.bias, self.in_features, self.tensor_parallel_rank)

    def set_tensor_parallel_runtime(self, runtime: Any | None) -> None:
        self.tensor_parallel_group = _runtime_tp_group(runtime)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(x, self.weight, self.bias)
        if not self.gather_output:
            return local_logits
        return _gather_from_tp(
            local_logits,
            group=self.tensor_parallel_group,
            world_size=self.tensor_parallel_size,
            rank=self.tensor_parallel_rank,
            sizes=partition_sizes(self.out_features, self.tensor_parallel_size),
        )

    def parallel_cross_entropy(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        ignore_index: int = -100,
        exclude_index: int | None = None,
        vocab_block_size: int = 0,
        reduction: str = "none",
        logit_softcap: float | None = None,
    ) -> torch.Tensor:
        if self.tensor_parallel_size <= 1:
            return tiled_linear_cross_entropy(
                x,
                self.weight,
                labels,
                bias=self.bias,
                vocab_block_size=int(vocab_block_size),
                reduction=str(reduction),
                ignore_index=int(ignore_index),
                exclude_index=exclude_index,
                logit_softcap=logit_softcap,
                dtype=x.dtype,
                weight_layout="vocab_first",
            )
        return _VocabParallelCrossEntropy.apply(
            x,
            self.weight,
            self.bias,
            labels,
            int(self.vocab_start),
            int(self.out_features),
            int(ignore_index),
            -1 if exclude_index is None else int(exclude_index),
            int(vocab_block_size),
            _ensure_group(self.tensor_parallel_group, self.tensor_parallel_size),
            int(self.tensor_parallel_size),
            str(reduction),
            0.0 if logit_softcap is None else float(logit_softcap),
        )
