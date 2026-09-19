# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training performance profiling: step time, tokens/sec, MFU, comm, idle.

CUDA-event timers give step time / tokens-per-sec / MFU at low cost; a
scheduled ``torch.profiler`` capture gives the comm and idle breakdown from the
device timeline, then self-stops. ``FlopsModel`` and ``compute_breakdown`` are
pure functions; ``PerfProfiler`` is a lightweight callback-compatible helper
for legacy profiling tests and standalone instrumentation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from dllm_parallel.core.schedules.block import build_block_schedule
except ModuleNotFoundError:
    build_block_schedule = None

try:
    from dllm_parallel.core.attention.cp_backend import (
        cp_bp_counters_from_visits,
        prefix_visits_for_noisy_blocks,
    )
except ModuleNotFoundError:
    cp_bp_counters_from_visits = None
    prefix_visits_for_noisy_blocks = None


DEFAULT_HARDWARE = "h100_sxm_bf16"
DFLASH_WORK_COUNT_NAMES = (
    "context_token_rows",
    "valid_anchors",
    "full_context_pairs",
    "sliding_context_pairs",
    "supervised_token_rows",
)
HARDWARE_PEAK_FLOPS: dict[str, float] = {
    "h100_sxm_bf16": 989.0e12,
    "h100_pcie_bf16": 756.0e12,
    "h200_sxm_bf16": 989.0e12,
    "a100_sxm_bf16": 312.0e12,
}
DEFAULT_LOG_EVERY_N_STEPS = 50
DEFAULT_PROFILER_SCHEDULE: dict[str, int] = {
    "wait": 200,
    "warmup": 8,
    "active": 8,
    "repeat": 1,
}


def peak_flops_for_hardware(name: str) -> float:
    try:
        return HARDWARE_PEAK_FLOPS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown hardware preset {name!r}; known presets: "
            f"{sorted(HARDWARE_PEAK_FLOPS)}"
        ) from exc


def detect_hardware_preset(*, allow_default: bool = True) -> str:
    """Resolve the dense BF16 peak preset for the current CUDA device.

    Legacy standalone profiling may opt into the H100 default when CUDA is not
    available. Production MFU callers disable that behavior so an unknown
    accelerator cannot silently produce a plausible but incorrect percentage.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            if allow_default:
                return DEFAULT_HARDWARE
            raise RuntimeError(
                "MFU requires a CUDA device with a registered peak preset"
            )
        name = torch.cuda.get_device_name().upper()
    except Exception as exc:
        if allow_default:
            return DEFAULT_HARDWARE
        raise RuntimeError("MFU hardware detection requires PyTorch with CUDA") from exc
    if "H200" in name:
        return "h200_sxm_bf16"
    if "H100" in name:
        return "h100_pcie_bf16" if "PCIE" in name else "h100_sxm_bf16"
    if "A100" in name:
        return "a100_sxm_bf16"
    if allow_default:
        return DEFAULT_HARDWARE
    raise RuntimeError(f"no dense BF16 peak-FLOP preset is registered for {name!r}")


@dataclass(frozen=True)
class FlopsModel:
    """Approximate fwd+bwd FLOPs per data token.

    The dense term is the usual ``6N`` transformer accounting, scaled by the
    number of executed hidden positions per data token. The attention term is
    ``12 L d T`` scaled by executed query/key pairs relative to ``T ** 2``.

    ``seq_factor`` applies the same scale to dense positions and attention
    pairs; explicit dense/attention factors take precedence when supplied.
    """

    num_params: int
    num_layers: int
    hidden_size: int
    seq_len: int
    seq_factor: Optional[float] = None
    dense_token_factor: Optional[float] = None
    attention_pair_factor: Optional[float] = None

    def __post_init__(self):
        if self.dense_token_factor is None:
            dense_factor = 1.0 if self.seq_factor is None else float(self.seq_factor)
            object.__setattr__(self, "dense_token_factor", dense_factor)
        if self.attention_pair_factor is None:
            pair_factor = (
                1.0 if self.seq_factor is None else float(self.seq_factor) ** 2
            )
            object.__setattr__(self, "attention_pair_factor", pair_factor)

    def flops_per_data_token(self) -> float:
        dense = 6.0 * self.num_params * float(self.dense_token_factor)
        attn = (
            12.0
            * self.num_layers
            * self.hidden_size
            * self.seq_len
            * float(self.attention_pair_factor)
        )
        return dense + attn

    def flops_for_tokens(self, data_tokens: float) -> float:
        return self.flops_per_data_token() * float(data_tokens)


@dataclass(frozen=True)
class MegatronTransformerFlops:
    """Megatron-style useful training FLOPs for a dense decoder transformer.

    Counts forward, activation-gradient, and weight-gradient GEMMs (three
    executions, two FLOPs per fused multiply-add). Attention uses an exact
    query/key-pair count rather than Megatron's dense or causal approximation.
    Embedding lookup, normalization, activation functions, communication,
    optimizer work, and activation-checkpoint recomputation are excluded,
    matching the conventional MFU definition used by Megatron and TorchTitan.
    """

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def query_width(self) -> int:
        return int(self.num_attention_heads) * int(self.head_dim)

    @property
    def key_value_width(self) -> int:
        return int(self.num_key_value_heads) * int(self.head_dim)

    def flops_per_step(
        self,
        *,
        transformer_token_rows: float,
        attention_pairs: float,
        vocabulary_token_rows: float,
    ) -> float:
        """Return useful model FLOPs for one global optimizer step."""

        if min(transformer_token_rows, attention_pairs, vocabulary_token_rows) < 0:
            raise ValueError("FLOP row and pair counts must be non-negative")
        h = int(self.hidden_size)
        q = self.query_width
        kv = self.key_value_width
        i = int(self.intermediate_size)

        # Q/K/V projections and output projection.
        attention_projection = (
            6.0 * transformer_token_rows * (h * (q + kv + kv) + q * h)
        )
        # QK^T and probability/value products over exactly the allowed pairs.
        attention_core = 12.0 * attention_pairs * q
        # SwiGLU gate/up/down projections.
        mlp = 6.0 * transformer_token_rows * (3 * h * i)
        transformer = int(self.num_layers) * (
            attention_projection + attention_core + mlp
        )
        vocabulary = 6.0 * vocabulary_token_rows * h * int(self.vocab_size)
        return transformer + vocabulary


@dataclass(frozen=True)
class FastDLLMv2TransformerFlops:
    """Useful training FLOPs for Fast-dLLM v2 converted causal LMs.

    The accounting follows Megatron's training-MFU convention. Matrix
    products count forward, activation-gradient, and weight-gradient work;
    activation-checkpoint recomputation and non-model systems work are not
    counted. Gated-DeltaNet uses Megatron's architecture-specific estimate,
    while softmax attention uses the exact allowed query/key-pair count of the
    sampled block-diffusion execution.
    """

    layer_types: tuple[str, ...]
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    attention_output_gate: bool = False
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_conv_kernel_dim: int | None = None

    def __post_init__(self) -> None:
        if not self.layer_types:
            raise ValueError("layer_types must be non-empty")
        unsupported = sorted(
            set(self.layer_types)
            - {"full_attention", "sliding_attention", "linear_attention"}
        )
        if unsupported:
            raise ValueError(
                "unsupported Fast-dLLM v2 layer types: " + ", ".join(unsupported)
            )
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_linear_attention_layers:
            for name in (
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_conv_kernel_dim",
            ):
                value = getattr(self, name)
                if value is None or int(value) <= 0:
                    raise ValueError(
                        f"{name} must be positive for Gated-DeltaNet layers"
                    )

    @property
    def num_full_attention_layers(self) -> int:
        return self.layer_types.count("full_attention")

    @property
    def num_sliding_attention_layers(self) -> int:
        return self.layer_types.count("sliding_attention")

    @property
    def num_linear_attention_layers(self) -> int:
        return self.layer_types.count("linear_attention")

    def flops_breakdown_per_step(
        self,
        *,
        transformer_token_rows: float,
        full_attention_pairs: float,
        sliding_attention_pairs: float,
        vocabulary_token_rows: float,
    ) -> dict[str, float]:
        """Return independently auditable useful-FLOP terms for one step."""

        if (
            min(
                transformer_token_rows,
                full_attention_pairs,
                sliding_attention_pairs,
                vocabulary_token_rows,
            )
            < 0
        ):
            raise ValueError("FLOP row and pair counts must be non-negative")

        rows = float(transformer_token_rows)
        h = int(self.hidden_size)
        q = int(self.num_attention_heads) * int(self.head_dim)
        kv = int(self.num_key_value_heads) * int(self.head_dim)
        softmax_layers = (
            self.num_full_attention_layers + self.num_sliding_attention_layers
        )
        q_and_gate = q * (2 if self.attention_output_gate else 1)
        attention_projection = (
            softmax_layers * 6.0 * rows * (h * (q_and_gate + 2 * kv) + q * h)
        )
        attention_core = (
            12.0
            * q
            * (
                self.num_full_attention_layers * float(full_attention_pairs)
                + self.num_sliding_attention_layers * float(sliding_attention_pairs)
            )
        )

        gated_delta_net = 0.0
        if self.num_linear_attention_layers:
            qk_dim = int(self.linear_key_head_dim) * int(self.linear_num_key_heads)
            value_dim = int(self.linear_value_head_dim) * int(
                self.linear_num_value_heads
            )
            input_projection_dim = (
                2 * qk_dim + 2 * value_dim + 2 * int(self.linear_num_value_heads)
            )
            per_layer = (
                6.0
                * rows
                * (
                    h * input_projection_dim
                    + int(self.linear_conv_kernel_dim) * (2 * qk_dim + value_dim)
                    + 4
                    * int(self.linear_num_value_heads)
                    * int(self.linear_value_head_dim) ** 2
                    + h * value_dim
                )
            )
            gated_delta_net = self.num_linear_attention_layers * per_layer

        mlp = len(self.layer_types) * 6.0 * rows * (3 * h * int(self.intermediate_size))
        vocabulary = 6.0 * float(vocabulary_token_rows) * h * int(self.vocab_size)
        return {
            "attention_projection": attention_projection,
            "attention_core": attention_core,
            "gated_delta_net": gated_delta_net,
            "mlp": mlp,
            "vocabulary": vocabulary,
        }

    def flops_per_step(self, **kwargs: float) -> float:
        return sum(self.flops_breakdown_per_step(**kwargs).values())


@dataclass(frozen=True)
class DiffusionGemmaTransformerFlops:
    """Useful training FLOPs for DiffusionGemma's text transformer.

    This follows Megatron's model-FLOP convention: GEMMs count the forward,
    activation-gradient, and weight-gradient executions, while normalization,
    routing decisions, communication, optimizer work, and activation-checkpoint
    recomputation are excluded. Routed-expert GEMMs count only the selected
    top-k experts. DiffusionGemma runs a dense SwiGLU branch and a routed MoE
    branch in every layer.
    """

    layer_types: tuple[str, ...]
    hidden_size: int
    intermediate_size: int
    expert_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    num_experts: int
    top_k_experts: int
    vocab_size: int

    def __post_init__(self) -> None:
        if not self.layer_types:
            raise ValueError("layer_types must be non-empty")
        unsupported = sorted(
            set(self.layer_types) - {"sliding_attention", "full_attention"}
        )
        if unsupported:
            raise ValueError(
                "unsupported DiffusionGemma layer types: " + ", ".join(unsupported)
            )
        for name in (
            "hidden_size",
            "intermediate_size",
            "expert_intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_global_key_value_heads",
            "head_dim",
            "global_head_dim",
            "num_experts",
            "top_k_experts",
            "vocab_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.top_k_experts) > int(self.num_experts):
            raise ValueError("top_k_experts cannot exceed num_experts")

    @property
    def num_sliding_layers(self) -> int:
        return self.layer_types.count("sliding_attention")

    @property
    def num_full_layers(self) -> int:
        return self.layer_types.count("full_attention")

    def flops_breakdown_per_step(
        self,
        *,
        transformer_token_rows: float,
        sliding_attention_pairs: float,
        full_attention_pairs: float,
        self_conditioning_token_rows: float,
        vocabulary_token_rows: float,
    ) -> dict[str, float]:
        """Return independently auditable useful-FLOP terms for one step."""

        values = (
            transformer_token_rows,
            sliding_attention_pairs,
            full_attention_pairs,
            self_conditioning_token_rows,
            vocabulary_token_rows,
        )
        if min(values) < 0:
            raise ValueError("FLOP row and pair counts must be non-negative")

        h = int(self.hidden_size)
        dense_i = int(self.intermediate_size)
        expert_i = int(self.expert_intermediate_size)
        sliding_q = int(self.num_attention_heads) * int(self.head_dim)
        sliding_kv = int(self.num_key_value_heads) * int(self.head_dim)
        global_q = int(self.num_attention_heads) * int(self.global_head_dim)
        global_kv = int(self.num_global_key_value_heads) * int(self.global_head_dim)

        # Sliding layers project Q, K, V, and O. Global layers reuse K as V,
        # matching DiffusionGemma's v_proj=None implementation.
        sliding_projection_per_layer = (
            6.0
            * transformer_token_rows
            * (h * (sliding_q + 2 * sliding_kv) + sliding_q * h)
        )
        global_projection_per_layer = (
            6.0 * transformer_token_rows * (h * (global_q + global_kv) + global_q * h)
        )
        attention_projection = (
            self.num_sliding_layers * sliding_projection_per_layer
            + self.num_full_layers * global_projection_per_layer
        )
        attention_core = 12.0 * (
            self.num_sliding_layers * sliding_attention_pairs * sliding_q
            + self.num_full_layers * full_attention_pairs * global_q
        )

        num_layers = len(self.layer_types)
        dense_mlp = num_layers * 6.0 * transformer_token_rows * (3 * h * dense_i)
        router = num_layers * 6.0 * transformer_token_rows * h * int(self.num_experts)
        routed_experts = (
            num_layers
            * 6.0
            * transformer_token_rows
            * int(self.top_k_experts)
            * (3 * h * expert_i)
        )
        self_conditioning = 6.0 * self_conditioning_token_rows * (3 * h * dense_i)
        vocabulary = 6.0 * vocabulary_token_rows * h * int(self.vocab_size)
        return {
            "attention_projection": attention_projection,
            "attention_core": attention_core,
            "dense_mlp": dense_mlp,
            "router": router,
            "routed_experts": routed_experts,
            "self_conditioning": self_conditioning,
            "vocabulary": vocabulary,
        }

    def flops_per_step(self, **kwargs: float) -> float:
        return sum(self.flops_breakdown_per_step(**kwargs).values())

    def native_flops_breakdown_per_step(
        self,
        *,
        clean_encoder_rows: float,
        detached_decoder_rows: float,
        trained_decoder_rows: float,
        clean_sliding_attention_pairs: float,
        clean_full_attention_pairs: float,
        detached_sliding_attention_pairs: float,
        detached_full_attention_pairs: float,
        trained_sliding_attention_pairs: float,
        trained_full_attention_pairs: float,
        self_conditioning_vocabulary_rows: float,
        decoder_loss_vocabulary_rows: float,
        encoder_ar_vocabulary_rows: float,
    ) -> dict[str, float]:
        """Exact GEMM work for split-stream native all-block SFT."""

        values = (
            clean_encoder_rows,
            detached_decoder_rows,
            trained_decoder_rows,
            clean_sliding_attention_pairs,
            clean_full_attention_pairs,
            detached_sliding_attention_pairs,
            detached_full_attention_pairs,
            trained_sliding_attention_pairs,
            trained_full_attention_pairs,
            self_conditioning_vocabulary_rows,
            decoder_loss_vocabulary_rows,
            encoder_ar_vocabulary_rows,
        )
        if min(values) < 0:
            raise ValueError("native DiffusionGemma FLOP counts must be non-negative")

        h = int(self.hidden_size)
        dense_i = int(self.intermediate_size)
        expert_i = int(self.expert_intermediate_size)
        experts = int(self.num_experts)
        top_k = int(self.top_k_experts)
        vocab = int(self.vocab_size)
        sliding_q = int(self.num_attention_heads) * int(self.head_dim)
        sliding_kv = int(self.num_key_value_heads) * int(self.head_dim)
        global_q = int(self.num_attention_heads) * int(self.global_head_dim)
        global_kv = int(self.num_global_key_value_heads) * int(self.global_head_dim)

        def projections(rows: float, multiplier: float) -> float:
            sliding = h * (sliding_q + 2 * sliding_kv) + sliding_q * h
            full = h * (global_q + global_kv) + global_q * h
            return (
                multiplier
                * rows
                * (self.num_sliding_layers * sliding + self.num_full_layers * full)
            )

        def feed_forward(rows: float, multiplier: float) -> float:
            per_layer = 3 * h * dense_i + top_k * (3 * h * expert_i)
            return multiplier * len(self.layer_types) * rows * per_layer

        def router(rows: float, multiplier: float) -> float:
            return multiplier * len(self.layer_types) * rows * h * experts

        return {
            "clean_attention_projection": projections(clean_encoder_rows, 6.0),
            "detached_attention_projection": projections(
                detached_decoder_rows,
                2.0,
            ),
            "trained_attention_projection": projections(
                trained_decoder_rows,
                6.0,
            ),
            "clean_attention_core": 12.0
            * (
                self.num_sliding_layers
                * clean_sliding_attention_pairs
                * sliding_q
                + self.num_full_layers * clean_full_attention_pairs * global_q
            ),
            "detached_attention_core": 4.0
            * (
                self.num_sliding_layers
                * detached_sliding_attention_pairs
                * sliding_q
                + self.num_full_layers * detached_full_attention_pairs * global_q
            ),
            "trained_attention_core": 12.0
            * (
                self.num_sliding_layers
                * trained_sliding_attention_pairs
                * sliding_q
                + self.num_full_layers * trained_full_attention_pairs * global_q
            ),
            "clean_feed_forward": feed_forward(clean_encoder_rows, 6.0),
            "detached_feed_forward": feed_forward(detached_decoder_rows, 2.0),
            "trained_feed_forward": feed_forward(trained_decoder_rows, 6.0),
            "clean_router": router(clean_encoder_rows, 4.0),
            "detached_router": router(detached_decoder_rows, 2.0),
            "trained_router": router(trained_decoder_rows, 4.0),
            "self_conditioning": (
                (6.0 * trained_decoder_rows + 2.0 * detached_decoder_rows)
                * (3 * h * dense_i)
            ),
            "self_conditioning_vocabulary": (
                4.0 * self_conditioning_vocabulary_rows * h * vocab
            ),
            "decoder_vocabulary": (6.0 * decoder_loss_vocabulary_rows * h * vocab),
            "encoder_vocabulary": (6.0 * encoder_ar_vocabulary_rows * h * vocab),
        }


@dataclass(frozen=True)
class DFlashTransformerFlops:
    """Useful training FLOPs for a verifier-conditioned DFlash drafter.

    The count follows the same GEMM-oriented convention as
    :class:`MegatronTransformerFlops`. It counts the trainable drafter's
    forward, activation-gradient, and weight-gradient matrix multiplies while
    excluding activation-checkpoint recomputation and distributed execution
    overhead. Frozen output heads include their forward products and, for the
    draft head, the hidden-state gradient required by training.
    """

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    target_hidden_size: int
    target_feature_width: int
    draft_vocab_size: int
    dflash2_conv_kernel_size: int = 0
    dflash2_conv_group_size: int = 0
    dflash2_selector_rank: int = 0
    dflash2_selector_top_k: int = 0

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "target_hidden_size",
            "target_feature_width",
            "draft_vocab_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def query_width(self) -> int:
        return int(self.num_attention_heads) * int(self.head_dim)

    @property
    def key_value_width(self) -> int:
        return int(self.num_key_value_heads) * int(self.head_dim)

    def flops_breakdown_per_step(
        self,
        *,
        context_token_rows: float,
        draft_token_rows: float,
        attention_pairs: float,
        supervised_token_rows: float,
        loss_kind: str,
    ) -> dict[str, float]:
        """Return useful global FLOPs for one optimizer step by component."""

        if (
            min(
                context_token_rows,
                draft_token_rows,
                attention_pairs,
                supervised_token_rows,
            )
            < 0
        ):
            raise ValueError("DFlash FLOP row and pair counts must be non-negative")
        hard_label_losses = {
            "paper_ce",
            "dflash",
            "dpace",
            "dpace-cumulative-confidence-only",
            "dpace-continuation-value-only",
        }
        if loss_kind not in {"speculators_kl", *hard_label_losses}:
            raise ValueError(f"unsupported DFlash loss {loss_kind!r}")

        layers = int(self.num_layers)
        h = int(self.hidden_size)
        i = int(self.intermediate_size)
        q = self.query_width
        kv = self.key_value_width
        target_h = int(self.target_hidden_size)
        target_features = int(self.target_feature_width)
        vocab = int(self.draft_vocab_size)
        context_rows = float(context_token_rows)
        draft_rows = float(draft_token_rows)
        supervised_rows = float(supervised_token_rows)

        # Target features are detached. Their trainable projection therefore
        # has a forward product and a weight gradient, but no input gradient.
        target_projection = 4.0 * context_rows * target_features * h
        attention_projection = (
            6.0
            * layers
            * (
                draft_rows * h * q
                + (draft_rows + context_rows) * h * (2 * kv)
                + draft_rows * q * h
            )
        )
        attention_core = 12.0 * float(attention_pairs) * q
        mlp = 6.0 * layers * draft_rows * (3 * h * i)

        dflash2 = 0.0
        if self.dflash2_conv_kernel_size and self.dflash2_conv_group_size:
            groups = h // int(self.dflash2_conv_group_size)
            dynamic_kernel_width = 2 * int(self.dflash2_conv_kernel_size) * groups
            # Two trainable kernel projections per layer.
            dflash2 += 12.0 * layers * draft_rows * h * dynamic_kernel_width
        if self.dflash2_selector_rank and self.dflash2_selector_top_k:
            rank = int(self.dflash2_selector_rank)
            candidates = int(self.dflash2_selector_top_k)
            dflash2 += 6.0 * supervised_rows * h * rank
            dflash2 += 6.0 * supervised_rows * candidates * rank

        # Both heads are frozen. The draft head additionally propagates a
        # gradient to the trainable draft hidden states.
        output = 4.0 * supervised_rows * h * vocab
        if loss_kind == "speculators_kl":
            output += 2.0 * supervised_rows * target_h * vocab

        breakdown = {
            "target_projection": target_projection,
            "attention_projection": attention_projection,
            "attention_core": attention_core,
            "mlp": mlp,
            "output": output,
        }
        if dflash2 > 0.0:
            breakdown["dflash2"] = dflash2
        return breakdown

    def flops_per_step(
        self,
        *,
        context_token_rows: float,
        draft_token_rows: float,
        attention_pairs: float,
        supervised_token_rows: float,
        loss_kind: str,
    ) -> float:
        return sum(
            self.flops_breakdown_per_step(
                context_token_rows=context_token_rows,
                draft_token_rows=draft_token_rows,
                attention_pairs=attention_pairs,
                supervised_token_rows=supervised_token_rows,
                loss_kind=loss_kind,
            ).values()
        )

    def executed_gemm_flops_breakdown_per_step(
        self,
        *,
        context_token_rows: float,
        draft_token_rows: float,
        attention_pairs: float,
        supervised_token_rows: float,
        loss_kind: str,
        draft_replication: float = 1.0,
        context_attention_pairs: float | None = None,
        local_attention_pairs: float | None = None,
    ) -> dict[str, float]:
        """Count GEMMs actually executed by the memory-bounded objective.

        Conventional useful MFU counts a frozen head's forward and the draft
        hidden-state gradient. DFlash deliberately recomputes streamed logits
        in backward to avoid retaining an ``[tokens, vocabulary]`` tensor.
        This second, explicitly named metric includes that real training work
        without pretending skipped 512K-token projections were executed.
        """

        if draft_replication < 1.0:
            raise ValueError("DFlash draft replication must be at least one")
        breakdown = self.flops_breakdown_per_step(
            context_token_rows=context_token_rows,
            draft_token_rows=draft_token_rows,
            attention_pairs=attention_pairs,
            supervised_token_rows=supervised_token_rows,
            loss_kind=loss_kind,
        )
        replication = float(draft_replication)
        if replication != 1.0:
            if context_attention_pairs is None or local_attention_pairs is None:
                raise ValueError(
                    "replicated DFlash accounting requires split attention pairs"
                )
            layers = int(self.num_layers)
            h = int(self.hidden_size)
            q = self.query_width
            kv = self.key_value_width
            context_rows = float(context_token_rows)
            draft_rows = float(draft_token_rows)
            context_projection = 6.0 * layers * context_rows * h * (2 * kv)
            draft_projection = (
                6.0
                * layers
                * (draft_rows * h * q + draft_rows * h * (2 * kv) + draft_rows * q * h)
            )
            breakdown["attention_projection"] = (
                context_projection + replication * draft_projection
            )
            breakdown["attention_core"] = (
                12.0
                * q
                * (
                    float(context_attention_pairs)
                    + replication * float(local_attention_pairs)
                )
            )
            for name in ("mlp", "output", "dflash2"):
                if name in breakdown:
                    breakdown[name] *= replication
        rows = float(supervised_token_rows)
        vocab = int(self.draft_vocab_size)
        replay_width = int(self.hidden_size)
        if loss_kind == "speculators_kl":
            replay_width += int(self.target_hidden_size)
        breakdown["output_recompute"] = 2.0 * rows * vocab * replay_width * replication
        return breakdown


def model_flops_utilization_pct(
    *,
    model_flops_per_step: float,
    elapsed_ms: float,
    world_size: int,
    peak_flops_per_gpu: float,
) -> float:
    """Return conventional model FLOPs utilization as a percentage."""

    if model_flops_per_step < 0:
        raise ValueError("model_flops_per_step must be non-negative")
    if elapsed_ms <= 0 or world_size <= 0 or peak_flops_per_gpu <= 0:
        raise ValueError("elapsed time, world size, and peak FLOPs must be positive")
    capacity = float(world_size) * float(peak_flops_per_gpu) * elapsed_ms / 1000.0
    return 100.0 * float(model_flops_per_step) / capacity


def block_diffusion_sparse_attention_pairs(*, seq_len: int, block_size: int) -> int:
    """Allowed attention pairs for the standard all-block objective."""

    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")
    num_blocks = seq_len // block_size
    return block_size * block_size * num_blocks * (num_blocks + 1)


def diffusiongemma_block_attention_pairs(
    *,
    seq_len: int,
    block_size: int,
    sliding_window: int | None,
) -> int:
    """Exact attention pairs in one packed DiffusionGemma BDLM layer.

    Noisy queries attend bidirectionally within their target block and to the
    preceding clean prefix. Clean queries use token-causal attention. Sliding
    layers retain ``window - 1`` clean-prefix states for noisy queries and an
    inclusive ``window`` for clean queries, matching the executor's bounds.
    """

    if seq_len <= 0 or block_size <= 0:
        raise ValueError("seq_len and block_size must be positive")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")
    if sliding_window is not None and int(sliding_window) <= 0:
        raise ValueError("sliding_window must be positive")

    num_blocks = seq_len // block_size
    noisy_local = num_blocks * block_size * block_size
    if sliding_window is None:
        noisy_clean = block_size * block_size * num_blocks * (num_blocks - 1) // 2
        clean_causal = seq_len * (seq_len + 1) // 2
    else:
        window = int(sliding_window)
        history = window - 1
        noisy_clean = block_size * sum(
            min(block * block_size, history) for block in range(num_blocks)
        )
        if window >= seq_len:
            clean_causal = seq_len * (seq_len + 1) // 2
        else:
            clean_causal = window * (window + 1) // 2 + (seq_len - window) * window
    return noisy_local + noisy_clean + clean_causal


def diffusiongemma_clean_attention_pairs(
    *,
    seq_len: int,
    sliding_window: int | None,
) -> int:
    """Causal clean-stream attention pairs for one DiffusionGemma layer."""

    if int(seq_len) <= 0:
        raise ValueError("seq_len must be positive")
    if sliding_window is None or int(sliding_window) >= int(seq_len):
        return int(seq_len) * (int(seq_len) + 1) // 2
    window = int(sliding_window)
    if window <= 0:
        raise ValueError("sliding_window must be positive")
    return window * (window + 1) // 2 + (int(seq_len) - window) * window


def block_diffusion_active_packed_factors(
    *,
    seq_len: int,
    block_size: int,
    active_blocks: list[int],
    sparse_attention: bool,
) -> tuple[float, float]:
    """Return ``(dense_token_factor, attention_pair_factor)`` for BP packing."""

    if not active_blocks:
        raise ValueError("active_blocks must be non-empty")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")

    active_blocks = sorted(int(b) for b in active_blocks)
    active_tokens = len(active_blocks) * block_size
    prefix_len = max(active_blocks) * block_size
    packed_len = active_tokens + prefix_len
    dense_factor = packed_len / seq_len
    if not sparse_attention:
        return dense_factor, dense_factor * dense_factor

    xt_pairs = sum((block + 1) * block_size * block_size for block in active_blocks)
    prefix_blocks = prefix_len // block_size
    x0_pairs = block_size * block_size * prefix_blocks * (prefix_blocks + 1) // 2
    pair_factor = (xt_pairs + x0_pairs) / float(seq_len * seq_len)
    return dense_factor, pair_factor


def block_diffusion_sharded_clean_factors(
    *,
    seq_len: int,
    block_size: int,
    active_blocks: list[int],
    context_parallel_size: int,
    context_parallel_rank: int,
) -> tuple[float, float]:
    """Return FLOP factors for ring CP/BP with sharded clean hidden state."""

    if not active_blocks:
        raise ValueError("active_blocks must be non-empty")
    if seq_len % block_size != 0:
        raise ValueError("seq_len must divide evenly by block_size")
    if context_parallel_size <= 0:
        raise ValueError("context_parallel_size must be positive")
    if not 0 <= context_parallel_rank < context_parallel_size:
        raise ValueError("context_parallel_rank out of range")

    clean_start, clean_stop = _balanced_token_shard(
        seq_len, context_parallel_size, context_parallel_rank
    )
    active_blocks = sorted(int(b) for b in active_blocks)
    active_tokens = len(active_blocks) * block_size
    clean_tokens = clean_stop - clean_start
    dense_factor = (active_tokens + clean_tokens) / float(seq_len)

    local_active_pairs = active_tokens * block_size
    active_to_clean_pairs = sum(
        block * block_size * block_size for block in active_blocks
    )
    clean_to_clean_pairs = sum(
        ((pos // block_size) + 1) * block_size for pos in range(clean_start, clean_stop)
    )
    pair_factor = (
        local_active_pairs + active_to_clean_pairs + clean_to_clean_pairs
    ) / float(seq_len * seq_len)
    return dense_factor, pair_factor


def block_diffusion_context_parallel_factors(
    *,
    seq_len: int,
    block_size: int,
    context_parallel_size: int,
    context_parallel_rank: int,
) -> tuple[float, float]:
    """Executed FLOP factors for full-mask token-row context parallelism."""

    if seq_len % block_size:
        raise ValueError("seq_len must divide evenly by block_size")
    if seq_len % context_parallel_size:
        raise ValueError("seq_len must divide evenly by context_parallel_size")
    if not 0 <= context_parallel_rank < context_parallel_size:
        raise ValueError("context_parallel_rank out of range")
    local_tokens = seq_len // context_parallel_size
    noisy_start = context_parallel_rank * local_tokens
    noisy_stop = noisy_start + local_tokens
    clean_start = seq_len - (context_parallel_rank + 1) * local_tokens
    clean_stop = clean_start + local_tokens
    pair_count = block_size * sum(
        position // block_size + 1 for position in range(noisy_start, noisy_stop)
    )
    pair_count += block_size * sum(
        position // block_size + 1 for position in range(clean_start, clean_stop)
    )
    return 2.0 / context_parallel_size, pair_count / float(seq_len * seq_len)


def _balanced_token_shard(
    num_items: int,
    num_shards: int,
    shard_index: int,
) -> tuple[int, int]:
    base, remainder = divmod(num_items, num_shards)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + (1 if shard_index < remainder else 0)
    return start, stop


def throughput_metrics(
    *,
    tokens: int,
    steps: int,
    elapsed_ms: float,
    world_size: int = 1,
    data_parallel_size: int | None = None,
    flops: Optional[FlopsModel] = None,
    peak_flops: float = 0.0,
    min_window_ms: float = 1.0,
) -> Optional[dict]:
    """perf/* throughput + MFU for one window, or None if degenerate.

    Returns None when the window is empty or shorter than ``min_window_ms`` so
    near-zero windows (e.g. the final partial window at end of training) never
    produce divide-by-zero / infinite rates.
    """
    if steps <= 0 or elapsed_ms <= min_window_ms:
        return None
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if data_parallel_size is None:
        data_parallel_size = world_size
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    elapsed_s = elapsed_ms / 1e3
    tok_s = tokens / elapsed_s
    metrics = {
        "perf/step_time_ms": elapsed_ms / steps,
        "perf/tokens_per_s_per_gpu": tok_s,
        "perf/tokens_per_s_global": tok_s * world_size,
        "perf/unique_tokens_per_s_global": tok_s * data_parallel_size,
    }
    if flops is not None and peak_flops > 0:
        achieved = flops.flops_for_tokens(tokens) / elapsed_s
        mfu = achieved / peak_flops
        metrics["perf/tflops_per_gpu"] = achieved / 1e12
        metrics["perf/mfu"] = mfu
        metrics["perf/mfu_pct"] = mfu * 100.0
    return metrics


_COMM_RE = re.compile(
    r"nccl|c10d|all_?reduce|all_?gather|reduce_?scatter|broadcast|"
    r"all_?to_?all|send|recv",
    re.IGNORECASE,
)


def is_comm_kernel(name: str) -> bool:
    """True if a device-kernel name looks like a collective / p2p transfer."""
    return bool(_COMM_RE.search(name or ""))


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _union_len(merged: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in merged)


def _overlap_len(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        i, j = (i + 1, j) if a[i][1] < b[j][1] else (i, j + 1)
    return total


@dataclass(frozen=True)
class ProfileBreakdown:
    """Comm / compute / idle decomposition of a profiled window (all ms)."""

    window_ms: float
    busy_ms: float
    compute_ms: float
    comm_ms: float
    exposed_comm_ms: float
    idle_ms: float
    n_device_events: int
    measured_tflops: Optional[float] = None

    def _frac(self, v: float) -> float:
        return v / self.window_ms if self.window_ms > 0 else 0.0

    compute_frac = property(lambda s: s._frac(s.compute_ms))
    comm_frac = property(lambda s: s._frac(s.comm_ms))
    exposed_comm_frac = property(lambda s: s._frac(s.exposed_comm_ms))
    idle_frac = property(lambda s: s._frac(s.idle_ms))

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "compute_frac": self.compute_frac,
            "comm_frac": self.comm_frac,
            "exposed_comm_frac": self.exposed_comm_frac,
            "idle_frac": self.idle_frac,
        }


def compute_breakdown(
    intervals: list[tuple[float, float, str, bool]],
    measured_tflops: Optional[float] = None,
) -> Optional[ProfileBreakdown]:
    """Decompose ``(start, end, name, is_comm)`` device kernels (overlap-aware).

    ``busy`` = union of all kernels; ``idle`` = window - busy; ``exposed_comm``
    = comm time not hidden under compute (i.e. on the critical path).
    """
    if not intervals:
        return None
    window_ms = max(e for _, e, _, _ in intervals) - min(s for s, _, _, _ in intervals)
    comm = _merge([(s, e) for s, e, _, c in intervals if c])
    compute = _merge([(s, e) for s, e, _, c in intervals if not c])
    busy_ms = _union_len(_merge([(s, e) for s, e, _, _ in intervals]))
    comm_ms = _union_len(comm)
    return ProfileBreakdown(
        window_ms=window_ms,
        busy_ms=busy_ms,
        compute_ms=_union_len(compute),
        comm_ms=comm_ms,
        exposed_comm_ms=max(0.0, comm_ms - _overlap_len(comm, compute)),
        idle_ms=max(0.0, window_ms - busy_ms),
        n_device_events=len(intervals),
        measured_tflops=measured_tflops,
    )


def _safe(obj, attr, default=None):
    try:
        return getattr(obj, attr)()
    except Exception:
        return default


def _event_span_ms(ev) -> Optional[tuple[float, float]]:
    for start_attr, dur_attr, scale in (
        ("start_ns", "duration_ns", 1e-6),
        ("start_us", "duration_us", 1e-3),
    ):
        if hasattr(ev, start_attr) and hasattr(ev, dur_attr):
            try:
                start = getattr(ev, start_attr)() * scale
                dur = getattr(ev, dur_attr)() * scale
            except Exception:
                continue
            if dur > 0:
                return start, start + dur
    return None


def device_intervals(prof) -> list[tuple[float, float, str, bool]]:
    """GPU kernel ``(start, end, name, is_comm)`` (ms) from a torch profiler."""
    kineto = getattr(getattr(prof, "profiler", None), "kineto_results", None)
    events = _safe(kineto, "events", None) if kineto is not None else None
    if not events:
        return []
    try:
        from torch._C._profiler import DeviceType

        cuda_dt = DeviceType.CUDA
    except Exception:
        cuda_dt = None

    out = []
    for ev in events:
        dt = _safe(ev, "device_type")
        if dt is None:
            continue
        is_cuda = dt == cuda_dt if cuda_dt is not None else "cuda" in str(dt).lower()
        if not is_cuda:
            continue
        span = _event_span_ms(ev)
        if span is None:
            continue
        name = _safe(ev, "name", "")
        out.append((span[0], span[1], name, is_comm_kernel(name)))
    return out


def summarize(prof) -> Optional[ProfileBreakdown]:
    """Device intervals from ``prof`` -> :class:`ProfileBreakdown`."""
    intervals = device_intervals(prof)
    if not intervals:
        return None
    window_ms = max(e for _, e, _, _ in intervals) - min(s for s, _, _, _ in intervals)
    tflops = None
    try:
        flops = sum(getattr(k, "flops", 0) or 0 for k in prof.key_averages())
        if flops > 0 and window_ms > 0:
            tflops = flops / (window_ms / 1e3) / 1e12
    except Exception:
        pass
    return compute_breakdown(intervals, tflops)


@dataclass
class PerfConfig:
    """Knobs mirroring the ``profile`` config group."""

    hardware: str = "auto"
    hardware_peak_flops: Optional[float] = None
    log_every_n_steps: int = DEFAULT_LOG_EVERY_N_STEPS
    start_step: int = 0
    cross_attn_seq_factor: Optional[float] = None
    torch_profiler: bool = True
    schedule: dict = field(default_factory=lambda: dict(DEFAULT_PROFILER_SCHEDULE))
    profiler_kwargs: dict = field(
        default_factory=lambda: {
            "record_shapes": False,
            "with_stack": False,
            "profile_memory": False,
            "with_flops": True,
        }
    )
    export_chrome_trace: bool = True
    trace_dir: str = "profiler"

    def __post_init__(self):
        if self.hardware == "auto":
            self.hardware = detect_hardware_preset()
        if self.hardware_peak_flops is None:
            self.hardware_peak_flops = peak_flops_for_hardware(self.hardware)

    @classmethod
    def from_node(cls, node) -> "PerfConfig":
        tp = node.get("torch_profiler", {}) or {}
        keep = ("record_shapes", "with_stack", "profile_memory", "with_flops")
        explicit_peak = node.get("hardware_peak_flops", None)
        return cls(
            hardware=str(node.get("hardware", "auto")),
            hardware_peak_flops=(
                None if explicit_peak is None else float(explicit_peak)
            ),
            log_every_n_steps=int(
                node.get("log_every_n_steps", DEFAULT_LOG_EVERY_N_STEPS)
            ),
            start_step=int(node.get("start_step", 0)),
            cross_attn_seq_factor=node.get("cross_attn_seq_factor", None),
            torch_profiler=bool(tp.get("enabled", True)),
            schedule={
                k: int(tp.get(k, d)) for k, d in DEFAULT_PROFILER_SCHEDULE.items()
            },
            profiler_kwargs={k: bool(tp.get(k, k == "with_flops")) for k in keep},
            export_chrome_trace=bool(tp.get("export_chrome_trace", True)),
            trace_dir=str(tp.get("trace_dir", "profiler")),
        )


class _Window:
    """One CUDA-event timing window (groups mutable callback state)."""

    def __init__(self):
        self.start = self.end = None
        self.steps = self.tokens = self.last_step = 0
        self.armed = False

    def reset(self):
        self.steps = self.tokens = 0
        self.armed = False


class PerfProfiler:
    """Callback-compatible helper emitting perf/* metrics for throughput and overlap."""

    def __init__(self, cfg: PerfConfig):
        self.cfg = cfg
        self._flops: Optional[FlopsModel] = None
        self._world_size = 1
        self._data_parallel_size = 1
        self._tensor_parallel_size = 1
        self._is_zero = True
        self._trainer = None
        self._win = _Window()
        self._prof = None
        self._pstep = self._cycles = 0
        self._cp_bp_static_metrics: dict[str, float] = {}

    @classmethod
    def from_config(cls, node) -> Optional["PerfProfiler"]:
        """Build from a ``profile`` config node, or None if absent/disabled."""
        if node is None or not node.get("enabled", False):
            return None
        return cls(PerfConfig.from_node(node))

    def setup(self, trainer, pl_module, stage=None):
        self._trainer = trainer

    def on_train_start(self, trainer, pl_module):
        import torch

        self._world_size = int(getattr(trainer, "world_size", 1) or 1)
        self._data_parallel_size = self._infer_data_parallel_size(
            pl_module, self._world_size
        )
        runtime = getattr(pl_module, "parallel_runtime", None)
        self._tensor_parallel_size = int(
            getattr(runtime, "tensor_parallel_size", 1) or 1
        )
        self._is_zero = bool(getattr(trainer, "is_global_zero", True))
        self._flops = self._build_flops_model(pl_module)
        self._cp_bp_static_metrics = self._build_cp_bp_static_metrics(pl_module)
        self._win.last_step = int(getattr(trainer, "global_step", 0))
        if torch.cuda.is_available():
            self._reset_peak_memory(torch)
            self._win.start = torch.cuda.Event(enable_timing=True)
            self._win.end = torch.cuda.Event(enable_timing=True)
            if self.cfg.torch_profiler and self._is_zero:
                self._start_torch_profiler(trainer)
        if self._is_zero and self._flops:
            logger.info(
                "[PerfProfiler] params=%.1fM layers=%d hidden=%d seq=%d "
                "dense_factor=%.3f attn_pair_factor=%.3f "
                "peak=%.0fTFLOP/s world=%d data_parallel=%d tensor_parallel=%d",
                self._flops.num_params / 1e6,
                self._flops.num_layers,
                self._flops.hidden_size,
                self._flops.seq_len,
                self._flops.dense_token_factor,
                self._flops.attention_pair_factor,
                self.cfg.hardware_peak_flops / 1e12,
                self._world_size,
                self._data_parallel_size,
                self._tensor_parallel_size,
            )

    @staticmethod
    def _infer_data_parallel_size(pl_module, world_size: int) -> int:
        runtime = getattr(pl_module, "parallel_runtime", None)
        if runtime is None or not bool(getattr(runtime, "enabled", False)):
            return int(world_size)

        group_ranks = getattr(runtime, "data_parallel_group_ranks", None)
        if group_ranks:
            return len(group_ranks)

        plan = getattr(runtime, "plan", None)
        if plan is not None and hasattr(plan, "data_parallel_size"):
            return int(plan.data_parallel_size)

        local_parallel_size = int(getattr(runtime, "local_parallel_size", 1) or 1)
        return max(1, int(world_size) // local_parallel_size)

    def _build_flops_model(self, pl_module) -> FlopsModel:
        cfg = pl_module.config
        backbone = getattr(pl_module, "backbone", pl_module)
        dense_factor, pair_factor = self._flop_factors(pl_module)
        return FlopsModel(
            num_params=sum(p.numel() for p in backbone.parameters()),
            num_layers=int(cfg.model.n_blocks),
            hidden_size=int(cfg.model.hidden_size),
            seq_len=int(cfg.model.length),
            dense_token_factor=dense_factor,
            attention_pair_factor=pair_factor,
        )

    def _flop_factors(self, pl_module) -> tuple[float, float]:
        cfg = pl_module.config
        factor = self.cfg.cross_attn_seq_factor
        if factor is not None:
            f = float(factor)
            return f, f * f
        if str(cfg.algo.get("name", "")) != "standard_block_diffusion":
            return 1.0, 1.0

        seq_len = int(cfg.model.length)
        block_size = int(cfg.block_size)
        attn_backend = str(getattr(cfg.model, "attn_backend", "sdpa"))
        runtime = getattr(pl_module, "parallel_runtime", None)
        active_mode = str(getattr(runtime, "active_block_mode", "disabled"))
        local_parallel_size = int(getattr(runtime, "local_parallel_size", 1) or 1)
        kv_backend = str(getattr(runtime, "kv_backend", "replicated"))
        block_parallel_size = int(
            getattr(runtime, "block_parallel_size", local_parallel_size) or 1
        )
        context_parallel_size = int(
            getattr(
                runtime,
                "configured_context_parallel_size",
                getattr(runtime, "context_attention_size", 1),
            )
            or 1
        )

        if active_mode == "dual_end" and local_parallel_size > 1:
            active_blocks = self._active_blocks_for_rank(
                seq_len=seq_len,
                block_size=block_size,
                runtime=runtime,
            )
            if kv_backend == "ring":
                return block_diffusion_sharded_clean_factors(
                    seq_len=seq_len,
                    block_size=block_size,
                    active_blocks=active_blocks,
                    context_parallel_size=int(
                        getattr(runtime, "context_attention_size", 1)
                    ),
                    context_parallel_rank=int(
                        getattr(runtime, "context_parallel_rank", 0)
                    ),
                )
            return block_diffusion_active_packed_factors(
                seq_len=seq_len,
                block_size=block_size,
                active_blocks=active_blocks,
                sparse_attention=(attn_backend == "flex"),
            )
        if (
            active_mode == "all_blocks"
            and kv_backend == "ring"
            and block_parallel_size == 1
            and context_parallel_size > 1
        ):
            return block_diffusion_context_parallel_factors(
                seq_len=seq_len,
                block_size=block_size,
                context_parallel_size=int(
                    getattr(runtime, "context_attention_size", 1)
                ),
                context_parallel_rank=int(getattr(runtime, "context_parallel_rank", 0)),
            )

        dense_factor = 2.0
        if attn_backend == "flex":
            pair_factor = block_diffusion_sparse_attention_pairs(
                seq_len=seq_len,
                block_size=block_size,
            ) / float(seq_len * seq_len)
        else:
            # SDPA and replicated streaming attention form dense score chunks
            # and apply the block mask after QK, so executed attention FLOPs
            # are dense even though the mathematical mask is sparse.
            pair_factor = dense_factor * dense_factor
        return dense_factor, pair_factor

    @staticmethod
    def _active_blocks_for_rank(*, seq_len: int, block_size: int, runtime) -> list[int]:
        if build_block_schedule is None:
            raise RuntimeError("dllm_parallel.core.schedules.block is unavailable")
        num_blocks = seq_len // block_size
        block_parallel_size = int(
            getattr(runtime, "block_parallel_size", runtime.local_parallel_size)
        )
        context_parallel_size = int(
            getattr(
                runtime,
                "configured_context_parallel_size",
                getattr(runtime, "context_attention_size", 1),
            )
        )
        schedule = build_block_schedule(
            num_blocks=num_blocks,
            block_parallel_size=block_parallel_size,
            context_parallel_size=context_parallel_size,
        )
        return schedule.active_blocks_by_worker[int(runtime.block_parallel_rank)]

    @staticmethod
    def _build_cp_bp_static_metrics(pl_module) -> dict[str, float]:
        if cp_bp_counters_from_visits is None or prefix_visits_for_noisy_blocks is None:
            return {}
        cfg = pl_module.config
        if str(cfg.algo.get("name", "")) != "standard_block_diffusion":
            return {}
        runtime = getattr(pl_module, "parallel_runtime", None)
        if runtime is None:
            return {}
        if str(getattr(runtime, "active_block_mode", "disabled")) != "dual_end":
            return {}
        local_parallel_size = int(getattr(runtime, "local_parallel_size", 1) or 1)
        if local_parallel_size <= 1:
            return {}

        seq_len = int(cfg.model.length)
        block_size = int(cfg.block_size)
        active_blocks = PerfProfiler._active_blocks_for_rank(
            seq_len=seq_len,
            block_size=block_size,
            runtime=runtime,
        )
        kv_backend = str(getattr(runtime, "kv_backend", "replicated"))
        if kv_backend == "ring":
            visits = prefix_visits_for_noisy_blocks(
                active_blocks=active_blocks,
                block_size=block_size,
                seq_len=seq_len,
                context_parallel_size=int(
                    getattr(runtime, "context_attention_size", 1)
                ),
            )
            return cp_bp_counters_from_visits(
                active_blocks=active_blocks,
                visits=visits,
            ).to_metrics()
        return {
            "perf/cp_bp/active_blocks": float(len(active_blocks)),
            "perf/cp_bp/prefix_shards_visited": 0.0,
            "perf/cp_bp/prefix_shards_skipped": 0.0,
            "perf/cp_bp/prefix_tokens_visited": 0.0,
            "perf/cp_bp/prefix_tokens_skipped": 0.0,
        }

    def _start_torch_profiler(self, trainer):
        from torch.profiler import ProfilerActivity, profile, schedule

        self.cfg.trace_dir = os.path.join(
            getattr(trainer, "default_root_dir", None) or os.getcwd(),
            self.cfg.trace_dir,
        )
        os.makedirs(self.cfg.trace_dir, exist_ok=True)
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(**self.cfg.schedule),
            on_trace_ready=self._on_trace_ready,
            **self.cfg.profiler_kwargs,
        )
        self._prof.start()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = int(getattr(trainer, "global_step", 0))
        if not self._win.armed and step < self.cfg.start_step:
            self._win.last_step = step
            return
        if not self._win.armed and self._win.start is not None:
            self._win.steps = 0
            self._win.tokens = 0
            self._win.last_step = step
            self._win.start.record()
            self._win.armed = True
        self._win.tokens += self._count_tokens(batch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._advance_torch_profiler()
        step = int(getattr(trainer, "global_step", 0))
        if not self._win.armed:
            self._win.last_step = step
            return
        if step > self._win.last_step:
            self._win.steps += step - self._win.last_step
            self._win.last_step = step
            if self._win.steps >= self.cfg.log_every_n_steps:
                self._flush(trainer)

    def on_train_end(self, trainer, pl_module):
        self._stop_torch_profiler()

    def _flush(self, trainer):
        import torch

        win = self._win
        if win.start is None or win.end is None or not win.armed:
            win.reset()
            return
        win.end.record()
        torch.cuda.synchronize()
        metrics = throughput_metrics(
            tokens=win.tokens,
            steps=win.steps,
            elapsed_ms=win.start.elapsed_time(win.end),
            world_size=self._world_size,
            data_parallel_size=self._data_parallel_size,
            flops=self._flops,
            peak_flops=self.cfg.hardware_peak_flops,
        )
        if metrics is None:  # degenerate window
            win.reset()
            return
        metrics.update(self._cp_bp_static_metrics)
        peak_memory_mb = self._max_peak_memory_mb(torch)
        if peak_memory_mb is not None:
            metrics["perf/peak_memory_mb"] = peak_memory_mb
        self._log(trainer, metrics)
        if self._is_zero:
            memory = metrics.get("perf/peak_memory_mb")
            memory_text = f" | peak_mem {memory:.0f} MiB" if memory is not None else ""
            logger.info(
                "[PerfProfiler] step=%d | %.1f ms/step | %.0f tok/s/gpu "
                "(%.0f global, %.0f unique) | %.2f TFLOP/s | MFU %.1f%%%s",
                win.last_step,
                metrics["perf/step_time_ms"],
                metrics["perf/tokens_per_s_per_gpu"],
                metrics["perf/tokens_per_s_global"],
                metrics["perf/unique_tokens_per_s_global"],
                metrics.get("perf/tflops_per_gpu", 0),
                metrics.get("perf/mfu_pct", 0),
                memory_text,
            )
        self._reset_peak_memory(torch)
        win.reset()

    @staticmethod
    def _max_peak_memory_mb(torch_module) -> Optional[float]:
        if not torch_module.cuda.is_available():
            return None
        peak_mb = float(torch_module.cuda.max_memory_allocated()) / (1024.0**2)
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                device = torch_module.device("cuda", torch_module.cuda.current_device())
                value = torch_module.tensor([peak_mb], device=device)
                dist.all_reduce(value, op=dist.ReduceOp.MAX)
                peak_mb = float(value.item())
        except Exception:
            pass
        return peak_mb

    @staticmethod
    def _reset_peak_memory(torch_module) -> None:
        try:
            torch_module.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    @staticmethod
    def _count_tokens(batch) -> int:
        if isinstance(batch, dict) and "input_ids" in batch:
            return int(batch["input_ids"].numel())
        return 0

    def _advance_torch_profiler(self):
        if self._prof is None:
            return
        self._prof.step()
        self._pstep += 1
        if self._cycles >= self.cfg.schedule["repeat"]:
            self._stop_torch_profiler()

    def _stop_torch_profiler(self):
        if self._prof is not None:
            try:
                self._prof.stop()
            except Exception:
                pass
            self._prof = None

    def _on_trace_ready(self, prof):
        self._cycles += 1
        try:
            breakdown = summarize(prof)
        except Exception as exc:
            logger.warning("[PerfProfiler] summarize failed: %r", exc)
            return
        if breakdown is None:
            return
        metrics = {
            "perf/compute_ms": breakdown.compute_ms,
            "perf/compute_frac": breakdown.compute_frac,
            "perf/comm_ms": breakdown.comm_ms,
            "perf/comm_frac": breakdown.comm_frac,
            "perf/exposed_comm_ms": breakdown.exposed_comm_ms,
            "perf/exposed_comm_frac": breakdown.exposed_comm_frac,
            "perf/idle_ms": breakdown.idle_ms,
            "perf/idle_frac": breakdown.idle_frac,
            "perf/window_ms": breakdown.window_ms,
        }
        if breakdown.measured_tflops is not None:
            metrics["perf/profiler_measured_tflops"] = breakdown.measured_tflops
        self._log(self._trainer, metrics)
        logger.info(
            "[PerfProfiler] capture: window=%.1fms compute=%.1f%% comm=%.1f%% "
            "(exposed %.1f%%) idle=%.1f%% (%d kernels)",
            breakdown.window_ms,
            breakdown.compute_frac * 100,
            breakdown.comm_frac * 100,
            breakdown.exposed_comm_frac * 100,
            breakdown.idle_frac * 100,
            breakdown.n_device_events,
        )
        self._dump(breakdown, prof)

    def _dump(self, breakdown, prof):
        try:
            os.makedirs(self.cfg.trace_dir, exist_ok=True)
            path = os.path.join(self.cfg.trace_dir, f"breakdown_step{self._pstep}.json")
            with open(path, "w") as fh:
                json.dump(breakdown.as_dict(), fh, indent=2)
            if self.cfg.export_chrome_trace:
                prof.export_chrome_trace(
                    os.path.join(self.cfg.trace_dir, f"trace_step{self._pstep}.json")
                )
        except Exception as exc:
            logger.warning("[PerfProfiler] dump failed: %r", exc)

    def _log(self, trainer, metrics: dict):
        lg = getattr(trainer, "logger", None) if trainer else None
        if lg is None or not self._is_zero:
            return
        try:
            lg.log_metrics(metrics, step=int(getattr(trainer, "global_step", 0)))
        except Exception as exc:
            logger.warning("[PerfProfiler] log_metrics failed: %r", exc)
