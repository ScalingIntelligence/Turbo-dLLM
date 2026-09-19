# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Typed production run specification for block-diffusion training."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

import yaml

from dllm_parallel.core.models.registry import (
    SUPPORTED_OPTIMIZER_BACKENDS,
    supported_families,
)


@dataclass(frozen=True)
class ModelRunSpec:
    id: str = field(
        default="nvidia/Nemotron-Labs-Diffusion-3B", metadata={"cli_dest": "model_id"}
    )
    revision: str | None = None
    family: str = field(default="auto", metadata={"cli_dest": "model_family"})
    seq_len: int = 1024
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    conditioning_dim: int | None = None
    vocab_size: int | None = None
    mask_token_id: int | None = None
    mask_token: str | None = None
    attention_backend: str = "flex"
    dtype: str = "bf16"
    trust_remote_code: bool = False
    max_position_embeddings: int | None = None
    max_layers: int | None = None
    verifier_id: str | None = None
    verifier_revision: str | None = None
    target_layer_ids: tuple[int, ...] = field(
        default=(),
        metadata={"cli_metavar": "LAYER"},
    )
    draft_vocab_path: str | None = None


@dataclass(frozen=True)
class ObjectiveRunSpec:
    name: str = field(
        default="standard_block_diffusion", metadata={"cli_dest": "objective_name"}
    )
    block_size: int | None = None
    bp_loss_scale: float | None = None
    skip_rng_steps: int = 0
    noise_schedule: str = "loglinear"
    loss_weighting: str = "inverse_move_chance"
    noise_schedule_epsilon: float = 1.0e-3
    sampling_epsilon_min: float = 1.0e-3
    sampling_epsilon_max: float = 1.0
    antithetic_sampling: bool = True
    self_conditioning_probability: float = 0.5
    encoder_loss_weight: float = 1.0
    self_conditioning_row_chunk_size: int = 256
    self_conditioning_vocab_chunk_size: int = 32768
    max_anchors: int | None = None
    anchor_sampling: str = "uniform"
    anchor_group_size: int = 0
    sample_from_anchor: bool = False
    dflash_loss: str = "speculators_kl"
    decay_gamma: float | None = None
    dpace_alpha: float = 0.5
    lk_loss_type: str | None = None
    kl_scale: float = 1.0
    kl_decay: float = 1.0
    selector_loss_alpha: float = 1.0
    selector_warmup_ratio: float = 0.0
    selector_ramp_ratio: float = 0.0
    selector_stop_gradient: bool = False
    dflash_vocab_block_size: int = 32768


@dataclass(frozen=True)
class DataRunSpec:
    input_mode: str = "random"
    vocab_sample_size: int | None = None
    dataset_path: str | None = None
    minimum_sequence_length: int | None = None
    blend_seed: int = 0
    shuffle: bool = True
    optimizer_step_unit: str = "microbatch"
    trajectory_loss_reduction: str = "token_mean"
    trajectory_terminal_turn_weight: float = 1.0
    target_features: str | None = None
    target_feature_path: str | None = None


@dataclass(frozen=True)
class TrainingRunSpec:
    batch_size: int = 1
    steps: int = 3
    max_duration_seconds: float | None = None
    gradient_accumulation_steps: int = 1
    activation_checkpointing: bool = True
    activation_checkpointing_scope: str = "full"
    skip_optimizer_step: bool = False


@dataclass(frozen=True)
class EvaluationRunSpec:
    dataset_path: str | None = field(
        default=None,
        metadata={"cli_dest": "evaluation_dataset_path"},
    )
    batches: int = field(default=0, metadata={"cli_dest": "evaluation_batches"})
    seed: int = field(default=314159, metadata={"cli_dest": "evaluation_seed"})


@dataclass(frozen=True)
class AdapterRunSpec:
    type: str = field(default="none", metadata={"cli_dest": "adapter_type"})
    rank: int | None = field(default=None, metadata={"cli_dest": "adapter_rank"})
    alpha: float | None = field(default=None, metadata={"cli_dest": "adapter_alpha"})
    dropout: float = field(default=0.0, metadata={"cli_dest": "adapter_dropout"})
    targets: tuple[str, ...] = field(
        default=(),
        metadata={"cli_dest": "adapter_target", "cli_metavar": "ROLE"},
    )


@dataclass(frozen=True)
class TopologyRunSpec:
    context_parallel_size: int = 4
    block_parallel_size: int = 4
    replicate_clean_prefix: bool = False
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    sequence_parallel: bool = False
    tensor_parallel_overlap: bool = True
    placement_policy: str = "auto"
    process_group_timeout_seconds: float | None = None
    process_group_timeout: tuple[str, ...] = field(
        default=(), metadata={"cli_metavar": "GROUP=SECONDS"}
    )


@dataclass(frozen=True)
class OptimizerRunSpec:
    backend: str = field(default="auto", metadata={"cli_dest": "optimizer_backend"})
    zero_optimizer_impl: str = "auto"
    lr: float = 1.0e-6
    adam_eps: float = 1.0e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    zero_param_group_max_elements: int = 0
    zero_reduce_bucket_size: int = 0
    zero_contiguous_gradients: bool = True
    zero_overlap_comm: bool = True
    gradient_clip_norm: float = 0.0
    allow_deepspeed_op_build: bool = False
    fsdp_mixed_precision: str | None = None
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_use_orig_params: bool = True


@dataclass(frozen=True)
class SchedulerRunSpec:
    type: str = field(default="constant", metadata={"cli_dest": "scheduler_type"})
    warmup_steps: int = field(default=0, metadata={"cli_dest": "lr_warmup_steps"})
    decay_steps: int | None = field(
        default=None, metadata={"cli_dest": "lr_decay_steps"}
    )
    min_lr: float = 0.0
    weight_decay_style: str = "constant"
    weight_decay_start: float | None = None
    weight_decay_end: float | None = None
    weight_decay_steps: int | None = None


@dataclass(frozen=True)
class CheckpointingRunSpec:
    save_checkpoint_dir: str | None = None
    save_checkpoint_interval: int = 0
    keep_last_n: int = 0
    load_checkpoint_dir: str | None = None
    checkpoint_tag: str = "latest"
    auto_resume: bool = False
    save_first_step: bool = False
    save_final: bool = False
    async_checkpoint_save: bool = False
    model_only: bool = False
    save_duration_fractions: tuple[float, ...] = field(
        default=(), metadata={"cli_dest": "save_duration_fraction"}
    )


@dataclass(frozen=True)
class ProfilerRunSpec:
    warmup_steps: int = field(default=0, metadata={"cli_dest": "profile_warmup_steps"})
    phase_timing: bool = False
    phase_timing_sync: bool = False
    system_trace: bool = False
    system_trace_backend: str = "kineto"
    system_trace_dir: str | None = None
    system_trace_start_step: int = 1
    system_trace_steps: int = 1


@dataclass(frozen=True)
class LoggingRunSpec:
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_group: str | None = None
    wandb_id: str | None = None
    wandb_mode: str = "online"
    wandb_dir: str | None = None
    wandb_tags: tuple[str, ...] = field(default=(), metadata={"cli_metavar": "TAG"})


@dataclass(frozen=True)
class DebugRunSpec:
    grad_finite: bool = field(default=False, metadata={"cli_dest": "debug_grad_finite"})
    optimizer_finite: bool = field(
        default=False, metadata={"cli_dest": "debug_optimizer_finite"}
    )
    trace: bool = field(default=False, metadata={"cli_dest": "debug_trace"})


@dataclass(frozen=True)
class KernelRunSpec:
    ring_attention_key_chunk_size: int = 0
    mlp_token_chunk_size: int = 0
    cp_bp_attention_policy: str = "production"
    cp_bp_clean_kv_layout: str = "zigzag"
    cp_bp_clean_kv_transport: str = "collective"
    cp_bp_debug_nonfinite_attention: bool = False
    runtime_jit: bool = False


@dataclass(frozen=True)
class LaunchRunSpec:
    seed: int = 2026
    recipe_kind: str = "smoke"


@dataclass(frozen=True)
class RunSpec:
    model: ModelRunSpec = field(default_factory=ModelRunSpec)
    objective: ObjectiveRunSpec = field(default_factory=ObjectiveRunSpec)
    data: DataRunSpec = field(default_factory=DataRunSpec)
    training: TrainingRunSpec = field(default_factory=TrainingRunSpec)
    evaluation: EvaluationRunSpec = field(default_factory=EvaluationRunSpec)
    adapter: AdapterRunSpec = field(default_factory=AdapterRunSpec)
    topology: TopologyRunSpec = field(default_factory=TopologyRunSpec)
    optimizer: OptimizerRunSpec = field(default_factory=OptimizerRunSpec)
    scheduler: SchedulerRunSpec = field(default_factory=SchedulerRunSpec)
    checkpointing: CheckpointingRunSpec = field(default_factory=CheckpointingRunSpec)
    profiler: ProfilerRunSpec = field(default_factory=ProfilerRunSpec)
    logging: LoggingRunSpec = field(default_factory=LoggingRunSpec)
    debug: DebugRunSpec = field(default_factory=DebugRunSpec)
    kernel: KernelRunSpec = field(default_factory=KernelRunSpec)
    launch: LaunchRunSpec = field(default_factory=LaunchRunSpec)

    @classmethod
    def default(cls) -> "RunSpec":
        return cls()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RunSpec":
        return cls._from_sections(_canonical_sections(raw)).validate()

    def with_overrides(self, overrides: Mapping[str, Any]) -> "RunSpec":
        sections = self.to_dict()
        cli_fields = run_spec_cli_fields()
        for flat_key, value in overrides.items():
            cli_field = cli_fields.get(str(flat_key))
            if cli_field is None:
                raise ValueError(f"unknown RunSpec override: {flat_key}")
            if value is None and not cli_field.allows_none:
                continue
            sections[cli_field.section][cli_field.field] = _normalize_override(
                flat_key,
                value,
            )
        return type(self)._from_sections(sections).validate()

    @classmethod
    def _from_sections(cls, sections: Mapping[str, Mapping[str, Any]]) -> "RunSpec":
        kwargs: dict[str, Any] = {}
        for name, section_type in _run_spec_section_types().items():
            values = dict(sections.get(name, {}))
            valid = {field.name for field in dataclasses.fields(section_type)}
            unknown = sorted(set(values) - valid)
            if unknown:
                raise ValueError(
                    f"unknown RunSpec {name} fields: " + ", ".join(unknown)
                )
            values = {
                field_name: _normalize_section_value(name, field_name, field_value)
                for field_name, field_value in values.items()
            }
            kwargs[name] = section_type(**values)
        return cls(**kwargs)

    def validate(self) -> "RunSpec":
        _choice(
            "model.family",
            self.model.family,
            ("auto", "hf", *supported_families()),
        )
        _choice(
            "model.attention_backend", self.model.attention_backend, ("sdpa", "flex")
        )
        _choice("model.dtype", self.model.dtype, ("bf16", "fp16"))
        _choice(
            "objective.name",
            self.objective.name,
            (
                "standard_block_diffusion",
                "diffusiongemma_native_sft",
                "fast_dllm_v2",
                "dflash_distillation",
            ),
        )
        _choice("data.input_mode", self.data.input_mode, ("random", "text", "dataset"))
        _choice(
            "data.optimizer_step_unit",
            self.data.optimizer_step_unit,
            ("microbatch", "trajectory"),
        )
        _choice(
            "data.trajectory_loss_reduction",
            self.data.trajectory_loss_reduction,
            ("token_mean", "turn_mean"),
        )
        _positive_float(
            "data.trajectory_terminal_turn_weight",
            self.data.trajectory_terminal_turn_weight,
        )
        if (
            self.data.optimizer_step_unit == "trajectory"
            and self.data.input_mode != "dataset"
        ):
            raise ValueError(
                "data.optimizer_step_unit=trajectory requires data.input_mode=dataset"
            )
        if (
            self.data.trajectory_loss_reduction != "token_mean"
            and self.data.optimizer_step_unit != "trajectory"
        ):
            raise ValueError(
                "data.trajectory_loss_reduction=turn_mean requires "
                "data.optimizer_step_unit=trajectory"
            )
        if (
            float(self.data.trajectory_terminal_turn_weight) != 1.0
            and self.data.optimizer_step_unit != "trajectory"
        ):
            raise ValueError(
                "data.trajectory_terminal_turn_weight requires "
                "data.optimizer_step_unit=trajectory"
            )
        _choice(
            "training.activation_checkpointing_scope",
            self.training.activation_checkpointing_scope,
            ("full", "mlp"),
        )
        _choice("adapter.type", self.adapter.type, ("none", "lora"))
        _nonnegative_float("adapter.dropout", self.adapter.dropout)
        if float(self.adapter.dropout) >= 1.0:
            raise ValueError("adapter.dropout must be less than 1")
        supported_adapter_targets = {"attention", "mlp", "experts"}
        unknown_adapter_targets = sorted(
            set(self.adapter.targets) - supported_adapter_targets
        )
        if unknown_adapter_targets:
            raise ValueError(
                "unknown adapter.targets: " + ", ".join(unknown_adapter_targets)
            )
        if self.adapter.type == "lora":
            if self.adapter.rank is None:
                raise ValueError("LoRA requires adapter.rank")
            if self.adapter.alpha is None:
                raise ValueError("LoRA requires adapter.alpha")
            _positive("adapter.rank", self.adapter.rank)
            _positive_float("adapter.alpha", self.adapter.alpha)
            if not self.adapter.targets:
                raise ValueError("LoRA requires at least one adapter target")
        if len(set(self.adapter.targets)) != len(self.adapter.targets):
            raise ValueError("adapter.targets must not contain duplicates")
        if (
            self.adapter.type == "lora"
            and self.topology.tensor_parallel_size > 1
            and not self.topology.sequence_parallel
        ):
            raise ValueError(
                "tensor-parallel packed LoRA requires topology.sequence_parallel=true"
            )
        _choice(
            "optimizer.backend", self.optimizer.backend, SUPPORTED_OPTIMIZER_BACKENDS
        )
        _choice(
            "scheduler.type",
            self.scheduler.type,
            ("constant", "linear", "cosine", "inverse_square_root"),
        )
        _choice(
            "scheduler.weight_decay_style",
            self.scheduler.weight_decay_style,
            ("constant", "linear", "cosine"),
        )
        _choice(
            "optimizer.zero_optimizer_impl",
            self.optimizer.zero_optimizer_impl,
            (
                "auto",
                "torch_adamw",
                "torch_fused_adamw",
                "deepspeed_fused_adam",
                "deepspeed_fused_adam_hybrid",
            ),
        )
        if self.optimizer.fsdp_mixed_precision is not None:
            _choice(
                "optimizer.fsdp_mixed_precision",
                self.optimizer.fsdp_mixed_precision,
                ("bf16", "fp16", "fp32"),
            )
        _choice(
            "optimizer.fsdp_sharding_strategy",
            self.optimizer.fsdp_sharding_strategy,
            ("full_shard", "shard_grad_op", "hybrid_shard", "no_shard"),
        )
        _choice(
            "kernel.cp_bp_attention_policy",
            self.kernel.cp_bp_attention_policy,
            ("production",),
        )
        _choice(
            "kernel.cp_bp_clean_kv_layout",
            self.kernel.cp_bp_clean_kv_layout,
            ("zigzag",),
        )
        _choice(
            "kernel.cp_bp_clean_kv_transport",
            self.kernel.cp_bp_clean_kv_transport,
            ("collective", "streaming"),
        )
        _choice(
            "logging.wandb_mode",
            self.logging.wandb_mode,
            ("online", "offline", "disabled"),
        )
        _choice(
            "launch.recipe_kind", self.launch.recipe_kind, ("prod", "profile", "smoke")
        )
        _positive("model.seq_len", self.model.seq_len)
        _positive("training.batch_size", self.training.batch_size)
        _nonnegative("training.steps", self.training.steps)
        _nonnegative("evaluation.batches", self.evaluation.batches)
        if self.evaluation.dataset_path is None and self.evaluation.batches > 0:
            raise ValueError("evaluation.batches requires evaluation.dataset_path")
        if self.evaluation.dataset_path is not None and self.evaluation.batches <= 0:
            raise ValueError(
                "evaluation.dataset_path requires positive evaluation.batches"
            )
        if self.data.minimum_sequence_length is not None:
            _positive(
                "data.minimum_sequence_length",
                self.data.minimum_sequence_length,
            )
            if int(self.data.minimum_sequence_length) > int(self.model.seq_len):
                raise ValueError(
                    "data.minimum_sequence_length cannot exceed model.seq_len"
                )
        if self.training.max_duration_seconds is not None:
            _positive_float(
                "training.max_duration_seconds",
                self.training.max_duration_seconds,
            )
        _nonnegative("profiler.warmup_steps", self.profiler.warmup_steps)
        _choice(
            "profiler.system_trace_backend",
            self.profiler.system_trace_backend,
            ("kineto", "nsys"),
        )
        _positive(
            "profiler.system_trace_start_step", self.profiler.system_trace_start_step
        )
        _positive("profiler.system_trace_steps", self.profiler.system_trace_steps)
        _positive(
            "training.gradient_accumulation_steps",
            self.training.gradient_accumulation_steps,
        )
        _positive("topology.context_parallel_size", self.topology.context_parallel_size)
        _positive("topology.block_parallel_size", self.topology.block_parallel_size)
        _positive("topology.tensor_parallel_size", self.topology.tensor_parallel_size)
        _positive("topology.expert_parallel_size", self.topology.expert_parallel_size)
        _choice(
            "topology.placement_policy",
            self.topology.placement_policy,
            ("auto", "inter_node_cp"),
        )
        _nonnegative(
            "optimizer.zero_param_group_max_elements",
            self.optimizer.zero_param_group_max_elements,
        )
        _nonnegative(
            "optimizer.zero_reduce_bucket_size", self.optimizer.zero_reduce_bucket_size
        )
        _nonnegative_float(
            "optimizer.gradient_clip_norm", self.optimizer.gradient_clip_norm
        )
        _positive_float("optimizer.lr", self.optimizer.lr)
        _positive_float("optimizer.adam_eps", self.optimizer.adam_eps)
        _nonnegative_float("optimizer.weight_decay", self.optimizer.weight_decay)
        _nonnegative("scheduler.warmup_steps", self.scheduler.warmup_steps)
        _nonnegative_float("scheduler.min_lr", self.scheduler.min_lr)
        if self.scheduler.decay_steps is not None:
            _positive("scheduler.decay_steps", self.scheduler.decay_steps)
        if self.scheduler.weight_decay_steps is not None:
            _positive("scheduler.weight_decay_steps", self.scheduler.weight_decay_steps)
        if self.scheduler.weight_decay_start is not None:
            _nonnegative_float(
                "scheduler.weight_decay_start", self.scheduler.weight_decay_start
            )
        if self.scheduler.weight_decay_end is not None:
            _nonnegative_float(
                "scheduler.weight_decay_end", self.scheduler.weight_decay_end
            )
        if (
            self.scheduler.weight_decay_start is not None
            and self.scheduler.weight_decay_end is not None
            and float(self.scheduler.weight_decay_end)
            < float(self.scheduler.weight_decay_start)
        ):
            raise ValueError(
                "scheduler.weight_decay_end must be >= scheduler.weight_decay_start"
            )
        if (
            self.scheduler.weight_decay_style == "constant"
            and self.scheduler.weight_decay_start is not None
            and self.scheduler.weight_decay_end is not None
            and float(self.scheduler.weight_decay_start)
            != float(self.scheduler.weight_decay_end)
        ):
            raise ValueError(
                "constant scheduler.weight_decay_style requires matching "
                "weight_decay_start and weight_decay_end"
            )
        if float(self.scheduler.min_lr) > float(self.optimizer.lr):
            raise ValueError("scheduler.min_lr must be <= optimizer.lr")
        _nonnegative(
            "kernel.ring_attention_key_chunk_size",
            self.kernel.ring_attention_key_chunk_size,
        )
        _nonnegative("kernel.mlp_token_chunk_size", self.kernel.mlp_token_chunk_size)
        _nonnegative(
            "checkpointing.save_checkpoint_interval",
            self.checkpointing.save_checkpoint_interval,
        )
        _nonnegative("checkpointing.keep_last_n", self.checkpointing.keep_last_n)
        fractions = tuple(
            float(value) for value in self.checkpointing.save_duration_fractions
        )
        if any(not 0.0 < value <= 1.0 for value in fractions):
            raise ValueError("checkpointing.save_duration_fractions must be in (0, 1]")
        if tuple(sorted(set(fractions))) != fractions:
            raise ValueError(
                "checkpointing.save_duration_fractions must be strictly increasing"
            )
        if len({int(round(value * 100.0)) for value in fractions}) != len(fractions):
            raise ValueError(
                "checkpoint duration fractions must map to unique percentage tags"
            )
        _nonnegative("data.blend_seed", self.data.blend_seed)
        _nonnegative("objective.skip_rng_steps", self.objective.skip_rng_steps)
        if self.objective.block_size is not None:
            _positive("objective.block_size", self.objective.block_size)
            if (
                self.objective.name
                in {
                    "standard_block_diffusion",
                    "diffusiongemma_native_sft",
                    "fast_dllm_v2",
                }
                and self.model.seq_len % int(self.objective.block_size) != 0
            ):
                raise ValueError(
                    "model.seq_len must divide evenly by objective.block_size"
                )
            if (
                self.topology.context_parallel_size > 1
                and self.model.seq_len % int(self.topology.context_parallel_size) != 0
            ):
                raise ValueError(
                    "context parallelism requires equal DualChunkSwap chunks; model.seq_len "
                    "must divide evenly by topology.context_parallel_size"
                )
        if self.objective.name == "standard_block_diffusion":
            if self.objective.noise_schedule != "loglinear":
                raise ValueError("objective.noise_schedule must be 'loglinear'")
            _choice(
                "objective.loss_weighting",
                self.objective.loss_weighting,
                ("inverse_move_chance", "unit"),
            )
            if not (0.0 < float(self.objective.noise_schedule_epsilon) < 1.0):
                raise ValueError("objective.noise_schedule_epsilon must be in (0, 1)")
            if not (
                0.0
                < float(self.objective.sampling_epsilon_min)
                <= float(self.objective.sampling_epsilon_max)
                <= 1.0
            ):
                raise ValueError(
                    "objective sampling epsilon bounds must satisfy 0 < min <= max <= 1"
                )
        elif self.objective.name == "diffusiongemma_native_sft":
            if self.objective.block_size is None:
                raise ValueError(
                    "DiffusionGemma native SFT requires objective.block_size"
                )
            if self.objective.noise_schedule != "loglinear":
                raise ValueError(
                    "DiffusionGemma native SFT requires "
                    "objective.noise_schedule=loglinear"
                )
            if self.objective.loss_weighting != "unit":
                raise ValueError(
                    "DiffusionGemma native SFT requires objective.loss_weighting=unit"
                )
            probability = float(self.objective.self_conditioning_probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "objective.self_conditioning_probability must be in [0, 1]"
                )
            _nonnegative_float(
                "objective.encoder_loss_weight",
                self.objective.encoder_loss_weight,
            )
            _positive(
                "objective.self_conditioning_row_chunk_size",
                self.objective.self_conditioning_row_chunk_size,
            )
            _positive(
                "objective.self_conditioning_vocab_chunk_size",
                self.objective.self_conditioning_vocab_chunk_size,
            )
            if not (
                0.0
                < float(self.objective.sampling_epsilon_min)
                <= float(self.objective.sampling_epsilon_max)
                <= 1.0
            ):
                raise ValueError(
                    "objective sampling epsilon bounds must satisfy 0 < min <= max <= 1"
                )
        elif self.objective.name == "fast_dllm_v2":
            if self.objective.block_size is None:
                raise ValueError("Fast-dLLM v2 requires objective.block_size")
            if self.model.family == "dflash":
                raise ValueError(
                    "Fast-dLLM v2 converts Hugging Face causal LMs; use "
                    "model.family=auto, hf, causal_lm, or qwen3_8"
                )
            if self.model.mask_token_id is None and self.model.mask_token is None:
                raise ValueError(
                    "Fast-dLLM v2 requires model.mask_token or model.mask_token_id"
                )
            if not (0.0 < float(self.objective.noise_schedule_epsilon) < 1.0):
                raise ValueError("objective.noise_schedule_epsilon must be in (0, 1)")
            if self.objective.noise_schedule != "linear_mask":
                raise ValueError(
                    "Fast-dLLM v2 requires objective.noise_schedule=linear_mask"
                )
            if self.objective.loss_weighting != "unit":
                raise ValueError("Fast-dLLM v2 requires objective.loss_weighting=unit")
            if not self.objective.antithetic_sampling:
                raise ValueError(
                    "Fast-dLLM v2 requires complementary paired corruption"
                )
        else:
            if self.model.family != "dflash":
                raise ValueError("dflash_distillation requires model.family=dflash")
            if self.objective.block_size is None:
                raise ValueError("DFlash requires objective.block_size")
            if self.objective.max_anchors is None:
                raise ValueError("DFlash requires objective.max_anchors")
            _positive("objective.max_anchors", self.objective.max_anchors)
            _choice(
                "objective.anchor_sampling",
                self.objective.anchor_sampling,
                ("uniform", "locality"),
            )
            _nonnegative(
                "objective.anchor_group_size", self.objective.anchor_group_size
            )
            if self.objective.anchor_sampling == "locality" and int(
                self.objective.anchor_group_size
            ) > int(self.objective.max_anchors):
                raise ValueError(
                    "objective.anchor_group_size cannot exceed objective.max_anchors"
                )
            if int(self.topology.block_parallel_size) > 1 and int(
                self.objective.max_anchors
            ) < int(self.topology.block_parallel_size):
                raise ValueError(
                    "DFlash requires at least one anchor per block-parallel rank"
                )
            if self.objective.decay_gamma is None:
                raise ValueError("DFlash requires objective.decay_gamma")
            _positive_float("objective.decay_gamma", self.objective.decay_gamma)
            _choice(
                "objective.dflash_loss",
                self.objective.dflash_loss,
                (
                    "paper_ce",
                    "speculators_kl",
                    "dflash",
                    "dpace",
                    "dpace-cumulative-confidence-only",
                    "dpace-continuation-value-only",
                ),
            )
            if not 0.0 <= float(self.objective.dpace_alpha) <= 1.0:
                raise ValueError("objective.dpace_alpha must be in [0, 1]")
            if self.objective.lk_loss_type not in {None, "alpha", "lambda", "tv"}:
                raise ValueError(
                    "objective.lk_loss_type must be null, alpha, lambda, or tv"
                )
            _nonnegative_float(
                "objective.selector_loss_alpha",
                self.objective.selector_loss_alpha,
            )
            for field_name in ("selector_warmup_ratio", "selector_ramp_ratio"):
                value = float(getattr(self.objective, field_name))
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"objective.{field_name} must be in [0, 1]")
            _positive(
                "objective.dflash_vocab_block_size",
                self.objective.dflash_vocab_block_size,
            )
            if not self.model.verifier_id:
                raise ValueError("DFlash requires model.verifier_id")
            if not self.model.target_layer_ids:
                raise ValueError("DFlash requires model.target_layer_ids")
            if len(set(self.model.target_layer_ids)) != len(
                self.model.target_layer_ids
            ):
                raise ValueError("DFlash model.target_layer_ids must be unique")
            if any(int(layer) < 0 for layer in self.model.target_layer_ids):
                raise ValueError("DFlash model.target_layer_ids must be non-negative")
            _choice(
                "data.target_features",
                str(self.data.target_features),
                ("offline", "synthetic"),
            )
            if (
                self.data.target_features == "offline"
                and not self.data.target_feature_path
            ):
                raise ValueError("offline DFlash requires data.target_feature_path")
            if (
                self.data.target_features == "synthetic"
                and self.launch.recipe_kind == "prod"
            ):
                raise ValueError(
                    "synthetic DFlash features are restricted to smoke/profile recipes"
                )
        pure_bp = (
            int(self.topology.block_parallel_size) > 1
            and int(self.topology.context_parallel_size) == 1
        )
        if pure_bp and not self.topology.replicate_clean_prefix:
            raise ValueError(
                "BP without context parallelism replicates the clean prefix; "
                "set topology.replicate_clean_prefix=true explicitly"
            )
        if self.topology.replicate_clean_prefix and not pure_bp:
            raise ValueError(
                "topology.replicate_clean_prefix is only valid for BP-only execution "
                "with context_parallel_size=1 and block_parallel_size>1"
            )
        if self.topology.sequence_parallel and self.topology.tensor_parallel_size <= 1:
            raise ValueError("sequence_parallel requires tensor_parallel_size > 1")
        if (
            self.topology.context_parallel_size > 1
            and self.kernel.cp_bp_attention_policy != "production"
        ):
            raise ValueError("CP/BP supports only the validated production policy")
        if self.launch.recipe_kind == "prod" and self.data.input_mode in {
            "random",
            "text",
        }:
            raise ValueError(
                "production recipes must use a real data module; random/text inputs "
                "are reserved for smoke/profile runs"
            )
        if self.launch.recipe_kind == "prod":
            if self.objective.name == "dflash_distillation":
                if not self.data.target_feature_path:
                    raise ValueError(
                        "production DFlash recipes require data.target_feature_path"
                    )
            else:
                if self.data.input_mode != "dataset" or not self.data.dataset_path:
                    raise ValueError("production recipes require data.dataset_path")
                if str(self.data.dataset_path) == "/data/dllm/packed_tokens/train.pt":
                    raise ValueError(
                        "production recipes must replace the placeholder dataset_path"
                    )
        if self.launch.recipe_kind == "prod" and self.kernel.runtime_jit:
            raise ValueError("production recipes must use packaged native kernels")
        if self.profiler.system_trace:
            if (
                self.profiler.system_trace_backend == "kineto"
                and not self.profiler.system_trace_dir
            ):
                raise ValueError(
                    "profiler.system_trace_dir is required for Kineto tracing"
                )
            trace_stop = (
                int(self.profiler.system_trace_start_step)
                + int(self.profiler.system_trace_steps)
                - 1
            )
            if trace_stop > int(self.training.steps):
                raise ValueError(
                    "profiler system-trace window must end within training.steps"
                )
        if (
            self.launch.recipe_kind == "prod"
            and self.model.family != "dflash"
            and not self.model.revision
        ):
            raise ValueError("production Hugging Face models require model.revision")
        if (
            self.launch.recipe_kind == "prod"
            and self.objective.name == "dflash_distillation"
            and not self.model.verifier_revision
        ):
            raise ValueError("production DFlash requires model.verifier_revision")
        if (
            self.launch.recipe_kind == "prod"
            and self.optimizer.allow_deepspeed_op_build
        ):
            raise ValueError("production recipes must use prebuilt DeepSpeed CUDA ops")
        if (
            self.checkpointing.auto_resume
            and not self.checkpointing.save_checkpoint_dir
        ):
            raise ValueError("checkpointing.auto_resume requires save_checkpoint_dir")
        if (
            self.checkpointing.save_first_step
            and not self.checkpointing.save_checkpoint_dir
        ):
            raise ValueError(
                "checkpointing.save_first_step requires save_checkpoint_dir"
            )
        if self.checkpointing.save_final and not self.checkpointing.save_checkpoint_dir:
            raise ValueError("checkpointing.save_final requires save_checkpoint_dir")
        if fractions and not self.checkpointing.save_checkpoint_dir:
            raise ValueError(
                "checkpointing.save_duration_fractions requires save_checkpoint_dir"
            )
        if fractions and self.training.max_duration_seconds is None:
            raise ValueError(
                "checkpointing.save_duration_fractions requires max_duration_seconds"
            )
        if self.checkpointing.model_only and self.checkpointing.async_checkpoint_save:
            raise ValueError("model-only checkpointing does not support async saves")
        if self.checkpointing.model_only and self.checkpointing.auto_resume:
            raise ValueError(
                "model-only checkpoints cannot auto-resume optimizer training state"
            )
        if (
            self.launch.recipe_kind == "prod"
            and self.logging.wandb
            and self.logging.wandb_mode != "disabled"
            and not self.logging.wandb_project
        ):
            raise ValueError("production W&B logging requires logging.wandb_project")
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def load_run_spec(path: str | Path) -> RunSpec:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("RunSpec config must be a mapping")
    return RunSpec.from_mapping(raw)


def run_spec_from_args(args: Any) -> RunSpec:
    raw = vars(args)
    config_path = raw.get("config")
    spec = load_run_spec(config_path) if config_path else RunSpec.default()
    control_keys = {"config", "print_supported_configs", "validate_only"}
    overrides = {key: value for key, value in raw.items() if key not in control_keys}
    return spec.with_overrides(overrides)


def run_spec_from_argv(
    parser: Any,
    argv: list[str] | None,
) -> tuple[RunSpec, Any]:
    args = parser.parse_args(argv)
    return run_spec_from_args(args), args


def preflight_run_spec_from_argv(argv: list[str] | tuple[str, ...]) -> RunSpec:
    """Resolve only launch-relevant spec fields without importing trainer code."""

    parser = build_run_spec_arg_parser(add_help=False)
    args = parser.parse_args(list(argv))
    return run_spec_from_args(args)


def _canonical_sections(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    unknown_sections = sorted(
        set(str(key) for key in config) - set(_run_spec_section_types())
    )
    if unknown_sections:
        raise ValueError("unknown RunSpec sections: " + ", ".join(unknown_sections))
    sections = RunSpec.default().to_dict()
    for section_name, values in config.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"RunSpec section {section_name!r} must be a mapping")
        sections[str(section_name)].update(dict(values))
    return sections


def _normalize_override(key: str, value: Any) -> Any:
    if key in {
        "process_group_timeout",
        "wandb_tags",
        "adapter_target",
        "save_duration_fraction",
    }:
        if isinstance(value, list):
            if key == "save_duration_fraction":
                return tuple(float(item) for item in value)
            return tuple(str(item) for item in value)
    if key == "target_layer_ids" and isinstance(value, list):
        return tuple(int(item) for item in value)
    return value


def _normalize_section_value(section: str, field_name: str, value: Any) -> Any:
    if section == "topology" and field_name == "process_group_timeout":
        if isinstance(value, Mapping):
            raise ValueError(
                "topology.process_group_timeout must be a list of GROUP=SECONDS strings"
            )
        if isinstance(value, list):
            return tuple(str(item) for item in value)
    if section == "logging" and field_name == "wandb_tags":
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list):
            return tuple(str(item) for item in value)
    if section == "model" and field_name == "target_layer_ids":
        if isinstance(value, int):
            return (value,)
        if isinstance(value, list):
            return tuple(int(item) for item in value)
    if section == "adapter" and field_name == "targets":
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list):
            return tuple(str(item) for item in value)
    if section == "checkpointing" and field_name == "save_duration_fractions":
        if isinstance(value, (int, float)):
            return (float(value),)
        if isinstance(value, list):
            return tuple(float(item) for item in value)
    return value


@dataclass(frozen=True)
class RunSpecCLIField:
    dest: str
    section: str
    field: str
    annotation: Any
    default: Any
    allows_none: bool
    metavar: str | None


def build_run_spec_arg_parser(
    *,
    description: str | None = None,
    add_help: bool = True,
    include_control_flags: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or "Run production block-diffusion optimizer steps.",
        allow_abbrev=False,
        add_help=add_help,
    )
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Typed RunSpec YAML. Explicit CLI flags override it.",
    )
    for cli_field in run_spec_cli_fields().values():
        flag = f"--{cli_field.dest.replace('_', '-')}"
        if _is_bool_annotation(cli_field.annotation):
            parser.add_argument(
                flag,
                dest=cli_field.dest,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
            )
            continue
        if _is_tuple(cli_field.annotation):
            parser.add_argument(
                flag,
                dest=cli_field.dest,
                action="append",
                default=argparse.SUPPRESS,
                metavar=cli_field.metavar or "VALUE",
                type=_tuple_item_type(cli_field.annotation),
            )
            continue
        parser.add_argument(
            flag,
            dest=cli_field.dest,
            type=_argparse_type(cli_field.annotation),
            default=argparse.SUPPRESS,
        )
    if include_control_flags:
        parser.add_argument(
            "--print-supported-configs",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Print the supported backbones and parallel axes and exit.",
        )
        parser.add_argument(
            "--validate-only",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Load the recipe, apply overrides, print the resolved spec, and exit before CUDA initialization.",
        )
    return parser


def run_spec_cli_fields() -> dict[str, RunSpecCLIField]:
    fields: dict[str, RunSpecCLIField] = {}
    for section, section_type in _run_spec_section_types().items():
        type_hints = get_type_hints(section_type)
        for spec_field in dataclasses.fields(section_type):
            dest = str(spec_field.metadata.get("cli_dest", spec_field.name))
            if dest in fields:
                other = fields[dest]
                raise ValueError(
                    "duplicate RunSpec CLI destination "
                    f"{dest!r}: {other.section}.{other.field} and "
                    f"{section}.{spec_field.name}"
                )
            fields[dest] = RunSpecCLIField(
                dest=dest,
                section=section,
                field=spec_field.name,
                annotation=type_hints[spec_field.name],
                default=_field_default(spec_field),
                allows_none=_allows_none(type_hints[spec_field.name]),
                metavar=spec_field.metadata.get("cli_metavar"),
            )
    return fields


def _run_spec_section_types() -> dict[str, type[Any]]:
    type_hints = get_type_hints(RunSpec)
    section_types: dict[str, type[Any]] = {}
    for spec_field in dataclasses.fields(RunSpec):
        section_type = type_hints[spec_field.name]
        if not isinstance(section_type, type):
            raise TypeError(
                f"RunSpec section {spec_field.name!r} must be a dataclass type"
            )
        section_types[spec_field.name] = section_type
    return section_types


def _field_default(spec_field: dataclasses.Field[Any]) -> Any:
    if spec_field.default is not dataclasses.MISSING:
        return spec_field.default
    if spec_field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
        return spec_field.default_factory()  # type: ignore[misc]
    return dataclasses.MISSING


def _allows_none(annotation: Any) -> bool:
    return type(None) in get_args(annotation)


def _strip_optional(annotation: Any) -> Any:
    args = get_args(annotation)
    if not args or type(None) not in args:
        return annotation
    non_none = [arg for arg in args if arg is not type(None)]
    return non_none[0] if len(non_none) == 1 else annotation


def _is_bool_annotation(annotation: Any) -> bool:
    return _strip_optional(annotation) is bool


def _is_tuple(annotation: Any) -> bool:
    annotation = _strip_optional(annotation)
    origin = get_origin(annotation)
    return origin is tuple


def _tuple_item_type(annotation: Any) -> type[Any]:
    annotation = _strip_optional(annotation)
    args = get_args(annotation)
    return args[0] if args and args[0] in (str, int, float) else str


def _argparse_type(annotation: Any) -> type[Any]:
    annotation = _strip_optional(annotation)
    if annotation in (str, int, float):
        return annotation
    return str


def _choice(name: str, value: str, valid: tuple[str, ...]) -> None:
    if str(value) not in valid:
        raise ValueError(f"{name} must be one of: {', '.join(valid)}")


def _positive(name: str, value: int | float) -> None:
    if value is None or int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _nonnegative(name: str, value: int | float) -> None:
    if value is None or int(value) < 0:
        raise ValueError(f"{name} must be non-negative")


def _positive_float(name: str, value: int | float) -> None:
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _nonnegative_float(name: str, value: int | float) -> None:
    if value is None or float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")
