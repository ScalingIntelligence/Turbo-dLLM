# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Model-family contracts: the spec and executor interfaces backbones implement.

This is the single, torch-free contract surface for the model layer. It unifies
the normalized model-family spec (``ModelFamilySpec``) with the backbone
behavioral contract (``BackboneExecutor`` plus its capability/kernel-policy
dataclasses) so registry, loader, planner, and backbone modules import their
interfaces from one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelFamilySpec:
    model_id: str
    family: str
    model_type: str | None
    architecture: str | None
    remote_code_required: bool = False
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    num_global_key_value_heads: int | None = None
    layer_types: tuple[str, ...] | None = None
    head_dim: int | None = None
    global_head_dim: int | None = None
    vocab_size: int | None = None
    intermediate_size: int | None = None
    max_position_embeddings: int | None = None
    block_size: int | None = None
    canvas_length: int | None = None
    diffusion_paradigm: str | None = None
    mask_token_id: int | None = None
    num_experts: int | None = None
    top_k_experts: int | None = None
    expert_intermediate_size: int | None = None
    sliding_window: int | None = None
    target_hidden_size: int | None = None
    target_feature_width: int | None = None
    draft_vocab_size: int | None = None
    attention_layer_types: tuple[str, ...] | None = None
    sliding_window_non_causal: bool | None = None
    dflash2_conv_kernel_size: int | None = None
    dflash2_conv_group_size: int | None = None
    dflash2_selector_rank: int | None = None
    dflash2_selector_top_k: int | None = None
    attention_output_gate: bool | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_conv_kernel_dim: int | None = None
    attn_implementation: str | None = None
    use_cache: bool | None = None
    use_bidirectional_attention: str | None = None
    dtype: str | None = None
    transformers_version: str | None = None

    @property
    def requires_remote_code(self) -> bool:
        return self.remote_code_required


@dataclass(frozen=True)
class BackboneCapabilities:
    family: str
    packed_block_diffusion: bool
    tensor_parallel: bool
    sequence_parallel: bool
    checkpoint_hooks: bool
    sharded_state_dict: bool
    tokenizer_required_for_text: bool
    tokenizer_required_for_dataset: bool = True
    uses_transformers: bool = True


@dataclass(frozen=True)
class BackboneKernelPolicy:
    cp_bp_attention_policy: str = "production"
    clean_kv_layout: str = "zigzag"
    supports_vocab_parallel_ce: bool = True


@runtime_checkable
class BackboneExecutor(Protocol):
    family: str

    def metadata(self, config: Any, *, model_id: str) -> Any:
        """Return normalized model-family metadata."""

    def capabilities(self) -> BackboneCapabilities:
        """Return topology and lifecycle capabilities for this backbone."""

    def validate_run_spec(self, spec: Any, *, config: Any | None = None) -> None:
        """Fail closed for unsupported model/topology/kernel combinations."""

    def build_model(
        self,
        *,
        model_id: str,
        revision: str | None,
        config: Any,
        runtime: Any | None,
        dtype: Any,
        device: Any,
        trust_remote_code: bool,
    ) -> Any:
        """Construct the concrete model on the target device."""

    def prepare_tokenizer_and_config(
        self,
        spec: Any,
        *,
        config: Any,
        tokenizer: Any | None,
    ) -> None:
        """Finalize tokenizer-dependent config before checkpoint construction."""

    def build_packed_block_diffusion_model(
        self,
        model: Any,
        *,
        runtime: Any,
        seq_len: int,
        block_size: int,
        ring_attention_key_chunk_size: int = 0,
        activation_checkpointing: bool = True,
        activation_checkpointing_scope: str = "full",
        mlp_token_chunk_size: int = 0,
    ) -> Any:
        """Wrap the model with the optimized packed block-diffusion executor."""

    def build_training_model(
        self,
        model: Any,
        *,
        runtime: Any,
        spec: Any,
    ) -> Any:
        """Bind the model to its objective-specific distributed executor."""

    def fsdp_modules(self, model: Any) -> tuple[Any, ...]:
        """Return nested FSDP2 units reached through registered forward methods."""

    def build_training_task(
        self,
        *,
        spec: Any,
        runtime: Any,
        device: Any,
        seed: int,
        data_parallel_seed: int,
        mask_token_id: int,
        vocab_size: int,
        token_accounting: Any,
        block_size: int,
        model_metadata: ModelFamilySpec,
    ) -> Any:
        """Construct the objective-bound task consumed by the trainer loop."""

    def build_data_runtime(
        self,
        *,
        spec: Any,
        tokenizer: Any | None,
        vocab_size: int,
        mask_token_id: int,
        device: Any,
        dtype: Any,
        seed: int,
        token_pool: Any | None,
        runtime: Any | None,
        rank: int,
        world_size: int,
    ) -> Any:
        """Construct the restartable input runtime for this backbone."""

    def verify_native_kernels(self, spec: Any) -> dict[str, Any]:
        """Verify packaged native operators and return startup metadata."""

    def parallel_work_units(self, spec: Any) -> int:
        """Return the objective work units used to build BP ownership."""

    def tokenizer_model_id(self, spec: Any) -> str:
        """Return the checkpoint whose tokenizer defines model token IDs."""

    def build_objective_schedule(
        self,
        spec: Any,
        *,
        sequence_length: int | None = None,
    ) -> Any:
        """Construct the objective schedule for this family."""

    def kernel_policy(self, spec: Any) -> BackboneKernelPolicy:
        """Return the effective kernel policy for the resolved spec."""

    def validate_tokenizer_data_compatibility(self, spec: Any, tokenizer: Any | None) -> None:
        """Validate tokenizer/data requirements before training."""

    def load_checkpoint_hooks(self, checkpoint: Any, model: Any) -> None:
        """Backbone-specific checkpoint load hook."""

    def save_checkpoint_hooks(self, model: Any) -> dict[str, Any]:
        """Backbone-specific checkpoint save hook."""

    def sharded_state_dict(self, model: Any) -> dict[str, Any]:
        """Return a backbone-owned sharded state dict."""

    def migrate_config(self, raw_config: Any) -> Any:
        """Migrate older family config shapes into the current schema."""
