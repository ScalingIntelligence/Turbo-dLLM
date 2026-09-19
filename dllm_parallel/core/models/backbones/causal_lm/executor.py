# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native Fast-dLLM v2 conversion for standard Hugging Face causal LMs."""

from __future__ import annotations

from typing import Any

from dllm_parallel.core.models.backbones.config_utils import (
    first,
    optional_int,
)
from dllm_parallel.core.models.contracts import (
    BackboneCapabilities,
    BackboneKernelPolicy,
    ModelFamilySpec,
)
from dllm_parallel.core.models.compatibility import validate_objective_for_family
from dllm_parallel.core.objectives.fast_dllm_v2 import fast_dllm_v2_schedule


FAMILY = "causal_lm"


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    raise TypeError("causal-LM config must be a dict or expose to_dict()")


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("text_config")
    return value if isinstance(value, dict) else config


def summarize_config(model_id: str, config: dict[str, Any]) -> ModelFamilySpec:
    text = _text_config(config)
    return ModelFamilySpec(
        model_id=model_id,
        family=FAMILY,
        model_type=str(text.get("model_type") or config.get("model_type") or "")
        or None,
        architecture=first(config.get("architectures") or text.get("architectures")),
        remote_code_required=bool(config.get("auto_map") or text.get("auto_map")),
        hidden_size=optional_int(text.get("hidden_size")),
        num_layers=optional_int(text.get("num_hidden_layers")),
        num_attention_heads=optional_int(text.get("num_attention_heads")),
        num_key_value_heads=optional_int(text.get("num_key_value_heads")),
        head_dim=optional_int(text.get("head_dim")),
        vocab_size=optional_int(text.get("vocab_size")),
        intermediate_size=optional_int(text.get("intermediate_size")),
        max_position_embeddings=optional_int(text.get("max_position_embeddings")),
        mask_token_id=optional_int(
            text.get("mask_token_id", config.get("mask_token_id"))
        ),
        num_experts=optional_int(text.get("num_experts", text.get("n_routed_experts"))),
        top_k_experts=optional_int(
            text.get("num_experts_per_tok", text.get("num_experts_per_token"))
        ),
        expert_intermediate_size=optional_int(text.get("moe_intermediate_size")),
        sliding_window=optional_int(text.get("sliding_window")),
        attn_implementation=text.get("attn_implementation"),
        use_cache=text.get("use_cache"),
        dtype=text.get("torch_dtype") or text.get("dtype"),
        transformers_version=config.get("transformers_version"),
    )


def _validate_decoder_architecture(config: Any) -> None:
    raw = _config_to_dict(config)
    text = _text_config(raw)
    architectures = tuple(raw.get("architectures") or text.get("architectures") or ())
    if architectures and not any(
        str(name).endswith("ForCausalLM") for name in architectures
    ):
        raise ValueError(
            "Fast-dLLM v2 conversion requires a decoder-only *ForCausalLM checkpoint"
        )

    layer_types = tuple(text.get("layer_types") or ())
    unsupported_mixers = sorted(
        {
            str(kind)
            for kind in layer_types
            if str(kind) not in {"full_attention", "sliding_attention"}
        }
    )
    if unsupported_mixers:
        raise ValueError(
            "Fast-dLLM v2 conversion currently requires softmax-attention decoder "
            "layers; unsupported sequence mixers: " + ", ".join(unsupported_mixers)
        )

    latent_attention_fields = (
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "index_head_dim",
        "index_topk",
    )
    present = [name for name in latent_attention_fields if text.get(name) is not None]
    if present:
        raise ValueError(
            "Fast-dLLM v2 conversion does not yet support latent or compressed-sparse "
            "attention; detected " + ", ".join(present)
        )


class CausalLMBackboneExecutor:
    family = FAMILY

    def metadata(self, config: Any, *, model_id: str) -> ModelFamilySpec:
        return summarize_config(model_id, _config_to_dict(config))

    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            family=FAMILY,
            packed_block_diffusion=True,
            tensor_parallel=True,
            sequence_parallel=True,
            checkpoint_hooks=True,
            sharded_state_dict=True,
            tokenizer_required_for_text=True,
        )

    def validate_run_spec(self, spec: Any, *, config: Any | None = None) -> None:
        validate_objective_for_family(
            family=self.family,
            objective=str(getattr(getattr(spec, "objective", None), "name", "")),
        )
        if spec.model.family not in {"auto", "hf", FAMILY}:
            raise ValueError(
                "causal_lm conversion requires model.family=auto, hf, or causal_lm"
            )
        if config is not None:
            _validate_decoder_architecture(config)
            metadata = self.metadata(config, model_id=str(spec.model.id))
            if metadata.num_experts:
                raise ValueError(
                    "generic causal-LM conversion does not yet support MoE layers; "
                    "expert-parallel conversion requires a model-family adapter"
                )
        if int(spec.topology.expert_parallel_size) != 1:
            raise ValueError(
                "causal_lm conversion currently requires expert_parallel_size=1"
            )
        if (
            spec.topology.sequence_parallel
            and int(spec.topology.tensor_parallel_size) <= 1
        ):
            raise ValueError("sequence_parallel requires tensor_parallel_size > 1")
        if spec.model.mask_token_id is None:
            raw = _config_to_dict(config) if config is not None else {}
            text = _text_config(raw)
            if text.get("mask_token_id", raw.get("mask_token_id")) is None:
                raise ValueError(
                    "autoregressive conversion requires model.mask_token_id to name an "
                    "existing reserved vocabulary token"
                )

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
        from dllm_parallel.core.models.hf_loader import load_hf_model_from_config

        return load_hf_model_from_config(
            model_id,
            config=config,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_auto_class="causal_lm",
            parallel_runtime=runtime,
            torch_dtype=dtype,
            device=device,
            model_kwargs={"low_cpu_mem_usage": True},
        )

    def build_packed_block_diffusion_model(self, model: Any, **kwargs: Any) -> Any:
        from dllm_parallel.core.models.backbones.nemotron import (
            build_packed_block_diffusion_model,
        )

        return build_packed_block_diffusion_model(
            model,
            self_condition_clean_tokens=False,
            **kwargs,
        )

    def build_training_model(self, model: Any, *, runtime: Any, spec: Any) -> Any:
        block_size = spec.objective.block_size
        if block_size is None:
            raise ValueError("Fast-dLLM v2 conversion requires objective.block_size")
        return self.build_packed_block_diffusion_model(
            model,
            runtime=runtime,
            seq_len=int(spec.model.seq_len),
            block_size=int(block_size),
            ring_attention_key_chunk_size=int(
                spec.kernel.ring_attention_key_chunk_size
            ),
            activation_checkpointing=bool(spec.training.activation_checkpointing),
            activation_checkpointing_scope=str(
                spec.training.activation_checkpointing_scope
            ),
            mlp_token_chunk_size=int(spec.kernel.mlp_token_chunk_size),
        )

    def fsdp_modules(self, model: Any) -> tuple[Any, ...]:
        from dllm_parallel.core.models.backbones.nemotron.model import (
            packed_fsdp_modules,
        )

        return packed_fsdp_modules(model)

    def build_training_task(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.objectives.training import (
            build_fast_dllm_v2_training_task,
        )

        return build_fast_dllm_v2_training_task(**kwargs)

    def build_data_runtime(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.data import build_standard_data_runtime

        return build_standard_data_runtime(**kwargs)

    def verify_native_kernels(self, spec: Any) -> dict[str, Any]:
        del spec
        from dllm_parallel.core.attention.flex import verify_flex_attention_runtime

        return verify_flex_attention_runtime().to_log_dict()

    def parallel_work_units(self, spec: Any) -> int:
        return int(spec.model.seq_len) // int(spec.objective.block_size)

    def tokenizer_model_id(self, spec: Any) -> str:
        return str(spec.model.id)

    def build_objective_schedule(
        self,
        spec: Any,
        *,
        sequence_length: int | None = None,
    ) -> Any:
        return fast_dllm_v2_schedule(
            sequence_length=int(sequence_length or spec.model.seq_len),
            block_size=int(spec.objective.block_size),
            mask_token_id=spec.model.mask_token_id,
        )

    def kernel_policy(self, spec: Any) -> BackboneKernelPolicy:
        del spec
        return BackboneKernelPolicy()

    def validate_tokenizer_data_compatibility(
        self, spec: Any, tokenizer: Any | None
    ) -> None:
        if spec.data.input_mode in {"text", "dataset"} and tokenizer is None:
            raise ValueError("causal-LM text and dataset inputs require a tokenizer")

    def load_checkpoint_hooks(self, checkpoint: Any, model: Any) -> None:
        del model
        state = (checkpoint or {}).get("backbone_state") or {}
        family = state.get("family")
        if family is not None and str(family) != FAMILY:
            raise RuntimeError(
                f"checkpoint backbone family {family!r} cannot be loaded by {FAMILY!r}"
            )

    def save_checkpoint_hooks(self, model: Any) -> dict[str, Any]:
        return {
            "family": FAMILY,
            "model_class": model.__class__.__name__,
            "checkpoint_hooks_version": 1,
        }

    def sharded_state_dict(self, model: Any) -> dict[str, Any]:
        state_dict = getattr(model, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("causal-LM model does not expose state_dict")
        return {
            "format": "dllm_parallel.backbone_state_dict.v1",
            "family": FAMILY,
            "state_dict": state_dict(),
        }

    def migrate_config(self, raw_config: Any) -> Any:
        return raw_config


def build_executor() -> CausalLMBackboneExecutor:
    return CausalLMBackboneExecutor()


__all__ = [
    "CausalLMBackboneExecutor",
    "build_executor",
    "summarize_config",
]
