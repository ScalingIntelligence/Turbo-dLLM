# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Production text-only executor for Qwen3.8 hybrid checkpoints."""

from __future__ import annotations

from typing import Any

from dllm_parallel.core.models.backbones.causal_lm.executor import _config_to_dict
from dllm_parallel.core.models.backbones.config_utils import (
    first,
    optional_int,
    string_tuple,
)
from dllm_parallel.core.models.contracts import (
    BackboneCapabilities,
    BackboneKernelPolicy,
    ModelFamilySpec,
)
from dllm_parallel.core.models.compatibility import validate_objective_for_family
from dllm_parallel.core.objectives.fast_dllm_v2 import fast_dllm_v2_schedule


FAMILY = "qwen3_8"
_SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})
_SUPPORTED_LAYER_TYPES = frozenset({"linear_attention", "full_attention"})
_MASK_TOKEN_MIGRATION_VERSION = 1


def _text_config(config: Any) -> Any:
    if isinstance(config, dict):
        value = config.get("text_config")
    else:
        value = getattr(config, "text_config", None)
    return value if value is not None else config


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _set_value(config: Any, name: str, value: Any) -> None:
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def _checkpoint_config(model: Any) -> Any:
    """Return the Qwen text config through supported training wrappers."""

    current = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        module = getattr(current, "module", None)
        if module is not None and module is not current:
            current = module
            continue
        config = getattr(current, "config", None)
        if config is not None:
            return config
        nested_model = getattr(current, "model", None)
        if nested_model is not None and nested_model is not current:
            current = nested_model
            continue
        break
    raise TypeError("Qwen3.8 checkpoint model does not expose its text config")


def summarize_config(model_id: str, config: Any) -> ModelFamilySpec:
    raw = _config_to_dict(config)
    text = raw.get("text_config") or raw
    return ModelFamilySpec(
        model_id=model_id,
        family=FAMILY,
        model_type=str(text.get("model_type") or raw.get("model_type") or "") or None,
        architecture=first(raw.get("architectures") or text.get("architectures")),
        hidden_size=optional_int(text.get("hidden_size")),
        num_layers=optional_int(text.get("num_hidden_layers")),
        num_attention_heads=optional_int(text.get("num_attention_heads")),
        num_key_value_heads=optional_int(text.get("num_key_value_heads")),
        head_dim=optional_int(text.get("head_dim")),
        vocab_size=optional_int(text.get("vocab_size")),
        intermediate_size=optional_int(text.get("intermediate_size")),
        max_position_embeddings=optional_int(text.get("max_position_embeddings")),
        mask_token_id=optional_int(text.get("mask_token_id")),
        attention_layer_types=string_tuple(text.get("layer_types")),
        attention_output_gate=True,
        linear_key_head_dim=optional_int(text.get("linear_key_head_dim")),
        linear_value_head_dim=optional_int(text.get("linear_value_head_dim")),
        linear_num_key_heads=optional_int(text.get("linear_num_key_heads")),
        linear_num_value_heads=optional_int(text.get("linear_num_value_heads")),
        linear_conv_kernel_dim=optional_int(text.get("linear_conv_kernel_dim")),
        dtype=text.get("dtype") or text.get("torch_dtype"),
        transformers_version=raw.get("transformers_version"),
    )


def _validate_config(config: Any) -> None:
    text = _text_config(config)
    model_type = str(_value(text, "model_type", ""))
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "Qwen3.8 requires its official Transformers architecture identifier "
            "(qwen3_5 or qwen3_5_text)"
        )
    layer_types = tuple(str(value) for value in (_value(text, "layer_types", ()) or ()))
    if not layer_types or set(layer_types) - _SUPPORTED_LAYER_TYPES:
        raise ValueError(
            "Qwen3.8 executor requires an explicit hybrid linear/full-attention layer plan"
        )
    if "linear_attention" not in layer_types or "full_attention" not in layer_types:
        raise ValueError("Qwen3.8 executor requires both Gated DeltaNet and full-attention layers")


def _configure_mask_token(spec: Any, *, config: Any, tokenizer: Any | None) -> None:
    text = _text_config(config)
    explicit_id = spec.model.mask_token_id
    token = spec.model.mask_token
    if tokenizer is None or token is None:
        raise ValueError("Qwen3.8 conversion requires model.mask_token and its tokenizer")

    token = str(token)
    original_size = len(tokenizer)
    added = int(tokenizer.add_special_tokens({"mask_token": token}))
    mask_id = int(tokenizer.mask_token_id)
    if explicit_id is not None and mask_id != int(explicit_id):
        raise ValueError("model.mask_token_id does not match the tokenizer's mask token")

    prior_version = _value(text, "dllm_mask_token_migration_version")
    prior_token = _value(text, "dllm_mask_token")
    prior_id = _value(text, "dllm_mask_token_id")
    prior_original_size = _value(text, "dllm_mask_token_original_vocab_size")
    if added == 0:
        if (
            prior_version != _MASK_TOKEN_MIGRATION_VERSION
            or prior_token != token
            or prior_id is None
            or int(prior_id) != mask_id
            or prior_original_size is None
            or not 0 <= int(prior_original_size) <= mask_id
        ):
            raise ValueError(
                "model.mask_token aliases an existing tokenizer token without "
                "Qwen3.8 migration metadata"
            )
        original_size = int(prior_original_size)
    elif added != 1 or mask_id != original_size:
        raise RuntimeError("tokenizer returned an inconsistent mask-token migration")

    vocab_size = int(_value(text, "vocab_size"))
    if not 0 <= mask_id < vocab_size:
        raise ValueError(
            f"mask token id {mask_id} does not fit checkpoint vocab_size={vocab_size}"
        )
    _set_value(text, "mask_token_id", mask_id)
    _set_value(config, "mask_token_id", mask_id)
    _set_value(text, "dllm_mask_token", token)
    _set_value(text, "dllm_mask_token_id", mask_id)
    _set_value(text, "dllm_mask_token_original_vocab_size", original_size)
    _set_value(
        text,
        "dllm_mask_token_migration_version",
        _MASK_TOKEN_MIGRATION_VERSION,
    )
    _set_value(config, "dllm_mask_token", token)
    _set_value(config, "dllm_mask_token_id", mask_id)
    _set_value(config, "dllm_mask_token_original_vocab_size", original_size)
    _set_value(text, "dllm_model_family", FAMILY)
    _set_value(config, "dllm_model_family", FAMILY)
    _set_value(
        config,
        "dllm_mask_token_migration_version",
        _MASK_TOKEN_MIGRATION_VERSION,
    )

    layer_count = int(_value(text, "num_hidden_layers"))
    layer_types = tuple(_value(text, "layer_types", ()) or ())
    if len(layer_types) < layer_count:
        raise ValueError(
            "Qwen3.8 layer_types must contain an entry for every loaded layer"
        )
    if len(layer_types) != layer_count:
        _set_value(text, "layer_types", list(layer_types[:layer_count]))


class Qwen38BackboneExecutor:
    family = FAMILY

    def metadata(self, config: Any, *, model_id: str) -> ModelFamilySpec:
        return summarize_config(model_id, config)

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
            raise ValueError("Qwen3.8 conversion requires model.family=auto, hf, or qwen3_8")
        if spec.model.mask_token is None:
            raise ValueError("Qwen3.8 conversion requires model.mask_token")
        if int(spec.topology.expert_parallel_size) != 1:
            raise ValueError("Qwen3.8-27B is dense and requires expert_parallel_size=1")
        if spec.topology.sequence_parallel and int(spec.topology.tensor_parallel_size) <= 1:
            raise ValueError("sequence_parallel requires tensor_parallel_size > 1")
        if config is not None:
            _validate_config(config)
            text = _text_config(config)
            tp_size = int(spec.topology.tensor_parallel_size)
            cp_size = int(spec.topology.context_parallel_size)
            for field in (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "num_attention_heads",
                "num_key_value_heads",
            ):
                heads = int(_value(text, field))
                if heads % tp_size:
                    raise ValueError(f"{field}={heads} must divide evenly by TP={tp_size}")
            for field in ("linear_num_key_heads", "linear_num_value_heads"):
                heads = int(_value(text, field))
                if heads % (tp_size * cp_size):
                    raise ValueError(
                        f"{field}={heads} must divide evenly by TP*CP="
                        f"{tp_size * cp_size} for Gated DeltaNet context parallelism"
                    )

    def prepare_tokenizer_and_config(
        self, spec: Any, *, config: Any, tokenizer: Any | None
    ) -> None:
        _configure_mask_token(spec, config=config, tokenizer=tokenizer)

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
        from dllm_parallel.core.models.backbones.qwen3_8.model import load_text_checkpoint

        return load_text_checkpoint(
            model_id=model_id,
            revision=revision,
            config=config,
            runtime=runtime,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    def build_packed_block_diffusion_model(self, model: Any, **kwargs: Any) -> Any:
        from dllm_parallel.core.models.backbones.qwen3_8.model import (
            Qwen38PackedBlockDiffusionModel,
        )

        return Qwen38PackedBlockDiffusionModel(model, **kwargs)

    def build_training_model(self, model: Any, *, runtime: Any, spec: Any) -> Any:
        return self.build_packed_block_diffusion_model(
            model,
            runtime=runtime,
            seq_len=int(spec.model.seq_len),
            block_size=int(spec.objective.block_size),
            ring_attention_key_chunk_size=int(spec.kernel.ring_attention_key_chunk_size),
            activation_checkpointing=bool(spec.training.activation_checkpointing),
            activation_checkpointing_scope=str(spec.training.activation_checkpointing_scope),
            mlp_token_chunk_size=int(spec.kernel.mlp_token_chunk_size),
        )

    def fsdp_modules(self, model: Any) -> tuple[Any, ...]:
        return tuple(getattr(model, "layers", ()))

    def build_training_task(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.objectives.training import build_fast_dllm_v2_training_task

        return build_fast_dllm_v2_training_task(**kwargs)

    def build_data_runtime(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.data import build_standard_data_runtime

        return build_standard_data_runtime(**kwargs)

    def verify_native_kernels(self, spec: Any) -> dict[str, Any]:
        del spec
        from dllm_parallel.core.attention.flex import verify_flex_attention_runtime
        from dllm_parallel.core.models.backbones.qwen3_8.model import verify_qwen38_runtime

        attention = verify_flex_attention_runtime().to_log_dict()
        return {
            **attention,
            **verify_qwen38_runtime(),
            "backend": "qwen3_8_hybrid_training",
            "softmax_attention_forward": "fa4_and_compiled_flex",
            "softmax_attention_backward": "compiled_flex_d256",
            "linear_attention": "fla_tilelang",
            "projection_mlp": "transformer_engine",
        }

    def parallel_work_units(self, spec: Any) -> int:
        return int(spec.model.seq_len) // int(spec.objective.block_size)

    def tokenizer_model_id(self, spec: Any) -> str:
        return str(spec.model.id)

    def build_objective_schedule(self, spec: Any, *, sequence_length: int | None = None) -> Any:
        return fast_dllm_v2_schedule(
            sequence_length=int(sequence_length or spec.model.seq_len),
            block_size=int(spec.objective.block_size),
            mask_token_id=spec.model.mask_token_id,
        )

    def kernel_policy(self, spec: Any) -> BackboneKernelPolicy:
        del spec
        return BackboneKernelPolicy()

    def validate_tokenizer_data_compatibility(self, spec: Any, tokenizer: Any | None) -> None:
        del spec
        if tokenizer is None:
            raise ValueError("Qwen3.8 text-only training requires its tokenizer")

    def load_checkpoint_hooks(self, checkpoint: Any, model: Any) -> None:
        state = (checkpoint or {}).get("backbone_state") or {}
        config = _checkpoint_config(model)
        family = state.get("family")
        if str(family) != FAMILY:
            raise RuntimeError(f"checkpoint family {family!r} is incompatible with {FAMILY!r}")
        expected_mask_id = state.get("mask_token_id")
        observed_mask_id = getattr(config, "mask_token_id", None)
        if expected_mask_id is None or observed_mask_id is None:
            raise RuntimeError("checkpoint is missing Qwen3.8 mask-token migration metadata")
        if int(expected_mask_id) != int(observed_mask_id):
            raise RuntimeError(
                "checkpoint mask token does not match the resolved Qwen3.8 tokenizer"
            )
        migration_version = state.get("mask_token_migration_version")
        if migration_version is None or int(migration_version) != _MASK_TOKEN_MIGRATION_VERSION:
            raise RuntimeError("checkpoint uses an unsupported Qwen3.8 mask migration")
        expected_token = state.get("mask_token")
        observed_token = getattr(config, "dllm_mask_token", None)
        if expected_token is None or observed_token is None:
            raise RuntimeError("checkpoint is missing Qwen3.8 mask-token migration metadata")
        if str(expected_token) != str(observed_token):
            raise RuntimeError(
                "checkpoint mask token does not match the resolved Qwen3.8 tokenizer"
            )
        expected_original_size = state.get("mask_token_original_vocab_size")
        observed_original_size = getattr(
            config,
            "dllm_mask_token_original_vocab_size",
            None,
        )
        if expected_original_size is None or observed_original_size is None:
            raise RuntimeError("checkpoint is missing Qwen3.8 mask-token migration metadata")
        if int(expected_original_size) != int(observed_original_size):
            raise RuntimeError(
                "checkpoint tokenizer vocabulary does not match the Qwen3.8 migration"
            )

    def save_checkpoint_hooks(self, model: Any) -> dict[str, Any]:
        config = _checkpoint_config(model)
        return {
            "family": FAMILY,
            "mask_token": getattr(config, "dllm_mask_token", None),
            "mask_token_id": getattr(config, "mask_token_id", None),
            "mask_token_migration_version": getattr(
                config,
                "dllm_mask_token_migration_version",
                _MASK_TOKEN_MIGRATION_VERSION,
            ),
            "mask_token_original_vocab_size": getattr(
                config,
                "dllm_mask_token_original_vocab_size",
                None,
            ),
        }

    def sharded_state_dict(self, model: Any) -> dict[str, Any]:
        state_dict = getattr(model, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("Qwen3.8 model does not expose state_dict")
        return {
            "format": "dllm_parallel.backbone_state_dict.v1",
            "family": FAMILY,
            "state_dict": state_dict(),
        }

    def migrate_config(self, raw_config: Any) -> Any:
        return raw_config


def build_executor() -> Qwen38BackboneExecutor:
    return Qwen38BackboneExecutor()
