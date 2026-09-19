# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DiffusionGemma packed block-diffusion executor."""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dllm_parallel.core.models.backbones.config_utils import optional_int
from dllm_parallel.core.models.backbones.diffusiongemma.metadata import (
    FAMILY,
    build_schedule,
    summarize_config,
)
from dllm_parallel.core.models.backbones.moe_checkpoint import (
    record_moe_route_indices,
    replay_moe_route_indices,
)
from dllm_parallel.core.models.contracts import (
    BackboneCapabilities,
    BackboneKernelPolicy,
)
from dllm_parallel.core.models.compatibility import validate_objective_for_family


class DiffusionGemmaBackboneExecutor:
    family = FAMILY

    def metadata(self, config: Any, *, model_id: str) -> Any:
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

    def prepare_tokenizer_and_config(
        self,
        spec: Any,
        *,
        config: Any,
        tokenizer: Any | None,
    ) -> None:
        """DiffusionGemma's pinned checkpoint already defines its token IDs."""

        del spec, config, tokenizer

    def validate_run_spec(self, spec: Any, *, config: Any | None = None) -> None:
        validate_objective_for_family(
            family=self.family,
            objective=str(getattr(getattr(spec, "objective", None), "name", "")),
        )
        if spec.model.family not in {"auto", "hf", FAMILY}:
            raise ValueError(
                "DiffusionGemma executor requires "
                "model.family=auto, hf, or diffusion_gemma"
            )
        if spec.kernel.cp_bp_attention_policy != "production":
            raise ValueError(
                "DiffusionGemma CP/BP requires the production attention policy"
            )
        if (
            spec.topology.sequence_parallel
            and int(spec.topology.tensor_parallel_size) <= 1
        ):
            raise ValueError("sequence_parallel requires tensor_parallel_size > 1")
        native_objective = (
            str(getattr(getattr(spec, "objective", None), "name", ""))
            == "diffusiongemma_native_sft"
        )
        context_parallel_size = int(
            getattr(spec.topology, "context_parallel_size", 1) or 1
        )
        block_parallel_size = int(getattr(spec.topology, "block_parallel_size", 1) or 1)
        native_block_size = getattr(
            getattr(spec, "objective", None), "block_size", None
        )
        if (
            native_objective
            and context_parallel_size > 1
            and block_parallel_size == 1
            and native_block_size is not None
            and int(native_block_size) % context_parallel_size
        ):
            raise ValueError(
                "DiffusionGemma native pure CP requires objective.block_size divisible "
                "by topology.context_parallel_size"
            )
        training = getattr(spec, "training", None)
        if (
            native_objective
            and training is not None
            and (
                not bool(getattr(training, "activation_checkpointing", False))
                or str(getattr(training, "activation_checkpointing_scope", ""))
                != "full"
            )
        ):
            raise ValueError(
                "DiffusionGemma native SFT requires full activation checkpointing"
            )
        if int(getattr(spec.topology, "expert_parallel_size", 1) or 1) > 1 and str(
            spec.optimizer.backend
        ) not in {"auto", "deepspeed_zero2"}:
            raise ValueError(
                "DiffusionGemma expert parallelism requires the DeepSpeed "
                "ZeRO-2 optimizer backend"
            )
        if config is not None:
            _prepare_config(config)
            text_config = getattr(config, "text_config", None)
            if text_config is None:
                raise ValueError("DiffusionGemma config must expose text_config")
            if optional_int(getattr(text_config, "num_experts", None)) is None:
                raise ValueError(
                    "DiffusionGemma production support requires "
                    "MoE text_config.num_experts"
                )
            layer_types = tuple(getattr(text_config, "layer_types", ()) or ())
            num_layers = optional_int(getattr(text_config, "num_hidden_layers", None))
            if num_layers is None or len(layer_types) != num_layers:
                raise ValueError(
                    "DiffusionGemma text_config.layer_types must describe every layer"
                )
            unsupported_layer_types = sorted(
                set(layer_types) - {"sliding_attention", "full_attention"}
            )
            if unsupported_layer_types:
                raise ValueError(
                    "DiffusionGemma has unsupported attention layer types: "
                    + ", ".join(unsupported_layer_types)
                )
            sliding_window = optional_int(getattr(text_config, "sliding_window", None))
            if "sliding_attention" in layer_types and (
                sliding_window is None or sliding_window <= 0
            ):
                raise ValueError(
                    "DiffusionGemma sliding-attention layers require a positive "
                    "text_config.sliding_window"
                )
            if not bool(getattr(config, "tie_word_embeddings", False)):
                raise ValueError(
                    "DiffusionGemma packed training requires its shared encoder/decoder "
                    "text backbone"
                )
            num_experts = int(text_config.num_experts)
            ep_size = int(getattr(spec.topology, "expert_parallel_size", 1) or 1)
            if num_experts % ep_size != 0:
                raise ValueError(
                    "DiffusionGemma num_experts must be divisible by "
                    "expert_parallel_size"
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

        _prepare_config(config)
        return load_hf_model_from_config(
            model_id,
            config=config,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_auto_class="block_diffusion",
            parallel_runtime=runtime,
            torch_dtype=dtype,
            device=device,
            model_kwargs={"low_cpu_mem_usage": True},
        )

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
        native_objective: bool = False,
    ) -> Any:
        return build_packed_block_diffusion_model(
            model,
            runtime=runtime,
            seq_len=seq_len,
            block_size=block_size,
            ring_attention_key_chunk_size=ring_attention_key_chunk_size,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_scope=activation_checkpointing_scope,
            mlp_token_chunk_size=mlp_token_chunk_size,
            native_objective=bool(native_objective),
        )

    def build_training_model(self, model: Any, *, runtime: Any, spec: Any) -> Any:
        block_size = spec.objective.block_size
        if block_size is None:
            raise ValueError("DiffusionGemma training requires objective.block_size")
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
            native_objective=(str(spec.objective.name) == "diffusiongemma_native_sft"),
        )

    def fsdp_modules(self, model: Any) -> tuple[Any, ...]:
        from dllm_parallel.core.models.backbones.nemotron.model import (
            packed_fsdp_modules,
        )

        return packed_fsdp_modules(model)

    def build_training_task(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.objectives.training import (
            build_diffusiongemma_native_training_task,
            build_standard_training_task,
        )

        spec = kwargs.get("spec")
        if (
            str(getattr(getattr(spec, "objective", None), "name", ""))
            == "diffusiongemma_native_sft"
        ):
            return build_diffusiongemma_native_training_task(**kwargs)
        return build_standard_training_task(**kwargs)

    def build_data_runtime(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.data import build_standard_data_runtime

        return build_standard_data_runtime(**kwargs)

    def verify_native_kernels(self, spec: Any) -> dict[str, Any]:
        del spec
        from dllm_parallel.core.attention.flex import verify_flex_attention_runtime

        return verify_flex_attention_runtime().to_log_dict()

    def parallel_work_units(self, spec: Any) -> int:
        block_size = spec.objective.block_size
        if block_size is None:
            raise ValueError("DiffusionGemma training requires objective.block_size")
        return int(spec.model.seq_len) // int(block_size)

    def tokenizer_model_id(self, spec: Any) -> str:
        return str(spec.model.id)

    def build_objective_schedule(
        self,
        spec: Any,
        *,
        sequence_length: int | None = None,
    ) -> Any:
        return build_schedule(spec, sequence_length=sequence_length)

    def kernel_policy(self, spec: Any) -> BackboneKernelPolicy:
        del spec
        return BackboneKernelPolicy()

    def validate_tokenizer_data_compatibility(
        self,
        spec: Any,
        tokenizer: Any | None,
    ) -> None:
        if spec.data.input_mode in {"text", "dataset"} and tokenizer is None:
            raise ValueError("DiffusionGemma text/dataset input requires a tokenizer")
        if tokenizer is not None and getattr(tokenizer, "mask_token_id", None) is None:
            raise ValueError("DiffusionGemma tokenizer must expose mask_token_id")

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
            raise TypeError("DiffusionGemma model does not expose state_dict")
        return {
            "format": "dllm_parallel.backbone_state_dict.v1",
            "family": FAMILY,
            "state_dict": state_dict(),
        }

    def migrate_config(self, raw_config: Any) -> Any:
        _prepare_config(raw_config)
        return raw_config


def build_executor() -> DiffusionGemmaBackboneExecutor:
    return DiffusionGemmaBackboneExecutor()


def build_packed_block_diffusion_model(
    model: Any,
    *,
    runtime: Any,
    seq_len: int,
    block_size: int,
    ring_attention_key_chunk_size: int = 0,
    activation_checkpointing: bool = True,
    activation_checkpointing_scope: str = "full",
    mlp_token_chunk_size: int = 0,
    native_objective: bool = False,
) -> Any:
    from dllm_parallel.core.models.backbones.diffusiongemma.expert_parallel import (
        install_diffusion_gemma_expert_parallel,
    )
    from dllm_parallel.core.models.backbones.nemotron.model import (
        build_packed_block_diffusion_model as build_nemotron_execution_model,
    )

    _prepare_config(getattr(model, "config", None))
    _validate_shared_text_backbone(model)
    _install_checkpoint_stable_diffusion_gemma_routers(model)
    if native_objective:
        _freeze_diffusion_gemma_routers(model)
    install_diffusion_gemma_expert_parallel(model, runtime)
    _release_unused_conditioning_modules(model)
    return build_nemotron_execution_model(
        model,
        runtime=runtime,
        seq_len=int(seq_len),
        block_size=int(block_size),
        ring_attention_key_chunk_size=int(ring_attention_key_chunk_size),
        activation_checkpointing=bool(activation_checkpointing),
        activation_checkpointing_scope=str(activation_checkpointing_scope),
        mlp_token_chunk_size=int(mlp_token_chunk_size),
        self_condition_clean_tokens=False,
        encoder_causal_attention=True,
    )


class _ReleasedDiffusionGemmaConditioningEncoder(nn.Module):
    """Sentinel for encoder/vision modules unused by packed decoder denoising."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "DiffusionGemma conditioning encoder was released for packed "
            "block-diffusion decoder training"
        )


class _CheckpointStableDiffusionGemmaRouter(nn.Module):
    """Gemma4 router with exact route replay during activation recomputation."""

    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        norm = getattr(router, "norm", None)
        proj = getattr(router, "proj", None)
        scale = getattr(router, "scale", None)
        per_expert_scale = getattr(router, "per_expert_scale", None)
        scalar_root_size = getattr(router, "scalar_root_size", None)
        config = getattr(router, "config", None)
        if (
            not isinstance(norm, nn.Module)
            or not isinstance(proj, nn.Module)
            or not isinstance(scale, torch.Tensor)
            or not isinstance(per_expert_scale, torch.Tensor)
            or scalar_root_size is None
            or config is None
        ):
            raise TypeError(
                "DiffusionGemma router must expose norm, proj, scale, "
                "scalar_root_size, per_expert_scale, and config"
            )
        self.config = config
        self.norm = norm
        self.proj = proj
        self.scale = scale
        self.scalar_root_size = float(scalar_root_size)
        self.per_expert_scale = per_expert_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_states = self.norm(hidden_states)
        router_states = router_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(router_states)
        router_probabilities = F.softmax(expert_scores, dim=-1, dtype=torch.float32)
        with torch.no_grad():
            top_k_index = replay_moe_route_indices(self)
            if top_k_index is None:
                top_k_index = torch.topk(
                    router_probabilities.detach(),
                    k=int(self.config.top_k_experts),
                    dim=-1,
                ).indices.contiguous()
                record_moe_route_indices(self, top_k_index)
            else:
                top_k_index = top_k_index.to(
                    device=router_probabilities.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
        top_k_weights = router_probabilities.gather(-1, top_k_index)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index


def _install_checkpoint_stable_diffusion_gemma_routers(model: Any) -> None:
    root = getattr(model, "model", None)
    decoder = (
        getattr(root, "decoder", None)
        if root is not None
        else getattr(model, "decoder", None)
    )
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return
    for layer in layers:
        router = getattr(layer, "router", None)
        if router is None or isinstance(router, _CheckpointStableDiffusionGemmaRouter):
            continue
        layer.router = _CheckpointStableDiffusionGemmaRouter(router)


def _freeze_diffusion_gemma_routers(model: Any) -> None:
    """Freeze every native router parameter without changing legacy SFT."""

    root = getattr(model, "model", None)
    decoder = (
        getattr(root, "decoder", None)
        if root is not None
        else getattr(model, "decoder", None)
    )
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return
    for layer in layers:
        router = getattr(layer, "router", None)
        if isinstance(router, nn.Module):
            for parameter in router.parameters():
                parameter.requires_grad_(False)


def _release_unused_conditioning_modules(model: Any) -> None:
    root = getattr(model, "model", None)
    if root is None:
        return
    encoder = getattr(root, "encoder", None)
    decoder = getattr(root, "decoder", None)
    if encoder is None or decoder is None:
        return
    root.encoder = _ReleasedDiffusionGemmaConditioningEncoder()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_shared_text_backbone(model: Any) -> None:
    """Verify that releasing the duplicate encoder keeps every trainable weight.

    DiffusionGemma represents one tied text transformer as encoder and decoder
    modules. Packed training executes that transformer once with the exact
    clean/noisy attention roles, so retaining both module trees would only
    duplicate parameter registration and memory. Validate parameter identity
    before removing the encoder tree; value equality is insufficient because
    independent parameters would require independent gradients and updates.
    """

    root = getattr(model, "model", None)
    encoder = getattr(root, "encoder", None) if root is not None else None
    decoder = getattr(root, "decoder", None) if root is not None else None
    language_model = getattr(encoder, "language_model", None)
    if language_model is None or decoder is None:
        raise TypeError(
            "DiffusionGemma packed training requires tied encoder.language_model "
            "and decoder modules"
        )
    encoder_parameters = dict(language_model.named_parameters())
    decoder_parameters = {
        name: parameter
        for name, parameter in decoder.named_parameters()
        if not name.startswith("self_conditioning.")
    }
    missing: list[str] = []
    untied: list[str] = []
    for name, decoder_parameter in decoder_parameters.items():
        encoder_parameter = encoder_parameters.get(name)
        if encoder_parameter is None:
            missing.append(name)
        elif encoder_parameter is not decoder_parameter:
            untied.append(name)
    extra = sorted(encoder_parameters.keys() - decoder_parameters.keys())
    if missing or untied or extra:
        details = []
        if missing:
            details.append(f"missing={missing[:3]}")
        if untied:
            details.append(f"untied={untied[:3]}")
        if extra:
            details.append(f"encoder_only={extra[:3]}")
        raise RuntimeError(
            "DiffusionGemma encoder/decoder text weights are not fully tied: "
            + ", ".join(details)
        )


def _prepare_config(config: Any) -> None:
    if config is None:
        return
    text_config = getattr(config, "text_config", None)
    canvas_length = getattr(config, "canvas_length", None)
    if canvas_length is not None and getattr(config, "block_size", None) is None:
        setattr(config, "block_size", int(canvas_length))
    if text_config is not None:
        if getattr(text_config, "num_experts", None) is not None:
            setattr(text_config, "enable_moe_block", True)
        setattr(text_config, "attention_k_eq_v", True)
        if hasattr(text_config, "use_cache"):
            text_config.use_cache = False
        if (
            getattr(config, "vocab_size", None) is None
            and getattr(text_config, "vocab_size", None) is not None
        ):
            setattr(config, "vocab_size", int(text_config.vocab_size))
        if (
            getattr(config, "max_position_embeddings", None) is None
            and getattr(text_config, "max_position_embeddings", None) is not None
        ):
            setattr(
                config,
                "max_position_embeddings",
                int(text_config.max_position_embeddings),
            )
        if (
            getattr(config, "mask_token_id", None) is None
            and getattr(text_config, "mask_token_id", None) is not None
        ):
            setattr(config, "mask_token_id", int(text_config.mask_token_id))
    if hasattr(config, "use_cache"):
        config.use_cache = False


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        if isinstance(value, dict):
            return value
    raise TypeError("DiffusionGemma config must be a dict or expose to_dict()")
