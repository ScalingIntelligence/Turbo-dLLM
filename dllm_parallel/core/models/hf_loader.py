# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Hugging Face loading front door for diffusion-LM models.

This module owns checkpoint/config/tokenizer/model loading. Objective selection
and distributed execution remain separate layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

from dllm_parallel.core.parallel.hf_tensor_parallel import (
    configure_hf_config_for_tensor_parallel,
    hf_from_pretrained_tensor_parallel_kwargs,
    validate_hf_tensor_parallel_model,
)
from dllm_parallel.core.models.registry import summarize_config
from dllm_parallel.core.models.contracts import ModelFamilySpec


ModelAutoClass = Literal["causal_lm", "model", "block_diffusion"]


@dataclass(frozen=True)
class HuggingFaceModelBundle:
    model_id: str
    config: Any
    config_dict: dict[str, Any]
    spec: ModelFamilySpec
    tokenizer: Any | None = None
    model: Any | None = None


def summarize_hf_config(
    model_id: str,
    config: Any,
) -> ModelFamilySpec:
    """Summarize a HF config object or plain config dictionary."""

    return summarize_config(model_id, _config_to_dict(config))


def load_hf_config(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
    family: str | None = None,
    **kwargs: Any,
) -> HuggingFaceModelBundle:
    """Load and summarize only the Hugging Face config."""

    transformers = _import_transformers()
    config = transformers.AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )
    config_dict = _config_to_dict(config)
    spec = (
        summarize_config(model_id, config_dict)
        if family is None
        else _summarize_for_family(family, model_id=model_id, config=config)
    )
    return HuggingFaceModelBundle(
        model_id=model_id,
        config=config,
        config_dict=config_dict,
        spec=spec,
    )


def load_hf_model_bundle(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
    family: str | None = None,
    load_tokenizer: bool = True,
    load_model: bool = False,
    model_auto_class: ModelAutoClass = "causal_lm",
    parallel_runtime: Any | None = None,
    torch_dtype: Any | None = None,
    device: torch.device | str | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
) -> HuggingFaceModelBundle:
    """Load HF components and return normalized planning metadata.

    The default is config+tokenizer only. Weight loading is explicit because
    planning and capability checks should not require model weights.
    """

    transformers = _import_transformers()
    config_bundle = load_hf_config(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
        family=family,
        **(config_kwargs or {}),
    )

    tokenizer = None
    if load_tokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **(tokenizer_kwargs or {}),
        )

    model = None
    if load_model:
        model = load_hf_model_from_config(
            model_id,
            config=config_bundle.config,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_auto_class=model_auto_class,
            parallel_runtime=parallel_runtime,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs,
        )

    return HuggingFaceModelBundle(
        model_id=model_id,
        config=config_bundle.config,
        config_dict=config_bundle.config_dict,
        spec=config_bundle.spec,
        tokenizer=tokenizer,
        model=model,
    )


def load_hf_model_from_config(
    model_id: str,
    *,
    config: Any,
    trust_remote_code: bool = False,
    revision: str | None = None,
    model_auto_class: ModelAutoClass = "causal_lm",
    parallel_runtime: Any | None = None,
    torch_dtype: Any | None = None,
    device: torch.device | str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load HF weights from an already prepared config.

    This is the shared weight-loading path for packaged trainers.  It keeps the
    HF ``tp_plan``/``DeviceMesh`` contract in one place while leaving objective
    execution to the caller.
    """

    transformers = _import_transformers()
    torch = _import_torch()
    if parallel_runtime is not None:
        configure_hf_config_for_tensor_parallel(config, parallel_runtime)
    if model_auto_class == "causal_lm":
        auto_cls = transformers.AutoModelForCausalLM
    elif model_auto_class == "model":
        auto_cls = transformers.AutoModel
    elif model_auto_class == "block_diffusion":
        auto_cls = _block_diffusion_model_class(transformers, config)
    else:
        raise ValueError(f"unsupported HF auto class: {model_auto_class!r}")
    kwargs = dict(model_kwargs or {})
    if torch_dtype is not None:
        kwargs.setdefault("torch_dtype", torch_dtype)
    if parallel_runtime is not None:
        device_type = (
            torch.device(device).type
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        kwargs.update(
            hf_from_pretrained_tensor_parallel_kwargs(
                parallel_runtime,
                device_type=device_type,
            )
        )
    if (
        device is not None
        and torch.device(device).type == "cuda"
        and kwargs.get("tp_plan") is None
        and kwargs.get("device_mesh") is None
    ):
        # Avoid a full host-memory model copy per non-TP worker.
        kwargs.setdefault("device_map", {"": torch.device(device)})
    model = auto_cls.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )
    if parallel_runtime is not None:
        validate_hf_tensor_parallel_model(model, parallel_runtime)
    if device is not None and not kwargs.get("device_mesh"):
        model.to(device)
    return model


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        if isinstance(value, dict):
            return value
    raise TypeError("config must be a dict or expose to_dict()")


def _summarize_for_family(family: str, *, model_id: str, config: Any) -> ModelFamilySpec:
    from dllm_parallel.core.models.registry import executor_for_family

    return executor_for_family(str(family)).metadata(config, model_id=model_id)


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Hugging Face loading requires the optional transformers package"
        ) from exc
    return transformers


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Hugging Face model loading requires the optional torch package"
        ) from exc
    return torch


def _block_diffusion_model_class(transformers: Any, config: Any) -> Any:
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if "DiffusionGemmaForBlockDiffusion" in architectures or getattr(
        config,
        "model_type",
        None,
    ) == "diffusion_gemma":
        module = transformers.models.diffusion_gemma.modeling_diffusion_gemma
        return module.DiffusionGemmaForBlockDiffusion
    raise ValueError(
        "block_diffusion HF loading requires a config architecture with a "
        "packaged block-diffusion model class"
    )
