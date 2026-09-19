# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DFlash checkpoint loading and production execution contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import hashlib
import shutil
import tempfile
from typing import Any

import torch

from dllm_parallel.core.models.contracts import (
    BackboneCapabilities,
    BackboneKernelPolicy,
    ModelFamilySpec,
)
from dllm_parallel.core.models.compatibility import validate_objective_for_family
from dllm_parallel.core.models.backbones.dflash.model import (
    DFlashModel,
    DFlashModelConfig,
    dflash_block_size,
    dflash_target_layer_ids,
)


FAMILY = "dflash"


class DFlashBackboneExecutor:
    family = FAMILY

    def load_config_from_run_spec(self, spec: Any) -> dict[str, Any]:
        config = _load_json_file(
            spec.model.id,
            "config.json",
            revision=spec.model.revision,
        )
        checkpoint_block_size = dflash_block_size(config)
        requested_block_size = int(spec.objective.block_size)
        if checkpoint_block_size != requested_block_size:
            raise ValueError(
                "DFlash checkpoint and objective block sizes differ: "
                f"{checkpoint_block_size} != {requested_block_size}"
            )
        checkpoint_layers = dflash_target_layer_ids(config)
        requested_layers = tuple(int(layer) for layer in spec.model.target_layer_ids)
        if checkpoint_layers != requested_layers:
            raise ValueError(
                "DFlash checkpoint and requested target layers differ: "
                f"{checkpoint_layers} != {requested_layers}"
            )
        checkpoint_verifier = _checkpoint_verifier_id(config)
        requested_verifier = str(spec.model.verifier_id)
        if checkpoint_verifier is not None and checkpoint_verifier != requested_verifier:
            raise ValueError(
                "DFlash checkpoint and requested verifier differ: "
                f"{checkpoint_verifier!r} != {requested_verifier!r}"
            )
        checkpoint_sample_from_anchor = _sample_from_anchor_contract(config)
        requested_sample_from_anchor = bool(spec.objective.sample_from_anchor)
        if checkpoint_sample_from_anchor != requested_sample_from_anchor:
            raise ValueError(
                "DFlash checkpoint and objective sample_from_anchor contracts differ: "
                f"{checkpoint_sample_from_anchor} != {requested_sample_from_anchor}"
            )
        config["sample_from_anchor"] = bool(spec.objective.sample_from_anchor)
        config["verifier_id"] = str(spec.model.verifier_id)
        config["verifier_revision"] = spec.model.verifier_revision
        config["draft_vocab_path"] = spec.model.draft_vocab_path
        # The shared input pipeline operates in verifier-token space. Reduced
        # draft vocabularies remain private to the DFlash model and loss.
        model_config = DFlashModelConfig.from_mapping(config)
        config["vocab_size"] = model_config.verifier_vocab_size
        config["mask_token_id"] = model_config.mask_token_id
        return config

    def metadata(self, config: Any, *, model_id: str) -> ModelFamilySpec:
        values = _config_mapping(config)
        model_config = DFlashModelConfig.from_mapping(values)
        return ModelFamilySpec(
            model_id=model_id,
            family=FAMILY,
            model_type=FAMILY,
            architecture=model_config.architecture,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            vocab_size=model_config.verifier_vocab_size,
            intermediate_size=model_config.intermediate_size,
            block_size=model_config.block_size,
            mask_token_id=model_config.mask_token_id,
            sliding_window=model_config.sliding_window,
            target_hidden_size=model_config.target_hidden_size,
            target_feature_width=(
                len(model_config.target_layer_ids) * model_config.target_hidden_size
            ),
            draft_vocab_size=model_config.draft_vocab_size,
            attention_layer_types=model_config.layer_types,
            sliding_window_non_causal=model_config.sliding_window_non_causal,
            dflash2_conv_kernel_size=model_config.conv_kernel_size,
            dflash2_conv_group_size=model_config.conv_group_size,
            dflash2_selector_rank=model_config.selector_rank,
            dflash2_selector_top_k=model_config.selector_top_k,
            attn_implementation="dflash_flex_fa4",
        )

    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(
            family=FAMILY,
            packed_block_diffusion=True,
            tensor_parallel=False,
            sequence_parallel=False,
            checkpoint_hooks=True,
            sharded_state_dict=True,
            tokenizer_required_for_text=True,
            tokenizer_required_for_dataset=False,
            uses_transformers=False,
        )

    def prepare_tokenizer_and_config(
        self,
        spec: Any,
        *,
        config: Any,
        tokenizer: Any | None,
    ) -> None:
        """DFlash uses the verifier and draft metadata already in its config."""

        del spec, config, tokenizer

    def validate_run_spec(self, spec: Any, *, config: Any | None = None) -> None:
        validate_objective_for_family(
            family=self.family,
            objective=str(getattr(getattr(spec, "objective", None), "name", "")),
        )
        if spec.adapter.type != "none":
            raise ValueError("DFlash verifier training does not support LoRA")
        if spec.model.family != FAMILY:
            raise ValueError("DFlash executor requires model.family=dflash")
        if int(spec.topology.expert_parallel_size) != 1:
            raise ValueError("DFlash is dense and does not support expert parallelism")
        if int(spec.topology.tensor_parallel_size) != 1:
            raise ValueError("DFlash tensor parallelism is not yet implemented")
        if bool(spec.topology.sequence_parallel):
            raise ValueError("DFlash sequence parallelism is not yet implemented")
        if spec.kernel.cp_bp_attention_policy != "production":
            raise ValueError("DFlash CP/BP requires the production attention policy")
        if config is not None:
            from dllm_parallel.core.attention.dflash_attention import (
                validate_dflash_attention_head_dim,
            )

            model_config = DFlashModelConfig.from_mapping(_config_mapping(config))
            validate_dflash_attention_head_dim(model_config.head_dim)
            if model_config.block_size != int(spec.objective.block_size):
                raise ValueError("DFlash checkpoint and objective block sizes differ")
            if model_config.target_layer_ids != tuple(spec.model.target_layer_ids):
                raise ValueError("DFlash checkpoint and requested target layers differ")
            if spec.objective.dflash_loss == "paper_ce" and (
                model_config.draft_vocab_size != model_config.verifier_vocab_size
            ):
                raise ValueError("paper_ce requires the verifier's full vocabulary")
            if spec.objective.dflash_loss in {
                "dflash",
                "dpace",
                "dpace-cumulative-confidence-only",
                "dpace-continuation-value-only",
            } and model_config.architecture != "DFlash2DraftModel":
                raise ValueError(
                    "SpecForge hard-label objectives require DFlash2DraftModel"
                )
            if model_config.architecture == "DFlash2DraftModel" and (
                spec.objective.dflash_loss in {"paper_ce", "speculators_kl"}
            ):
                raise ValueError(
                    "DFlash2DraftModel requires a SpecForge hard-label objective"
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
    ) -> DFlashModel:
        del trust_remote_code
        from dllm_parallel.core.attention.dflash_attention import (
            dflash_attention,
            validate_dflash_attention_head_dim,
        )

        model_config = DFlashModelConfig.from_mapping(_config_mapping(config))
        validate_dflash_attention_head_dim(model_config.head_dim)
        if model_config.architecture == "DFlash2DraftModel":
            from dllm_parallel.core.models.backbones.dflash.dflash2 import DFlash2Model

            model = DFlash2Model(model_config, attention_op=dflash_attention)
        else:
            model = DFlashModel(model_config, attention_op=dflash_attention)
        draft_state = _load_safetensor_state(model_id, revision=revision)
        missing, unexpected = model.load_state_dict(
            draft_state,
            strict=False,
            assign=True,
        )
        if unexpected:
            raise RuntimeError(f"DFlash checkpoint has unexpected tensors: {unexpected}")
        _load_vocabulary_mapping(model, config.get("draft_vocab_path"))
        verifier_names = _verifier_weight_aliases()
        unresolved = set(missing)
        if unresolved.intersection(verifier_names):
            verifier_state = _load_selected_tensors(
                str(_required_value(config, "verifier_id")),
                {name for aliases in verifier_names.values() for name in aliases},
                revision=config.get("verifier_revision"),
            )
            replacement: dict[str, torch.Tensor] = {}
            for destination, aliases in verifier_names.items():
                if destination not in unresolved:
                    continue
                source = next((verifier_state[name] for name in aliases if name in verifier_state), None)
                if source is None:
                    raise RuntimeError(f"verifier checkpoint is missing {aliases}")
                replacement[destination] = _select_draft_vocabulary(
                    source,
                    model=model,
                    destination=destination,
                )
            _, rejected = model.load_state_dict(
                replacement,
                strict=False,
                assign=True,
            )
            if rejected:
                raise RuntimeError(f"verifier checkpoint has incompatible tensors: {rejected}")
            unresolved.difference_update(replacement)
        unresolved.difference_update({"t2d", "d2t"})
        if unresolved:
            raise RuntimeError(f"DFlash checkpoint is missing trainable tensors: {sorted(unresolved)}")
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"DFlash checkpoint tensor {name!r} is nonfinite")
        model.to(device=device, dtype=dtype)
        model.runtime = runtime
        model.speculators_config = _deployment_config(_config_mapping(config))
        return model

    def build_training_model(self, model: DFlashModel, *, runtime: Any, spec: Any) -> DFlashModel:
        model.runtime = runtime
        model.activation_checkpointing = bool(spec.training.activation_checkpointing)
        candidate_selector = getattr(model, "candidate_selector", None)
        if isinstance(candidate_selector, torch.nn.Module):
            enabled = float(spec.objective.selector_loss_alpha) > 0.0
            model.selector_objective_enabled = enabled
            if not enabled:
                candidate_selector.requires_grad_(False)
        return model

    def fsdp_modules(self, model: Any) -> tuple[Any, ...]:
        return tuple(
            layer
            for layer in getattr(model, "layers", ())
            if isinstance(layer, torch.nn.Module)
        )

    def build_training_task(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.models.backbones.dflash.training import (
            build_training_task,
        )

        return build_training_task(**kwargs)

    def build_data_runtime(self, **kwargs: Any) -> Any:
        from dllm_parallel.core.models.backbones.dflash.data import (
            DFlashFeatureDataRuntime,
            DFlashSyntheticFeatureDataRuntime,
        )
        from dllm_parallel.core.parallel.runtime import data_parallel_coordinates

        spec = kwargs["spec"]
        data_parallel_rank, data_parallel_size = data_parallel_coordinates(
            kwargs.get("runtime"),
            rank=int(kwargs.get("rank", 0)),
            world_size=int(kwargs.get("world_size", 1)),
            distributed_data_parallel=True,
        )
        if spec.data.target_features == "synthetic":
            model_config = DFlashModelConfig.from_mapping(
                self.load_config_from_run_spec(spec)
            )
            return DFlashSyntheticFeatureDataRuntime(
                batch_size=int(spec.training.batch_size),
                seq_len=int(spec.model.seq_len),
                target_feature_width=(
                    len(model_config.target_layer_ids)
                    * model_config.target_hidden_size
                ),
                verifier_hidden_size=model_config.target_hidden_size,
                vocab_size=model_config.verifier_vocab_size,
                device=kwargs["device"],
                dtype=kwargs["dtype"],
                seed=int(kwargs["seed"]),
                runtime=kwargs.get("runtime"),
                data_parallel_rank=data_parallel_rank,
                data_parallel_size=data_parallel_size,
                require_teacher_features=(
                    spec.objective.dflash_loss == "speculators_kl"
                ),
            )
        feature_path = spec.data.target_feature_path
        if feature_path is None:
            raise ValueError("offline DFlash requires data.target_feature_path")
        return DFlashFeatureDataRuntime.from_path(
            path=str(feature_path),
            batch_size=int(spec.training.batch_size),
            seq_len=int(spec.model.seq_len),
            device=kwargs["device"],
            dtype=kwargs["dtype"],
            require_teacher_features=(spec.objective.dflash_loss == "speculators_kl"),
            verifier_id=str(spec.model.verifier_id),
            verifier_revision=spec.model.verifier_revision,
            target_layer_ids=tuple(spec.model.target_layer_ids),
            runtime=kwargs.get("runtime"),
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
        )

    def verify_native_kernels(self, spec: Any) -> dict[str, Any]:
        from dllm_parallel.core.attention.dflash_fa4 import (
            verify_dflash_fa4_runtime,
        )
        from dllm_parallel.core.attention.flex import verify_flex_attention_runtime
        from dllm_parallel.core.kernels import (
            chunked_linear_ce_native,
            dflash_cp_fusion,
        )

        metadata = verify_flex_attention_runtime().to_log_dict()
        metadata["backend"] = "dflash_flex_fa4"
        metadata["local_block_kernel"] = verify_dflash_fa4_runtime()
        dflash_cp_fusion.verify()
        chunked_linear_ce_native.verify_dflash_symbols(
            loss_kind=str(spec.objective.dflash_loss)
        )
        return metadata

    def parallel_work_units(self, spec: Any) -> int:
        max_anchors = spec.objective.max_anchors
        if max_anchors is None:
            raise ValueError("DFlash training requires objective.max_anchors")
        return int(max_anchors)

    def tokenizer_model_id(self, spec: Any) -> str:
        verifier_id = spec.model.verifier_id
        if verifier_id is None:
            raise ValueError("DFlash training requires model.verifier_id")
        return str(verifier_id)

    def build_packed_block_diffusion_model(self, model: Any, **kwargs: Any) -> Any:
        del kwargs
        return model

    def build_objective_schedule(self, spec: Any, *, sequence_length: int | None = None) -> None:
        del spec, sequence_length
        return None

    def kernel_policy(self, spec: Any) -> BackboneKernelPolicy:
        del spec
        return BackboneKernelPolicy()

    def validate_tokenizer_data_compatibility(self, spec: Any, tokenizer: Any | None) -> None:
        if spec.data.input_mode == "text" and tokenizer is None:
            raise ValueError("DFlash text input requires the verifier tokenizer")

    def load_checkpoint_hooks(self, checkpoint: Any, model: Any) -> None:
        del model
        family = ((checkpoint or {}).get("backbone_state") or {}).get("family")
        if family is not None and family != FAMILY:
            raise RuntimeError(f"cannot restore {family!r} state into DFlash")

    def save_checkpoint_hooks(self, model: Any) -> dict[str, Any]:
        return {"family": FAMILY, "model_class": model.__class__.__name__, "checkpoint_hooks_version": 1}

    def sharded_state_dict(self, model: Any) -> dict[str, Any]:
        return {
            "format": "dllm_parallel.backbone_state_dict.v1",
            "family": FAMILY,
            "state_dict": model.state_dict(),
        }

    def migrate_config(self, raw_config: Any) -> Any:
        return raw_config


def build_executor() -> DFlashBackboneExecutor:
    return DFlashBackboneExecutor()


def export_speculators_checkpoint(
    model: DFlashModel,
    output_dir: str | Path,
) -> Path:
    """Write a deployable Hugging Face Speculators DFlash checkpoint."""

    if not isinstance(model, DFlashModel):
        raise TypeError("DFlash export requires an unwrapped DFlashModel")
    config = getattr(model, "speculators_config", None)
    if not isinstance(config, dict):
        raise RuntimeError("DFlash model does not retain its checkpoint config")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if name not in (
            {
                "embed_tokens.weight",
                "lm_head.weight",
                "verifier_lm_head.weight",
                "verifier_norm.weight",
                "t2d",
                "d2t",
            }
            if model.config.architecture == "DFlash2DraftModel"
            else {"verifier_lm_head.weight", "verifier_norm.weight"}
        )
    }
    from safetensors.torch import save_file

    weights_path = destination / "model.safetensors"
    temporary_weights = destination / ".model.safetensors.tmp"
    save_file(state, str(temporary_weights))
    os.replace(temporary_weights, weights_path)
    config_path = destination / "config.json"
    temporary_config = destination / ".config.json.tmp"
    temporary_config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_config, config_path)
    return destination


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sglang_dflash2_config(
    config: dict[str, Any],
    *,
    expected_block_size: int | None = None,
) -> dict[str, Any]:
    """Return a standalone config accepted by SGLang's DFLASH loader."""

    normalized = json.loads(json.dumps(config))
    model_type = normalized.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("SGLang DFlash2 requires a nonempty model_type")
    if normalized.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("SGLang DFlash2 requires architectures=['DFlash2DraftModel']")
    method = normalized.get("dflash_config")
    if not isinstance(method, dict):
        raise ValueError("SGLang DFlash2 requires a dflash_config mapping")
    block_size = normalized.get("block_size", method.get("block_size"))
    if expected_block_size is not None and block_size != int(expected_block_size):
        raise ValueError(
            f"exported block_size={block_size!r}, expected {int(expected_block_size)}"
        )
    for name in (
        "block_size",
        "conv_group_size",
        "conv_kernel_size",
        "selector_rank",
        "selector_top_k",
    ):
        value = method.get(name) if name != "block_size" else block_size
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"DFlash2 requires positive integer {name}, got {value!r}")
    attention_mode = str(method.get("attention_mode", "gqa")).lower()
    if attention_mode not in {"gqa", "mha"}:
        raise ValueError(
            "SGLang DFlash2 serving supports GQA/MHA only, "
            f"got attention_mode={attention_mode!r}"
        )
    method["attention_mode"] = attention_mode
    normalized.pop("auto_map", None)
    rope = normalized.get("rope_parameters")
    if isinstance(rope, dict):
        if "rope_theta" in rope:
            normalized["rope_theta"] = rope["rope_theta"]
        rope_type = rope.get("rope_type", rope.get("type"))
        if rope_type not in {None, "default"} and not normalized.get("rope_scaling"):
            normalized["rope_scaling"] = {
                key: value for key, value in rope.items() if key != "rope_theta"
            }
    return normalized


def validate_sglang_dflash2_export(
    export_dir: str | Path,
    *,
    expected_block_size: int | None = None,
) -> dict[str, Any]:
    """Validate config, provenance, hash, and serving tensor inventory."""

    root = Path(export_dir).expanduser().resolve()
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    provenance_path = root / "training_export.json"
    for path in (config_path, weights_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    normalized = normalize_sglang_dflash2_config(
        json.loads(config_path.read_text(encoding="utf-8")),
        expected_block_size=expected_block_size,
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("format") != "dllm_parallel.dflash2_training_export.v1":
        raise RuntimeError("unsupported DFlash2 training export provenance")
    weights = provenance.get("weights") or {}
    if weights.get("sha256") != _file_sha256(weights_path):
        raise RuntimeError("DFlash2 export weights SHA-256 mismatch")
    from safetensors import safe_open

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        observed_shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in names}
    if int(weights.get("tensor_count", -1)) != len(names):
        raise RuntimeError("DFlash2 export tensor count mismatch")
    forbidden = {
        "embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight",
        "verifier_norm.weight", "t2d", "d2t",
    }
    if names & forbidden:
        raise RuntimeError("DFlash2 export contains verifier-only tensors")
    expected_shapes = _expected_dflash2_serving_shapes(normalized)
    missing = sorted(expected_shapes.keys() - names)
    extra = sorted(names - expected_shapes.keys())
    wrong_shapes = sorted(
        name for name in names & expected_shapes.keys()
        if observed_shapes[name] != expected_shapes[name]
    )
    if missing or extra or wrong_shapes:
        raise RuntimeError(
            "DFlash2 export serving tensor inventory mismatch: "
            f"missing={missing[:3]} extra={extra[:3]} wrong_shapes={wrong_shapes[:3]}"
        )
    return {"path": str(root), "tensor_count": len(names), "config": "DFlash2DraftModel"}


def _expected_dflash2_serving_shapes(
    config: dict[str, Any],
) -> dict[str, tuple[int, ...]]:
    model = DFlashModelConfig.from_mapping(config)
    hidden = model.hidden_size
    intermediate = model.intermediate_size
    q_width = model.num_attention_heads * model.head_dim
    kv_width = model.num_key_value_heads * model.head_dim
    assert model.conv_kernel_size is not None
    assert model.conv_group_size is not None
    assert model.selector_rank is not None
    groups = hidden // model.conv_group_size
    shapes: dict[str, tuple[int, ...]] = {
        "fc.weight": (hidden, len(model.target_layer_ids) * model.target_hidden_size),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
        "candidate_selector.predecessor_codebook": (
            model.verifier_vocab_size, model.selector_rank
        ),
        "candidate_selector.successor_codebook": (
            model.verifier_vocab_size, model.selector_rank
        ),
        "candidate_selector.hidden_projection.weight": (model.selector_rank, hidden),
    }
    for index in range(model.num_hidden_layers):
        prefix = f"layers.{index}."
        shapes.update({
            prefix + "self_attn.q_proj.weight": (q_width, hidden),
            prefix + "self_attn.k_proj.weight": (kv_width, hidden),
            prefix + "self_attn.v_proj.weight": (kv_width, hidden),
            prefix + "self_attn.o_proj.weight": (hidden, q_width),
            prefix + "self_attn.q_norm.weight": (model.head_dim,),
            prefix + "self_attn.k_norm.weight": (model.head_dim,),
            prefix + "mlp.gate_proj.weight": (intermediate, hidden),
            prefix + "mlp.up_proj.weight": (intermediate, hidden),
            prefix + "mlp.down_proj.weight": (hidden, intermediate),
            prefix + "input_layernorm.weight": (hidden,),
            prefix + "post_attention_layernorm.weight": (hidden,),
        })
        for name in ("attention_conv", "mlp_conv"):
            shapes[prefix + name + ".base_kernel"] = (
                2, model.conv_kernel_size, hidden
            )
            shapes[prefix + name + ".kernel_projection.weight"] = (
                2 * model.conv_kernel_size * groups, hidden
            )
        if model.attention_bias:
            shapes.update({
                prefix + "self_attn.q_proj.bias": (q_width,),
                prefix + "self_attn.k_proj.bias": (kv_width,),
                prefix + "self_attn.v_proj.bias": (kv_width,),
                prefix + "self_attn.o_proj.bias": (hidden,),
            })
        if model.mlp_bias:
            shapes.update({
                prefix + "mlp.gate_proj.bias": (intermediate,),
                prefix + "mlp.up_proj.bias": (intermediate,),
                prefix + "mlp.down_proj.bias": (hidden,),
            })
    return shapes


def export_speculators_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    base_model_dir: str | Path,
    output_dir: str | Path,
    checkpoint_tag: str = "latest",
    expected_model_id: str,
    expected_model_revision: str,
    expected_block_size: int | None = None,
) -> Path:
    """Export one replicated native training shard as a serving checkpoint.

    The optimized DFlash CP/BP trainer keeps a replicated model on every rank;
    only data and activations are partitioned.  This exporter deliberately
    rejects FSDP/ZeRO/partial state so rank zero can never be mistaken for a
    complete deployment checkpoint.
    """

    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    metadata_path = checkpoint_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    from dllm_parallel.core.checkpoint.io import CHECKPOINT_FORMAT_VERSION

    if metadata.get("format") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError("unsupported DFlash training checkpoint format")
    if metadata.get("checkpoint_backend") != "rank_local":
        raise RuntimeError("DFlash deployment export requires replicated rank state")
    if metadata.get("model_state_scope") != "full_training_state":
        raise RuntimeError("DFlash deployment export requires full training state")
    world_size = metadata.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise RuntimeError("DFlash checkpoint world size is invalid")
    tag = metadata.get("latest_tag") if checkpoint_tag == "latest" else checkpoint_tag
    if not isinstance(tag, str) or not tag:
        raise RuntimeError("DFlash checkpoint tag is invalid")
    expected_step_tag = f"step_{int(metadata.get('step', -1)):08d}"
    if checkpoint_tag == "latest" and tag != expected_step_tag:
        raise RuntimeError("DFlash latest checkpoint tag and step disagree")
    rank_pattern = metadata.get("rank_state_pattern")
    if not isinstance(rank_pattern, str) or "{rank:05d}" not in rank_pattern:
        raise RuntimeError("DFlash checkpoint rank-state pattern is invalid")
    rank_filename = rank_pattern.format(rank=0)
    rank_path = (checkpoint_root / tag / rank_filename).resolve()
    if checkpoint_root not in rank_path.parents or not rank_path.is_file():
        raise FileNotFoundError(rank_path)
    config = metadata.get("config")
    model_spec = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model_spec, dict) or (
        model_spec.get("id") != str(expected_model_id)
        or model_spec.get("revision") != str(expected_model_revision)
    ):
        raise RuntimeError("DFlash training checkpoint model identity mismatch")

    base_root = Path(base_model_dir).expanduser().resolve()
    base_config_path = base_root / "config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(base_config_path)
    deployment_config = normalize_sglang_dflash2_config(
        json.loads(base_config_path.read_text(encoding="utf-8")),
        expected_block_size=expected_block_size,
    )

    checkpoint = torch.load(
        rank_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_backend") != "rank_local":
        raise RuntimeError("DFlash rank checkpoint backend is inconsistent")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError("DFlash rank checkpoint has no model state")
    excluded = {
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
        "t2d",
        "d2t",
    }
    exported: dict[str, torch.Tensor] = {}
    for name, value in model_state.items():
        if name in excluded:
            continue
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise RuntimeError("DFlash model state contains a non-tensor entry")
        tensor = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"DFlash trained tensor {name!r} is non-finite")
        exported[name] = tensor
    if not exported or not any(name.startswith("layers.") for name in exported):
        raise RuntimeError("DFlash exported state is incomplete")
    if not any(name.startswith("candidate_selector.") for name in exported):
        raise RuntimeError("DFlash2 exported state is missing its candidate selector")

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        from safetensors.torch import save_file

        weights_path = staging / "model.safetensors"
        save_file(exported, str(weights_path))
        (staging / "config.json").write_text(
            json.dumps(deployment_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provenance = {
            "format": "dllm_parallel.dflash2_training_export.v1",
            "model_id": str(expected_model_id),
            "model_revision": str(expected_model_revision),
            "checkpoint_tag": tag,
            "checkpoint_step": int(metadata["step"]),
            "checkpoint_world_size": world_size,
            "checkpoint_metadata_sha256": _file_sha256(metadata_path),
            "checkpoint_rank_zero": {
                "path": rank_path.name,
                "bytes": rank_path.stat().st_size,
                "sha256": _file_sha256(rank_path),
            },
            "weights": {
                "path": weights_path.name,
                "bytes": weights_path.stat().st_size,
                "sha256": _file_sha256(weights_path),
                "tensor_count": len(exported),
            },
        }
        (staging / "training_export.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_sglang_dflash2_export(
            staging, expected_block_size=expected_block_size
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _config_mapping(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError("DFlash config must be a mapping")


def _deployment_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in config.items()
        if name not in {"draft_vocab_path", "verifier_id", "verifier_revision"}
    }


def _required_value(config: Any, name: str) -> Any:
    value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
    if value is None:
        raise ValueError(f"DFlash config requires {name}")
    return value


def _sample_from_anchor_contract(config: dict[str, Any]) -> bool:
    explicit = config.get("sample_from_anchor")
    if explicit is not None:
        return bool(explicit)
    dflash_config = config.get("dflash_config")
    if isinstance(dflash_config, dict):
        nested_explicit = dflash_config.get("sample_from_anchor")
        if nested_explicit is not None:
            return bool(nested_explicit)
    speculators_config = config.get("speculators_config")
    if not isinstance(speculators_config, dict):
        if isinstance(dflash_config, dict) and dflash_config.get("target_layer_ids"):
            # Native z-lab DFlash uses the anchor as a bonus token and trains
            # block_size - 1 prediction slots. This is the source schema's
            # documented contract, not a model-name-specific default.
            return False
        raise ValueError("DFlash checkpoint does not declare its slot contract")
    block_size = dflash_block_size(config)
    default_method = speculators_config.get("default_proposal_method")
    proposal_methods = speculators_config.get("proposal_methods")
    if not isinstance(proposal_methods, list):
        raise ValueError("DFlash checkpoint does not declare proposal methods")
    for proposal in proposal_methods:
        if not isinstance(proposal, dict):
            continue
        proposal_type = proposal.get("proposal_type")
        if default_method is not None and proposal_type != default_method:
            continue
        speculative_tokens = proposal.get("speculative_tokens")
        if speculative_tokens is None:
            continue
        speculative_tokens = int(speculative_tokens)
        if speculative_tokens == block_size:
            return True
        if speculative_tokens == block_size - 1:
            return False
        raise ValueError(
            "DFlash proposal contract is incompatible with checkpoint block_size"
        )
    raise ValueError("DFlash checkpoint does not declare its default proposal shape")


def _checkpoint_verifier_id(config: dict[str, Any]) -> str | None:
    speculators_config = config.get("speculators_config")
    if not isinstance(speculators_config, dict):
        return None
    verifier = speculators_config.get("verifier")
    if not isinstance(verifier, dict):
        return None
    model_id = verifier.get("name_or_path")
    return str(model_id) if model_id is not None else None


def _verifier_weight_aliases() -> dict[str, tuple[str, ...]]:
    embeddings = (
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    output_heads = ("lm_head.weight", *embeddings)
    final_norms = (
        "model.language_model.norm.weight",
        "model.norm.weight",
        "norm.weight",
    )
    return {
        "embed_tokens.weight": embeddings,
        "lm_head.weight": output_heads,
        "verifier_lm_head.weight": output_heads,
        "verifier_norm.weight": final_norms,
    }


def _repository_file(
    model_id: str,
    filename: str,
    *,
    revision: str | None = None,
) -> Path:
    local = Path(model_id).expanduser()
    if local.is_dir():
        path = local / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=model_id, filename=filename, revision=revision)
    )


def _load_json_file(
    model_id: str,
    filename: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(
        _repository_file(model_id, filename, revision=revision).read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return payload


def _weight_files(model_id: str, *, revision: str | None = None) -> tuple[Path, ...]:
    local = Path(model_id).expanduser()
    if local.is_dir():
        index_path = local / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            return tuple(local / name for name in sorted(set(index["weight_map"].values())))
        files = tuple(sorted(local.glob("*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors checkpoint found in {local}")
        return files
    from huggingface_hub.errors import EntryNotFoundError

    try:
        index = _load_json_file(
            model_id,
            "model.safetensors.index.json",
            revision=revision,
        )
    except EntryNotFoundError:
        return (
            _repository_file(
                model_id,
                "model.safetensors",
                revision=revision,
            ),
        )
    return tuple(
        _repository_file(model_id, name, revision=revision)
        for name in sorted(set(index["weight_map"].values()))
    )


def _load_safetensor_state(
    model_id: str,
    *,
    revision: str | None = None,
) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for path in _weight_files(model_id, revision=revision):
        shard = load_file(str(path), device="cpu")
        overlap = state.keys() & shard.keys()
        if overlap:
            raise RuntimeError(f"duplicate DFlash checkpoint tensors: {sorted(overlap)}")
        state.update(shard)
    return state


def _load_selected_tensors(
    model_id: str,
    candidates: set[str],
    *,
    revision: str | None = None,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    selected: dict[str, torch.Tensor] = {}
    for path in _selected_weight_files(model_id, candidates, revision=revision):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in candidates.intersection(handle.keys()):
                selected[name] = handle.get_tensor(name)
        if candidates.issubset(selected):
            break
    return selected


def _selected_weight_files(
    model_id: str,
    candidates: set[str],
    *,
    revision: str | None = None,
) -> tuple[Path, ...]:
    local = Path(model_id).expanduser()
    if local.is_dir():
        index_path = local / "model.safetensors.index.json"
        if not index_path.is_file():
            return _weight_files(model_id, revision=revision)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = {
            index["weight_map"][name]
            for name in candidates
            if name in index["weight_map"]
        }
        return tuple(local / name for name in sorted(names))
    from huggingface_hub.errors import EntryNotFoundError

    try:
        index = _load_json_file(
            model_id,
            "model.safetensors.index.json",
            revision=revision,
        )
    except EntryNotFoundError:
        return (
            _repository_file(
                model_id,
                "model.safetensors",
                revision=revision,
            ),
        )
    names = {
        index["weight_map"][name]
        for name in candidates
        if name in index["weight_map"]
    }
    return tuple(
        _repository_file(model_id, name, revision=revision)
        for name in sorted(names)
    )


def _select_draft_vocabulary(
    tensor: torch.Tensor,
    *,
    model: DFlashModel,
    destination: str,
) -> torch.Tensor:
    if destination in {"lm_head.weight", "verifier_lm_head.weight"} and (
        model.config.draft_vocab_size != model.config.verifier_vocab_size
    ):
        if model.t2d is None or not bool(model.t2d.any()):
            raise RuntimeError("reduced DFlash vocabulary mapping was not loaded")
        return tensor.index_select(0, torch.nonzero(model.t2d, as_tuple=False).flatten())
    return tensor


def _load_vocabulary_mapping(model: DFlashModel, mapping_path: str | None) -> None:
    if model.config.draft_vocab_size == model.config.verifier_vocab_size:
        return
    if mapping_path is None:
        if model.t2d is None or model.d2t is None:
            raise RuntimeError("reduced DFlash checkpoint has no vocabulary mapping")
        selected = torch.nonzero(model.t2d, as_tuple=False).flatten()
        expected_offsets = selected - torch.arange(
            model.config.draft_vocab_size,
            device=selected.device,
            dtype=selected.dtype,
        )
        if int(selected.numel()) != model.config.draft_vocab_size or not torch.equal(
            model.d2t.cpu(),
            expected_offsets.cpu(),
        ):
            raise RuntimeError("reduced DFlash checkpoint vocabulary mapping is invalid")
        return
    from safetensors.torch import load_file

    values = load_file(str(Path(mapping_path).expanduser()), device="cpu")
    t2d = values.get("t2d")
    d2t = values.get("d2t")
    if t2d is None or d2t is None:
        raise ValueError("DFlash vocabulary mapping must contain t2d and d2t")
    model.load_vocabulary_mapping(t2d=t2d, d2t=d2t)


__all__ = [
    "DFlashBackboneExecutor",
    "build_executor",
    "export_speculators_checkpoint",
    "export_speculators_training_checkpoint",
    "normalize_sglang_dflash2_config",
    "validate_sglang_dflash2_export",
]
