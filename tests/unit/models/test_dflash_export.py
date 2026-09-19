from __future__ import annotations

import pytest
import json
import torch
from safetensors.torch import load_file, save_file
import hashlib

from dllm_parallel.core.models.backbones.dflash.executor import (
    normalize_sglang_dflash2_config,
    export_speculators_training_checkpoint,
    validate_sglang_dflash2_export,
)
from dllm_parallel.core.checkpoint.io import CHECKPOINT_FORMAT_VERSION
from dllm_parallel.core.models.backbones.dflash.dflash2 import DFlash2Model
from dllm_parallel.core.models.backbones.dflash.model import DFlashModelConfig


def _config(**method_overrides):
    method = {
        "block_size": 16,
        "conv_group_size": 4,
        "conv_kernel_size": 2,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [1, 13, 25],
        "mask_token_id": 31,
        "attention_mode": "gqa",
        **method_overrides,
    }
    return {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "dflash_config": method,
        "auto_map": {"AutoModel": "private.Model"},
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 4.0,
            "rope_theta": 1_000_000.0,
        },
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "layer_types": ["full_attention"],
        "initializer_range": 0.02,
    }


def test_sglang_config_normalization_is_standalone_and_legacy_rope_compatible():
    source = _config()
    normalized = normalize_sglang_dflash2_config(source, expected_block_size=16)

    assert "auto_map" not in normalized
    assert normalized["architectures"] == ["DFlash2DraftModel"]
    assert normalized["rope_theta"] == 1_000_000.0
    assert normalized["rope_scaling"] == {"rope_type": "yarn", "factor": 4.0}
    assert "auto_map" in source


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(attention_mode="mla"), "GQA/MHA"),
        (_config(conv_kernel_size=0), "conv_kernel_size"),
    ],
)
def test_sglang_config_normalization_rejects_unsupported_exports(config, message):
    with pytest.raises(ValueError, match=message):
        normalize_sglang_dflash2_config(config, expected_block_size=16)


def test_sglang_config_normalization_rejects_block_size_mismatch():
    with pytest.raises(ValueError, match="block_size=16.*expected 8"):
        normalize_sglang_dflash2_config(_config(), expected_block_size=8)


def test_sglang_config_normalization_is_not_qwen_specific():
    config = _config()
    config["model_type"] = "gemma3_text"
    config["transformer_layer_config"] = {"model_type": "gemma3_text"}
    assert normalize_sglang_dflash2_config(config)["model_type"] == "gemma3_text"


def test_training_checkpoint_export_is_sglang_loadable_and_excludes_verifier(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    tag = "step_00000004"
    tag_dir = checkpoint / tag
    tag_dir.mkdir(parents=True)
    model_id, revision = "org/draft", "abc123"
    metadata = {
        "format": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_backend": "rank_local",
        "model_state_scope": "full_training_state",
        "world_size": 2,
        "step": 4,
        "latest_tag": tag,
        "rank_state_pattern": "rank_{rank:05d}.pt",
        "config": {"model": {"id": model_id, "revision": revision}},
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    model_config = _config()
    model_config["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 1_000_000.0,
    }
    model = DFlash2Model(
        DFlashModelConfig.from_mapping(model_config), attention_op=lambda **kwargs: None
    )
    state = model.state_dict()
    torch.save(
        {"checkpoint_backend": "rank_local", "model": state},
        tag_dir / "rank_00000.pt",
    )
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(model_config), encoding="utf-8")

    output = export_speculators_training_checkpoint(
        checkpoint,
        base_model_dir=base,
        output_dir=tmp_path / "export",
        expected_model_id=model_id,
        expected_model_revision=revision,
        expected_block_size=16,
    )

    exported = load_file(str(output / "model.safetensors"))
    assert "layers.0.self_attn.q_proj.weight" in exported
    assert "candidate_selector.hidden_projection.weight" in exported
    assert "embed_tokens.weight" not in exported
    assert (
        json.loads((output / "training_export.json").read_text())["checkpoint_step"]
        == 4
    )
    validated = validate_sglang_dflash2_export(output, expected_block_size=16)
    assert validated["tensor_count"] == len(exported)

    incomplete = dict(exported)
    incomplete.pop("layers.0.self_attn.k_proj.weight")
    save_file(incomplete, str(output / "model.safetensors"))
    provenance = json.loads((output / "training_export.json").read_text())
    provenance["weights"]["tensor_count"] = len(incomplete)
    provenance["weights"]["sha256"] = hashlib.sha256(
        (output / "model.safetensors").read_bytes()
    ).hexdigest()
    (output / "training_export.json").write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="inventory mismatch.*k_proj"):
        validate_sglang_dflash2_export(output, expected_block_size=16)

    (output / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256"):
        validate_sglang_dflash2_export(output, expected_block_size=16)
