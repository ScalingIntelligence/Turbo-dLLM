from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dllm_parallel.core.models.backbones.causal_lm import build_executor
from dllm_parallel.core.models.backbones.nemotron.model import (
    _nemotronlabsdiffusion_components,
)
from dllm_parallel.training.run_spec import RunSpec


def _spec(**topology):
    return RunSpec.from_mapping(
        {
            "model": {
                "id": "Qwen/Qwen3-8B",
                "family": "causal_lm",
                "seq_len": 1024,
                "mask_token_id": 151665,
            },
            "objective": {
                "name": "fast_dllm_v2",
                "block_size": 32,
                "noise_schedule": "linear_mask",
                "loss_weighting": "unit",
            },
            "topology": {
                "context_parallel_size": 1,
                "block_parallel_size": 1,
                **topology,
            },
        }
    )


def _qwen_config():
    return {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
    }


def test_standard_softmax_causal_lm_passes_preflight() -> None:
    executor = build_executor()
    executor.validate_run_spec(_spec(), config=_qwen_config())
    metadata = executor.metadata(_qwen_config(), model_id="Qwen/Qwen3-8B")

    assert metadata.family == "causal_lm"
    assert metadata.num_layers == 36
    assert metadata.num_key_value_heads == 8


def test_hybrid_sequence_mixer_fails_before_model_loading() -> None:
    config = _qwen_config()
    config["architectures"] = ["QwenHybridForCausalLM"]
    config["layer_types"] = ["linear_attention", "full_attention"]

    with pytest.raises(ValueError, match="unsupported sequence mixers"):
        build_executor().validate_run_spec(_spec(), config=config)


def test_latent_attention_fails_before_model_loading() -> None:
    config = _qwen_config()
    config["architectures"] = ["DeepseekForCausalLM"]
    config["kv_lora_rank"] = 512

    with pytest.raises(ValueError, match="latent or compressed-sparse attention"):
        build_executor().validate_run_spec(_spec(), config=config)


def test_generic_moe_requires_family_specific_executor() -> None:
    config = _qwen_config()
    config["num_experts"] = 64

    with pytest.raises(ValueError, match="does not yet support MoE"):
        build_executor().validate_run_spec(_spec(), config=config)


def test_qwen3_decoder_contract_is_extracted_without_remote_model_code() -> None:
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    model = Qwen3ForCausalLM(config)
    components = _nemotronlabsdiffusion_components(model)

    assert components.encoder is model.model
    assert len(components.layers) == 1
    assert components.output_head is model.lm_head
    assert components.rotary_emb_accepts_layer_type is False
    assert components.layers[0].attention.head_dim == 16
    assert isinstance(components.output_head.weight, torch.Tensor)
