from __future__ import annotations

from types import SimpleNamespace

import pytest

from dllm_parallel.core.models.backbones.qwen3_8.executor import (
    Qwen38BackboneExecutor,
    summarize_config,
)
from dllm_parallel.core.models.backbones.qwen3_8.model import (
    _unexpected_text_checkpoint_keys,
)


class _Tokenizer:
    def __init__(self, tokens: tuple[str, ...] = ("a", "b", "c")) -> None:
        self._tokens = {token: index for index, token in enumerate(tokens)}
        self.mask_token_id: int | None = None

    def __len__(self) -> int:
        return len(self._tokens)

    def add_special_tokens(self, values: dict[str, str]) -> int:
        token = values["mask_token"]
        added = int(token not in self._tokens)
        if added:
            self._tokens[token] = len(self._tokens)
        self.mask_token_id = self._tokens[token]
        return added


def _spec(*, mask_token_id: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(mask_token="<|mask|>", mask_token_id=mask_token_id)
    )


def _config() -> dict[str, object]:
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 8,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    }


def test_qwen38_metadata_exposes_hybrid_mfu_architecture() -> None:
    config = _config()
    text = config["text_config"]
    assert isinstance(text, dict)
    text.update(
        {
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 4,
        }
    )
    metadata = summarize_config("Qwen/Qwen3.8-27B", config)

    assert metadata.attention_layer_types == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
    assert metadata.attention_output_gate is True
    assert metadata.linear_key_head_dim == 16
    assert metadata.linear_value_head_dim == 16
    assert metadata.linear_num_key_heads == 2
    assert metadata.linear_num_value_heads == 4
    assert metadata.linear_conv_kernel_dim == 4


def test_qwen38_kernel_metadata_reports_each_production_backend(monkeypatch) -> None:
    from dllm_parallel.core.attention import flex
    from dllm_parallel.core.models.backbones.qwen3_8 import model

    runtime = SimpleNamespace(
        to_log_dict=lambda: {
            "backend": "torch_compiled_flex_attention",
            "fa4_version": "test",
        }
    )
    monkeypatch.setattr(flex, "verify_flex_attention_runtime", lambda: runtime)
    monkeypatch.setattr(
        model,
        "verify_qwen38_runtime",
        lambda: {"gated_delta_net": "flash-linear-attention"},
    )

    metadata = Qwen38BackboneExecutor().verify_native_kernels(SimpleNamespace())

    assert metadata["backend"] == "qwen3_8_hybrid_training"
    assert metadata["softmax_attention_forward"] == "fa4_and_compiled_flex"
    assert metadata["softmax_attention_backward"] == "compiled_flex_d256"
    assert metadata["linear_attention"] == "fla_tilelang"
    assert metadata["projection_mlp"] == "transformer_engine"
    assert metadata["fa4_version"] == "test"


def test_text_checkpoint_filter_only_accepts_vision_and_truncated_layers() -> None:
    keys = (
        "model.visual.blocks.0.attn.qkv.weight",
        "model.layers.4.input_layernorm.weight",
        "model.layers.63.mlp.down_proj.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.norm.weight",
        "model.language_model.layers.7.self_attn.q_proj.weight",
    )

    assert _unexpected_text_checkpoint_keys(keys, loaded_layers=4) == (
        "model.layers.3.self_attn.q_proj.weight",
        "model.norm.weight",
        "model.language_model.layers.7.self_attn.q_proj.weight",
    )


def test_mask_migration_uses_first_checkpoint_padding_row() -> None:
    executor = Qwen38BackboneExecutor()
    config = _config()
    tokenizer = _Tokenizer()

    executor.prepare_tokenizer_and_config(
        _spec(),
        config=config,
        tokenizer=tokenizer,
    )

    text = config["text_config"]
    assert isinstance(text, dict)
    assert tokenizer.mask_token_id == 3
    assert text["mask_token_id"] == 3
    assert text["dllm_mask_token_original_vocab_size"] == 3
    assert text["dllm_mask_token_migration_version"] == 1
    assert text["dllm_model_family"] == "qwen3_8"
    assert config["dllm_model_family"] == "qwen3_8"

    executor.prepare_tokenizer_and_config(
        _spec(),
        config=config,
        tokenizer=tokenizer,
    )
    assert len(tokenizer) == 4


def test_mask_migration_rejects_untracked_token_alias() -> None:
    tokenizer = _Tokenizer(("a", "<|mask|>"))
    with pytest.raises(ValueError, match="aliases an existing tokenizer token"):
        Qwen38BackboneExecutor().prepare_tokenizer_and_config(
            _spec(),
            config=_config(),
            tokenizer=tokenizer,
        )


def test_mask_migration_rejects_explicit_id_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        Qwen38BackboneExecutor().prepare_tokenizer_and_config(
            _spec(mask_token_id=7),
            config=_config(),
            tokenizer=_Tokenizer(),
        )


def test_checkpoint_hook_records_and_validates_migration() -> None:
    executor = Qwen38BackboneExecutor()
    config = _config()
    tokenizer = _Tokenizer()
    executor.prepare_tokenizer_and_config(_spec(), config=config, tokenizer=tokenizer)
    text = config["text_config"]
    assert isinstance(text, dict)
    model = SimpleNamespace(config=SimpleNamespace(**text))

    state = executor.save_checkpoint_hooks(model)
    executor.load_checkpoint_hooks({"backbone_state": state}, model)

    state["mask_token_id"] = 4
    with pytest.raises(RuntimeError, match="mask token"):
        executor.load_checkpoint_hooks({"backbone_state": state}, model)


def test_checkpoint_hook_resolves_distributed_model_wrappers() -> None:
    executor = Qwen38BackboneExecutor()
    config = _config()
    tokenizer = _Tokenizer()
    executor.prepare_tokenizer_and_config(_spec(), config=config, tokenizer=tokenizer)
    text = config["text_config"]
    assert isinstance(text, dict)
    base = SimpleNamespace(config=SimpleNamespace(**text))
    wrapped = SimpleNamespace(
        config=SimpleNamespace(train_batch_size=1),
        module=SimpleNamespace(model=base),
    )

    state = executor.save_checkpoint_hooks(wrapped)
    executor.load_checkpoint_hooks({"backbone_state": state}, wrapped)


@pytest.mark.parametrize(
    "missing",
    [
        "family",
        "mask_token",
        "mask_token_id",
        "mask_token_migration_version",
        "mask_token_original_vocab_size",
    ],
)
def test_checkpoint_hook_rejects_incomplete_migration_metadata(missing: str) -> None:
    executor = Qwen38BackboneExecutor()
    config = _config()
    tokenizer = _Tokenizer()
    executor.prepare_tokenizer_and_config(_spec(), config=config, tokenizer=tokenizer)
    text = config["text_config"]
    assert isinstance(text, dict)
    model = SimpleNamespace(config=SimpleNamespace(**text))
    state = executor.save_checkpoint_hooks(model)
    state.pop(missing)

    with pytest.raises(RuntimeError):
        executor.load_checkpoint_hooks({"backbone_state": state}, model)


def test_sharded_state_dict_has_stable_family_envelope() -> None:
    model = SimpleNamespace(state_dict=lambda: {"weight": "sentinel"})
    state = Qwen38BackboneExecutor().sharded_state_dict(model)
    assert state == {
        "format": "dllm_parallel.backbone_state_dict.v1",
        "family": "qwen3_8",
        "state_dict": {"weight": "sentinel"},
    }
