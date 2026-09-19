from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dllm_parallel.core.attention.dflash_attention import (
    validate_dflash_attention_head_dim,
)
from dllm_parallel.core.models.backbones.dflash.model import (
    DFlashModel,
    DFlashModelConfig,
    RMSNorm,
)
from dllm_parallel.core.models.backbones.dflash.executor import (
    DFlashBackboneExecutor,
    _checkpoint_verifier_id,
    _load_vocabulary_mapping,
    _sample_from_anchor_contract,
    _verifier_weight_aliases,
    export_speculators_checkpoint,
)
from dllm_parallel.core.objectives.dflash import (
    DFlashObjectiveRuntime,
)


DFLASH2_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/dflash2"


def test_muse_glimmer_dflash2_published_checkpoint_matches_generic_contract() -> None:
    raw = json.loads((DFLASH2_FIXTURES / "muse_glimmer_30b_config.json").read_text())
    published = json.loads(
        (DFLASH2_FIXTURES / "muse_glimmer_30b_weight_map.json").read_text()
    )

    config = DFlashModelConfig.from_mapping(raw)

    assert config.architecture == "DFlash2DraftModel"
    assert config.hidden_size == 6_656
    assert config.intermediate_size == 19_968
    assert config.num_hidden_layers == 5
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.target_hidden_size == 6_656
    assert config.target_layer_ids == (2, 14, 26, 38, 50)
    assert len(config.target_layer_ids) * config.target_hidden_size == 33_280
    assert config.verifier_vocab_size == config.draft_vocab_size == 202_048
    assert config.block_size == 16
    assert config.layer_types == ("sliding_attention",) * 5
    assert config.sliding_window == 2_048
    assert config.sliding_window_non_causal
    assert config.output_multiplier == 0.19611613513818404
    assert config.final_logit_softcapping == 20.0

    shapes = published["draft_tensor_shapes"]
    assert shapes["fc.weight"] == [6_656, 33_280]
    assert shapes["candidate_selector.predecessor_codebook"] == [202_048, 256]
    assert shapes["candidate_selector.successor_codebook"] == [202_048, 256]
    assert shapes["candidate_selector.hidden_projection.weight"] == [256, 6_656]
    assert shapes["layers.0.self_attn.q_proj.weight"] == [4_096, 6_656]
    assert shapes["layers.0.self_attn.k_proj.weight"] == [1_024, 6_656]
    assert shapes["layers.0.self_attn.v_proj.weight"] == [1_024, 6_656]
    assert shapes["layers.0.mlp.gate_proj.weight"] == [19_968, 6_656]
    assert shapes["layers.0.mlp.down_proj.weight"] == [6_656, 19_968]
    assert shapes["layers.0.attention_conv.base_kernel"] == [2, 2, 6_656]

    verifier_weight_map = published["verifier_weight_map"]
    for aliases in _verifier_weight_aliases().values():
        assert any(alias in verifier_weight_map for alias in aliases)
    assert {
        verifier_weight_map["model.language_model.embed_tokens.weight"],
        verifier_weight_map["model.language_model.norm.weight"],
        verifier_weight_map["lm_head.weight"],
    } == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def _reference_attention(
    *,
    query,
    local_key,
    local_value,
    global_key,
    global_value,
    local_attn_mask,
    global_attn_mask,
    scale,
    runtime,
    global_seq_len=None,
    interval_plan_cache=None,
):
    del interval_plan_cache, runtime, global_seq_len
    batch_size, query_length, query_heads, head_dim = query.shape
    key_heads = global_key.shape[2]
    groups = query_heads // key_heads
    context_key = global_key.repeat_interleave(groups, dim=2)
    context_value = global_value.repeat_interleave(groups, dim=2)
    draft_key = local_key.repeat_interleave(groups, dim=2)
    draft_value = local_value.repeat_interleave(groups, dim=2)
    key = torch.cat((context_key, draft_key), dim=1)
    value = torch.cat((context_value, draft_value), dim=1)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * scale

    context_length = global_key.shape[1]
    block_size = global_attn_mask.block_size
    anchor_for_query = torch.arange(query_length) // block_size
    query_slot = torch.arange(query_length) % block_size
    context_position = torch.arange(context_length)
    context_allowed = (
        context_position.view(1, 1, -1)
        >= global_attn_mask.context_starts[:, anchor_for_query].unsqueeze(-1)
    ) & (
        context_position.view(1, 1, -1)
        < global_attn_mask.context_stops[:, anchor_for_query].unsqueeze(-1)
    )
    query_valid = global_attn_mask.anchor_valid[:, anchor_for_query]
    context_allowed &= query_valid.unsqueeze(-1)

    local_anchor = torch.arange(query_length) // block_size
    local_slot = torch.arange(query_length) % block_size
    local_allowed = anchor_for_query.view(-1, 1).eq(local_anchor.view(1, -1))
    if local_attn_mask.causal:
        local_allowed &= local_slot.view(1, -1) <= query_slot.view(-1, 1)
    local_allowed = local_allowed.unsqueeze(0).expand(batch_size, -1, -1)
    local_allowed &= query_valid.unsqueeze(-1)
    allowed = torch.cat((context_allowed, local_allowed), dim=-1)
    scores.masked_fill_(~allowed.unsqueeze(1), -torch.inf)
    probabilities = torch.softmax(scores.float(), dim=-1)
    probabilities = torch.nan_to_num(probabilities)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities.to(value.dtype), value)
    return output * query_valid.unsqueeze(-1).unsqueeze(-1)


def test_dflash_rms_norm_matches_reference_forward_and_backward() -> None:
    torch.manual_seed(7)
    module = RMSNorm(16, 1e-6)
    reference_weight = module.weight.detach().clone().requires_grad_(True)
    actual_input = torch.randn(3, 5, 16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(actual_input)

    actual = module(actual_input)
    normalized = reference_input.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    reference = reference_weight * (
        normalized * torch.rsqrt(variance + module.eps)
    ).to(reference_input.dtype)
    actual.backward(grad_output)
    reference.backward(grad_output)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    torch.testing.assert_close(module.weight.grad, reference_weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16")
def test_dflash_rms_norm_matches_bf16_reference() -> None:
    torch.manual_seed(11)
    module = RMSNorm(128, 1e-6).to(device="cuda", dtype=torch.bfloat16)
    actual_input = torch.randn(
        4,
        17,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_input = actual_input.detach().clone().requires_grad_(True)
    reference_weight = module.weight.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(actual_input)

    actual = module(actual_input)
    normalized = reference_input.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    reference = reference_weight * (
        normalized * torch.rsqrt(variance + module.eps)
    ).to(reference_input.dtype)
    actual.backward(grad_output)
    reference.backward(grad_output)

    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        module.weight.grad,
        reference_weight.grad,
        atol=2e-2,
        rtol=2e-2,
    )


def _config() -> DFlashModelConfig:
    return DFlashModelConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        target_hidden_size=8,
        target_layer_ids=(1, 3),
        verifier_vocab_size=32,
        draft_vocab_size=32,
        block_size=4,
        mask_token_id=31,
        rms_norm_eps=1.0e-6,
        rope_theta=10_000.0,
        rope_scaling=None,
        pad_token_id=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        hidden_activation="silu",
        sliding_window=None,
        layer_types=("full_attention", "full_attention"),
        sliding_window_non_causal=False,
        sample_from_anchor=False,
    )


def test_dflash_config_reads_published_nested_rope_parameters() -> None:
    config = DFlashModelConfig.from_mapping(
        {
            "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
            "block_size": 8,
            "draft_vocab_size": 32_000,
            "mask_token_id": 151_669,
            "transformer_layer_config": {
                "hidden_size": 4096,
                "intermediate_size": 12_288,
                "num_hidden_layers": 5,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151_936,
                "rms_norm_eps": 1.0e-6,
                "hidden_act": "silu",
                "layer_types": ["full_attention"] * 5,
                "rope_parameters": {
                    "rope_theta": 10_000.0,
                    "rope_type": "default",
                },
            },
        }
    )

    assert config.rope_theta == 10_000.0
    assert config.rope_scaling == {
        "rope_theta": 10_000.0,
        "rope_type": "default",
    }
    assert config.target_layer_ids == (2, 10, 18, 26, 34)


@pytest.mark.parametrize(
    ("raw", "expected_layers", "expected_block_size", "expected_mask_id"),
    (
        (
            {
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 6,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 248320,
                "hidden_act": "silu",
                "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
                "sliding_window": 4096,
                "dflash_config": {
                    "block_size": 16,
                    "mask_token_id": 248077,
                    "target_layer_ids": [1, 10, 18, 27, 35, 44, 52, 61],
                },
            },
            (2, 11, 19, 28, 36, 45, 53, 62),
            16,
            248077,
        ),
        (
            {
                "hidden_size": 5376,
                "intermediate_size": 10752,
                "num_hidden_layers": 5,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 262144,
                "hidden_act": "silu",
                "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
                "sliding_window": 2048,
                "block_size": 16,
                "final_logit_softcapping": 30.0,
                "dflash_config": {
                    "mask_token_id": 4,
                    "target_layer_ids": [1, 12, 23, 35, 46, 57],
                },
            },
            (2, 13, 24, 36, 47, 58),
            16,
            4,
        ),
    ),
)
def test_dflash_config_reads_published_zlab_schema(
    raw,
    expected_layers,
    expected_block_size,
    expected_mask_id,
) -> None:
    config = DFlashModelConfig.from_mapping(raw)

    assert config.target_layer_ids == expected_layers
    assert config.block_size == expected_block_size
    assert config.mask_token_id == expected_mask_id
    assert not _sample_from_anchor_contract(raw)
    if expected_mask_id == 4:
        assert config.final_logit_softcapping == 30.0


def test_dflash_verifier_weight_aliases_cover_conditional_wrappers() -> None:
    aliases = _verifier_weight_aliases()

    assert "model.language_model.embed_tokens.weight" in aliases["embed_tokens.weight"]
    assert "model.language_model.norm.weight" in aliases["verifier_norm.weight"]
    assert "model.language_model.embed_tokens.weight" in aliases["lm_head.weight"]


def test_dflash_checkpoint_declares_its_verifier_identity() -> None:
    assert _checkpoint_verifier_id(
        {
            "speculators_config": {
                "verifier": {"name_or_path": "Qwen/Qwen3-8B"}
            }
        }
    ) == "Qwen/Qwen3-8B"


def test_dflash_loader_rejects_target_layer_contract_drift(monkeypatch) -> None:
    checkpoint_config = {
        "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
        "block_size": 8,
        "speculators_config": {
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {"proposal_type": "greedy", "speculative_tokens": 7}
            ],
            "verifier": {"name_or_path": "Qwen/Qwen3-8B"},
        },
    }
    monkeypatch.setattr(
        "dllm_parallel.core.models.backbones.dflash.executor._load_json_file",
        lambda *_args, **_kwargs: dict(checkpoint_config),
    )
    spec = SimpleNamespace(
        model=SimpleNamespace(
            id="draft",
            revision=None,
            verifier_id="Qwen/Qwen3-8B",
            verifier_revision=None,
            target_layer_ids=(2, 10, 18),
            draft_vocab_path=None,
        ),
        objective=SimpleNamespace(block_size=8, sample_from_anchor=False),
    )
    with pytest.raises(ValueError, match="target layers differ"):
        DFlashBackboneExecutor().load_config_from_run_spec(spec)


def test_dflash_loader_exposes_verifier_input_vocabulary(monkeypatch) -> None:
    checkpoint_config = {
        "aux_hidden_state_layer_ids": [2, 10],
        "block_size": 8,
        "draft_vocab_size": 32_000,
        "mask_token_id": 151_669,
        "transformer_layer_config": {
            "hidden_size": 256,
            "intermediate_size": 768,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 151_936,
            "hidden_act": "silu",
            "layer_types": ["full_attention", "full_attention"],
        },
        "speculators_config": {
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {"proposal_type": "greedy", "speculative_tokens": 7}
            ],
            "verifier": {"name_or_path": "Qwen/Qwen3-8B"},
        },
    }
    monkeypatch.setattr(
        "dllm_parallel.core.models.backbones.dflash.executor._load_json_file",
        lambda *_args, **_kwargs: dict(checkpoint_config),
    )
    spec = SimpleNamespace(
        model=SimpleNamespace(
            id="draft",
            revision=None,
            verifier_id="Qwen/Qwen3-8B",
            verifier_revision=None,
            target_layer_ids=(2, 10),
            draft_vocab_path=None,
        ),
        objective=SimpleNamespace(block_size=8, sample_from_anchor=False),
    )

    loaded = DFlashBackboneExecutor().load_config_from_run_spec(spec)

    assert loaded["vocab_size"] == 151_936
    assert loaded["mask_token_id"] == 151_669
    assert loaded["draft_vocab_size"] == 32_000


def test_reduced_vocabulary_uses_checkpoint_mapping_without_sidecar() -> None:
    config = _config()
    config = replace(
        config,
        verifier_vocab_size=32,
        draft_vocab_size=8,
    )
    model = DFlashModel(config, attention_op=_reference_attention)
    selected = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])
    with torch.no_grad():
        model.t2d.zero_()
        model.t2d[selected] = True
        model.d2t.copy_(selected - torch.arange(selected.numel()))

    _load_vocabulary_mapping(model, None)


def test_dflash_export_matches_speculators_checkpoint_contract(tmp_path) -> None:
    from safetensors.torch import load_file

    model = DFlashModel(_config(), attention_op=_reference_attention)
    model.speculators_config = {
        "architectures": ["DFlashSpeculator"],
        "speculators_model_type": "dflash",
        "block_size": 4,
    }

    export_speculators_checkpoint(model, tmp_path)

    assert (tmp_path / "config.json").is_file()
    exported = load_file(str(tmp_path / "model.safetensors"), device="cpu")
    assert "layers.0.self_attn.q_proj.weight" in exported
    assert "embed_tokens.weight" in exported
    assert "lm_head.weight" in exported
    assert "verifier_lm_head.weight" not in exported
    assert "verifier_norm.weight" not in exported


def test_slot_contract_is_derived_from_checkpoint_proposal_shape() -> None:
    base = {
        "block_size": 8,
        "speculators_config": {
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {"proposal_type": "greedy", "speculative_tokens": 7}
            ],
        },
    }
    assert not _sample_from_anchor_contract(base)
    base["speculators_config"]["proposal_methods"][0]["speculative_tokens"] = 8
    assert _sample_from_anchor_contract(base)


def test_dflash_supports_rectangular_attention_projection_width() -> None:
    config = replace(
        _config(),
        hidden_size=10,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        target_hidden_size=10,
    )
    model = DFlashModel(config, attention_op=_reference_attention)
    assert model.layers[0].self_attn.q_proj.weight.shape == (8, 10)
    assert model.layers[0].self_attn.o_proj.weight.shape == (10, 8)


def test_production_dflash_attention_validates_kernel_head_dimensions() -> None:
    validate_dflash_attention_head_dim(128)
    with pytest.raises(ValueError, match="head_dim <= 256"):
        validate_dflash_attention_head_dim(512)
    with pytest.raises(ValueError, match="divisible by 8"):
        validate_dflash_attention_head_dim(12)


def test_dflash_model_preserves_frozen_verifier_boundary() -> None:
    torch.manual_seed(3)
    config = _config()
    model = DFlashModel(config, attention_op=_reference_attention)
    with torch.no_grad():
        model.embed_tokens.weight.normal_()
        model.lm_head.weight.normal_()
        model.verifier_lm_head.weight.normal_()
    objective_runtime = DFlashObjectiveRuntime(
        block_size=4,
        max_anchors=2,
        mask_token_id=31,
        decay_gamma=4.0,
        sample_from_anchor=False,
        device=torch.device("cpu"),
        seed=5,
    )
    input_ids = torch.arange(12).view(1, -1) % config.verifier_vocab_size
    objective = objective_runtime.prepare(
        input_ids,
        torch.ones_like(input_ids, dtype=torch.bool),
    )
    target_hidden = torch.randn(1, 12, 16, requires_grad=True)
    verifier_last_hidden = torch.randn(1, 12, 8, requires_grad=True)
    teacher_hidden = verifier_last_hidden.gather(
        1,
        objective.teacher_source_positions.reshape(1, -1)
        .unsqueeze(-1)
        .expand(-1, -1, 8),
    )
    output = model(
        target_hidden_states=target_hidden,
        teacher_hidden_states=teacher_hidden,
        input_ids=input_ids,
        position_ids=torch.arange(12).view(1, -1),
        objective=objective,
    )
    assert output.hidden_states.shape == (1, 8, config.hidden_size)
    assert output.teacher_hidden_states.shape == (1, 8, config.target_hidden_size)
    loss = model.distributed_dflash_loss(
        output,
        loss_kind="speculators_kl",
        normalization_count=objective.valid_supervised_tokens,
    )
    loss.backward()

    assert model.fc.weight.grad is not None
    assert model.layers[0].self_attn.k_proj.weight.grad is not None
    assert target_hidden.grad is None
    assert verifier_last_hidden.grad is None
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert model.verifier_lm_head.weight.grad is None
    assert model.verifier_norm.weight.grad is None


def test_dflash_draft_rope_positions_are_anchor_relative() -> None:
    config = replace(_config(), num_hidden_layers=1, layer_types=("full_attention",))

    class PositionRecorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.positions = []

        def prepare(self, positions, *, dtype):
            self.positions.append(positions.detach().clone())
            shape = (*positions.shape, 1, config.head_dim)
            return SimpleNamespace(
                cos=torch.ones(shape, dtype=dtype),
                sin=torch.zeros(shape, dtype=dtype),
            )

    def capture_attention(**kwargs):
        return torch.zeros_like(kwargs["query"])

    model = DFlashModel(config, attention_op=capture_attention)
    recorder = PositionRecorder()
    model.rotary_emb = recorder
    objective_runtime = DFlashObjectiveRuntime(
        block_size=4,
        max_anchors=1,
        mask_token_id=31,
        decay_gamma=4.0,
        sample_from_anchor=False,
        device=torch.device("cpu"),
        seed=5,
    )
    input_ids = torch.arange(12).view(1, -1) % config.verifier_vocab_size
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    loss_mask[:, 6] = True
    objective = objective_runtime.prepare(input_ids, loss_mask)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]])

    model(
        target_hidden_states=torch.randn(1, 12, 16),
        teacher_hidden_states=torch.randn(1, 4, 8),
        input_ids=input_ids,
        position_ids=position_ids,
        objective=objective,
    )

    expected = torch.arange(4).view(1, -1)
    assert torch.equal(recorder.positions[0], expected)
    assert torch.equal(recorder.positions[1], position_ids)


def test_reference_attention_matches_explicit_allowed_edges() -> None:
    query = torch.ones(1, 4, 1, 2)
    context_key = torch.ones(1, 5, 1, 2)
    context_value = torch.arange(10, dtype=torch.float32).view(1, 5, 1, 2)
    local_key = torch.ones(1, 4, 1, 2)
    local_value = torch.full((1, 4, 1, 2), 20.0)
    from dllm_parallel.core.attention.masks import (
        DFlashGlobalContextMask,
        DFlashLocalBlockMask,
    )

    output = _reference_attention(
        query=query,
        local_key=local_key,
        local_value=local_value,
        global_key=context_key,
        global_value=context_value,
        local_attn_mask=DFlashLocalBlockMask(
            anchor_valid=torch.tensor([[True]]),
            block_size=4,
        ),
        global_attn_mask=DFlashGlobalContextMask(
            context_starts=torch.tensor([[1]], dtype=torch.int32),
            context_stops=torch.tensor([[3]], dtype=torch.int32),
            anchor_valid=torch.tensor([[True]]),
            block_size=4,
        ),
        scale=1.0 / math.sqrt(2),
        runtime=None,
    )
    expected_context = context_value[:, 1:3].sum(dim=1)
    expected_local = local_value.sum(dim=1)
    expected = (expected_context + expected_local) / 6.0
    assert torch.allclose(output[:, 0], expected)
