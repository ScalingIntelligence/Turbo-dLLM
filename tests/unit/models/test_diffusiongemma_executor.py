from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from dllm_parallel.core.models.backbones.diffusiongemma import expert_parallel
from dllm_parallel.core.models.backbones.diffusiongemma.model import (
    DiffusionGemmaBackboneExecutor,
    _CheckpointStableDiffusionGemmaRouter,
    _validate_shared_text_backbone,
)
from dllm_parallel.core.models.backbones.moe_checkpoint import (
    moe_route_checkpoint_context_fn,
)
from dllm_parallel.core.models.backbones.nemotron.model import (
    NemotronLabsDiffusionBackboneExecutor,
    NemotronLabsDiffusionPackedBlockDiffusionModel,
)


class _TextBackbone(torch.nn.Module):
    def __init__(self, projection: torch.nn.Module) -> None:
        super().__init__()
        self.projection = projection


class _Decoder(_TextBackbone):
    def __init__(self, projection: torch.nn.Module) -> None:
        super().__init__(projection)
        self.self_conditioning = torch.nn.Linear(2, 2, bias=False)


def _model_with_text_backbones(*, tied: bool) -> SimpleNamespace:
    encoder_projection = torch.nn.Linear(2, 2, bias=False)
    decoder_projection = (
        encoder_projection if tied else torch.nn.Linear(2, 2, bias=False)
    )
    return SimpleNamespace(
        model=SimpleNamespace(
            encoder=SimpleNamespace(
                language_model=_TextBackbone(encoder_projection),
            ),
            decoder=_Decoder(decoder_projection),
        )
    )


def test_diffusiongemma_requires_identity_tied_text_parameters() -> None:
    _validate_shared_text_backbone(_model_with_text_backbones(tied=True))

    with pytest.raises(RuntimeError, match="not fully tied"):
        _validate_shared_text_backbone(_model_with_text_backbones(tied=False))

    model = _model_with_text_backbones(tied=True)
    model.model.encoder.language_model.encoder_only = torch.nn.Linear(
        2,
        2,
        bias=False,
    )
    with pytest.raises(RuntimeError, match="encoder_only"):
        _validate_shared_text_backbone(model)


def test_shared_packed_executor_declares_clean_self_conditioning_policy() -> None:
    parameters = inspect.signature(
        NemotronLabsDiffusionPackedBlockDiffusionModel.__init__
    ).parameters
    assert parameters["self_condition_clean_tokens"].default is True
    assert parameters["encoder_causal_attention"].default is False


def test_diffusiongemma_router_matches_native_topk_and_replays_checkpoint_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = SimpleNamespace(
        norm=torch.nn.RMSNorm(3, elementwise_affine=False),
        proj=torch.nn.Linear(3, 5, bias=False),
        scale=torch.nn.Parameter(torch.tensor([0.75, 1.25, 0.5])),
        scalar_root_size=3**-0.5,
        per_expert_scale=torch.nn.Parameter(torch.ones(5)),
        config=SimpleNamespace(top_k_experts=2),
    )
    wrapped = _CheckpointStableDiffusionGemmaRouter(router)
    hidden = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]])
    router_states = router.norm(hidden) * router.scale * router.scalar_root_size
    expected_probabilities = torch.softmax(
        router.proj(router_states), dim=-1, dtype=torch.float32
    )
    expected_indices = torch.topk(expected_probabilities, k=2, dim=-1).indices
    record_context, replay_context = moe_route_checkpoint_context_fn()

    with record_context:
        probabilities, _, recorded_indices = wrapped(hidden)

    torch.testing.assert_close(probabilities, expected_probabilities)
    torch.testing.assert_close(recorded_indices, expected_indices)
    monkeypatch.setattr(
        torch,
        "topk",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint replay must not rerun topk")
        ),
    )
    with replay_context:
        _, _, replayed_indices = wrapped(hidden)
    torch.testing.assert_close(replayed_indices, recorded_indices)


def test_shared_packed_executor_propagates_clean_self_conditioning_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_builder(model: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return model

    monkeypatch.setattr(
        "dllm_parallel.core.models.backbones.nemotron.model."
        "build_packed_block_diffusion_model",
        fake_builder,
    )
    model = object()
    result = NemotronLabsDiffusionBackboneExecutor().build_packed_block_diffusion_model(
        model,
        runtime=object(),
        seq_len=8,
        block_size=4,
        self_condition_clean_tokens=False,
    )

    assert result is model
    assert captured["self_condition_clean_tokens"] is False


def test_diffusiongemma_ep_rejects_unverified_optimizer_backends() -> None:
    spec = SimpleNamespace(
        model=SimpleNamespace(family="hf"),
        objective=SimpleNamespace(name="standard_block_diffusion"),
        kernel=SimpleNamespace(cp_bp_attention_policy="production"),
        topology=SimpleNamespace(
            sequence_parallel=False,
            tensor_parallel_size=1,
            expert_parallel_size=2,
        ),
        optimizer=SimpleNamespace(backend="fsdp2"),
    )

    with pytest.raises(ValueError, match="DeepSpeed ZeRO-2"):
        DiffusionGemmaBackboneExecutor().validate_run_spec(spec)


def _native_diffusiongemma_spec(
    *,
    tensor_parallel_size: int,
    sequence_parallel: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(family="hf"),
        objective=SimpleNamespace(
            name="diffusiongemma_native_sft",
            block_size=256,
        ),
        kernel=SimpleNamespace(cp_bp_attention_policy="production"),
        topology=SimpleNamespace(
            sequence_parallel=sequence_parallel,
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=1,
            context_parallel_size=8,
            block_parallel_size=8,
        ),
        training=SimpleNamespace(
            activation_checkpointing=True,
            activation_checkpointing_scope="full",
        ),
        optimizer=SimpleNamespace(backend="auto"),
    )


@pytest.mark.parametrize(
    ("tensor_parallel_size", "sequence_parallel"),
    [(2, False), (2, True), (4, True)],
)
def test_diffusiongemma_native_accepts_generalized_tensor_parallelism(
    tensor_parallel_size: int,
    sequence_parallel: bool,
) -> None:
    spec = _native_diffusiongemma_spec(
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    DiffusionGemmaBackboneExecutor().validate_run_spec(spec)


def test_diffusiongemma_native_rejects_sequence_parallel_with_tp1() -> None:
    spec = _native_diffusiongemma_spec(
        tensor_parallel_size=1,
        sequence_parallel=True,
    )

    with pytest.raises(ValueError, match="sequence_parallel requires"):
        DiffusionGemmaBackboneExecutor().validate_run_spec(spec)


def test_diffusiongemma_validates_complete_attention_schedule() -> None:
    spec = SimpleNamespace(
        model=SimpleNamespace(family="hf"),
        objective=SimpleNamespace(name="standard_block_diffusion"),
        kernel=SimpleNamespace(cp_bp_attention_policy="production"),
        topology=SimpleNamespace(
            sequence_parallel=False,
            tensor_parallel_size=1,
            expert_parallel_size=1,
        ),
        optimizer=SimpleNamespace(backend="auto"),
    )
    text_config = SimpleNamespace(
        num_experts=4,
        num_hidden_layers=6,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        sliding_window=1024,
    )
    config = SimpleNamespace(
        text_config=text_config,
        tie_word_embeddings=True,
    )

    DiffusionGemmaBackboneExecutor().validate_run_spec(spec, config=config)

    text_config.layer_types = ["sliding_attention"] * 5
    with pytest.raises(ValueError, match="describe every layer"):
        DiffusionGemmaBackboneExecutor().validate_run_spec(spec, config=config)


def test_diffusiongemma_layer_clean_context_bounds() -> None:
    executor = SimpleNamespace(
        encoder_causal_attention=True,
        block_size=4,
    )
    query_positions = torch.tensor([8, 8], dtype=torch.int32)
    query_is_clean = torch.tensor([False, True])
    sliding_layer = SimpleNamespace(
        layer_type="sliding_attention",
        attention=SimpleNamespace(sliding_window=4),
    )
    full_layer = SimpleNamespace(
        layer_type="full_attention",
        attention=SimpleNamespace(sliding_window=4),
    )

    sliding = NemotronLabsDiffusionPackedBlockDiffusionModel._clean_attention_bounds(
        executor,
        query_positions=query_positions,
        query_is_clean=query_is_clean,
        layer_ops=sliding_layer,
    )
    full = NemotronLabsDiffusionPackedBlockDiffusionModel._clean_attention_bounds(
        executor,
        query_positions=query_positions,
        query_is_clean=query_is_clean,
        layer_ops=full_layer,
    )

    torch.testing.assert_close(
        sliding,
        torch.tensor([[5, 8], [5, 9]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        full,
        torch.tensor([[0, 8], [0, 9]], dtype=torch.int32),
    )


def test_diffusiongemma_six_layer_attention_schedule_preserves_sliding_then_full_semantics() -> (
    None
):
    def dense_clean_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        clean_positions = torch.arange(
            key.shape[1], device=query.device, dtype=torch.int32
        )
        allowed = (clean_positions[None] >= bounds[:, 0:1]) & (
            clean_positions[None] < bounds[:, 1:2]
        )
        expanded_key = key.float().repeat_interleave(
            query.shape[2] // key.shape[2], dim=2
        )
        expanded_value = value.float().repeat_interleave(
            query.shape[2] // value.shape[2], dim=2
        )
        scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), expanded_key) * (
            query.shape[-1] ** -0.5
        )
        scores = scores.masked_fill(~allowed[None, None], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", probabilities, expanded_value)

    executor = SimpleNamespace(
        encoder_causal_attention=True,
        block_size=4,
    )
    query_positions = torch.tensor([4, 8, 8, 12], dtype=torch.int32)
    query_is_clean = torch.tensor([False, False, True, True])
    sliding_layer = SimpleNamespace(
        layer_type="sliding_attention",
        attention=SimpleNamespace(sliding_window=4),
    )
    full_layer = SimpleNamespace(
        layer_type="full_attention",
        attention=SimpleNamespace(sliding_window=4),
    )
    layers = [sliding_layer] * 5 + [full_layer]
    bounds = [
        NemotronLabsDiffusionPackedBlockDiffusionModel._clean_attention_bounds(
            executor,
            query_positions=query_positions,
            query_is_clean=query_is_clean,
            layer_ops=layer_ops,
        )
        for layer_ops in layers
    ]

    for layer_bounds in bounds[:5]:
        torch.testing.assert_close(
            layer_bounds,
            torch.tensor([[1, 4], [5, 8], [5, 9], [9, 13]], dtype=torch.int32),
        )
    torch.testing.assert_close(
        bounds[5],
        torch.tensor([[0, 4], [0, 8], [0, 9], [0, 13]], dtype=torch.int32),
    )

    torch.manual_seed(23)
    query = torch.randn(1, query_positions.numel(), 4, 8, requires_grad=True)
    key = torch.randn(1, 16, 2, 8, requires_grad=True)
    value = torch.randn(1, 16, 2, 8, requires_grad=True)
    grad_output = torch.randn(1, query_positions.numel(), 4, 8)

    sliding_output = dense_clean_attention(query, key, value, bounds[0])
    full_output = dense_clean_attention(query, key, value, bounds[5])
    sliding_grads = torch.autograd.grad(
        sliding_output,
        (query, key, value),
        grad_output.float(),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    full_grads = torch.autograd.grad(
        full_output,
        (query, key, value),
        grad_output.float(),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    assert torch.count_nonzero(sliding_grads[1][:, :1]) == 0
    assert torch.count_nonzero(full_grads[1][:, :1]) > 0
    assert not torch.allclose(sliding_output.float(), full_output.float())


def test_diffusiongemma_ep_rejects_deepep_incompatible_hidden_size() -> None:
    experts = SimpleNamespace(
        hidden_dim=128,
        intermediate_dim=64,
        num_experts=4,
    )
    runtime = SimpleNamespace(
        expert_parallel_size=2,
        expert_parallel_rank=0,
        expert_parallel_group=object(),
    )

    with pytest.raises(ValueError, match="hidden size divisible by 256"):
        expert_parallel.DiffusionGemmaExpertParallelExperts(experts, runtime)


def test_diffusiongemma_ep_transport_follows_physical_group_placement() -> None:
    assert (
        expert_parallel._expert_transport(
            SimpleNamespace(
                expert_parallel_group_ranks=[2, 6],
                node_size=8,
            )
        )
        == "nvlink"
    )
    assert (
        expert_parallel._expert_transport(
            SimpleNamespace(
                expert_parallel_group_ranks=[2, 10],
                node_size=8,
            )
        )
        == "elastic"
    )


def test_diffusiongemma_grouped_experts_match_reference_outputs_and_gradients() -> None:
    torch.manual_seed(17)
    counts = torch.tensor([2, 0, 3], dtype=torch.int64)
    hidden = torch.randn(5, 4, requires_grad=True)
    gate_up = torch.randn(3, 12, 4, requires_grad=True)
    down = torch.randn(3, 4, 6, requires_grad=True)
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (hidden, gate_up, down)
    )
    observed_offsets: list[list[int]] = []

    def fake_grouped_mm(
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        offs: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        assert offs.dtype == torch.int32
        assert out_dtype == left.dtype
        boundaries = [int(value) for value in offs.tolist()]
        observed_offsets.append(boundaries)
        outputs: list[torch.Tensor] = []
        start = 0
        for expert_index, stop in enumerate(boundaries):
            outputs.append(left[start:stop] @ right[expert_index])
            start = stop
        return torch.cat(outputs, dim=0)

    actual = expert_parallel._torch_grouped_gated_experts(
        fake_grouped_mm,
        hidden,
        gate_up,
        down,
        counts,
        "silu",
    )
    expected = expert_parallel._reference_grouped_gated_experts(
        *reference_inputs,
        counts,
        "silu",
    )
    grad_output = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (hidden, gate_up, down), grad_output)
    expected_gradients = torch.autograd.grad(
        expected,
        reference_inputs,
        grad_output,
    )

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert observed_offsets == [[2, 2, 5], [2, 2, 5]]


def test_diffusiongemma_local_grouped_routing_matches_token_major_reference() -> None:
    torch.manual_seed(29)
    hidden = torch.randn(5, 4, requires_grad=True)
    gate_up = torch.randn(3, 12, 4, requires_grad=True)
    down = torch.randn(3, 4, 6, requires_grad=True)
    weights = torch.softmax(torch.randn(5, 2), dim=-1).requires_grad_(True)
    indices = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 1], [0, 2]])
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (hidden, gate_up, down, weights)
    )

    actual = expert_parallel._local_grouped_routed_experts(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        "silu",
    )
    ref_hidden, ref_gate_up, ref_down, ref_weights = reference_inputs
    expected = torch.zeros_like(ref_hidden)
    for token in range(indices.shape[0]):
        for route in range(indices.shape[1]):
            expert = int(indices[token, route])
            gate, up = torch.nn.functional.linear(
                ref_hidden[token : token + 1], ref_gate_up[expert]
            ).chunk(2, dim=-1)
            expert_output = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                ref_down[expert],
            )
            expected[token : token + 1] = (
                expected[token : token + 1] + expert_output * ref_weights[token, route]
            )

    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(
        actual,
        (hidden, gate_up, down, weights),
        grad_output,
    )
    expected_grads = torch.autograd.grad(
        expected,
        reference_inputs,
        grad_output,
    )
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_diffusiongemma_ep1_installs_grouped_experts_without_rekeying_weights() -> None:
    experts = torch.nn.Module()
    experts.hidden_dim = 4
    experts.intermediate_dim = 6
    experts.num_experts = 3
    experts.act_fn = torch.nn.functional.silu
    experts.gate_up_proj = torch.nn.Parameter(torch.randn(3, 12, 4))
    experts.down_proj = torch.nn.Parameter(torch.randn(3, 4, 6))
    original_gate_up = experts.gate_up_proj
    original_down = experts.down_proj
    layer = torch.nn.Module()
    layer.experts = experts
    decoder = torch.nn.Module()
    decoder.layers = torch.nn.ModuleList([layer])
    model = SimpleNamespace(model=SimpleNamespace(decoder=decoder))

    expert_parallel.install_diffusion_gemma_expert_parallel(
        model,
        SimpleNamespace(expert_parallel_size=1),
    )

    installed = decoder.layers[0].experts
    assert isinstance(installed, expert_parallel.DiffusionGemmaGroupedExperts)
    assert installed.gate_up_proj is original_gate_up
    assert installed.down_proj is original_down
    assert set(installed.state_dict()) == {"gate_up_proj", "down_proj"}


def test_diffusiongemma_grouped_experts_use_portable_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.nn.functional,
        "grouped_mm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CPU execution must use the portable expert loop")
        ),
        raising=False,
    )
    hidden = torch.randn(3, 4, requires_grad=True)
    gate_up = torch.randn(2, 8, 4, requires_grad=True)
    down = torch.randn(2, 4, 4, requires_grad=True)

    output = expert_parallel._grouped_gated_experts(
        hidden,
        gate_up,
        down,
        torch.tensor([1, 2]),
        "gelu_tanh",
    )
    output.square().mean().backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None
    assert gate_up.grad is not None
    assert down.grad is not None


def test_diffusiongemma_grouped_experts_validate_expert_count_shape() -> None:
    with pytest.raises(ValueError, match="count and projection sizes differ"):
        expert_parallel._grouped_gated_experts(
            torch.randn(3, 4),
            torch.randn(2, 8, 4),
            torch.randn(2, 4, 4),
            torch.tensor([3]),
            "silu",
        )
