from __future__ import annotations

import math
import os
from importlib.util import find_spec
import tempfile
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from dllm_parallel.core.models.backbones.qwen3_8.model import (
    Qwen38PackedBlockDiffusionModel,
    _build_qwen38_te_layernorm_column_linear,
    _build_qwen38_te_layernorm_mlp,
    _causal_convolution_boundary_states,
    _chunk_gated_delta_rule,
    _ordered_gdn_boundary_states,
    _qwen38_context_attention,
    _uses_pure_cp_token_local_mlp,
)


@pytest.mark.parametrize("branch_ids", [(3, 0, 2), (3, 2), (0,)])
def test_ordered_gdn_boundary_states_matches_slice_stack_and_gradients(
    branch_ids: tuple[int, ...],
) -> None:
    nonzero_blocks = tuple(block_id for block_id in branch_ids if block_id != 0)
    nonzero_states = torch.randn(
        2,
        len(nonzero_blocks),
        3,
        4,
        requires_grad=True,
    )
    zero_state = torch.zeros(2, 3, 4)
    state_by_block = {
        block_id: nonzero_states[:, slot]
        for slot, block_id in enumerate(nonzero_blocks)
    }
    reference = torch.stack(
        [
            zero_state if block_id == 0 else state_by_block[block_id]
            for block_id in branch_ids
        ],
        dim=1,
    )
    actual = _ordered_gdn_boundary_states(
        None if not nonzero_blocks else nonzero_states,
        zero_state,
        branch_ids,
    )

    torch.testing.assert_close(actual, reference)
    if nonzero_blocks:
        grad_output = torch.randn_like(reference)
        reference_grad = torch.autograd.grad(
            reference,
            nonzero_states,
            grad_output,
            retain_graph=True,
        )[0]
        actual_grad = torch.autograd.grad(actual, nonzero_states, grad_output)[0]
        torch.testing.assert_close(actual_grad, reference_grad)


def test_qwen38_context_attention_keeps_bp_dispatch_out_of_pure_cp() -> None:
    pure = _qwen38_context_attention(SimpleNamespace(block_parallel_size=1))
    fused = _qwen38_context_attention(SimpleNamespace(block_parallel_size=4))

    assert pure.__name__ == "pure_context_block_denoising_attention_bshd"
    assert fused.__name__ == "fused_block_context_attention_bshd"


def test_pure_cp_token_local_mlp_dispatch_excludes_bp_and_unsupported_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    mlp = nn.Linear(4, 4, bias=False)
    packed = SimpleNamespace(mlp=mlp)
    runtime = SimpleNamespace(
        block_parallel_size=1,
        configured_context_parallel_size=4,
        context_block_parallel_group=object(),
        sequence_parallel=True,
        tensor_parallel_size=2,
    )

    assert _uses_pure_cp_token_local_mlp(runtime, packed)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert not _uses_pure_cp_token_local_mlp(runtime, packed)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    for override in (
        {"block_parallel_size": 4},
        {"configured_context_parallel_size": 1},
        {"context_block_parallel_group": None},
        {"sequence_parallel": False},
        {"tensor_parallel_size": 1},
    ):
        rejected = SimpleNamespace(**{**vars(runtime), **override})
        assert not _uses_pure_cp_token_local_mlp(rejected, packed)
    assert not _uses_pure_cp_token_local_mlp(runtime, None)
    assert not _uses_pure_cp_token_local_mlp(
        runtime,
        SimpleNamespace(mlp=nn.Linear(4, 4, bias=True)),
    )


class _GatedNorm(nn.Module):
    def forward(self, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return value * torch.nn.functional.silu(gate)


class _Mixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_k_heads = 1
        self.num_v_heads = 1
        self.head_k_dim = 8
        self.head_v_dim = 8
        self.key_dim = 8
        self.value_dim = 8
        self.conv_dim = 24
        self.activation = "silu"
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=4,
            groups=self.conv_dim,
            bias=True,
        )
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = _GatedNorm()


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.float().square().mean(dim=-1, keepdim=True)
        return (value.float() * torch.rsqrt(variance + self.variance_epsilon)).to(
            dtype=value.dtype
        ) * self.weight


class _MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu


class _QwenLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float) -> None:
        super().__init__()
        self.input_layernorm = _RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = _RMSNorm(hidden_size, eps)
        self.mlp = _MLP(hidden_size, intermediate_size)


def _owner(block_size: int) -> Qwen38PackedBlockDiffusionModel:
    owner = object.__new__(Qwen38PackedBlockDiffusionModel)
    nn.Module.__init__(owner)
    owner.block_size = int(block_size)
    owner.sequence_parallel = False
    owner.runtime = SimpleNamespace(tensor_parallel_size=1)
    owner._clean_gather_order_cache = {}
    owner._clean_rank_major_order_cache = {}
    owner._gdn_boundary_plan_cache = {}
    return owner


def _assert_bf16_gradient_equivalent(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Compare fused-kernel gradients without requiring identical reduction order."""

    actual_float = actual.float()
    reference_float = reference.float()
    difference = actual_float - reference_float
    absolute_tolerance = 5e-4
    relative_tolerance = 1.5e-2
    l2_limit = (
        relative_tolerance * reference_float.norm()
        + absolute_tolerance * math.sqrt(reference.numel())
    )
    max_limit = (
        relative_tolerance * reference_float.abs().max()
        + absolute_tolerance
    )
    assert float(difference.norm()) <= float(l2_limit)
    assert float(difference.abs().max()) <= float(max_limit)


def _projected(
    *,
    batch_size: int,
    tokens: int,
    mixer: _Mixer,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(
            batch_size,
            tokens,
            mixer.conv_dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            batch_size,
            tokens,
            mixer.value_dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            batch_size,
            tokens,
            mixer.num_v_heads,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            batch_size,
            tokens,
            mixer.num_v_heads,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine requires CUDA")
def test_te_layernorm_column_linear_matches_qwen_projection_and_gradients() -> None:
    pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(17)
    device = torch.device("cuda")
    hidden_size = 128
    layer = _QwenLayer(hidden_size, 256, 1e-6).to(
        device=device,
        dtype=torch.bfloat16,
    )
    first = nn.Linear(hidden_size, 192, bias=False, device=device, dtype=torch.bfloat16)
    second = nn.Linear(hidden_size, 64, bias=False, device=device, dtype=torch.bfloat16)
    fused = _build_qwen38_te_layernorm_column_linear(
        layer.input_layernorm,
        ("first", first),
        ("second", second),
        runtime=SimpleNamespace(tensor_parallel_size=1, tensor_parallel_group=None),
        dtype=torch.bfloat16,
        device=device,
        name="qwen38_test_mixer_input",
    )

    reference_input = torch.randn(
        257,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    fused_input = reference_input.detach().clone().requires_grad_(True)
    normalized = layer.input_layernorm(reference_input)
    reference = torch.cat((first(normalized), second(normalized)), dim=-1)
    actual = fused(fused_input)
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)

    output_gradient = torch.randn_like(reference)
    reference.backward(output_gradient)
    actual.backward(output_gradient)
    _assert_bf16_gradient_equivalent(fused_input.grad, reference_input.grad)
    _assert_bf16_gradient_equivalent(
        fused.layer_norm_weight.grad,
        layer.input_layernorm.weight.grad,
    )
    _assert_bf16_gradient_equivalent(fused.weight.grad[:192], first.weight.grad)
    _assert_bf16_gradient_equivalent(fused.weight.grad[192:], second.weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine requires CUDA")
def test_te_layernorm_mlp_matches_qwen_mlp_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(19)
    device = torch.device("cuda")
    hidden_size = 128
    intermediate_size = 256
    layer = _QwenLayer(hidden_size, intermediate_size, 1e-6).to(
        device=device,
        dtype=torch.bfloat16,
    )
    fused = _build_qwen38_te_layernorm_mlp(
        layer,
        runtime=SimpleNamespace(tensor_parallel_size=1, tensor_parallel_group=None),
        dtype=torch.bfloat16,
        device=device,
        name="qwen38_test_mlp",
    )
    reference_input = torch.randn(
        257,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    fused_input = reference_input.detach().clone().requires_grad_(True)
    normalized = layer.post_attention_layernorm(reference_input)
    reference = layer.mlp.down_proj(
        F.silu(layer.mlp.gate_proj(normalized)) * layer.mlp.up_proj(normalized)
    )
    actual = fused(fused_input)
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)

    output_gradient = torch.randn_like(reference)
    reference.backward(output_gradient)
    actual.backward(output_gradient)
    _assert_bf16_gradient_equivalent(fused_input.grad, reference_input.grad)
    _assert_bf16_gradient_equivalent(
        fused.layer_norm_weight.grad,
        layer.post_attention_layernorm.weight.grad,
    )
    gate_rows = intermediate_size
    _assert_bf16_gradient_equivalent(
        fused.fc1_weight.grad[:gate_rows],
        layer.mlp.gate_proj.weight.grad,
    )
    _assert_bf16_gradient_equivalent(
        fused.fc1_weight.grad[gate_rows:],
        layer.mlp.up_proj.weight.grad,
    )
    _assert_bf16_gradient_equivalent(
        fused.fc2_weight.grad,
        layer.mlp.down_proj.weight.grad,
    )
    with monkeypatch.context() as dispatch_monkeypatch:
        dispatch_monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        assert _uses_pure_cp_token_local_mlp(
            SimpleNamespace(
                block_parallel_size=1,
                configured_context_parallel_size=4,
                context_block_parallel_group=object(),
                sequence_parallel=True,
                tensor_parallel_size=2,
            ),
            SimpleNamespace(mlp=fused),
        )


def _te_userbuffer_parity_worker(
    rank: int,
    world_size: int,
    init_file: str,
    queue,
) -> None:
    userbuffers_initialized = False
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        from transformer_engine.pytorch import (
            LayerNormMLP,
            UserBufferQuantizationMode,
            destroy_ub,
            initialize_ub,
        )

        device = torch.device("cuda", rank)
        hidden_size = 128
        ffn_hidden_size = 256
        global_rows = 512
        initialize_ub(
            shape=[global_rows, hidden_size],
            tp_size=world_size,
            quantization_modes=[UserBufferQuantizationMode.NONE],
            dtype=torch.bfloat16,
            bootstrap_backend="nccl",
        )
        userbuffers_initialized = True
        common = {
            "sequence_parallel": True,
            "return_bias": False,
            "tp_group": dist.group.WORLD,
            "tp_size": world_size,
            "bias": False,
            "normalization": "RMSNorm",
            "activation": "swiglu",
            "params_dtype": torch.bfloat16,
            "set_parallel_mode": True,
            "device": device,
        }
        torch.manual_seed(41)
        reference = LayerNormMLP(hidden_size, ffn_hidden_size, **common)
        overlapped = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            ub_overlap_ag=True,
            ub_overlap_rs=True,
            **common,
        )
        overlapped.load_state_dict(reference.state_dict())

        for local_rows in (global_rows // world_size,):
            reference.zero_grad(set_to_none=True)
            overlapped.zero_grad(set_to_none=True)
            base = (
                torch.arange(local_rows * hidden_size, device=device, dtype=torch.float32)
                .view(local_rows, hidden_size)
                .remainder(103)
                .add(rank)
                .div(59.0)
                .to(torch.bfloat16)
            )
            reference_input = base.clone().requires_grad_(True)
            overlapped_input = base.clone().requires_grad_(True)
            expected = reference(reference_input)
            actual = overlapped(overlapped_input)
            torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
            output_gradient = (
                torch.arange(actual.numel(), device=device, dtype=torch.float32)
                .view_as(actual)
                .remainder(89)
                .add(rank)
                .div(47.0)
                .to(torch.bfloat16)
            )
            expected.backward(output_gradient)
            actual.backward(output_gradient)
            torch.testing.assert_close(
                overlapped_input.grad,
                reference_input.grad,
                rtol=8e-2,
                atol=8e-2,
            )
            for actual_parameter, expected_parameter in zip(
                overlapped.parameters(),
                reference.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(
                    actual_parameter.grad,
                    expected_parameter.grad,
                    rtol=8e-2,
                    atol=8e-2,
                )
        torch.cuda.synchronize(device)
        queue.put((rank, "ok", ""))
    except Exception:  # pragma: no cover - child reports failure to parent.
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if userbuffers_initialized:
            destroy_ub()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_te_userbuffer_mlp_matches_nccl_at_full_rows() -> None:
    if not dist.is_nccl_available() or torch.cuda.device_count() < 2:
        pytest.skip("CUDA/NCCL with two devices is required")
    pytest.importorskip("transformer_engine.pytorch")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as init:
        init_file = init.name
    try:
        processes = [
            ctx.Process(
                target=_te_userbuffer_parity_worker,
                args=(rank, 2, init_file, queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=180)
        for process in processes:
            assert process.exitcode == 0
        results = [queue.get(timeout=5) for _ in processes]
        errors = [message for _, status, message in results if status != "ok"]
        assert not errors, "\n".join(errors)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_causal_convolution_boundary_states_are_exact_and_differentiable() -> None:
    inputs = torch.arange(2 * 9 * 3, dtype=torch.float32).view(2, 9, 3)
    inputs.requires_grad_(True)

    states = _causal_convolution_boundary_states(
        inputs,
        boundary_tokens=(0, 2, 7),
        state_width=4,
    )

    expected = torch.stack(
        (
            F.pad(inputs[:, :0], (0, 0, 4, 0)),
            F.pad(inputs[:, :2], (0, 0, 2, 0)),
            inputs[:, 3:7],
        ),
        dim=1,
    ).permute(0, 1, 3, 2).reshape_as(states)
    torch.testing.assert_close(states, expected)

    states.sum().backward()
    expected_counts = torch.tensor(
        [1, 1, 0, 1, 1, 1, 1, 0, 0],
        dtype=inputs.dtype,
    ).view(1, 9, 1).expand_as(inputs)
    torch.testing.assert_close(inputs.grad, expected_counts)


@pytest.mark.skipif(find_spec("fla") is None, reason="FLA is not installed")
def test_gated_delta_boundary_plan_supports_sequence_endpoint() -> None:
    from dllm_parallel.core.kernels.gated_delta_boundaries import (
        build_gated_delta_boundary_plan,
    )

    plan = build_gated_delta_boundary_plan(
        tokens=256,
        boundary_tokens=(64, 192, 256),
        device=torch.device("cpu"),
    )

    assert plan.boundary_tokens == (64, 192, 256)
    assert plan.final_slot == 2
    assert plan.slots.tolist() == [-1, 0, -1, 1]


@pytest.mark.parametrize(
    ("layout", "block_parallel_size", "expected"),
    (
        ("contiguous", 1, tuple(range(16))),
        ("zigzag", 4, (0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9)),
    ),
)
def test_clean_gather_order_restores_global_sequence(
    layout: str,
    block_parallel_size: int,
    expected: tuple[int, ...],
) -> None:
    owner = _owner(block_size=4)
    owner.seq_len = 16
    owner.runtime = SimpleNamespace(
        block_parallel_size=block_parallel_size,
        cp_bp_policy=SimpleNamespace(clean_kv_layout=layout),
    )

    order = owner._clean_gather_order(
        device=torch.device("cpu"),
        world_size=4,
        local_tokens=4,
    )

    rank_major = torch.tensor(expected)
    torch.testing.assert_close(rank_major.index_select(0, order), torch.arange(16))


def _independent_target_reference(
    owner: Qwen38PackedBlockDiffusionModel,
    mixer: _Mixer,
    clean: tuple[torch.Tensor, ...],
    active: tuple[torch.Tensor, ...],
    block_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for branch, block_id_tensor in enumerate(block_ids):
        block_id = int(block_id_tensor)
        prefix_length = block_id * owner.block_size
        start = branch * owner.block_size
        stop = start + owner.block_size
        branch_projected = tuple(
            torch.cat((clean_value[:, :prefix_length], active_value[:, start:stop]), dim=1)
            for clean_value, active_value in zip(clean, active, strict=True)
        )
        q, k, v, z, beta, gate, _ = owner._gdn_convolve(mixer, branch_projected)
        output, _ = _chunk_gated_delta_rule(
            q,
            k,
            v,
            gate,
            beta,
            mixer=mixer,
        )
        output = mixer.norm(
            output.reshape(-1, mixer.head_v_dim),
            z.reshape(-1, mixer.head_v_dim),
        ).reshape_as(z).flatten(2)
        outputs.append(output[:, -owner.block_size :])
    return torch.cat(outputs, dim=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA requires CUDA")
def test_clean_boundary_state_reuse_matches_independent_branches_and_gradients() -> None:
    pytest.importorskip("fla")
    torch.manual_seed(11)
    device = torch.device("cuda")
    block_size = 64
    owner = _owner(block_size)
    mixer = _Mixer().to(device=device, dtype=torch.bfloat16)
    block_ids = torch.tensor([3, 0, 2], device=device, dtype=torch.long)

    clean = _projected(batch_size=1, tokens=3 * block_size, mixer=mixer, device=device)
    active = _projected(
        batch_size=1,
        tokens=int(block_ids.numel()) * block_size,
        mixer=mixer,
        device=device,
    )
    clean_scan, recurrent_states, convolution_states = owner._gdn_clean_forward(
        mixer,
        clean,
        tuple(int(value) for value in block_ids.tolist()),
    )
    actual = owner._gdn_target_forward(
        mixer,
        active,
        recurrent_states=recurrent_states,
        convolution_states=convolution_states,
    )
    reference = _independent_target_reference(owner, mixer, clean, active, block_ids)

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    actual_inputs = (*clean, *active, *tuple(mixer.parameters()))
    actual_grads = torch.autograd.grad(
        actual.float().sum() + clean_scan.float().sum(),
        actual_inputs,
        retain_graph=True,
        allow_unused=True,
    )

    reference_q, reference_k, reference_v, reference_z, reference_beta, reference_gate, _ = (
        owner._gdn_convolve(mixer, clean)
    )
    reference_clean, _ = _chunk_gated_delta_rule(
        reference_q,
        reference_k,
        reference_v,
        reference_gate,
        reference_beta,
        mixer=mixer,
    )
    reference_clean = mixer.norm(
        reference_clean.reshape(-1, mixer.head_v_dim),
        reference_z.reshape(-1, mixer.head_v_dim),
    ).reshape_as(reference_z).flatten(2)
    reference_grads = torch.autograd.grad(
        reference.float().sum() + reference_clean.float().sum(),
        actual_inputs,
        allow_unused=True,
    )
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        if actual_grad is None or reference_grad is None:
            assert actual_grad is reference_grad
        else:
            _assert_bf16_gradient_equivalent(actual_grad, reference_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA requires CUDA")
def test_fused_gated_deltanet_matches_official_qwen_forward_and_gradients() -> None:
    pytest.importorskip("fla")
    configuration = pytest.importorskip(
        "transformers.models.qwen3_5.configuration_qwen3_5"
    )
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    torch.manual_seed(19)
    device = torch.device("cuda")
    config = configuration.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention"],
    )
    mixer = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(
        device=device,
        dtype=torch.bfloat16,
    )
    owner = _owner(block_size=16)
    hidden = torch.randn(
        2,
        64,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    reference = mixer(hidden)
    projected = owner._gdn_project(mixer, hidden)
    query, key, value, z, beta, gate, _ = owner._gdn_convolve(mixer, projected)
    assert query.shape[2] == config.linear_num_key_heads
    assert key.shape[2] == config.linear_num_key_heads
    assert value.shape[2] == config.linear_num_value_heads
    mixed, _ = _chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        mixer=mixer,
    )
    mixed = mixer.norm(
        mixed.reshape(-1, mixer.head_v_dim),
        z.reshape(-1, mixer.head_v_dim),
    ).reshape(hidden.shape[0], hidden.shape[1], -1)
    actual = F.linear(mixed, mixer.out_proj.weight, mixer.out_proj.bias)

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    inputs = (hidden, *tuple(mixer.parameters()))
    actual_grads = torch.autograd.grad(
        actual.float().sum(),
        inputs,
        retain_graph=True,
        allow_unused=True,
    )
    reference_grads = torch.autograd.grad(
        reference.float().sum(),
        inputs,
        allow_unused=True,
    )
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        if actual_grad is None or reference_grad is None:
            assert actual_grad is reference_grad
        else:
            _assert_bf16_gradient_equivalent(actual_grad, reference_grad)
