import tempfile
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from benchmarks.tools.inspect_hf_dllm_models import summarize
from dllm_parallel.core.models import summarize_hf_config
from dllm_parallel.core.models.hf_loader import load_hf_model_from_config
import dllm_parallel.core.models.hf_loader as hf_loader
from dllm_parallel.core.parallel.hf_tensor_parallel import (
    configure_hf_config_for_tensor_parallel,
    prepare_hf_tensor_parallel_subgroup,
    validate_hf_tensor_parallel_model,
)
from dllm_parallel.training.block_diffusion import resolve_zero_optimizer_impl
from dllm_parallel.training import block_diffusion_trainer
import dllm_parallel.training.block_diffusion as block_diffusion
from dllm_parallel.core.models.backbones.nemotron.model import (
    _rope_scale_bshd,
    _rotary_bshd_adapter,
)


class _ConfigLike:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)


class _RuntimeLike:
    tensor_parallel_size = 2


class _ModelLike:
    def __init__(self, tp_size, *, sharded=True):
        self._tp_size = tp_size
        placement_type = type("Shard", (), {})
        placements = (placement_type(),) if sharded else ()
        self._parameters = (SimpleNamespace(placements=placements),)

    def parameters(self):
        return iter(self._parameters)

    def modules(self):
        return iter(())


class _NativeHfTpModule:
    _hf_tp_plan = "colwise"

    def __init__(self, tp_size):
        self._hf_device_mesh = SimpleNamespace(size=lambda: tp_size)
        self._parameters = (SimpleNamespace(placements=()),)

    def parameters(self, recurse=True):
        del recurse
        return iter(self._parameters)


class _NativeHfTpModel(_ModelLike):
    def __init__(self, tp_size):
        super().__init__(tp_size, sharded=False)
        self._module = _NativeHfTpModule(tp_size)

    def modules(self):
        return iter((self, self._module))

    def parameters(self, recurse=True):
        del recurse
        return iter(self._parameters)


class _LoadedModelLike:
    def __init__(self) -> None:
        self.to_device = None

    def to(self, device):
        self.to_device = device
        return self


def test_summarize_diffusion_gemma_nested_text_config() -> None:
    summary = summarize(
        "google/diffusiongemma-26B-A4B-it",
        {
            "model_type": "diffusion_gemma",
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
            "canvas_length": 256,
            "dtype": "bfloat16",
            "transformers_version": "5.8.0.dev0",
            "text_config": {
                "hidden_size": 2816,
                "intermediate_size": 2112,
                "num_hidden_layers": 30,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ]
                * 5,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704,
                "sliding_window": 1024,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_global_key_value_heads": 2,
            },
        },
    )

    assert summary.family == "diffusion_gemma"
    assert summary.architecture == "DiffusionGemmaForBlockDiffusion"
    assert summary.hidden_size == 2816
    assert summary.intermediate_size == 2112
    assert summary.num_layers == 30
    assert summary.canvas_length == 256
    assert summary.layer_types == (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ) * 5
    assert summary.head_dim == 256
    assert summary.global_head_dim == 512
    assert summary.num_global_key_value_heads == 2
    assert summary.num_experts == 128
    assert summary.top_k_experts == 8
    assert summary.expert_intermediate_size == 704
    assert summary.attention_layer_types == (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ) * 5
    assert summary.requires_remote_code is False


def test_diffusiongemma_mfu_is_architecture_aware_and_topology_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_diffusion_trainer,
        "detect_hardware_preset",
        lambda *, allow_default: "h200_sxm_bf16",
    )
    model_spec = SimpleNamespace(
        family="diffusion_gemma",
        num_layers=2,
        attention_layer_types=("sliding_attention", "full_attention"),
        layer_types=("sliding_attention", "full_attention"),
        hidden_size=8,
        intermediate_size=12,
        expert_intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        num_experts=4,
        top_k_experts=2,
        sliding_window=4,
    )

    def rank_metrics(valid_and_active: list[tuple[int, int]]) -> list[dict[str, int]]:
        return [
            {
                "measured_steps": 2,
                "measured_valid_tokens": valid,
                "measured_active_tokens": active,
            }
            for valid, active in valid_and_active
        ]

    concentrated = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=rank_metrics([(16, 8), (0, 0), (0, 0), (0, 0)]),
        model_spec=model_spec,
        objective_name="standard_block_diffusion",
        seq_len=8,
        block_size=2,
        vocab_size=16,
        world_size=4,
        elapsed_ms=100.0,
        adapter_type="none",
    )
    distributed = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=rank_metrics([(8, 4), (8, 4), (0, 0), (0, 0)]),
        model_spec=model_spec,
        objective_name="standard_block_diffusion",
        seq_len=8,
        block_size=2,
        vocab_size=16,
        world_size=4,
        elapsed_ms=100.0,
        adapter_type="none",
    )

    assert concentrated["mfu_unavailable_reason"] is None
    assert concentrated["mfu_method"] == "megatron_diffusiongemma_moe_v1"
    assert concentrated["mfu_transformer_token_rows_per_step"] == 16
    assert concentrated["mfu_self_conditioning_token_rows_per_step"] == 8
    assert concentrated["mfu_vocabulary_token_rows_per_step"] == 4
    assert concentrated["mfu_sliding_attention_pairs_per_step"] == 58
    assert concentrated["mfu_full_attention_pairs_per_step"] == 76
    assert concentrated["mfu_active_experts_per_token"] == 2
    assert concentrated["model_flops_per_step"] == pytest.approx(
        sum(concentrated["model_flops_breakdown_per_step"].values())
    )
    assert distributed["model_flops_per_step"] == pytest.approx(
        concentrated["model_flops_per_step"]
    )
    assert distributed["mfu_pct"] == pytest.approx(concentrated["mfu_pct"])


def test_diffusiongemma_adapter_mfu_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_diffusion_trainer,
        "detect_hardware_preset",
        lambda *, allow_default: "h200_sxm_bf16",
    )
    metrics = block_diffusion_trainer._megatron_mfu_metrics(
        rank_metrics=[
            {
                "measured_steps": 1,
                "measured_valid_tokens": 8,
                "measured_active_tokens": 4,
            }
        ],
        model_spec=SimpleNamespace(family="diffusion_gemma"),
        objective_name="standard_block_diffusion",
        seq_len=8,
        block_size=2,
        vocab_size=16,
        world_size=1,
        elapsed_ms=100.0,
        adapter_type="lora",
    )
    assert metrics["mfu_pct"] is None
    assert "adapter 'lora'" in metrics["mfu_unavailable_reason"]


def test_summarize_nemotron_labs_diffusion_remote_config() -> None:
    summary = summarize(
        "nvidia/Nemotron-Labs-Diffusion-8B",
        {
            "model_type": "nemotron_labs_diffusion",
            "architectures": ["NemotronLabsDiffusionModel"],
            "auto_map": {
                "AutoConfig": "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
                "AutoModel": "modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel",
            },
            "hidden_size": 4096,
            "num_hidden_layers": 34,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 131072,
            "intermediate_size": 14336,
            "max_position_embeddings": 262144,
            "block_size": 32,
            "dlm_paradigm": "bidirectional",
            "mask_token_id": 100,
            "attn_implementation": "sdpa",
            "use_cache": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "5.0.0",
        },
    )

    assert summary.family == "nemotron_labs_diffusion"
    assert summary.architecture == "NemotronLabsDiffusionModel"
    assert summary.hidden_size == 4096
    assert summary.num_layers == 34
    assert summary.intermediate_size == 14336
    assert summary.block_size == 32
    assert summary.diffusion_paradigm == "bidirectional"
    assert summary.mask_token_id == 100
    assert summary.attn_implementation == "sdpa"
    assert summary.use_cache is False
    assert summary.requires_remote_code is True


def test_hf_loader_summarizes_config_objects_without_weights() -> None:
    summary = summarize_hf_config(
        "nvidia/Nemotron-Labs-Diffusion-3B",
        _ConfigLike(
            {
                "model_type": "nemotron_labs_diffusion",
                "architectures": ["NemotronLabsDiffusionModel"],
                "hidden_size": 3072,
                "num_hidden_layers": 26,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 131072,
                "max_position_embeddings": 262144,
                "block_size": 32,
                "dlm_paradigm": "bidirectional",
                "mask_token_id": 100,
            }
        ),
    )

    assert summary.family == "nemotron_labs_diffusion"
    assert summary.head_dim == 128
    assert summary.mask_token_id == 100


def test_hf_tp_config_adds_vocab_parallel_heads_and_disables_cache() -> None:
    config = _ConfigLike({})
    config.use_cache = True

    configure_hf_config_for_tensor_parallel(config, _RuntimeLike())

    assert config.use_cache is False
    assert config.base_model_tp_plan["diffusion_head"] == "colwise"
    assert config.base_model_tp_plan["lm_head"] == "colwise"
    assert config.base_model_tp_plan["embed_out"] == "colwise"


def test_diffusiongemma_tp_config_uses_text_plan_without_expanding_paths() -> None:
    text_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
    }
    config = SimpleNamespace(
        model_type="diffusion_gemma",
        text_config=SimpleNamespace(base_model_tp_plan=text_plan),
        base_model_tp_plan=None,
        use_cache=True,
    )

    configure_hf_config_for_tensor_parallel(config, _RuntimeLike())

    assert config.base_model_tp_plan == text_plan
    assert config.base_model_tp_plan is not text_plan
    assert config.use_cache is False


def test_hf_tp_validation_fails_if_transformers_ignores_requested_tp() -> None:
    validate_hf_tensor_parallel_model(_ModelLike(2), _RuntimeLike())
    validate_hf_tensor_parallel_model(_NativeHfTpModel(2), _RuntimeLike())
    with pytest.raises(RuntimeError, match="did not attach"):
        validate_hf_tensor_parallel_model(_ModelLike(1), _RuntimeLike())
    with pytest.raises(RuntimeError, match="did not apply"):
        validate_hf_tensor_parallel_model(_ModelLike(2, sharded=False), _RuntimeLike())
    wrong_mesh_model = _NativeHfTpModel(2)
    wrong_mesh_model._module._hf_device_mesh = SimpleNamespace(size=lambda: 4)
    with pytest.raises(RuntimeError, match="unexpected mesh"):
        validate_hf_tensor_parallel_model(wrong_mesh_model, _RuntimeLike())


def test_hf_tp_subgroup_preparation_uses_explicit_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Mesh:
        ndim = 1

        def size(self) -> int:
            return 2

    config = SimpleNamespace(tp_size=2, fsdp_size=1, tp_plan=None)
    mesh = Mesh()
    calls: list[dict[str, object]] = []

    def initialize(tp_plan, **kwargs):
        calls.append({"tp_plan": tp_plan, **kwargs})
        return "cuda:0", kwargs["device_mesh"]

    distributed_utils = ModuleType("transformers.distributed.utils")
    distributed_utils.initialize_tensor_parallelism = initialize
    monkeypatch.setitem(sys.modules, "transformers.distributed.utils", distributed_utils)
    prepared, device_map, prepared_mesh = prepare_hf_tensor_parallel_subgroup(
        config,
        device_mesh=mesh,
    )

    assert prepared is config
    assert prepared.tp_plan == "auto"
    assert device_map == "cuda:0"
    assert prepared_mesh is mesh
    assert calls == [
        {
            "tp_plan": "auto",
            "tp_size": 2,
            "device_mesh": mesh,
            "device_map": None,
        }
    ]


def test_hf_tp_subgroup_preparation_rejects_wrong_mesh_and_fsdp() -> None:
    mesh = SimpleNamespace(ndim=1, size=lambda: 4)
    with pytest.raises(ValueError, match="does not match"):
        prepare_hf_tensor_parallel_subgroup(
            SimpleNamespace(tp_size=2, fsdp_size=1, tp_plan="auto"),
            device_mesh=mesh,
        )
    with pytest.raises(ValueError, match="cannot own an FSDP axis"):
        prepare_hf_tensor_parallel_subgroup(
            SimpleNamespace(tp_size=2, fsdp_size=2, tp_plan="auto"),
            device_mesh=mesh,
        )


def test_block_diffusion_training_export_uses_canonical_entrypoint() -> None:
    config = block_diffusion_trainer.config_from_args(
        block_diffusion_trainer.build_arg_parser().parse_args(
            ["--seq-len", "2048", "--steps", "2"]
        )
    )
    assert config.spec.model.seq_len == 2048
    assert config.spec.training.steps == 2
    assert not config.debug.grad_finite


def test_hf_training_yaml_config_sets_defaults_and_cli_overrides() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
        handle.write(
            "\n".join(
                [
                    "model:",
                    "  id: test/model",
                    "  seq_len: 4096",
                    "training:",
                    "  steps: 7",
                    "topology:",
                    "  context_parallel_size: 2",
                    "  block_parallel_size: 2",
                    "  process_group_timeout_seconds: 900",
                    "  process_group_timeout:",
                    "    - model_parallel_group=1200",
                    "debug:",
                    "  grad_finite: true",
                ]
            )
        )
        handle.flush()
        parser = block_diffusion_trainer.build_arg_parser()
        config = block_diffusion_trainer.config_from_args(
            parser.parse_args(["--config", handle.name, "--steps", "3"])
        )
    assert config.spec.model.id == "test/model"
    assert config.spec.model.seq_len == 4096
    assert config.spec.training.steps == 3
    assert config.spec.topology.context_parallel_size == 2
    assert config.spec.training.activation_checkpointing_scope == "full"
    assert config.spec.topology.process_group_timeout_seconds == 900
    assert config.spec.topology.process_group_timeout == ("model_parallel_group=1200",)
    assert config.debug.grad_finite


def test_synthetic_random_token_pool_excludes_nonfinite_reserved_rows() -> None:
    embedding = torch.nn.Embedding(8, 4)
    with torch.no_grad():
        embedding.weight[2, 0] = float("nan")
        embedding.weight[5, 1] = float("inf")
    model = SimpleNamespace(get_input_embeddings=lambda: embedding)

    token_pool = block_diffusion_trainer._synthetic_random_token_pool(
        model,
        runtime=None,
        vocab_size=8,
        sample_vocab_size=None,
        mask_token_id=3,
        device=torch.device("cpu"),
    )

    assert token_pool is not None
    assert token_pool.tolist() == [0, 1, 4, 6, 7]


def test_packed_hf_builder_dispatches_through_backbone_registry(monkeypatch) -> None:
    calls = {}
    packed = object()

    def _fake_builder(hf_model, **kwargs):
        calls["hf_model"] = hf_model
        calls["kwargs"] = kwargs
        return packed

    monkeypatch.setattr(
        block_diffusion,
        "build_backbone_packed_block_diffusion_model",
        _fake_builder,
    )
    hf_model = SimpleNamespace(config=_ConfigLike({"model_type": "custom"}))
    runtime = SimpleNamespace()

    result = block_diffusion.build_packed_block_diffusion_model(
        hf_model,
        runtime=runtime,
        seq_len=1024,
        block_size=32,
        ring_attention_key_chunk_size=256,
        activation_checkpointing=False,
        mlp_token_chunk_size=128,
    )

    assert result is packed
    assert calls["hf_model"] is hf_model
    assert calls["kwargs"] == {
        "runtime": runtime,
        "seq_len": 1024,
        "block_size": 32,
        "ring_attention_key_chunk_size": 256,
        "activation_checkpointing": False,
        "activation_checkpointing_scope": "full",
        "mlp_token_chunk_size": 128,
    }


@pytest.mark.parametrize(
    ("device", "overrides", "expected_map"),
    [
        ("cuda:2", {}, {"": torch.device("cuda:2")}),
        ("cpu", {}, None),
        (None, {}, None),
        ("cuda:2", {"device_map": {"": "cuda:2"}}, {"": "cuda:2"}),
        ("cuda:2", {"device_map": None}, None),
        ("cuda:2", {"tp_plan": {}}, None),
        ("cuda:2", {"tp_plan": "auto", "device_mesh": "tp-mesh"}, None),
    ],
)
def test_hf_model_from_config_is_shared_weight_loading_front_door(
    monkeypatch, device, overrides, expected_map,
) -> None:
    calls = {}
    model = _LoadedModelLike()

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            return model

    monkeypatch.setattr(
        hf_loader,
        "_import_transformers",
        lambda: SimpleNamespace(AutoModel=_AutoModel),
    )
    config = _ConfigLike({"model_type": "nemotron_labs_diffusion"})

    loaded = load_hf_model_from_config(
        "nvidia/Nemotron-Labs-Diffusion-3B",
        config=config,
        trust_remote_code=True,
        model_auto_class="model",
        device=device,
        model_kwargs={"low_cpu_mem_usage": True, **overrides},
    )

    assert loaded is model
    assert model.to_device == (None if overrides.get("device_mesh") else device)
    assert calls["model_id"] == "nvidia/Nemotron-Labs-Diffusion-3B"
    assert calls["kwargs"]["config"] is config
    assert calls["kwargs"]["trust_remote_code"] is True
    assert calls["kwargs"]["low_cpu_mem_usage"] is True
    assert calls["kwargs"].get("device_map") == expected_map
    for name, value in overrides.items():
        assert calls["kwargs"][name] == value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_model_loading_preserves_checkpoint_weights(tmp_path, monkeypatch) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(tmp_path)
    placements = []
    from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def load_and_record(*args, **kwargs):
        model = from_pretrained(*args, **kwargs)
        placements.append({parameter.device for parameter in model.parameters()})
        return model

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(load_and_record),
    )
    cpu_model = load_hf_model_from_config(
        str(tmp_path), config=config, torch_dtype=torch.bfloat16, device="cpu",
    )
    device = torch.device("cuda", torch.cuda.current_device())
    gpu_model = load_hf_model_from_config(
        str(tmp_path), config=config, torch_dtype=torch.bfloat16, device=device,
    )
    assert placements == [{torch.device("cpu")}, {device}]
    assert all(parameter.device == device for parameter in gpu_model.parameters())
    cpu_state, gpu_state = cpu_model.state_dict(), gpu_model.state_dict()
    assert cpu_state.keys() == gpu_state.keys()
    for name, expected in cpu_state.items():
        torch.testing.assert_close(gpu_state[name].cpu(), expected, atol=0, rtol=0)


def test_zero_optimizer_auto_uses_hybrid_fused_adam_for_tp_dual_end() -> None:
    runtime = SimpleNamespace(tensor_parallel_size=2, active_block_mode="dual_end")

    assert (
        resolve_zero_optimizer_impl(requested="auto", runtime=runtime)
        == "deepspeed_fused_adam_hybrid"
    )
    with pytest.raises(RuntimeError, match="not a valid explicit optimizer"):
        resolve_zero_optimizer_impl(requested="deepspeed_fused_adam", runtime=runtime)


def test_zero_optimizer_auto_uses_fused_adam_without_unsafe_tp_bp() -> None:
    assert (
        resolve_zero_optimizer_impl(
            requested="auto",
            runtime=SimpleNamespace(tensor_parallel_size=1, active_block_mode="dual_end"),
        )
        == "deepspeed_fused_adam"
    )
    assert (
        resolve_zero_optimizer_impl(
            requested="auto",
            runtime=SimpleNamespace(tensor_parallel_size=2, active_block_mode="all_blocks"),
        )
        == "deepspeed_fused_adam"
    )


def test_nemotron_rotary_adapter_preserves_bhsd_formula_without_layout_copy() -> None:
    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        left, right = value.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def apply_rotary(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids=None,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position_ids
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (
            query * cos + rotate_half(query) * sin,
            key * cos + rotate_half(key) * sin,
        )

    query = torch.randn(2, 7, 4, 8)
    key = torch.randn(2, 7, 2, 8)
    cos = torch.randn(2, 7, 8)
    sin = torch.randn(2, 7, 8)
    apply_bshd = _rotary_bshd_adapter(apply_rotary)

    actual_query, actual_key = apply_bshd(query, key, cos, sin, None)
    expected_query, expected_key = apply_rotary(
        query.transpose(1, 2),
        key.transpose(1, 2),
        cos,
        sin,
    )

    torch.testing.assert_close(actual_query, expected_query.transpose(1, 2))
    torch.testing.assert_close(actual_key, expected_key.transpose(1, 2))
    assert actual_query.is_contiguous()
    assert actual_key.is_contiguous()


def test_nemotron_rope_scale_is_broadcast_over_bshd_heads() -> None:
    scale = torch.arange(1, 8, dtype=torch.float32).unsqueeze(-1)
    actual = _rope_scale_bshd(scale, seq_len=7, dtype=torch.bfloat16)

    assert actual.shape == (1, 7, 1, 1)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.flatten().float(), scale.flatten())
