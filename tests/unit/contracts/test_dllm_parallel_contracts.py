# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import subprocess
import sys
import os
import tempfile
from pathlib import Path

import pytest
import torch

import dllm_parallel
from dllm_parallel.core.parallel import (
    build_distributed_training_plan,
    infer_parallel_mesh_sizes,
    launch_environment_metadata,
    prepare_parallel_environment,
    require_idle_assigned_gpus,
    validate_supported_training_axes,
)
from dllm_parallel.core.parallel.preflight import launch_environment_policy
from dllm_parallel.core.models import (
    build_schedule,
    summarize_config,
    validate_parallel_spec,
)
from dllm_parallel.core.models.registry import (
    describe_supported_configs,
    executor_for_family,
)
from dllm_parallel.core.models.contracts import BackboneExecutor
from dllm_parallel.core.objectives import (
    STANDARD_BLOCK_DIFFUSION_OBJECTIVE,
    active_block_loss_region,
    reduce_token_losses,
)
from dllm_parallel.core.specs import ParallelSpec
from dllm_parallel.core.parallel.runtime import ParallelRuntime


class _AttrDict(dict):
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value):
        self[key] = value


def test_top_level_import_does_not_eagerly_import_experiment_frameworks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, dllm_parallel; "
                "assert dllm_parallel.ParallelSpec().model_parallel_size == 1; "
                "print('lightning' in sys.modules, "
                "'deepspeed' in sys.modules, 'wandb' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False"


def test_model_metadata_import_uses_backbone_registry() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from dllm_parallel.core.models import summarize_config; "
                "spec = summarize_config('nemotron', {"
                "'model_type': 'nemotron_labs_diffusion', "
                "'num_attention_heads': 32, "
                "'num_key_value_heads': 8, "
                "'head_dim': 128, "
                "'block_size': 32}); "
                "print(spec.family, spec.head_dim, 'torch' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "nemotron_labs_diffusion 128 False"


@pytest.mark.parametrize(
    "family",
    ("dflash", "diffusion_gemma", "nemotron_labs_diffusion"),
)
def test_backbone_executors_implement_the_complete_training_contract(
    family: str,
) -> None:
    assert isinstance(executor_for_family(family), BackboneExecutor)


def test_public_package_namespaces_do_not_eagerly_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import dllm_parallel.core.attention, dllm_parallel.core.kernels, "
                "dllm_parallel.core.models, dllm_parallel.core.profiling, "
                "dllm_parallel.core.parallel, dllm_parallel.training; "
                "print('torch' in sys.modules, "
                "'transformers' in sys.modules, 'lightning' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False"


def test_hydra_configs_do_not_use_arbitrary_eval_resolver() -> None:
    config_root = (
        Path(__file__).resolve().parents[1] / "dllm_parallel" / "train" / "configs"
    )
    offenders = [
        str(path.relative_to(config_root))
        for path in config_root.rglob("*.yaml")
        if "${eval:" in path.read_text(errors="ignore")
    ]

    assert offenders == []


def test_hidden_debug_environment_switches_are_not_library_contracts() -> None:
    root = Path(__file__).resolve().parents[1] / "dllm_parallel"
    hidden_switches = (
        "DLLM_PROFILE_TRACE",
        "DLLM_PRINT_BATCH",
        "BDLM_PRINT_NANS",
        "DLLM_VERBOSE_EXTENSIONS",
    )
    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml"}:
            continue
        text = path.read_text(errors="ignore")
        for switch in hidden_switches:
            if switch in text:
                offenders.append(f"{path.relative_to(root)}:{switch}")

    assert offenders == []


def test_train_package_exports_single_endpoint_without_heavy_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from dllm_parallel.training import train; "
                "print(train.__module__, "
                "'torch' in sys.modules, 'deepspeed' in sys.modules, "
                "'lightning' in sys.modules, 'transformers' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == (
        "dllm_parallel.training.entrypoint False False False False"
    )


def test_train_entrypoint_routes_supported_frontends_without_heavy_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from dllm_parallel.training.entrypoint import _validate_canonical_args\n"
                "print(_validate_canonical_args(['--config', 'recipe.yaml', '--seq-len', '1024']))\n"
                "print(_validate_canonical_args(['--config=recipes/smoke/nemotron_3b.yaml', '--model-id', 'nvidia/x']))\n"
                "try:\n"
                "    _validate_canonical_args(['--seq-len', '1024'])\n"
                "except ValueError as exc:\n"
                "    print(type(exc).__name__, 'requires --config' in str(exc))\n"
                "for args in (['hf', '--seq-len', '1024'], ['causal_lm', '--seq-len', '1024']):\n"
                "    try:\n"
                "        _validate_canonical_args(args)\n"
                "    except ValueError as exc:\n"
                "        print(type(exc).__name__, 'unknown positional argument' in str(exc))\n"
                "try:\n"
                "    _validate_canonical_args(['model=medium'])\n"
                "except ValueError as exc:\n"
                "    print(type(exc).__name__, 'RunSpec' in str(exc))\n"
                "print('torch' in sys.modules, 'deepspeed' in sys.modules, "
                "'lightning' in sys.modules, 'transformers' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "None",
        "None",
        "ValueError True",
        "ValueError True",
        "ValueError True",
        "ValueError True",
        "False False False False",
    ]


def test_parallel_preflight_infers_dp_and_validates_tp_overlap_env(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    mesh = infer_parallel_mesh_sizes(
        world_size=16,
        context_parallel_size=2,
        block_parallel_size=2,
        tensor_parallel_size=2,
    )

    prepare_parallel_environment(
        sequence_parallel=False,
        tensor_parallel_size=2,
        tensor_parallel_overlap=True,
    )

    assert mesh.data_parallel_size == 4
    assert mesh.model_parallel_size == 4
    assert mesh.local_parallel_size == 2
    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


def test_parallel_preflight_rejects_unsupported_axes_and_sets_overlap_env(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dllm_parallel.core.parallel.preflight._validate_expert_parallel_environment",
        lambda: None,
    )
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        validate_supported_training_axes(pipeline_parallel_size=2)
    validate_supported_training_axes()

    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    prepare_parallel_environment(
        sequence_parallel=False,
        tensor_parallel_size=2,
        tensor_parallel_overlap=True,
    )
    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"

    prepare_parallel_environment(
        sequence_parallel=False,
        tensor_parallel_size=1,
        tensor_parallel_overlap=True,
        context_parallel_size=1,
        block_parallel_size=1,
        expert_parallel_size=2,
    )
    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


def test_parallel_preflight_sets_cp_bp_overlap_env(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)

    prepare_parallel_environment(
        sequence_parallel=False,
        context_parallel_size=4,
        block_parallel_size=4,
        tensor_parallel_size=1,
        tensor_parallel_overlap=True,
    )

    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


def test_parallel_preflight_sets_ordering_constraint_for_pure_cp(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)

    prepare_parallel_environment(
        sequence_parallel=False,
        context_parallel_size=4,
        block_parallel_size=1,
        tensor_parallel_size=1,
        tensor_parallel_overlap=True,
    )

    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


def test_parallel_preflight_keeps_expandable_segments_for_sequence_parallel(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        "garbage_collection_threshold:0.8,expandable_segments:True",
    )

    prepare_parallel_environment(
        sequence_parallel=True,
        tensor_parallel_size=1,
        tensor_parallel_overlap=True,
    )

    assert (
        os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        == "garbage_collection_threshold:0.8,expandable_segments:True"
    )


def test_parallel_preflight_orders_pure_tp_sequence_parallel_collectives(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    prepare_parallel_environment(
        sequence_parallel=True,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=4,
        expert_parallel_size=1,
        tensor_parallel_overlap=False,
    )
    policy = launch_environment_policy(
        sequence_parallel=True,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=4,
        expert_parallel_size=1,
        tensor_parallel_overlap=False,
    )

    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert policy["requires_cuda_device_max_connections_one"] is True


def test_launch_environment_metadata_logs_performance_relevant_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    metadata = launch_environment_metadata()

    assert metadata["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert metadata["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert metadata["LOCAL_WORLD_SIZE"] == "4"
    assert "WORLD_SIZE" in metadata


def test_gpu_launch_preflight_checks_only_assigned_devices() -> None:
    inventory = "0, GPU-a\n1, GPU-b\n2, GPU-c\n3, GPU-d\n"
    processes = "GPU-d, 91, python, 12000\n"
    outputs = iter((inventory, processes))

    require_idle_assigned_gpus(
        local_processes=2,
        environ={"CUDA_VISIBLE_DEVICES": "0,1"},
        command_runner=lambda _command: next(outputs),
    )


def test_gpu_launch_preflight_reports_process_on_assigned_device() -> None:
    inventory = "0, GPU-a\n1, GPU-b\n"
    processes = "GPU-b, 1234, python, 49152\n"
    outputs = iter((inventory, processes))

    with pytest.raises(RuntimeError, match=r"GPU-b.*pid=1234.*memory_mib=49152"):
        require_idle_assigned_gpus(
            local_processes=2,
            environ={"SLURM_STEP_GPUS": "0,1"},
            command_runner=lambda _command: next(outputs),
        )


def test_hf_trainer_config_accepts_gradient_accumulation_steps() -> None:
    from dllm_parallel.training.block_diffusion_trainer import (
        build_arg_parser,
        config_from_args,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(
            """
training:
  batch_size: 2
  steps: 4
  gradient_accumulation_steps: 3
profiler:
  warmup_steps: 1
topology:
  context_parallel_size: 2
  block_parallel_size: 2
"""
        )
        path = handle.name
    try:
        parser = build_arg_parser()
        config = config_from_args(
            parser.parse_args(["--config", path, "--gradient-accumulation-steps", "5"])
        )
    finally:
        os.unlink(path)

    assert config.spec.training.batch_size == 2
    assert config.spec.training.steps == 4
    assert config.spec.training.gradient_accumulation_steps == 5


def test_hf_trainer_config_accepts_scheduler_and_gradient_clipping() -> None:
    from dllm_parallel.training.block_diffusion_trainer import (
        _RunSpecLRScheduler,
        build_arg_parser,
        config_from_args,
    )

    parser = build_arg_parser()
    config = config_from_args(
        parser.parse_args(
            [
                "--scheduler-type",
                "linear",
                "--lr-warmup-steps",
                "2",
                "--lr-decay-steps",
                "4",
                "--min-lr",
                "0.1",
                "--lr",
                "1.0",
                "--gradient-clip-norm",
                "0.5",
            ]
        )
    )

    assert config.spec.scheduler.type == "linear"
    assert config.spec.scheduler.warmup_steps == 2
    assert config.spec.scheduler.decay_steps == 4
    assert config.spec.scheduler.min_lr == 0.1
    assert config.spec.optimizer.gradient_clip_norm == 0.5

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = _RunSpecLRScheduler(
        optimizer,
        scheduler_type="linear",
        warmup_steps=2,
        decay_steps=4,
        min_lr=0.1,
        weight_decay_style="linear",
        weight_decay_start=0.01,
        weight_decay_end=0.05,
        weight_decay_steps=4,
    )
    assert scheduler.step(1) == pytest.approx(0.55)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)
    assert scheduler.step(2) == pytest.approx(1.0)
    assert scheduler.step(4) == pytest.approx(0.55)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)
    state = scheduler.state_dict()

    restored_optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    restored = _RunSpecLRScheduler(
        restored_optimizer,
        scheduler_type="linear",
        warmup_steps=2,
        decay_steps=4,
        min_lr=0.1,
        weight_decay_style="linear",
        weight_decay_start=0.01,
        weight_decay_end=0.05,
        weight_decay_steps=4,
    )
    restored.load_state_dict(state)
    assert restored.current_lr() == pytest.approx(scheduler.current_lr())
    assert restored.step(5) == pytest.approx(0.325)
    assert restored_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)

    inverse_optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    inverse = _RunSpecLRScheduler(
        inverse_optimizer,
        scheduler_type="inverse_square_root",
        warmup_steps=4,
        decay_steps=16,
        min_lr=0.0,
    )
    assert inverse.step(4) == pytest.approx(1.0)
    assert inverse.step(16) == pytest.approx(0.5)


def test_trainer_global_gradient_clipping_uses_fp32_l2_norm() -> None:
    from dllm_parallel.training.optimizer_setup import _clip_gradients_for_optimizer

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])

    norm = _clip_gradients_for_optimizer(
        model,
        ds_engine=None,
        max_norm=1.0,
        runtime=None,
    )
    assert norm == pytest.approx(5.0)
    torch.testing.assert_close(
        model.weight.grad,
        torch.tensor([[0.6, 0.8]]),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_trainer_global_gradient_clipping_rejects_nonfinite_norm() -> None:
    from dllm_parallel.training.optimizer_setup import _clip_gradients_for_optimizer

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.tensor([[float("inf")]])

    with pytest.raises(RuntimeError, match="nonfinite gradients"):
        _clip_gradients_for_optimizer(
            model,
            ds_engine=None,
            max_norm=1.0,
            runtime=None,
        )


def test_fsdp2_gradient_clipping_reduces_data_shards_globally(monkeypatch) -> None:
    from types import SimpleNamespace

    from dllm_parallel.training.optimizer_setup import _clip_gradients_for_optimizer

    class Shard:
        pass

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    model.weight.placements = (Shard(),)
    model.weight.device_mesh = SimpleNamespace(mesh_dim_names=("data_parallel",))
    observed_groups = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "group",
        SimpleNamespace(WORLD="world"),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)

    def all_reduce(value, *, op, group):
        assert op == torch.distributed.ReduceOp.SUM
        observed_groups.append(group)
        value.mul_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    runtime = SimpleNamespace(
        model_parallel_group="model",
        model_parallel_size=1,
        local_parallel_rank=0,
        tensor_parallel_rank=0,
        expert_parallel_rank=0,
    )

    norm = _clip_gradients_for_optimizer(
        model,
        ds_engine=None,
        max_norm=1.0,
        runtime=runtime,
    )

    assert norm == pytest.approx(50.0**0.5)
    assert observed_groups == [torch.distributed.group.WORLD]
    torch.testing.assert_close(
        model.weight.grad,
        torch.tensor([[3.0, 4.0]]) / (50.0**0.5 + 1.0e-6),
    )


def test_tensor_parallel_linear_parameters_declare_shard_ownership() -> None:
    from dllm_parallel.core.parallel.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    column = ColumnParallelLinear(8, 16, tensor_parallel_size=2, bias=True)
    row = RowParallelLinear(16, 8, tensor_parallel_size=2, bias=True)

    assert column.weight._dllm_tensor_parallel_sharded
    assert column.bias._dllm_tensor_parallel_sharded
    assert row.weight._dllm_tensor_parallel_sharded
    assert not bool(getattr(row.bias, "_dllm_tensor_parallel_sharded", False))


def test_trainer_delegates_gradient_clipping_to_deepspeed_step() -> None:
    from types import SimpleNamespace

    from dllm_parallel.training.optimizer_setup import _clip_gradients_for_optimizer

    engine = SimpleNamespace(optimizer=SimpleNamespace(clip_grad=1.0))
    assert (
        _clip_gradients_for_optimizer(
            object(),
            ds_engine=engine,
            max_norm=1.0,
        )
        is None
    )

    engine.optimizer.clip_grad = 0.5
    with pytest.raises(RuntimeError, match="does not match"):
        _clip_gradients_for_optimizer(
            object(),
            ds_engine=engine,
            max_norm=1.0,
        )


def test_trainer_reads_deepspeed_cached_global_gradient_norm_safely() -> None:
    from types import SimpleNamespace

    from dllm_parallel.training.optimizer_setup import _deepspeed_global_grad_norm

    optimizer = SimpleNamespace(
        _global_grad_norm=torch.tensor(3.5),
        overflow=False,
    )
    engine = SimpleNamespace(optimizer=optimizer)
    assert _deepspeed_global_grad_norm(engine) == pytest.approx(3.5)

    optimizer.overflow = True
    assert _deepspeed_global_grad_norm(engine) is None

    optimizer.overflow = False
    optimizer._global_grad_norm = torch.tensor([1.0, 2.0])
    assert _deepspeed_global_grad_norm(engine) is None


def test_run_spec_rejects_placeholder_production_dataset_path() -> None:
    from dllm_parallel.training.run_spec import RunSpec

    with pytest.raises(ValueError, match="placeholder dataset_path"):
        RunSpec.from_mapping(
            {
                "launch": {"recipe_kind": "prod"},
                "model": {"revision": "immutable-test-revision"},
                "data": {
                    "input_mode": "dataset",
                    "dataset_path": "/data/dllm/packed_tokens/train.pt",
                },
            }
        )


def test_run_spec_requires_wandb_project_for_production_logging() -> None:
    from dllm_parallel.training.run_spec import RunSpec

    with pytest.raises(ValueError, match="production W&B logging requires"):
        RunSpec.from_mapping(
            {
                "launch": {"recipe_kind": "prod"},
                "model": {"revision": "immutable-test-revision"},
                "data": {
                    "input_mode": "dataset",
                    "dataset_path": "/mnt/data/train.pt",
                },
                "logging": {"wandb": True},
            }
        )


def test_run_spec_accepts_explicit_fsdp_policy() -> None:
    from dllm_parallel.training.run_spec import RunSpec

    spec = RunSpec.from_mapping(
        {
            "optimizer": {
                "backend": "fsdp",
                "fsdp_mixed_precision": "bf16",
                "fsdp_sharding_strategy": "shard_grad_op",
                "fsdp_use_orig_params": False,
            },
            "topology": {
                "context_parallel_size": 1,
                "block_parallel_size": 1,
                "tensor_parallel_size": 1,
            },
        }
    )
    assert spec.optimizer.backend == "fsdp"
    assert spec.optimizer.fsdp_mixed_precision == "bf16"
    assert spec.optimizer.fsdp_sharding_strategy == "shard_grad_op"
    assert spec.optimizer.fsdp_use_orig_params is False


def test_run_spec_supports_cp1_and_requires_equal_multirank_cp_chunks() -> None:
    from dllm_parallel.training.run_spec import RunSpec

    spec = RunSpec.from_mapping(
        {
            "topology": {
                "context_parallel_size": 1,
                "block_parallel_size": 1,
                "tensor_parallel_size": 2,
                "sequence_parallel": True,
            }
        }
    )
    assert spec.topology.context_parallel_size == 1
    assert spec.topology.tensor_parallel_size == 2
    with pytest.raises(ValueError, match="equal DualChunkSwap chunks"):
        RunSpec.from_mapping(
            {
                "model": {"seq_len": 30},
                "objective": {"block_size": 2},
                "topology": {
                    "context_parallel_size": 4,
                    "block_parallel_size": 1,
                },
            }
        )


def test_production_run_spec_has_one_packed_cp_mode() -> None:
    from dllm_parallel.training.run_spec import RunSpec

    spec = RunSpec.from_mapping(
        {
            "topology": {
                "context_parallel_size": 2,
                "block_parallel_size": 1,
            }
        }
    )
    assert not hasattr(spec.topology, "cp_execution_mode")

    with pytest.raises(ValueError, match="unknown RunSpec topology fields"):
        RunSpec.from_mapping(
            {
                "topology": {
                    "context_parallel_size": 2,
                    "block_parallel_size": 1,
                    "cp_execution_mode": "monolithic",
                }
            }
        )


def test_run_spec_selects_fused_or_replicated_bp_from_topology() -> None:
    from dllm_parallel.training.run_spec import RunSpec

    fused = RunSpec.from_mapping(
        {
            "objective": {"block_size": 32},
            "topology": {
                "context_parallel_size": 4,
                "block_parallel_size": 4,
            },
        }
    )
    assert fused.topology.replicate_clean_prefix is False

    with pytest.raises(ValueError, match="replicate_clean_prefix=true"):
        RunSpec.from_mapping(
            {
                "objective": {"block_size": 32},
                "topology": {
                    "context_parallel_size": 1,
                    "block_parallel_size": 4,
                },
            }
        )

    replicated = RunSpec.from_mapping(
        {
            "objective": {"block_size": 32},
            "topology": {
                "context_parallel_size": 1,
                "block_parallel_size": 4,
                "replicate_clean_prefix": True,
            },
        }
    )
    assert replicated.topology.replicate_clean_prefix is True

    with pytest.raises(ValueError, match="only valid for BP-only execution"):
        RunSpec.from_mapping(
            {
                "objective": {"block_size": 32},
                "topology": {
                    "context_parallel_size": 4,
                    "block_parallel_size": 4,
                    "replicate_clean_prefix": True,
                },
            }
        )


@pytest.mark.parametrize(
    "removed_backend", ["distributed", "fused_bp_cp", "production"]
)
def test_run_spec_has_no_backend_selector(removed_backend: str) -> None:
    from dllm_parallel.training.run_spec import RunSpec

    with pytest.raises(ValueError, match="unknown RunSpec topology fields: backend"):
        RunSpec.from_mapping({"topology": {"backend": removed_backend}})


def test_run_spec_exposes_only_zigzag_cp_ownership() -> None:
    from dllm_parallel.training.run_spec import RunSpec

    with pytest.raises(ValueError, match="must be one of: zigzag"):
        RunSpec.from_mapping(
            {
                "kernel": {"cp_bp_clean_kv_layout": "contiguous"},
            }
        )


def test_run_spec_validates_clean_kv_transport() -> None:
    from dllm_parallel.training.run_spec import RunSpec

    default = RunSpec.from_mapping({})
    assert default.kernel.cp_bp_clean_kv_transport == "collective"

    streaming = RunSpec.from_mapping(
        {"kernel": {"cp_bp_clean_kv_transport": "streaming"}}
    )
    assert streaming.kernel.cp_bp_clean_kv_transport == "streaming"

    with pytest.raises(ValueError, match="collective, streaming"):
        RunSpec.from_mapping({"kernel": {"cp_bp_clean_kv_transport": "invalid"}})


def test_hf_trainer_microbatch_no_sync_context_is_explicit() -> None:
    from dllm_parallel.training.block_diffusion_trainer import (
        _gradient_accumulation_context,
        _microbatch_no_sync_context,
    )

    class Module:
        def __init__(self) -> None:
            self.calls = 0
            self.active = 0

        def no_sync(self):
            module = self

            class Context:
                def __enter__(self):
                    module.calls += 1
                    module.active += 1

                def __exit__(self, exc_type, exc, tb):
                    module.active -= 1

            return Context()

    module = Module()
    with _microbatch_no_sync_context(
        module,
        optimizer_backend="fsdp",
        enabled=False,
    ):
        pass
    assert module.calls == 0
    with _microbatch_no_sync_context(
        module,
        optimizer_backend="fsdp",
        enabled=True,
    ):
        assert module.active == 1
    assert module.calls == 1
    assert module.active == 0

    with _gradient_accumulation_context(
        module,
        optimizer_backend="fsdp",
        accumulation_steps=2,
    ):
        pass


def test_hf_trainer_uses_deepspeed_coalesced_gradient_reduction() -> None:
    from dllm_parallel.training.block_diffusion_trainer import (
        _gradient_accumulation_context,
    )

    class Engine:
        def __init__(self) -> None:
            self.calls = 0
            self.active = 0

        def coalesce_grad_reduction(self):
            engine = self

            class Context:
                def __enter__(self):
                    engine.calls += 1
                    engine.active += 1

                def __exit__(self, exc_type, exc, tb):
                    engine.active -= 1

            return Context()

    engine = Engine()
    with _gradient_accumulation_context(
        engine,
        optimizer_backend="deepspeed_zero2",
        accumulation_steps=4,
    ):
        assert engine.active == 1
    assert engine.calls == 1
    assert engine.active == 0


def test_hf_trainer_rejects_deepspeed_without_coalesced_gradient_reduction() -> None:
    from dllm_parallel.training.block_diffusion_trainer import (
        _gradient_accumulation_context,
    )

    with pytest.raises(RuntimeError, match="DeepSpeed >= 0.19.2"):
        _gradient_accumulation_context(
            object(),
            optimizer_backend="deepspeed_zero2",
            accumulation_steps=2,
        )


def test_hf_trainer_uses_fsdp2_gradient_sync_control() -> None:
    from dllm_parallel.training.block_diffusion_trainer import (
        _microbatch_no_sync_context,
    )

    class Module:
        def __init__(self) -> None:
            self.values: list[bool] = []

        def set_requires_gradient_sync(self, enabled: bool) -> None:
            self.values.append(enabled)

    module = Module()
    with _microbatch_no_sync_context(
        module,
        optimizer_backend="fsdp2",
        enabled=True,
    ):
        assert module.values == [False]
    assert module.values == [False, True]


def test_diffusion_gemma_no_weight_config_planning() -> None:
    spec = summarize_config(
        "google/diffusiongemma-26B-A4B-it",
        {
            "model_type": "diffusion_gemma",
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
            "canvas_length": 256,
            "text_config": {
                "hidden_size": 2816,
                "num_hidden_layers": 30,
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
    schedule = build_schedule(
        spec,
        sequence_length=2048,
    )

    assert spec.family == "diffusion_gemma"
    assert spec.requires_remote_code is False
    assert spec.head_dim == 256
    assert spec.global_head_dim == 512
    assert spec.num_global_key_value_heads == 2
    assert spec.expert_intermediate_size == 704
    assert spec.block_size == 256
    assert schedule.attention_mode == "block_causal"
    assert schedule.objective is not None
    assert schedule.objective.kind == "block_denoising"
    assert schedule.objective.exact_block_parallel
    assert schedule.regions[-1].length == 256

    validate_parallel_spec(
        spec,
        ParallelSpec(
            data_parallel_size=1,
            context_parallel_size=1,
            tensor_parallel_size=4,
            expert_parallel_size=4,
        ),
    )
    executor = executor_for_family("diffusion_gemma")
    capabilities = executor.capabilities()
    assert capabilities.packed_block_diffusion
    assert capabilities.tensor_parallel
    assert capabilities.sequence_parallel
    supported = describe_supported_configs()
    assert "DiffusionGemma" in supported["parallel_axes"]["expert_parallel"]


def test_diffusion_gemma_rejects_inefficient_or_invalid_parallel_specs() -> None:
    spec = summarize_config(
        "google/diffusiongemma-26B-A4B-it",
        {
            "model_type": "diffusion_gemma",
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
            "canvas_length": 256,
            "text_config": {
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "num_experts": 128,
                "top_k_experts": 8,
            },
        },
    )
    with pytest.raises(ValueError, match="expert_parallel_size"):
        validate_parallel_spec(
            spec,
            ParallelSpec(expert_parallel_size=3),
        )
    with pytest.raises(ValueError, match="num_key_value_heads"):
        validate_parallel_spec(
            spec,
            ParallelSpec(tensor_parallel_size=16, expert_parallel_size=4),
        )
    validate_parallel_spec(
        spec,
        ParallelSpec(
            context_parallel_size=2,
            block_parallel_size=4,
            tensor_parallel_size=2,
            expert_parallel_size=4,
            kv_backend="ring",
        ),
    )


def test_nemotron_no_weight_config_planning() -> None:
    spec = summarize_config(
        "nvidia/Nemotron-Labs-Diffusion-8B",
        {
            "model_type": "nemotron_labs_diffusion",
            "architectures": ["NemotronLabsDiffusionModel"],
            "auto_map": {"AutoConfig": "configuration_nemotron_labs_diffusion.X"},
            "hidden_size": 4096,
            "num_hidden_layers": 34,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 131072,
            "intermediate_size": 14336,
            "max_position_embeddings": 262144,
            "block_size": 32,
            "dlm_paradigm": "bidirectional",
            "mask_token_id": 100,
            "attn_implementation": "sdpa",
            "use_cache": False,
        },
    )
    schedule = build_schedule(
        spec,
        sequence_length=1024,
    )

    assert spec.family == "nemotron_labs_diffusion"
    assert spec.requires_remote_code
    assert spec.head_dim == 128
    assert spec.intermediate_size == 14336
    assert spec.attn_implementation == "sdpa"
    assert spec.use_cache is False
    assert schedule.num_blocks == 32
    assert schedule.attention_mode == "block_causal"
    assert schedule.objective is not None
    assert schedule.objective.name == STANDARD_BLOCK_DIFFUSION_OBJECTIVE
    assert schedule.objective.mask_token_id == 100
    assert schedule.objective.supports_fused_cp_bp

    validate_parallel_spec(
        spec,
        ParallelSpec(
            data_parallel_size=2,
            context_parallel_size=2,
            block_parallel_size=2,
            tensor_parallel_size=4,
            kv_backend="ring",
        ),
    )


def test_nemotron_rejects_invalid_tp_and_kv_layouts() -> None:
    spec = summarize_config(
        "nvidia/Nemotron-Labs-Diffusion-8B",
        {
            "model_type": "nemotron_labs_diffusion",
            "architectures": ["NemotronLabsDiffusionModel"],
            "hidden_size": 4096,
            "num_hidden_layers": 34,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "block_size": 32,
            "dlm_paradigm": "bidirectional",
            "mask_token_id": 100,
        },
    )
    with pytest.raises(ValueError, match="num_key_value_heads"):
        validate_parallel_spec(spec, ParallelSpec(tensor_parallel_size=16))
    with pytest.raises(ValueError, match="ring K/V"):
        validate_parallel_spec(
            spec,
            ParallelSpec(tensor_parallel_size=4, kv_backend="ring"),
        )
    with pytest.raises(ValueError, match="sequence_parallel"):
        validate_parallel_spec(spec, ParallelSpec(sequence_parallel=True))


def test_nemotron_distributed_plan_uses_fused_block_context_topology() -> None:
    spec = summarize_config(
        "nvidia/Nemotron-Labs-Diffusion-3B",
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
        },
    )
    schedule = build_schedule(
        spec,
        sequence_length=1024,
    )
    plan = build_distributed_training_plan(
        spec,
        ParallelSpec(
            data_parallel_size=2,
            context_parallel_size=2,
            block_parallel_size=2,
            tensor_parallel_size=4,
            kv_backend="ring",
        ),
        schedule,
        world_size=16,
    )

    assert plan.topology is not None
    assert plan.topology.layout == "dp_x_fused_cp_bp_x_tp"
    assert plan.topology.block_schedule.kv_replication_factor == 1.0


def test_model_identity_and_standard_block_objective_are_separate() -> None:
    nemotron = summarize_config(
        "nvidia/Nemotron-Labs-Diffusion-3B",
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
        },
    )

    nemotron_schedule = build_schedule(
        nemotron,
        sequence_length=1024,
    )

    assert nemotron.family == "nemotron_labs_diffusion"
    assert nemotron_schedule.objective is not None
    assert nemotron_schedule.objective.name == STANDARD_BLOCK_DIFFUSION_OBJECTIVE
    assert nemotron_schedule.objective.supports_fused_cp_bp


def test_unknown_block_diffusion_config_requires_explicit_backbone() -> None:
    with pytest.raises(ValueError, match="supported model_type"):
        summarize_config(
            "org/custom-block-diffusion-lm",
            {
                "model_type": "custom_diffusion_lm",
                "architectures": ["CustomForBlockDiffusion"],
                "hidden_size": 2048,
                "num_hidden_layers": 20,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 65536,
                "max_position_embeddings": 8192,
                "block_size": 64,
                "dlm_paradigm": "block_denoising",
                "mask_token_id": 9,
            },
        )


def test_resolver_rejects_unsupported_pipeline_parallel_for_nemotron() -> None:
    spec = summarize_config(
        "nvidia/Nemotron-Labs-Diffusion-3B",
        {
            "model_type": "nemotron_labs_diffusion",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "block_size": 32,
        },
    )

    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        validate_parallel_spec(
            spec,
            ParallelSpec(pipeline_parallel_size=2),
        )


def test_topology_represents_expert_parallel_as_a_model_axis() -> None:
    gemma = summarize_config(
        "google/diffusiongemma-26B-A4B-it",
        {
            "model_type": "diffusion_gemma",
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
            "canvas_length": 256,
            "text_config": {
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "num_experts": 128,
                "top_k_experts": 8,
            },
        },
    )
    parallel = ParallelSpec(expert_parallel_size=4)
    validate_parallel_spec(gemma, parallel)
    schedule = build_schedule(gemma, sequence_length=512)
    plan = build_distributed_training_plan(
        gemma,
        parallel,
        schedule,
        world_size=4,
    )

    assert plan.topology is not None
    assert plan.parallel.model_parallel_size == 4
    assert plan.topology.expert_parallel_size == 4


def test_resolver_rejects_world_size_mismatch_across_fused_local_axis() -> None:
    spec = summarize_config(
        "google/diffusiongemma-26B-A4B-it",
        {
            "model_type": "diffusion_gemma",
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
            "canvas_length": 256,
            "text_config": {
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "num_experts": 128,
                "top_k_experts": 8,
            },
        },
    )
    schedule = build_schedule(spec, sequence_length=512)

    with pytest.raises(ValueError, match="data_parallel_size \\* model_parallel_size"):
        build_distributed_training_plan(
            spec,
            ParallelSpec(tensor_parallel_size=2, expert_parallel_size=2),
            schedule,
            world_size=2,
        )


def test_block_diffusion_loss_region_scales_before_dp_average() -> None:
    runtime = ParallelRuntime(
        enabled=True,
        active_block_mode="dual_end",
        rank=0,
        world_size=2,
        data_parallel_rank=0,
        local_parallel_rank=0,
        context_parallel_rank=0,
        block_parallel_rank=0,
        context_block_parallel_group_ranks=[0, 1],
        data_parallel_group_ranks=[0],
    )
    region = active_block_loss_region(
        attention_mask=torch.ones(1, 8),
        block_size=1,
        runtime=runtime,
    )
    token_losses = torch.arange(1, 9, dtype=torch.float32).unsqueeze(0)
    local = reduce_token_losses(token_losses, region)

    assert region.scale_before_dp_average == 2.0
    assert torch.allclose(local, torch.tensor((1 + 4 + 5 + 8) / 8 * 2.0))


def test_standard_block_diffusion_loss_region_is_model_family_neutral() -> None:
    runtime = ParallelRuntime(
        enabled=True,
        active_block_mode="dual_end",
        rank=1,
        world_size=2,
        data_parallel_rank=0,
        local_parallel_rank=1,
        context_parallel_rank=1,
        block_parallel_rank=1,
        context_block_parallel_group_ranks=[0, 1],
        data_parallel_group_ranks=[0],
    )
    region = active_block_loss_region(
        attention_mask=torch.ones(1, 8),
        block_size=1,
        runtime=runtime,
    )

    assert region.token_mask.tolist() == [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]
    assert region.denominator.equal(torch.tensor(8.0))
    assert region.reduction == "token_count"
    assert region.scale_before_dp_average == 2.0


def test_standard_block_diffusion_loss_region_honors_1d_attention_mask() -> None:
    runtime = ParallelRuntime(
        enabled=True,
        active_block_mode="dual_end",
        rank=1,
        world_size=2,
        data_parallel_rank=0,
        local_parallel_rank=1,
        context_parallel_rank=1,
        block_parallel_rank=1,
        context_block_parallel_group_ranks=[0, 1],
        data_parallel_group_ranks=[0],
    )
    attention_mask = torch.tensor([1, 1, 0, 1, 1, 1, 1, 0])
    region = active_block_loss_region(
        attention_mask=attention_mask,
        block_size=1,
        runtime=runtime,
    )

    assert region.token_mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert region.denominator.equal(torch.tensor(6.0))
