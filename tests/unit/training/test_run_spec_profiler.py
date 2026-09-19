from __future__ import annotations

from types import SimpleNamespace

import pytest

from dllm_parallel.training import block_diffusion_trainer
from dllm_parallel.training.run_spec import RunSpec


def test_training_duration_limit_is_typed_and_positive() -> None:
    spec = RunSpec.from_mapping({"training": {"max_duration_seconds": 28_800.0}})
    assert spec.training.max_duration_seconds == 28_800.0

    with pytest.raises(ValueError, match="training.max_duration_seconds"):
        RunSpec.from_mapping({"training": {"max_duration_seconds": 0.0}})


def test_run_spec_accepts_trajectory_optimizer_step_dataset() -> None:
    spec = RunSpec.from_mapping(
        {
            "data": {
                "input_mode": "dataset",
                "dataset_path": "/datasets/agent-sft",
                "optimizer_step_unit": "trajectory",
            }
        }
    )

    assert spec.data.optimizer_step_unit == "trajectory"


def test_run_spec_accepts_turn_balanced_trajectory_loss() -> None:
    spec = RunSpec.from_mapping(
        {
            "data": {
                "input_mode": "dataset",
                "optimizer_step_unit": "trajectory",
                "trajectory_loss_reduction": "turn_mean",
                "trajectory_terminal_turn_weight": 4.0,
            }
        }
    )

    assert spec.data.trajectory_loss_reduction == "turn_mean"
    assert spec.data.trajectory_terminal_turn_weight == 4.0


def test_run_spec_rejects_turn_balancing_outside_trajectory_steps() -> None:
    with pytest.raises(ValueError, match="requires.*optimizer_step_unit=trajectory"):
        RunSpec.from_mapping(
            {
                "data": {
                    "input_mode": "dataset",
                    "trajectory_loss_reduction": "turn_mean",
                }
            }
        )


@pytest.mark.parametrize("weight", [0.0, -1.0, float("inf"), float("nan")])
def test_run_spec_rejects_invalid_terminal_turn_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="terminal_turn_weight.*positive"):
        RunSpec.from_mapping(
            {
                "data": {
                    "input_mode": "dataset",
                    "optimizer_step_unit": "trajectory",
                    "trajectory_terminal_turn_weight": weight,
                }
            }
        )


def test_run_spec_rejects_trajectory_steps_for_non_dataset_input() -> None:
    with pytest.raises(ValueError, match="requires data.input_mode=dataset"):
        RunSpec.from_mapping(
            {"data": {"input_mode": "random", "optimizer_step_unit": "trajectory"}}
        )


def test_data_minimum_sequence_length_is_typed_and_bounded() -> None:
    spec = RunSpec.from_mapping(
        {
            "model": {"seq_len": 8192},
            "data": {"minimum_sequence_length": 4096},
        }
    )
    assert spec.data.minimum_sequence_length == 4096

    with pytest.raises(ValueError, match="cannot exceed model.seq_len"):
        RunSpec.from_mapping(
            {
                "model": {"seq_len": 4096},
                "data": {"minimum_sequence_length": 8192},
            }
        )


def test_evaluation_requires_a_dataset_and_positive_batch_count() -> None:
    spec = RunSpec.from_mapping(
        {
            "evaluation": {
                "dataset_path": "/datasets/validation",
                "batches": 8,
                "seed": 17,
            }
        }
    )
    assert spec.evaluation.batches == 8
    assert spec.evaluation.seed == 17

    with pytest.raises(ValueError, match="requires evaluation.dataset_path"):
        RunSpec.from_mapping({"evaluation": {"batches": 1}})
    with pytest.raises(ValueError, match="requires positive evaluation.batches"):
        RunSpec.from_mapping({"evaluation": {"dataset_path": "/datasets/validation"}})


def test_system_trace_window_must_fit_training_steps() -> None:
    with pytest.raises(ValueError, match="system-trace window"):
        RunSpec.from_mapping(
            {
                "training": {"steps": 3},
                "profiler": {
                    "system_trace": True,
                    "system_trace_dir": "/tmp/trace",
                    "system_trace_start_step": 3,
                    "system_trace_steps": 2,
                },
            }
        )


def test_system_trace_fields_are_generated_cli_overrides() -> None:
    spec = RunSpec.default().with_overrides(
        {
            "system_trace": True,
            "system_trace_dir": "/tmp/trace",
            "system_trace_start_step": 2,
            "system_trace_steps": 2,
            "steps": 3,
        }
    )
    assert spec.profiler.system_trace
    assert spec.profiler.system_trace_start_step == 2
    assert spec.profiler.system_trace_steps == 2


def test_kineto_system_trace_requires_artifact_directory() -> None:
    with pytest.raises(ValueError, match="system_trace_dir"):
        RunSpec.from_mapping(
            {
                "training": {"steps": 1},
                "profiler": {"system_trace": True},
            }
        )


def test_nsys_system_trace_does_not_require_internal_artifact_directory() -> None:
    spec = RunSpec.from_mapping(
        {
            "training": {"steps": 1},
            "profiler": {
                "system_trace": True,
                "system_trace_backend": "nsys",
            },
        }
    )
    assert spec.profiler.system_trace_backend == "nsys"


def test_profile_measurement_excludes_warmup_and_system_trace_steps() -> None:
    assert hasattr(block_diffusion_trainer, "_profile_step_is_measured")
    profiler = SimpleNamespace(
        warmup_steps=1,
        system_trace=True,
        system_trace_start_step=3,
        system_trace_steps=2,
    )

    measured = block_diffusion_trainer._profile_step_is_measured
    assert not measured(1, profiler)
    assert measured(2, profiler)
    assert not measured(3, profiler)
    assert not measured(4, profiler)
    assert measured(5, profiler)

    profiler.system_trace = False
    assert measured(3, profiler)


def test_inter_node_cp_is_a_typed_topology_override() -> None:
    spec = RunSpec.default().with_overrides({"placement_policy": "inter_node_cp"})
    assert spec.topology.placement_policy == "inter_node_cp"

    with pytest.raises(ValueError, match="topology.placement_policy"):
        RunSpec.from_mapping({"topology": {"placement_policy": "silently_wrong"}})
