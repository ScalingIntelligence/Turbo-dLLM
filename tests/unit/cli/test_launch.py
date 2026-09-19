from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dllm_parallel.cli import launch
from dllm_parallel.cli.main import run


@pytest.fixture(autouse=True)
def _runtime_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launch,
        "inspect_training_runtime",
        lambda *args, **kwargs: SimpleNamespace(require_ready=lambda: None),
    )


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(
        """
model:
  id: test/model
  family: causal_lm
  seq_len: 16
  dtype: bf16
  mask_token_id: 99
objective:
  name: fast_dllm_v2
  block_size: 4
  noise_schedule: linear_mask
  loss_weighting: unit
data:
  input_mode: random
training:
  steps: 1
topology:
  context_parallel_size: 1
  block_parallel_size: 1
  tensor_parallel_size: 1
kernel:
  runtime_jit: false
optimizer:
  backend: torch_adamw
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_builds_installed_single_process_command(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = launch.build_launch_command(
        config=config,
        trainer_args=("--steps", "2"),
        nproc_per_node=1,
    )

    assert command == [
        sys.executable,
        "-m",
        "dllm_parallel.training",
        "--config",
        str(config),
        "--steps",
        "2",
    ]


def test_builds_torchrun_command_with_rendezvous_before_module(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = launch.build_launch_command(
        config=config,
        trainer_args=(),
        nproc_per_node=8,
        nnodes=2,
        node_rank=1,
        master_addr="trainer-0",
        master_port=29600,
        rdzv_backend="c10d",
        rdzv_endpoint="trainer-0:29600",
        rdzv_id="job-7",
        max_restarts=2,
        monitor_interval=3.5,
    )

    module_index = command.index("--module")
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc-per-node=8" in command
    assert "--rdzv-endpoint=trainer-0:29600" in command[:module_index]
    assert command[module_index:] == [
        "--module",
        "dllm_parallel.training",
        "--config",
        str(config),
    ]


def test_single_process_elastic_options_still_use_torchrun(tmp_path: Path) -> None:
    config = _config(tmp_path)

    command = launch.build_launch_command(
        config=config,
        trainer_args=(),
        max_restarts=1,
    )

    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--max-restarts=1" in command


def test_elastic_node_range_is_forwarded_to_torchrun(tmp_path: Path) -> None:
    config = _config(tmp_path)

    command = launch.build_launch_command(
        config=config,
        trainer_args=(),
        nnodes="1:4",
        rdzv_backend="c10d",
        rdzv_endpoint="trainer-0:29600",
        rdzv_id="elastic-job",
    )

    assert "--nnodes=1:4" in command
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]


def test_elastic_node_range_requires_rendezvous(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="elastic nnodes range requires"):
        launch.build_launch_command(
            config=config,
            trainer_args=(),
            nnodes="1:4",
        )


def test_launch_propagates_environment_preflight_and_child_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []
    preflight: list[int] = []
    monkeypatch.setattr(
        launch,
        "require_idle_assigned_gpus",
        lambda *, local_processes: preflight.append(local_processes),
    )
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda command, **kwargs: (
            calls.append((list(command), dict(kwargs["env"])))
            or SimpleNamespace(returncode=17)
        ),
    )

    result = launch.run_launch(
        [
            "--config",
            str(config),
            "--nproc-per-node",
            "2",
            "--gpu-preflight",
            "idle",
            "--run-dir",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert result == 17
    assert preflight == [2]
    assert len(calls) == 1
    _, environment = calls[0]
    assert environment["DLLM_RUN_DIR"] == str((tmp_path / "runs").resolve())
    assert environment["DLLM_CACHE_DIR"] == str((tmp_path / "cache").resolve())
    assert environment["TRITON_CACHE_DIR"].endswith("/cache/triton")
    assert environment["DLLM_RUNTIME_PREFLIGHT_DONE"] == "1"
    assert environment.get("PYTHONPATH") == os.environ.get("PYTHONPATH")


def test_launch_preflight_receives_required_local_gpu_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    observed: list[tuple[bool, int]] = []
    monkeypatch.setattr(
        launch,
        "inspect_training_runtime",
        lambda spec, *, distributed, local_processes: (
            observed.append((distributed, local_processes))
            or SimpleNamespace(require_ready=lambda: None)
        ),
    )
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0),
    )

    assert (
        launch.run_launch(
            ["--config", str(config), "--nproc-per-node", "4"]
        )
        == 0
    )
    assert observed == [(True, 4)]


def test_explicit_cache_dir_overrides_inherited_dependency_caches(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"

    environment = launch.build_launch_environment(
        run_dir=tmp_path / "run",
        cache_dir=cache,
        environ={
            "HF_HOME": "/inherited/hf",
            "TRITON_CACHE_DIR": "/inherited/triton",
            "XDG_CACHE_HOME": "/inherited/xdg",
        },
        force_cache_locations=True,
    )

    assert environment["XDG_CACHE_HOME"] == str(cache.resolve())
    assert environment["HF_HOME"] == str(cache.resolve() / "hf")
    assert environment["TRITON_CACHE_DIR"] == str(cache.resolve() / "triton")


def test_launch_environment_exposes_tools_installed_beside_python(
    tmp_path: Path,
) -> None:
    environment = launch.build_launch_environment(
        run_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        environ={"PATH": "/usr/bin"},
    )

    assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)


def test_dry_run_validates_but_does_not_spawn_or_create_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "not-created-run"
    cache_dir = tmp_path / "not-created-cache"
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry-run spawned a child"),
    )

    assert (
        launch.run_launch(
            [
                "--config",
                str(config),
                "--run-dir",
                str(run_dir),
                "--cache-dir",
                str(cache_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "DLLM_COMMAND=" in output
    assert "dllm_parallel.training" in output
    assert not run_dir.exists()
    assert not cache_dir.exists()


def test_packaged_recipe_is_materialized_for_child_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[bool, str]] = []

    def run(command, **kwargs):
        config = Path(command[command.index("--config") + 1])
        observed.append((config.is_file(), config.read_text(encoding="utf-8")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", run)

    assert launch.run_launch(["--recipe", "smoke/cpu-config"]) == 0
    assert observed and observed[0][0]
    assert "recipe_kind: smoke" in observed[0][1]


def test_signal_exit_is_mapped_to_shell_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=-15),
    )

    assert launch.run_launch(["--config", str(config)]) == 143


def test_launch_rejects_parallel_mesh_that_cannot_fit_world_size(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["topology"]["tensor_parallel_size"] = 2
    payload["topology"]["sequence_parallel"] = True
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="world_size"):
        launch.run_launch(["--config", str(config), "--dry-run"])


def test_launch_reports_malformed_yaml_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "broken.yaml"
    config.write_text("model: [unterminated\n", encoding="utf-8")

    assert run(["launch", "--config", str(config), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "invalid RunSpec YAML" in error
    assert "Traceback" not in error


@pytest.mark.parametrize(
    ("arguments", "match"),
    (
        (["--config", "missing.yaml", "--nproc-per-node", "0"], "positive"),
        (["--config", "missing.yaml", "--nnodes", "0"], "positive"),
        (
            ["--config", "missing.yaml", "--node-rank", "2", "--nnodes", "2"],
            "node-rank",
        ),
    ),
)
def test_invalid_launch_topology_fails_before_config_access(
    arguments: list[str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        launch.run_launch(arguments)
