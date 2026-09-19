from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from dllm_parallel.cli import launch
from dllm_parallel.cli.main import run
from dllm_parallel.core import diagnostics


def _spec(
    *,
    runtime_jit: bool = False,
    family: str = "causal_lm",
    data_path: str | None = None,
    evaluation_path: str | None = None,
    draft_vocab_path: str | None = None,
    checkpoint_path: str | None = None,
    optimizer: str = "torch_adamw",
):
    return SimpleNamespace(
        kernel=SimpleNamespace(runtime_jit=runtime_jit),
        model=SimpleNamespace(family=family, draft_vocab_path=draft_vocab_path),
        data=SimpleNamespace(
            input_mode="dataset" if data_path else "random",
            dataset_path=data_path,
            target_features=None,
            target_feature_path=None,
        ),
        topology=SimpleNamespace(
            tensor_parallel_size=1,
            expert_parallel_size=1,
        ),
        optimizer=SimpleNamespace(backend=optimizer),
        evaluation=SimpleNamespace(
            dataset_path=evaluation_path,
            batches=1 if evaluation_path else 0,
        ),
        checkpointing=SimpleNamespace(load_checkpoint_dir=checkpoint_path),
    )

def _stub_common_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostics,
        "_verify_cuda_runtime",
        lambda required_devices: f"CUDA 12.8, {required_devices} sm90 GPU(s)",
    )
    monkeypatch.setattr(
        diagnostics,
        "_verify_packaged_kernels",
        lambda required_devices: "3 packaged kernels",
    )
    monkeypatch.setattr(diagnostics, "_verify_fa4", lambda: "flash-attn-4 ready")
    monkeypatch.setattr(diagnostics, "_verify_fa3", lambda: "flash-attn-3 ready")


def test_training_readiness_uses_packaged_or_jit_path(monkeypatch) -> None:
    _stub_common_checks(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        diagnostics,
        "_verify_packaged_kernels",
        lambda required_devices: calls.append("packaged") or "ready",
    )
    monkeypatch.setattr(
        diagnostics,
        "_verify_jit_toolchain",
        lambda: calls.append("jit") or "ready",
    )

    diagnostics.inspect_training_runtime(_spec(runtime_jit=False)).require_ready()
    diagnostics.inspect_training_runtime(_spec(runtime_jit=True)).require_ready()

    assert calls == ["packaged", "jit"]


def test_training_readiness_reports_all_blockers_with_remediation(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostics,
        "_verify_cuda_runtime",
        lambda required_devices: (_ for _ in ()).throw(
            RuntimeError("no visible CUDA device")
        ),
    )
    monkeypatch.setattr(
        diagnostics,
        "_verify_packaged_kernels",
        lambda required_devices: (_ for _ in ()).throw(
            RuntimeError("source hash mismatch")
        ),
    )
    monkeypatch.setattr(diagnostics, "_verify_fa4", lambda: "ready")
    monkeypatch.setattr(diagnostics, "_verify_fa3", lambda: "ready")

    report = diagnostics.inspect_training_runtime(_spec())

    assert report.ready is False
    assert {check.name for check in report.failures} == {"cuda", "native_kernels"}
    with pytest.raises(RuntimeError, match="dllm bundle install"):
        report.require_ready()


def test_training_readiness_rejects_missing_prepared_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    _stub_common_checks(monkeypatch)

    report = diagnostics.inspect_training_runtime(
        _spec(data_path=str(tmp_path / "missing"))
    )

    assert report.ready is False
    assert any(check.name == "dataset" for check in report.failures)


def test_distributed_auto_optimizer_checks_deepspeed(monkeypatch) -> None:
    _stub_common_checks(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        diagnostics,
        "_verify_deepspeed",
        lambda: calls.append("deepspeed") or "ready",
    )

    diagnostics.inspect_training_runtime(
        _spec(optimizer="auto"), distributed=True
    ).require_ready()

    assert calls == ["deepspeed"]


def test_training_readiness_checks_every_required_local_gpu(monkeypatch) -> None:
    _stub_common_checks(monkeypatch)
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        diagnostics,
        "_verify_cuda_runtime",
        lambda count: calls.append(("cuda", count)) or "ready",
    )
    monkeypatch.setattr(
        diagnostics,
        "_verify_packaged_kernels",
        lambda count: calls.append(("native", count)) or "ready",
    )

    diagnostics.inspect_training_runtime(_spec(), local_processes=4).require_ready()

    assert calls == [("cuda", 4), ("native", 4)]


def test_cuda_readiness_rejects_too_few_visible_devices(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.10.0",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_capability=lambda index: (9, 0),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="requires 2 local GPU"):
        diagnostics._verify_cuda_runtime(2)


def test_normal_training_readiness_always_checks_fa3(monkeypatch) -> None:
    _stub_common_checks(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        diagnostics,
        "_verify_fa3",
        lambda: calls.append("fa3") or "ready",
    )

    diagnostics.inspect_training_runtime(_spec()).require_ready()

    assert calls == ["fa3"]


def test_full_bundle_checks_common_gpu_runtime_imports(monkeypatch) -> None:
    _stub_common_checks(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        diagnostics,
        "_verify_transformer_engine",
        lambda: calls.append("te") or "ready",
    )
    monkeypatch.setattr(
        diagnostics,
        "_verify_deepspeed",
        lambda: calls.append("deepspeed") or "ready",
    )

    diagnostics.inspect_training_runtime(full_bundle=True).require_ready()

    assert calls == ["te", "deepspeed"]


def test_training_readiness_checks_all_configured_local_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    _stub_common_checks(monkeypatch)

    report = diagnostics.inspect_training_runtime(
        _spec(
            evaluation_path=str(tmp_path / "evaluation"),
            draft_vocab_path=str(tmp_path / "draft-vocab"),
            checkpoint_path=str(tmp_path / "checkpoint"),
        )
    )

    assert {check.name for check in report.failures} == {
        "evaluation_dataset",
        "draft_vocab",
        "load_checkpoint",
    }


def test_doctor_recipe_reports_training_failure_as_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure = diagnostics.RuntimeReadinessReport(
        (
            diagnostics.RuntimeCheck(
                name="native_kernels",
                status="fail",
                detail="source hash mismatch",
                remediation="reinstall the bundle",
            ),
        )
    )
    monkeypatch.setattr(
        diagnostics,
        "inspect_training_runtime",
        lambda *args, **kwargs: failure,
    )

    assert run(["doctor", "--recipe", "smoke/cpu-config", "--json"]) == 1
    assert '"ready": false' in capsys.readouterr().out


def test_launch_fails_readiness_before_directories_or_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        launch,
        "inspect_training_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            require_ready=lambda: (_ for _ in ()).throw(RuntimeError("not ready"))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("readiness failure spawned workers"),
    )

    with pytest.raises(RuntimeError, match="not ready"):
        launch.run_launch(
            [
                "--recipe",
                "smoke/cpu-config",
                "--run-dir",
                str(run_dir),
                "--cache-dir",
                str(cache_dir),
            ]
        )

    assert not run_dir.exists()
    assert not cache_dir.exists()


def test_launch_runtime_preflight_has_explicit_expert_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launch,
        "inspect_training_runtime",
        lambda *args, **kwargs: pytest.fail("disabled readiness check ran"),
        raising=False,
    )
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    assert (
        launch.run_launch(
            [
                "--recipe",
                "smoke/cpu-config",
                "--runtime-preflight",
                "off",
            ]
        )
        == 0
    )
