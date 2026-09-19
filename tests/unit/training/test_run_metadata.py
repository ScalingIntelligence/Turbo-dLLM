from __future__ import annotations

import dllm_parallel.training.run_metadata as run_metadata_module
from dllm_parallel.training.run_metadata import (
    RunContext,
    device_metadata,
    package_versions,
    run_metadata,
)


def test_run_metadata_has_expected_keys() -> None:
    metadata = run_metadata()
    for key in (
        "git_sha",
        "git_branch",
        "git_dirty",
        "python_version",
        "platform",
        "packages",
        "device",
        "launch_environment",
    ):
        assert key in metadata


def test_run_metadata_uses_baked_source_when_git_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(run_metadata_module, "_git_output", lambda _args: None)
    monkeypatch.setattr(run_metadata_module, "_git_dirty", lambda: None)
    monkeypatch.setenv("DLLM_SOURCE_GIT_SHA", "deadbeef")
    monkeypatch.setenv("DLLM_SOURCE_GIT_BRANCH", "paper-profile")
    monkeypatch.setenv("DLLM_SOURCE_GIT_DIRTY", "false")

    payload = run_metadata()

    assert payload["git_sha"] == "deadbeef"
    assert payload["git_branch"] == "paper-profile"
    assert payload["git_dirty"] is False


def test_package_versions_includes_self_entry() -> None:
    versions = package_versions()
    assert "turbo-dllm" in versions
    assert "torch" in versions


def test_device_metadata_reports_cuda_flag() -> None:
    assert "cuda_available" in device_metadata()


def test_run_context_records_resolved_spec_and_policy() -> None:
    context = RunContext.create(resolved_spec={"model": {"id": "unit"}})
    context.with_runtime(
        runtime=None,
        kernel_metadata={"runtime_jit": False},
        optimizer_policy={"backend": "deepspeed_zero2"},
        checkpoint_policy={"save_checkpoint_interval": 10},
        profiler_policy={"phase_timing": False},
    )

    payload = context.to_log_dict()
    assert payload["resolved_spec"]["model"]["id"] == "unit"
    assert payload["kernel_metadata"]["runtime_jit"] is False
    assert payload["optimizer_policy"]["backend"] == "deepspeed_zero2"
