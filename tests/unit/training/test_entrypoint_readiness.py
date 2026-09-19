from __future__ import annotations

from types import SimpleNamespace

from dllm_parallel.training import entrypoint


def test_direct_training_entrypoint_runs_readiness_once(monkeypatch) -> None:
    spec = SimpleNamespace()
    calls: list[tuple[object, bool, int]] = []
    monkeypatch.delenv("DLLM_RUNTIME_PREFLIGHT", raising=False)
    monkeypatch.delenv("DLLM_RUNTIME_PREFLIGHT_DONE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(
        "dllm_parallel.core.diagnostics.inspect_training_runtime",
        lambda observed, *, distributed, local_processes: (
            calls.append((observed, distributed, local_processes))
            or SimpleNamespace(require_ready=lambda: None)
        ),
    )

    entrypoint._preflight_training_runtime(spec)

    assert calls == [(spec, False, 1)]


def test_training_entrypoint_skips_preflight_already_done_by_launcher(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DLLM_RUNTIME_PREFLIGHT_DONE", "1")
    monkeypatch.setattr(
        "dllm_parallel.core.diagnostics.inspect_training_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate preflight")
        ),
    )

    entrypoint._preflight_training_runtime(SimpleNamespace())
