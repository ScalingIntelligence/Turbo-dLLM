from __future__ import annotations

from types import SimpleNamespace

from dllm_parallel.core.profiling import release_gates


def test_release_runtime_preflight_verifies_turbo_dllm_native_kernels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dllm_parallel.core.kernels.runtime.verify_packaged_native_kernels",
        lambda: ("one", "two", "three"),
    )
    monkeypatch.setattr(
        "dllm_parallel.core.attention.flex.verify_flex_attention_runtime",
        lambda: SimpleNamespace(to_log_dict=lambda: {"backend": "test"}),
    )
    monkeypatch.setattr(
        "dllm_parallel.core.attention.fa3.verify_flash_attention_kernels",
        lambda: SimpleNamespace(to_log_dict=lambda: {"package": "test"}),
    )

    result = release_gates.run_runtime_preflight()

    assert result["native_kernels"] == ["one", "two", "three"]
