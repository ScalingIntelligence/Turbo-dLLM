from __future__ import annotations

from types import SimpleNamespace

import pytest

from dllm_parallel.core.kernels import runtime


def test_eager_preflight_verifies_every_declared_native_kernel(monkeypatch) -> None:
    specs = (
        SimpleNamespace(
            name="first",
            package_module="package.first",
            symbols=("run",),
            source_paths=lambda: ("first.cu",),
        ),
        SimpleNamespace(
            name="second",
            package_module="package.second",
            symbols=("launch",),
            source_paths=lambda: ("second.cu",),
        ),
    )
    modules = {
        "package.first": SimpleNamespace(run=lambda: None),
        "package.second": SimpleNamespace(launch=lambda: None),
    }
    verified: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        "dllm_parallel.core.kernels._native_specs.NATIVE_KERNEL_SPECS", specs
    )
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        runtime,
        "_verify_native_extension",
        lambda **kwargs: verified.append(
            (kwargs["extension_name"], kwargs["required_symbols"])
        ),
    )

    result = runtime.verify_packaged_native_kernels()

    assert result == ("first", "second")
    assert verified == [("first", ("run",)), ("second", ("launch",))]


def test_eager_preflight_explains_incoherent_native_bundle(monkeypatch) -> None:
    spec = SimpleNamespace(
        name="missing",
        package_module="package.missing",
        symbols=(),
        source_paths=lambda: (),
    )
    monkeypatch.setattr(
        "dllm_parallel.core.kernels._native_specs.NATIVE_KERNEL_SPECS", (spec,)
    )
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )

    with pytest.raises(RuntimeError, match="coordinated GPU bundle"):
        runtime.verify_packaged_native_kernels()


def test_jit_toolchain_preflight_requires_nvcc(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runtime, "_cuda_home", lambda: tmp_path)
    monkeypatch.setattr(runtime, "_verify_jit_host_compiler", lambda: "/usr/bin/c++")
    monkeypatch.setattr(runtime, "_verify_jit_ninja", lambda: None)
    monkeypatch.setattr(runtime, "_verify_jit_cuda_headers", lambda cuda_home: None)
    monkeypatch.setattr(runtime, "_verify_jit_cache", lambda: None)

    with pytest.raises(RuntimeError, match="nvcc"):
        runtime.verify_runtime_jit_toolchain()

    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime.os, "access", lambda path, mode: True)
    assert runtime.verify_runtime_jit_toolchain() == str(nvcc)


def test_packaged_preflight_verifies_every_assigned_gpu_architecture(
    monkeypatch,
) -> None:
    spec = SimpleNamespace(
        name="example",
        package_module="dllm_parallel._C.example",
        symbols=("run",),
        source_paths=lambda: (),
    )
    module = SimpleNamespace(run=lambda: None)
    monkeypatch.setattr(
        "dllm_parallel.core.kernels._native_specs.NATIVE_KERNEL_SPECS",
        (spec,),
    )
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(runtime, "_verify_native_extension", lambda **kwargs: None)
    observed: list[tuple[int, int] | None] = []
    monkeypatch.setattr(
        runtime,
        "_verify_manifest_entry",
        lambda **kwargs: observed.append(kwargs["device_capability"]),
    )

    assert runtime.verify_packaged_native_kernels(
        device_capabilities=((9, 0), (8, 9), (9, 0))
    ) == ("example",)
    assert observed == [(8, 9), (9, 0)]


def test_jit_toolchain_preflight_requires_host_compiler(monkeypatch) -> None:
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda command: None)

    with pytest.raises(RuntimeError, match=r"host C\+\+ compiler"):
        runtime._verify_jit_host_compiler()
