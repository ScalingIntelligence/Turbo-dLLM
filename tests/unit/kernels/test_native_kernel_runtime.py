from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from dllm_parallel.core.kernels import runtime
from dllm_parallel.core.kernels.build import _resolve_build_package_version


def _source_hash(name: str, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")
    return digest.hexdigest()


def test_native_kernel_build_accepts_explicit_preinstall_package_version() -> None:
    assert _resolve_build_package_version(" 0.1.0 ") == "0.1.0"
    with pytest.raises(ValueError, match="must not be empty"):
        _resolve_build_package_version("   ")


def test_native_kernel_build_uses_installed_package_version(monkeypatch) -> None:
    monkeypatch.setattr(
        "dllm_parallel.core.kernels.build.installed_package_version",
        lambda: "0.2.0",
    )
    assert _resolve_build_package_version(None) == "0.2.0"


def test_native_kernel_runtime_verifies_binary_hash_and_abi(tmp_path, monkeypatch) -> None:
    source = tmp_path / "kernel.py"
    source.write_text("# kernel\n", encoding="utf-8")
    binary = tmp_path / "kernel.so"
    binary.write_bytes(b"binary")
    manifest = tmp_path / "native_kernels.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "dllm_parallel.native_kernels.v1",
                "package_version": "0.1.0",
                "python_extension_suffix": ".so",
                "kernels": {
                    "unit_kernel": {
                        "binary": "kernel.so",
                        "binary_hash": hashlib.sha256(b"binary").hexdigest(),
                        "cuda_arch_list": "8.0;9.0",
                        "minimum_device_capability": [8, 0],
                        "supported_device_capabilities": [[8, 0], [9, 0]],
                        "ptx_device_capabilities": [],
                        "python_extension_suffix": ".so",
                        "required_symbols": ["run"],
                        "source_files": [str(source)],
                        "source_hash": _source_hash("kernel.py", b"# kernel\n"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "native_kernel_manifest_path", lambda: manifest)
    monkeypatch.setattr(runtime, "installed_package_version", lambda: "0.1.0")
    monkeypatch.setattr(runtime.sysconfig, "get_config_var", lambda name: ".so")

    module = SimpleNamespace(__file__=str(binary), run=lambda: None)

    runtime._verify_manifest_entry(
        module=module,
        extension_name="unit_kernel",
        source_files=(source,),
        device_capability=(9, 0),
    )

    binary.write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="binary hash mismatch"):
        runtime._verify_manifest_entry(
            module=module,
            extension_name="unit_kernel",
            source_files=(source,),
            device_capability=(9, 0),
        )


def test_native_kernel_runtime_rejects_uncompiled_device(tmp_path, monkeypatch) -> None:
    source = tmp_path / "kernel.py"
    source.write_text("# kernel\n", encoding="utf-8")
    binary = tmp_path / "kernel.so"
    binary.write_bytes(b"binary")
    manifest = tmp_path / "native_kernels.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "dllm_parallel.native_kernels.v1",
                "package_version": "0.1.0",
                "kernels": {
                    "unit_kernel": {
                        "binary": "kernel.so",
                        "binary_hash": hashlib.sha256(b"binary").hexdigest(),
                        "python_extension_suffix": ".so",
                        "source_hash": _source_hash("kernel.py", b"# kernel\n"),
                        "supported_device_capabilities": [[9, 0]],
                        "ptx_device_capabilities": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "native_kernel_manifest_path", lambda: manifest)
    monkeypatch.setattr(runtime, "installed_package_version", lambda: "0.1.0")
    monkeypatch.setattr(runtime.sysconfig, "get_config_var", lambda name: ".so")

    with pytest.raises(RuntimeError, match="was built for"):
        runtime._verify_manifest_entry(
            module=SimpleNamespace(__file__=str(binary)),
            extension_name="unit_kernel",
            source_files=(source,),
            device_capability=(8, 9),
        )


def test_native_kernel_runtime_rejects_package_version_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    binary = tmp_path / "kernel.so"
    binary.write_bytes(b"binary")
    manifest = tmp_path / "native_kernels.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "dllm_parallel.native_kernels.v1",
                "package_version": "0.0.9",
                "kernels": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "native_kernel_manifest_path", lambda: manifest)
    monkeypatch.setattr(runtime, "installed_package_version", lambda: "0.1.0")

    with pytest.raises(RuntimeError, match="package version mismatch"):
        runtime._verify_manifest_entry(
            module=SimpleNamespace(__file__=str(binary)),
            extension_name="unit_kernel",
            source_files=(),
            device_capability=None,
        )


def test_runtime_jit_policy_always_builds_instead_of_loading_packaged_binary(
    monkeypatch,
) -> None:
    built = SimpleNamespace(run=lambda: None)
    verified: list[bool] = []
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: pytest.fail(f"runtime JIT imported packaged module {name}"),
    )
    monkeypatch.setattr(
        runtime,
        "_verify_native_extension",
        lambda **kwargs: verified.append(kwargs["require_manifest"]),
    )
    runtime.configure_kernel_runtime(allow_runtime_jit=True)
    try:
        result = runtime.load_native_extension(
            package_module="package.kernel",
            extension_name="kernel",
            jit_builder=lambda: built,
            required_symbols=("run",),
        )
    finally:
        runtime.configure_kernel_runtime(allow_runtime_jit=False)

    assert result is built
    assert verified == [False]
