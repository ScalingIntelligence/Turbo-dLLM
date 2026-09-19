# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""FlashAttention-3 / BDLM kernel import and verification."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


_DISTRIBUTION_NAME = "bdlm-flash-attn-3"
_BDLM_VARIANT = "bdlm-cp-splitd-v3"
_BUILD_METADATA_FORMAT = "bdlm.flash_attn_3.build.v3"
_BUILD_METADATA_NAME = "build_metadata.json"
_CONFLICTING_DISTRIBUTIONS = ("flash-attn-3", "flash-attn")


@dataclass(frozen=True)
class FlashAttentionKernelMetadata:
    module: str
    path: str
    interface_path: str
    build_metadata_path: str
    package: str
    version: str
    build_id: str
    source_hash: str
    binary_hash: str
    torch_version: str
    torch_cuda_version: str
    cuda_toolkit_version: str
    cuda_architectures: tuple[str, ...]
    required_ops: tuple[str, ...]

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "path": self.path,
            "interface_path": self.interface_path,
            "build_metadata_path": self.build_metadata_path,
            "package": self.package,
            "version": self.version,
            "build_id": self.build_id,
            "source_hash": self.source_hash,
            "binary_hash": self.binary_hash,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_toolkit_version": self.cuda_toolkit_version,
            "cuda_architectures": list(self.cuda_architectures),
            "required_ops": list(self.required_ops),
        }


_fa3_flash_attn_func: Any | None = None
_bdlm_flash_attn_func: Any | None = None
_bdlm_flash_attn_module: Any | None = None


def require_bdlm_flash_attention() -> Any:
    global _bdlm_flash_attn_func
    if _bdlm_flash_attn_func is None:
        module = _import_module("flash_attn_interface")
        try:
            _bdlm_flash_attn_func = getattr(module, "bdlm_flash_attn_func")
        except AttributeError as exc:
            raise RuntimeError(
                "FlashAttention-3 is importable but does not expose "
                "bdlm_flash_attn_func. Install the BDLM-enabled Hopper build."
            ) from exc
    return _bdlm_flash_attn_func


def require_fa3_flash_attention() -> Any:
    global _fa3_flash_attn_func
    if _fa3_flash_attn_func is None:
        module = _import_module("flash_attn_interface")
        try:
            _fa3_flash_attn_func = getattr(module, "flash_attn_func")
        except AttributeError as exc:
            raise RuntimeError(
                "FlashAttention-3 is importable but does not expose flash_attn_func."
            ) from exc
    return _fa3_flash_attn_func


def require_bdlm_flash_attention_module(
    *,
    required_ops: tuple[str, ...] = (),
) -> Any:
    global _bdlm_flash_attn_module
    if _bdlm_flash_attn_module is None:
        _import_module("flash_attn_3._C")
        _import_module("flash_attn_3")
        _bdlm_flash_attn_module = torch.ops.flash_attn_3
    missing = [
        op_name
        for op_name in required_ops
        if not _has_torch_op(_bdlm_flash_attn_module, op_name)
    ]
    if missing:
        raise RuntimeError(
            "FlashAttention-3 extension is missing required operators: "
            f"{', '.join(missing)}. Install the matching BDLM FlashAttention build."
        )
    return _bdlm_flash_attn_module


def flash_attention_metadata() -> FlashAttentionKernelMetadata:
    interface_module = _import_module("flash_attn_interface")
    extension_module = _import_module("flash_attn_3._C")
    package_module = _import_module("flash_attn_3")
    return _verify_packaged_flash_attention(
        interface_module=interface_module,
        extension_module=extension_module,
        package_module=package_module,
    )


def verify_flash_attention_kernels() -> FlashAttentionKernelMetadata:
    require_fa3_flash_attention()
    bdlm_flash_attn_func = require_bdlm_flash_attention()
    flash_attn_3_gpu = require_bdlm_flash_attention_module(
        required_ops=_REQUIRED_BDLM_OPS,
    )
    _verify_bdlm_attention_abi(
        bdlm_flash_attn_func=bdlm_flash_attn_func,
        flash_attn_3_gpu=flash_attn_3_gpu,
        op_abi=_BDLM_OP_ABI,
    )
    return flash_attention_metadata()


def verify_dflash_attention_kernels() -> FlashAttentionKernelMetadata:
    metadata = verify_flash_attention_kernels()
    flash_attn_3_gpu = require_bdlm_flash_attention_module(
        required_ops=_REQUIRED_DFLASH_OPS,
    )
    _verify_operator_abi(flash_attn_3_gpu, _DFLASH_OP_ABI)
    return metadata


def _import_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(
            f"{name} is required for production CP/BP attention. Build and "
            "install the vendored Hopper package from this repository before launching "
            "training; public or runtime-JIT FlashAttention builds are not accepted."
        ) from exc


def _has_torch_op(namespace: Any, op_name: str) -> bool:
    try:
        getattr(namespace, op_name).default._schema
        return True
    except Exception:
        return False


def _verify_bdlm_attention_abi(
    *,
    bdlm_flash_attn_func: Any,
    flash_attn_3_gpu: Any,
    op_abi: Mapping[str, tuple[str, ...]],
) -> None:
    try:
        python_parameters = inspect.signature(bdlm_flash_attn_func).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "cannot inspect the packaged BDLM FlashAttention Python ABI"
        ) from exc
    python_suffix = tuple(python_parameters)[-_BDLM_PYTHON_ABI_SUFFIX_LENGTH:]
    if python_suffix != _BDLM_PYTHON_ABI_SUFFIX:
        raise RuntimeError(
            "packaged BDLM FlashAttention Python ABI suffix is stale or reordered: "
            f"{python_suffix} != {_BDLM_PYTHON_ABI_SUFFIX}"
        )

    _verify_operator_abi(flash_attn_3_gpu, op_abi)


def _verify_operator_abi(
    namespace: Any,
    op_abi: Mapping[str, tuple[str, ...]],
) -> None:
    for op_name, expected_arguments in op_abi.items():
        try:
            schema = getattr(namespace, op_name).default._schema
        except Exception as exc:
            raise RuntimeError(
                f"packaged flash_attn_3.{op_name} has no inspectable operator schema"
            ) from exc
        argument_names = tuple(str(argument.name) for argument in schema.arguments)
        if not _contains_contiguous(argument_names, expected_arguments):
            raise RuntimeError(
                f"packaged flash_attn_3.{op_name} ABI is stale or reordered: "
                f"expected contiguous arguments {expected_arguments}, observed "
                f"{argument_names}; reinstall the matching "
                "BDLM FlashAttention binary"
            )


def _contains_contiguous(
    values: tuple[str, ...],
    expected: tuple[str, ...],
) -> bool:
    width = len(expected)
    return any(values[index:index + width] == expected for index in range(len(values) - width + 1))


def _verify_packaged_flash_attention(
    *,
    interface_module: Any,
    extension_module: Any,
    package_module: Any,
) -> FlashAttentionKernelMetadata:
    interface_path = _required_module_path(interface_module, "flash_attn_interface")
    binary_path = _required_module_path(extension_module, "flash_attn_3._C")
    package_path = _required_module_path(package_module, "flash_attn_3")
    resolved_paths = (interface_path, binary_path, package_path)
    if any(_is_repo_checkout_flash_attention(path) for path in resolved_paths):
        raise RuntimeError(
            "production training requires an installed BDLM FlashAttention wheel; "
            f"a module resolved from the source checkout: {resolved_paths}"
        )
    if not any(
        binary_path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise RuntimeError(
            f"flash_attn_3._C is not a native Python extension: {binary_path}"
        )
    distribution = _required_distribution()
    _verify_package_identity(
        package_module,
        distribution_version=str(distribution.version),
    )
    metadata_path = binary_path.parent / _BUILD_METADATA_NAME
    _verify_distribution_files(
        distribution,
        interface_path=interface_path,
        binary_path=binary_path,
        package_path=package_path,
        metadata_path=metadata_path,
    )
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"BDLM FlashAttention build metadata is missing: {metadata_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"BDLM FlashAttention build metadata is invalid: {exc}"
        ) from exc

    device_capability = None
    if torch.cuda.is_available():
        device_capability = tuple(torch.cuda.get_device_capability())
    return _validate_build_metadata(
        payload,
        distribution_version=str(distribution.version),
        binary_path=binary_path,
        interface_path=interface_path,
        package_path=package_path,
        metadata_path=metadata_path,
        runtime_torch_version=str(torch.__version__),
        runtime_torch_cuda=str(torch.version.cuda),
        runtime_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        device_capability=device_capability,
    )


def _validate_build_metadata(
    payload: Mapping[str, Any],
    *,
    distribution_version: str,
    binary_path: Path,
    interface_path: Path,
    package_path: Path,
    metadata_path: Path,
    runtime_torch_version: str,
    runtime_torch_cuda: str,
    runtime_cxx11_abi: bool,
    device_capability: tuple[int, int] | None,
) -> FlashAttentionKernelMetadata:
    if payload.get("format") != _BUILD_METADATA_FORMAT:
        raise RuntimeError("installed FlashAttention has no recognized BDLM build metadata")
    if payload.get("variant") != _BDLM_VARIANT:
        raise RuntimeError(
            f"installed FlashAttention variant is not {_BDLM_VARIANT!r}"
        )
    if payload.get("module") != "flash_attn_3._C":
        raise RuntimeError("BDLM FlashAttention metadata names the wrong native module")
    if tuple(payload.get("required_ops") or ()) != _PACKAGED_ATTENTION_OPS:
        raise RuntimeError("BDLM FlashAttention metadata does not declare the required operators")

    distribution = _required_mapping(payload, "distribution")
    if _canonical_name(distribution.get("name")) != _canonical_name(_DISTRIBUTION_NAME):
        raise RuntimeError("flash_attn_3._C belongs to an unexpected distribution")
    if distribution.get("version") != distribution_version:
        raise RuntimeError(
            "BDLM FlashAttention wheel and build metadata versions do not match"
        )

    build_id = payload.get("build_id")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("build_id", None)
    expected_build_id = hashlib.sha256(
        json.dumps(unsigned_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if build_id != expected_build_id:
        raise RuntimeError("BDLM FlashAttention build metadata digest does not match")

    binary = _required_mapping(payload, "binary")
    if binary.get("path") != f"flash_attn_3/{binary_path.name}":
        raise RuntimeError("BDLM FlashAttention metadata names the wrong binary")
    extension_suffix = binary_path.name.removeprefix("_C")
    if binary.get("extension_suffix") != extension_suffix:
        raise RuntimeError("BDLM FlashAttention binary ABI suffix does not match metadata")
    binary_hash = _required_sha256(binary, "sha256", "binary")
    if _sha256(binary_path) != binary_hash:
        raise RuntimeError("BDLM FlashAttention native binary hash mismatch")

    python_interface = _required_mapping(payload, "python_interface")
    if python_interface.get("path") != interface_path.name:
        raise RuntimeError("BDLM FlashAttention metadata names the wrong Python interface")
    interface_hash = _required_sha256(python_interface, "sha256", "Python interface")
    if _sha256(interface_path) != interface_hash:
        raise RuntimeError("BDLM FlashAttention Python interface hash mismatch")

    _validate_python_package(
        _required_mapping(payload, "python_package"),
        package_root=package_path.parent,
    )

    source = _required_mapping(payload, "source")
    source_hash = _required_sha256(source, "sha256", "source")
    if not isinstance(source.get("file_count"), int) or source["file_count"] <= 0:
        raise RuntimeError("BDLM FlashAttention source metadata has no files")
    if not isinstance(source.get("revision"), str) or not source["revision"]:
        raise RuntimeError("BDLM FlashAttention source metadata has no revision")

    build = _required_mapping(payload, "build")
    build_torch = _required_string(build, "torch")
    build_torch_cuda = _required_string(build, "torch_cuda")
    cuda_toolkit = _required_string(build, "cuda_toolkit")
    if _major_minor(build_torch, "build torch") != _major_minor(
        runtime_torch_version, "runtime torch"
    ):
        raise RuntimeError(
            f"BDLM FlashAttention was built for torch {build_torch}, but runtime "
            f"torch is {runtime_torch_version}"
        )
    if _major_minor(build_torch_cuda, "build torch CUDA") != _major_minor(
        runtime_torch_cuda, "runtime torch CUDA"
    ):
        raise RuntimeError(
            f"BDLM FlashAttention was built for torch CUDA {build_torch_cuda}, "
            f"but runtime torch CUDA is {runtime_torch_cuda}"
        )
    if _major_minor(cuda_toolkit, "CUDA toolkit") < (12, 3):
        raise RuntimeError("BDLM FlashAttention requires a CUDA toolkit >= 12.3")
    if build.get("cxx11_abi") is not runtime_cxx11_abi:
        raise RuntimeError("BDLM FlashAttention C++ ABI does not match runtime torch")
    if build.get("flash_api_source") != "flash_api.cpp":
        raise RuntimeError("BDLM FlashAttention was not built with the complete operator API")
    feature_flags = _required_mapping(build, "feature_flags")
    if feature_flags.get("FLASHATTENTION_DISABLE_BACKWARD") is not False:
        raise RuntimeError("BDLM FlashAttention backward kernels were disabled at build time")

    architectures = _validate_architectures(
        build.get("cuda_architectures"),
        disable_sm8x=feature_flags.get("FLASHATTENTION_DISABLE_SM8x"),
        device_capability=device_capability,
    )
    return FlashAttentionKernelMetadata(
        module="flash_attn_3._C",
        path=str(binary_path),
        interface_path=str(interface_path),
        build_metadata_path=str(metadata_path),
        package=_DISTRIBUTION_NAME,
        version=distribution_version,
        build_id=str(build_id),
        source_hash=source_hash,
        binary_hash=binary_hash,
        torch_version=build_torch,
        torch_cuda_version=build_torch_cuda,
        cuda_toolkit_version=cuda_toolkit,
        cuda_architectures=architectures,
        required_ops=_PACKAGED_ATTENTION_OPS,
    )


def _validate_architectures(
    value: Any,
    *,
    disable_sm8x: Any,
    device_capability: tuple[int, int] | None,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("BDLM FlashAttention build metadata has no CUDA architectures")
    architectures = tuple(str(item) for item in value)
    if len(set(architectures)) != len(architectures):
        raise RuntimeError("BDLM FlashAttention build metadata repeats CUDA architectures")
    if any(re.fullmatch(r"sm_\d{2,3}a?", item) is None for item in architectures):
        raise RuntimeError("BDLM FlashAttention build metadata has invalid CUDA architectures")
    if not isinstance(disable_sm8x, bool):
        raise RuntimeError("BDLM FlashAttention metadata is missing the sm_80 build flag")
    expected_architectures = (
        ("sm_90a",)
        if disable_sm8x
        else ("sm_80", "sm_89", "sm_90a")
    )
    if architectures != expected_architectures:
        raise RuntimeError(
            "BDLM FlashAttention CUDA architecture metadata does not match the "
            f"build contract: {architectures} != {expected_architectures}"
        )
    if device_capability is not None and not any(
        _architecture_supports_device(item, device_capability) for item in architectures
    ):
        raise RuntimeError(
            "BDLM FlashAttention binary has no code for CUDA device capability "
            f"{device_capability}; packaged architectures are {architectures}"
        )
    return architectures


def _validate_python_package(
    payload: Mapping[str, Any],
    *,
    package_root: Path,
) -> None:
    expected_root = "flash_attn_3/bdlm_splitd"
    if payload.get("root") != expected_root:
        raise RuntimeError("BDLM Split-D metadata names the wrong Python package")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("BDLM Split-D metadata has no packaged source files")

    declared: dict[str, str] = {}
    for relative, digest in files.items():
        if not isinstance(relative, str) or not relative.startswith(expected_root + "/"):
            raise RuntimeError("BDLM Split-D metadata contains an invalid source path")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError("BDLM Split-D metadata contains an invalid source hash")
        declared[relative] = digest

    splitd_root = package_root / "bdlm_splitd"
    installed = {
        f"flash_attn_3/{path.relative_to(package_root).as_posix()}": path
        for path in splitd_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if set(installed) != set(declared):
        missing = sorted(set(declared) - set(installed))
        unexpected = sorted(set(installed) - set(declared))
        raise RuntimeError(
            "BDLM Split-D packaged source inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, path in installed.items():
        if _sha256(path) != declared[relative]:
            raise RuntimeError(f"BDLM Split-D packaged source hash mismatch: {relative}")

    package_hash = _required_sha256(payload, "sha256", "Split-D Python package")
    canonical = json.dumps(
        declared,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != package_hash:
        raise RuntimeError("BDLM Split-D package manifest digest does not match")


def _architecture_supports_device(
    architecture: str,
    device_capability: tuple[int, int],
) -> bool:
    match = re.fullmatch(r"sm_(\d{2,3})(a?)", architecture)
    if match is None:
        return False
    digits = match.group(1)
    target = (int(digits[:-1]), int(digits[-1]))
    return target == device_capability


def _verify_package_identity(
    package_module: Any,
    *,
    distribution_version: str,
) -> None:
    if getattr(package_module, "__bdlm_variant__", None) != _BDLM_VARIANT:
        raise RuntimeError("flash_attn_3 is not the repository's BDLM package")
    if getattr(package_module, "__build_metadata_format__", None) != _BUILD_METADATA_FORMAT:
        raise RuntimeError("flash_attn_3 package metadata contract is missing")
    if getattr(package_module, "__version__", None) != distribution_version:
        raise RuntimeError("flash_attn_3 package and distribution versions do not match")


def _required_distribution() -> Any:
    for name in _CONFLICTING_DISTRIBUTIONS:
        try:
            conflict = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        conflict_name = conflict.metadata.get("Name") or name
        if _canonical_name(conflict_name) != _canonical_name(_DISTRIBUTION_NAME):
            raise RuntimeError(
                f"conflicting FlashAttention distribution is installed: {conflict_name}; "
                f"remove it and install only {_DISTRIBUTION_NAME}"
            )
    try:
        distribution = importlib.metadata.distribution(_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{_DISTRIBUTION_NAME} distribution metadata is missing; install the "
            "wheel built from the vendored Hopper source in this repository"
        ) from exc
    installed_name = distribution.metadata.get("Name")
    if _canonical_name(installed_name) != _canonical_name(_DISTRIBUTION_NAME):
        raise RuntimeError("installed BDLM FlashAttention distribution name is invalid")
    return distribution


def _verify_distribution_files(distribution: Any, **expected_paths: Path) -> None:
    files = distribution.files
    if files is None:
        raise RuntimeError("BDLM FlashAttention distribution has no installed file record")
    installed_paths = {Path(distribution.locate_file(item)).resolve() for item in files}
    for label, path in expected_paths.items():
        if path.resolve() not in installed_paths:
            raise RuntimeError(
                f"BDLM FlashAttention distribution does not own {label}: {path}"
            )


def _installed_flash_attention_distribution() -> tuple[str | None, str | None]:
    try:
        distribution = importlib.metadata.distribution(_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None, None
    return _DISTRIBUTION_NAME, str(distribution.version)


def _required_module_path(module: Any, name: str) -> Path:
    path = _module_path(module)
    if path is None or not path.is_file():
        raise RuntimeError(f"{name} has no installed file path")
    return path


def _module_path(module: Any) -> Path | None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        module_spec = getattr(module, "__spec__", None)
        module_file = getattr(module_spec, "origin", None)
    if module_file is None:
        return None
    return Path(module_file).resolve()


def _required_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"BDLM FlashAttention metadata is missing {name}")
    return value


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value or value == "None":
        raise RuntimeError(f"BDLM FlashAttention metadata is missing {name}")
    return value


def _required_sha256(payload: Mapping[str, Any], name: str, label: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"BDLM FlashAttention metadata has no valid {label} hash")
    return value


def _major_minor(value: str, label: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise RuntimeError(f"BDLM FlashAttention {label} version is invalid: {value!r}")
    return int(match.group(1)), int(match.group(2))


def _canonical_name(value: Any) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "")).lower()


def _is_repo_checkout_flash_attention(path: Path) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    for index, part in enumerate(parts):
        if part == "flash-attention" and index + 1 < len(parts):
            if parts[index + 1] in {"hopper", "csrc"}:
                return True
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_REQUIRED_BDLM_OPS = (
    "bdlm_fwd_accum",
    "bdlm_ragged_prefix_fwd",
    "bdlm_ragged_prefix_bwd",
)
_REQUIRED_DFLASH_OPS = (
    "dflash_interval_fwd",
    "dflash_interval_bwd",
)
_PACKAGED_ATTENTION_OPS = _REQUIRED_BDLM_OPS + _REQUIRED_DFLASH_OPS

_BDLM_PYTHON_ABI_SUFFIX = (
    "key_start",
    "softmax_scale",
    "deterministic",
    "sm_margin",
    "return_softmax",
    "clean_offset",
)
_BDLM_PYTHON_ABI_SUFFIX_LENGTH = len(_BDLM_PYTHON_ABI_SUFFIX)
_BDLM_OP_ABI = {
    "fwd": (
        "bdlm_query_blocks",
        "bdlm_query_is_clean",
        "bdlm_block_size",
        "bdlm_key_start",
        "bdlm_clean_offset",
    ),
    "bwd": (
        "bdlm_query_blocks",
        "bdlm_query_is_clean",
        "bdlm_block_size",
        "bdlm_key_start",
        "softmax_lse_grad",
        "bdlm_clean_offset",
    ),
    "bdlm_fwd_accum": (
        "query_indices",
        "bdlm_query_blocks",
        "bdlm_query_is_clean",
        "bdlm_block_size",
        "bdlm_key_start",
        "bdlm_clean_offset",
        "softmax_scale",
        "initial",
    ),
}
_DFLASH_OP_ABI = {
    "dflash_interval_fwd": (
        "context_starts",
        "context_stops",
        "anchor_valid",
        "block_size",
        "key_start",
        "softmax_scale",
    ),
    "dflash_interval_bwd": (
        "context_starts",
        "context_stops",
        "anchor_valid",
        "block_size",
        "key_start",
        "softmax_scale",
    ),
}
