# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Release-blocking source, runtime, and artifact gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dllm_parallel.core.kernels._native_specs import spec_by_name
from dllm_parallel.core.kernels.runtime import NATIVE_KERNEL_MANIFEST_FORMAT

PRODUCTION_TRAINER = Path("dllm_parallel/training/block_diffusion_trainer.py")
REQUIRED_NATIVE_KERNELS = frozenset(
    {
        "bdlm_cp_fusion",
        "dflash_cp_fusion",
        "dllm_fused_linear_ce_v3",
    }
)
FA3_DISTRIBUTION = "bdlm-flash-attn-3"
FA3_BUILD_METADATA_FORMAT = "bdlm.flash_attn_3.build.v3"
FA3_VARIANT = "bdlm-cp-splitd-v3"
PRODUCTION_WHEEL_BUILDER = Path("scripts/build/build_cuda_wheels.sh")
REQUIRED_FA3_BUILD_POLICY = (
    "FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE",
    "FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE",
    "FLASH_ATTENTION_FORCE_LEGACY_API=1",
    "FLASH_ATTENTION_USE_STABLE_API=0",
    'BDLM_FA3_SOURCE_REVISION="$SOURCE_REVISION"',
)
_SYS_PATH_MUTATION_PATTERNS = tuple(f"sys.path.{name}" for name in ("insert", "remove"))


@dataclass(frozen=True)
class ReleaseGateViolation:
    name: str
    detail: str


def run_release_policy_gates(
    root: str | Path,
    *,
    require_native_artifacts: bool = True,
) -> list[ReleaseGateViolation]:
    root = Path(root)
    violations: list[ReleaseGateViolation] = []
    violations.extend(_check_single_trainer(root))
    violations.extend(_check_no_noncanonical_config_paths(root))
    violations.extend(_check_no_import_path_mutation(root))
    violations.extend(_check_no_hidden_flash_attention_path(root))
    violations.extend(_check_fa3_packaging(root))
    violations.extend(_check_production_wheel_builder(root))
    if require_native_artifacts:
        violations.extend(_check_native_kernel_manifest(root))
    violations.extend(_check_no_production_recipe_directory(root))
    return violations


def assert_release_policy_gates(
    root: str | Path,
    *,
    require_native_artifacts: bool = True,
) -> None:
    violations = run_release_policy_gates(
        root,
        require_native_artifacts=require_native_artifacts,
    )
    if violations:
        lines = [f"{item.name}: {item.detail}" for item in violations]
        raise AssertionError("release policy gates failed:\n" + "\n".join(lines))


def run_runtime_preflight(*, require_fa3: bool = True) -> dict[str, Any]:
    from dllm_parallel.core.attention.flex import verify_flex_attention_runtime
    from dllm_parallel.core.kernels.runtime import verify_packaged_native_kernels

    result = {
        "native_kernels": list(verify_packaged_native_kernels()),
        "flash_attention_4": verify_flex_attention_runtime().to_log_dict(),
    }
    if require_fa3:
        from dllm_parallel.core.attention.fa3 import verify_flash_attention_kernels

        result["flash_attention"] = verify_flash_attention_kernels().to_log_dict()
    return result


def verify_production_wheel_directory(directory: str | Path) -> dict[str, str]:
    """Verify the complete production wheel set."""

    directory = Path(directory).resolve()
    dllm_wheel = _exactly_one_wheel(directory, "turbo_dllm-")
    fa3_wheel = _exactly_one_wheel(directory, "bdlm_flash_attn_3-")
    fa4_wheel = _exactly_one_wheel(directory, "flash_attn_4-")
    _verify_platform_wheel(dllm_wheel)
    _verify_platform_wheel(fa3_wheel)
    with zipfile.ZipFile(dllm_wheel) as archive:
        names = set(archive.namelist())
        manifest_name = "dllm_parallel/core/_C/native_kernels.json"
        if manifest_name not in names:
            raise RuntimeError("Turbo-dLLM wheel is missing native_kernels.json")
        manifest = json.loads(archive.read(manifest_name))
        if manifest.get("format") != NATIVE_KERNEL_MANIFEST_FORMAT:
            raise RuntimeError(
                "Turbo-dLLM wheel has an invalid native manifest format"
            )
        kernels = manifest.get("kernels")
        if not isinstance(kernels, dict) or not kernels:
            raise RuntimeError("Turbo-dLLM wheel has no native kernel entries")
        for kernel, entry in kernels.items():
            binary = entry.get("binary") if isinstance(entry, dict) else None
            expected_hash = (
                entry.get("binary_hash") if isinstance(entry, dict) else None
            )
            if not isinstance(binary, str) or not binary:
                raise RuntimeError(f"native kernel {kernel!r} has no binary name")
            binary_name = f"dllm_parallel/core/_C/{binary}"
            if binary_name not in names:
                raise RuntimeError(
                    f"Turbo-dLLM wheel is missing {kernel} binary {binary_name}"
                )
            observed_hash = hashlib.sha256(archive.read(binary_name)).hexdigest()
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"Turbo-dLLM wheel has a binary hash mismatch for {kernel}"
                )
    with zipfile.ZipFile(fa3_wheel) as archive:
        names = set(archive.namelist())
        required = {
            "flash_attn_3/build_metadata.json",
            "flash_attn_interface.py",
        }
        missing = sorted(required.difference(names))
        if missing:
            raise RuntimeError(
                "bdlm-flash-attn-3 wheel is missing production metadata: "
                + ", ".join(missing)
            )
        binaries = [
            name
            for name in names
            if name.startswith("flash_attn_3/_C") and name.endswith((".so", ".pyd"))
        ]
        if len(binaries) != 1:
            raise RuntimeError(
                "bdlm-flash-attn-3 wheel must contain exactly one native extension"
            )
    with zipfile.ZipFile(fa4_wheel) as archive:
        names = set(archive.namelist())
        interface_name = "flash_attn/cute/interface.py"
        if interface_name not in names:
            raise RuntimeError("flash-attn-4 wheel is missing its CuTe interface")
        interface_source = archive.read(interface_name)
        if b"def _validate_pack_gqa_backward_capability(" not in interface_source:
            raise RuntimeError(
                "flash-attn-4 wheel lacks native Pack-GQA backward support"
            )
        postprocess_name = "flash_attn/cute/flash_bwd_postprocess.py"
        if postprocess_name not in names:
            raise RuntimeError("flash-attn-4 wheel is missing its dQ postprocess")
        postprocess_source = archive.read(postprocess_name)
        if b"PACK_GQA_DQACCUM_ALIGNMENT_PROPAGATED = True" not in postprocess_source:
            raise RuntimeError(
                "flash-attn-4 wheel lacks aligned packed-GQA dQ postprocessing"
            )
    return {
        "turbo_dllm": str(dllm_wheel),
        "flash_attention": str(fa3_wheel),
        "flash_attention_4": str(fa4_wheel),
    }


def _exactly_one_wheel(directory: Path, prefix: str) -> Path:
    wheels = sorted(directory.glob(f"{prefix}*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"expected exactly one {prefix} wheel in {directory}, found {len(wheels)}"
        )
    return wheels[0]


def _verify_platform_wheel(path: Path) -> None:
    if path.name.endswith("-none-any.whl"):
        raise RuntimeError(f"native wheel has a universal tag: {path.name}")
    with zipfile.ZipFile(path) as archive:
        wheel_metadata = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_metadata) != 1:
            raise RuntimeError(f"wheel metadata is missing or ambiguous: {path.name}")
        payload = archive.read(wheel_metadata[0]).decode("utf-8")
    if "Root-Is-Purelib: false" not in payload:
        raise RuntimeError(f"native wheel is marked as pure Python: {path.name}")
    tags = [
        line.removeprefix("Tag: ")
        for line in payload.splitlines()
        if line.startswith("Tag: ")
    ]
    if not tags or any(tag.endswith("-any") for tag in tags):
        raise RuntimeError(f"native wheel has an invalid platform tag: {path.name}")


def _check_single_trainer(root: Path) -> list[ReleaseGateViolation]:
    trainer_files = sorted((root / "dllm_parallel" / "training").glob("*.py"))
    forbidden = {
        "main.py",
        "hf_block_diffusion.py",
        "hf_block_diffusion_runner.py",
        "diffusion.py",
    }
    offenders = [path.name for path in trainer_files if path.name in forbidden]
    if offenders:
        return [
            ReleaseGateViolation(
                "duplicate_trainer_path",
                ", ".join(offenders),
            )
        ]
    if not (root / PRODUCTION_TRAINER).exists():
        return [
            ReleaseGateViolation("canonical_trainer_missing", str(PRODUCTION_TRAINER))
        ]
    return []


def _check_no_noncanonical_config_paths(root: Path) -> list[ReleaseGateViolation]:
    offenders: list[str] = []
    patterns = (
        "_" + "CONFIG" + "_" + "ALIASES",
        "_" + "SECTION" + "_" + "ALIASES",
        "config" + "_from" + "_yaml",
        "block_diffusion" + "_config" + "_from" + "_yaml",
        "_" + "HF" + "_" + "COMMANDS",
    )
    for path in (root / "dllm_parallel").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern in text for pattern in patterns):
            offenders.append(str(path.relative_to(root)))
    return (
        [ReleaseGateViolation("noncanonical_config_path", ", ".join(offenders))]
        if offenders
        else []
    )


def _check_no_import_path_mutation(root: Path) -> list[ReleaseGateViolation]:
    offenders: list[str] = []
    for base in (root / "dllm_parallel", root / "scripts"):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern in text for pattern in _SYS_PATH_MUTATION_PATTERNS):
                offenders.append(str(path.relative_to(root)))
    return (
        [ReleaseGateViolation("import_path_mutation", ", ".join(offenders))]
        if offenders
        else []
    )


def _check_no_hidden_flash_attention_path(root: Path) -> list[ReleaseGateViolation]:
    """Reject source-checkout coupling from importable runtime modules.

    Release tooling must name the vendored source tree in order to build the
    standalone FA3 wheel.  Runtime modules must resolve the installed package
    and may never reach back into that source checkout.
    """

    offenders: list[str] = []
    package = root / "dllm_parallel"
    if package.exists():
        for path in package.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ("third_party/flash-attention" + "/hopper") in text:
                offenders.append(str(path.relative_to(root)))
    return (
        [ReleaseGateViolation("hidden_flash_attention_path", ", ".join(offenders))]
        if offenders
        else []
    )


def _check_fa3_packaging(root: Path) -> list[ReleaseGateViolation]:
    violations: list[ReleaseGateViolation] = []
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return [ReleaseGateViolation("fa3_root_pyproject_missing", "pyproject.toml")]
    root_metadata = pyproject.read_text(encoding="utf-8", errors="ignore")
    public_requirement = re.search(
        r"[\"']flash[-_.]attn[-_.]3(?:[^a-zA-Z0-9_-]|[\"'])",
        root_metadata,
        flags=re.IGNORECASE,
    )
    if public_requirement is not None:
        violations.append(
            ReleaseGateViolation(
                "fa3_public_dependency",
                "root extras must not resolve public flash-attn-3; "
                "build the vendored Hopper source",
            )
        )

    hopper = root / "third_party" / "flash-attention" / "hopper"
    required_files = (
        hopper / "pyproject.toml",
        hopper / "setup.py",
        hopper / "flash_attn_3" / "__init__.py",
    )
    missing = [
        str(path.relative_to(root)) for path in required_files if not path.exists()
    ]
    if missing:
        violations.append(
            ReleaseGateViolation("fa3_packaging_files_missing", ", ".join(missing))
        )
        return violations

    setup_text = (hopper / "setup.py").read_text(encoding="utf-8", errors="ignore")
    forbidden_wheel_fallbacks = (
        "BASE_" + "WHEEL_URL",
        "Cached" + "WheelsCommand",
        "url" + "retrieve(",
    )
    present_fallbacks = [
        marker for marker in forbidden_wheel_fallbacks if marker in setup_text
    ]
    if present_fallbacks:
        violations.append(
            ReleaseGateViolation(
                "fa3_prebuilt_wheel_fallback", ", ".join(present_fallbacks)
            )
        )
    required_setup_markers = (
        FA3_DISTRIBUTION,
        FA3_BUILD_METADATA_FORMAT,
        FA3_VARIANT,
        "arch=compute_89,code=sm_89",
        "arch=compute_90a,code=sm_90a",
        "BDLMBuildExtension",
        "BDLMWheelCommand",
    )
    missing_markers = [
        marker for marker in required_setup_markers if marker not in setup_text
    ]
    if missing_markers:
        violations.append(
            ReleaseGateViolation(
                "fa3_build_metadata_contract", ", ".join(missing_markers)
            )
        )

    package_text = (hopper / "flash_attn_3" / "__init__.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    missing_identity = [
        marker
        for marker in (FA3_BUILD_METADATA_FORMAT, FA3_VARIANT)
        if marker not in package_text
    ]
    if missing_identity:
        violations.append(
            ReleaseGateViolation("fa3_package_identity", ", ".join(missing_identity))
        )
    return violations


def _check_production_wheel_builder(root: Path) -> list[ReleaseGateViolation]:
    builder = root / PRODUCTION_WHEEL_BUILDER
    if not builder.is_file():
        return [
            ReleaseGateViolation(
                "production_wheel_builder_missing",
                str(PRODUCTION_WHEEL_BUILDER),
            )
        ]
    text = builder.read_text(encoding="utf-8", errors="ignore")
    missing = [marker for marker in REQUIRED_FA3_BUILD_POLICY if marker not in text]
    if missing:
        return [
            ReleaseGateViolation(
                "fa3_build_policy_incomplete",
                ", ".join(missing),
            )
        ]
    return []


def _check_native_kernel_manifest(root: Path) -> list[ReleaseGateViolation]:
    manifest = root / "dllm_parallel" / "core" / "_C" / "native_kernels.json"
    if not manifest.exists():
        return [
            ReleaseGateViolation(
                "native_kernel_manifest_missing",
                "run dllm-build-kernels during production image/wheel build",
            )
        ]
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [ReleaseGateViolation("native_kernel_manifest_invalid", str(exc))]
    violations: list[ReleaseGateViolation] = []
    if payload.get("format") != NATIVE_KERNEL_MANIFEST_FORMAT:
        violations.append(
            ReleaseGateViolation(
                "native_kernel_manifest_format",
                repr(payload.get("format")),
            )
        )
    pyproject_path = root / "pyproject.toml"
    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        violations.append(ReleaseGateViolation("project_metadata_invalid", str(exc)))
        project = {}
    project_version = (project.get("project") or {}).get("version")
    if not isinstance(project_version, str) or not project_version:
        violations.append(
            ReleaseGateViolation("project_version_missing", "pyproject.toml")
        )
    elif payload.get("package_version") != project_version:
        violations.append(
            ReleaseGateViolation(
                "native_kernel_package_version",
                f"{payload.get('package_version')!r} != {project_version!r}",
            )
        )
    kernels = payload.get("kernels")
    if not isinstance(kernels, dict):
        return [
            ReleaseGateViolation(
                "native_kernel_manifest_invalid",
                "native_kernels.json must contain a kernels object",
            )
        ]
    missing = sorted(REQUIRED_NATIVE_KERNELS.difference(kernels))
    if missing:
        violations.append(
            ReleaseGateViolation(
                "native_kernel_manifest_missing_entries", ", ".join(missing)
            )
        )
    package_dir = manifest.parent
    for name in sorted(REQUIRED_NATIVE_KERNELS.intersection(kernels)):
        entry = kernels.get(name) or {}
        binary_name = entry.get("binary")
        expected_hash = entry.get("binary_hash")
        if not isinstance(binary_name, str) or not binary_name:
            violations.append(
                ReleaseGateViolation("native_kernel_binary_missing", name)
            )
            continue
        if not isinstance(expected_hash, str) or not expected_hash:
            violations.append(
                ReleaseGateViolation("native_kernel_binary_hash_missing", name)
            )
        binary_path = package_dir / binary_name
        if not binary_path.exists():
            violations.append(
                ReleaseGateViolation(
                    "native_kernel_binary_missing",
                    f"{name}: {binary_path.relative_to(root)}",
                )
            )
            continue
        if isinstance(expected_hash, str) and expected_hash:
            observed_hash = hashlib.sha256(binary_path.read_bytes()).hexdigest()
            if observed_hash != expected_hash:
                violations.append(
                    ReleaseGateViolation("native_kernel_binary_hash", name)
                )
        source_files = entry.get("source_files") or []
        expected_source_hash = entry.get("source_hash")
        if not isinstance(expected_source_hash, str) or not expected_source_hash:
            violations.append(
                ReleaseGateViolation("native_kernel_source_hash_missing", name)
            )
        if not isinstance(source_files, list) or not source_files:
            violations.append(
                ReleaseGateViolation("native_kernel_source_files_missing", name)
            )
        if isinstance(expected_source_hash, str) and source_files:
            digest = hashlib.sha256()
            for source in source_files:
                source_path = root / str(source)
                if not source_path.exists():
                    violations.append(
                        ReleaseGateViolation(
                            "native_kernel_source_missing", str(source)
                        )
                    )
                    continue
                digest.update(source_path.name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(source_path.read_bytes())
                digest.update(b"\0")
            if digest.hexdigest() != expected_source_hash:
                violations.append(
                    ReleaseGateViolation("native_kernel_source_hash", name)
                )
        if not entry.get("required_symbols"):
            violations.append(
                ReleaseGateViolation("native_kernel_symbols_missing", name)
            )
        expected_flags = {
            "extra_cflags": list(spec_by_name(name).extra_cflags),
            "extra_cuda_cflags": list(spec_by_name(name).extra_cuda_cflags),
            "extra_ldflags": list(spec_by_name(name).extra_ldflags),
        }
        if entry.get("build_flags") != expected_flags:
            violations.append(ReleaseGateViolation("native_kernel_build_flags", name))
        if not entry.get("python_extension_suffix"):
            violations.append(ReleaseGateViolation("native_kernel_abi_missing", name))
        minimum_capability = entry.get("minimum_device_capability")
        if (
            not isinstance(minimum_capability, list)
            or len(minimum_capability) != 2
            or any(not isinstance(item, int) for item in minimum_capability)
        ):
            violations.append(
                ReleaseGateViolation("native_kernel_capability_missing", name)
            )
        if not entry.get("cuda_arch_list"):
            violations.append(
                ReleaseGateViolation("native_kernel_arch_list_missing", name)
            )
        capabilities = entry.get("supported_device_capabilities")
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or any(
                not isinstance(capability, list)
                or len(capability) != 2
                or any(not isinstance(item, int) for item in capability)
                for capability in capabilities
            )
        ):
            violations.append(
                ReleaseGateViolation("native_kernel_capability_set_missing", name)
            )
    return violations


def _check_no_production_recipe_directory(root: Path) -> list[ReleaseGateViolation]:
    forbidden = (
        root / "recipes" / "prod",
        root / "recipes" / "production",
        root / "dllm_parallel" / "recipes" / "prod",
        root / "dllm_parallel" / "recipes" / "production",
    )
    present = [str(path.relative_to(root)) for path in forbidden if path.exists()]
    return (
        [ReleaseGateViolation("production_recipe_directory", ", ".join(present))]
        if present
        else []
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run DLLM release gates.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    policy = subparsers.add_parser("policy")
    policy.add_argument("--root", default=".")
    policy.add_argument(
        "--portable",
        action="store_true",
        help="run source policy without requiring generated native artifacts",
    )
    preflight = subparsers.add_parser(
        "preflight",
        help="verify installed production native artifacts and build metadata",
    )
    preflight.add_argument(
        "--skip-fa3",
        action="store_true",
        help="verify FA4 only for runtimes that do not execute FA3 kernels",
    )
    wheels = subparsers.add_parser(
        "wheels",
        help="verify built Turbo-dLLM and bdlm-flash-attn-3 wheels",
    )
    wheels.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "policy":
        assert_release_policy_gates(
            args.root,
            require_native_artifacts=not args.portable,
        )
        print(
            json.dumps(
                {"event": "release_policy_gates", "passed": True}, sort_keys=True
            )
        )
    elif args.command == "preflight":
        print(
            json.dumps(
                {
                    "event": "runtime_preflight",
                    "passed": True,
                    **run_runtime_preflight(require_fa3=not args.skip_fa3),
                },
                sort_keys=True,
            )
        )
    elif args.command == "wheels":
        print(
            json.dumps(
                {
                    "event": "production_wheel_gates",
                    "passed": True,
                    **verify_production_wheel_directory(args.directory),
                },
                sort_keys=True,
            )
        )
    else:  # pragma: no cover - argparse enforces choices.
        raise AssertionError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
