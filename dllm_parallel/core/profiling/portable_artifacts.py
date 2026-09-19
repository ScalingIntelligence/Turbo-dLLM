"""Verify the public boundary of portable wheel and source artifacts."""

from __future__ import annotations

import argparse
from email.parser import Parser
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Sequence


_REQUIRED_WHEEL_FILES = {
    "dllm_parallel/py.typed",
    "dllm_parallel/cli/main.py",
    "dllm_parallel/core/kernels/bundle_manifest.py",
    "dllm_parallel/recipes/manifest.yaml",
    "dllm_parallel/recipes/smoke/cpu-config.yaml",
}
_REQUIRED_SDIST_SUFFIXES = {
    "containers/cuda/Containerfile",
    "scripts/build/build_portable.sh",
    "third_party/flash-attention/LICENSE",
    "third_party/flash-attention/PROVENANCE.md",
}
_FORBIDDEN_PACKAGE_TOKENS = (
    "bd3lm",
    "coderforge",
    "dllm_baseline",
    "muse_glimmer",
    "openhands",
    "swebench",
    "tools/modal",
)
_NATIVE_SUFFIXES = (".dylib", ".pyd", ".so")


def _one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"portable artifact directory must contain exactly one {label}; "
            f"found {len(matches)}"
        )
    return matches[0]


def verify_portable_artifacts(directory: str | Path) -> dict[str, object]:
    """Inspect one portable wheel and sdist without extracting either artifact."""

    root = Path(directory)
    wheel = _one(root, "turbo_dllm-*.whl", "Turbo-dLLM wheel")
    sdist = _one(root, "turbo_dllm-*.tar.gz", "Turbo-dLLM sdist")

    with zipfile.ZipFile(wheel) as archive:
        wheel_files = set(archive.namelist())
        metadata_files = sorted(
            name for name in wheel_files if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_files) != 1:
            raise RuntimeError(
                "portable wheel must contain exactly one Core Metadata file; "
                f"found {len(metadata_files)}"
            )
        metadata = Parser().parsestr(
            archive.read(metadata_files[0]).decode("utf-8")
        )
    missing_wheel = sorted(_REQUIRED_WHEEL_FILES.difference(wheel_files))
    forbidden_wheel = sorted(
        name
        for name in wheel_files
        if "third_party/" in name.casefold()
        or name.casefold().endswith(_NATIVE_SUFFIXES)
        or any(token in name.casefold() for token in _FORBIDDEN_PACKAGE_TOKENS)
    )
    if missing_wheel:
        raise RuntimeError(f"missing required wheel content: {missing_wheel}")
    if forbidden_wheel:
        raise RuntimeError(f"forbidden wheel content: {forbidden_wheel}")
    direct_dependencies = sorted(
        dependency
        for dependency in metadata.get_all("Requires-Dist", [])
        if " @ " in dependency or "git+" in dependency.casefold()
    )
    if direct_dependencies:
        raise RuntimeError(
            "portable wheel contains direct URL dependencies rejected by public "
            f"indexes: {direct_dependencies}"
        )

    with tarfile.open(sdist, mode="r:gz") as archive:
        sdist_files = {member.name for member in archive.getmembers() if member.isfile()}
    missing_sdist = sorted(
        suffix
        for suffix in _REQUIRED_SDIST_SUFFIXES
        if not any(name.endswith(suffix) for name in sdist_files)
    )
    forbidden_sdist = sorted(
        name
        for name in sdist_files
        if "/dllm_parallel/" in f"/{name.casefold()}"
        and any(token in name.casefold() for token in _FORBIDDEN_PACKAGE_TOKENS)
    )
    if missing_sdist:
        raise RuntimeError(f"missing required sdist content: {missing_sdist}")
    if forbidden_sdist:
        raise RuntimeError(f"forbidden sdist package content: {forbidden_sdist}")

    return {
        "event": "portable_artifact_verification",
        "passed": True,
        "wheel": wheel.name,
        "wheel_files": len(wheel_files),
        "sdist": sdist.name,
        "sdist_files": len(sdist_files),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dllm_parallel.core.profiling.portable_artifacts"
    )
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(verify_portable_artifacts(args.directory), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ("verify_portable_artifacts",)
