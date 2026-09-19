from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from dllm_parallel.core.profiling.portable_artifacts import verify_portable_artifacts


WHEEL_FILES = (
    "dllm_parallel/py.typed",
    "dllm_parallel/cli/main.py",
    "dllm_parallel/core/kernels/bundle_manifest.py",
    "dllm_parallel/recipes/manifest.yaml",
    "dllm_parallel/recipes/smoke/cpu-config.yaml",
    "turbo_dllm-0.1.0.dist-info/METADATA",
)
SDIST_FILES = (
    "turbo_dllm-0.1.0/containers/cuda/Containerfile",
    "turbo_dllm-0.1.0/scripts/build/build_portable.sh",
    "turbo_dllm-0.1.0/third_party/flash-attention/LICENSE",
    "turbo_dllm-0.1.0/third_party/flash-attention/PROVENANCE.md",
)


def _artifacts(
    tmp_path: Path,
    *,
    extra_wheel_file: str | None = None,
    requires_dist: str | None = None,
) -> None:
    wheel = tmp_path / "turbo_dllm-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in (*WHEEL_FILES, *((extra_wheel_file,) if extra_wheel_file else ())):
            content = b"content"
            if name.endswith(".dist-info/METADATA"):
                dependency = (
                    f"Requires-Dist: {requires_dist}\n" if requires_dist else ""
                )
                content = (
                    "Metadata-Version: 2.4\n"
                    "Name: turbo-dllm\n"
                    "Version: 0.1.0\n"
                    f"{dependency}\n"
                ).encode()
            archive.writestr(name, content)

    sdist = tmp_path / "turbo_dllm-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in SDIST_FILES:
            info = tarfile.TarInfo(name)
            payload = b"content"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_portable_artifact_gate_accepts_release_boundaries(tmp_path: Path) -> None:
    _artifacts(tmp_path)

    report = verify_portable_artifacts(tmp_path)

    assert report["wheel_files"] == len(WHEEL_FILES)
    assert report["sdist_files"] == len(SDIST_FILES)


def test_portable_artifact_gate_rejects_removed_model_modules(tmp_path: Path) -> None:
    _artifacts(
        tmp_path,
        extra_wheel_file="dllm_parallel/core/models/backbones/bd3lm/model.py",
    )

    with pytest.raises(RuntimeError, match="forbidden wheel content"):
        verify_portable_artifacts(tmp_path)


def test_portable_artifact_gate_rejects_direct_url_dependencies(
    tmp_path: Path,
) -> None:
    _artifacts(
        tmp_path,
        requires_dist=(
            "deep-ep @ git+https://github.com/deepseek-ai/DeepEP.git@"
            "dd758caf451848bd150e1046af3d0a73e5fff38d"
        ),
    )

    with pytest.raises(RuntimeError, match="direct URL dependencies"):
        verify_portable_artifacts(tmp_path)
