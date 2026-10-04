from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dllm_parallel.core.kernels.bundle_manifest import create_bundle_manifest


ROOT = Path(__file__).resolve().parents[2]
BASE = "https://github.com/ScalingIntelligence/Turbo-dLLM/releases/download/v0.1.1"
REVISION = "a" * 40


def _prepare(tmp_path: Path, *, corrupt: bool = False, wrong_revision: bool = False):
    names = (
        "turbo_dllm-0.1.1-cp312-cp312-linux_x86_64.whl",
        "bdlm_flash_attn_3-0.1.0-cp312-cp312-linux_x86_64.whl",
        "flash_attn_4-4.0.0b19-py3-none-any.whl",
    )
    for name in names:
        (tmp_path / name).write_bytes(b"wheel fixture")
    create_bundle_manifest(
        directory=tmp_path,
        output=tmp_path / "gpu-sm90-cu128-cp312.json",
        base_url=BASE,
        cuda="12.8",
        architectures="9.0",
        python_abi="cp312",
        platform="linux_x86_64",
        package_version="0.1.1",
        source_revision="b" * 40 if wrong_revision else REVISION,
    )
    if corrupt:
        (tmp_path / names[0]).write_bytes(b"tampered wheel")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify/gpu_release_assets.py"),
            "--directory",
            str(tmp_path),
            "--release",
            "v0.1.1",
            "--source-revision",
            REVISION,
            "--base-url",
            BASE,
        ],
        capture_output=True,
        text=True,
    )


def test_release_catalog_and_checksums_cover_the_verified_upload(tmp_path: Path):
    completed = _prepare(tmp_path)
    assert completed.returncode == 0, completed.stderr
    catalog = json.loads((tmp_path / "gpu-bundles.json").read_text())
    assert catalog["release"] == "v0.1.1"
    assert catalog["bundles"][0]["url"] == f"{BASE}/gpu-sm90-cu128-cp312.json"
    records = (tmp_path / "SHA256SUMS-gpu").read_text().splitlines()
    assert len(records) == 5
    for record in records:
        digest, name = record.split("  ")
        assert digest == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", ["corrupt", "wrong_revision"])
def test_release_preparation_rejects_unpublishable_artifacts(
    tmp_path: Path, fault: str
):
    completed = _prepare(tmp_path, **{fault: True})
    assert completed.returncode != 0
    assert (
        "size mismatch" if fault == "corrupt" else "source revision mismatch"
    ) in completed.stderr
    assert not (tmp_path / "gpu-bundles.json").exists()
    assert not (tmp_path / "SHA256SUMS-gpu").exists()


def test_release_preparation_rejects_an_empty_bundle_directory(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify/gpu_release_assets.py"),
            "--directory",
            str(tmp_path),
            "--release",
            "v0.1.1",
            "--source-revision",
            REVISION,
            "--base-url",
            BASE,
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "at least one qualified GPU bundle" in completed.stderr
    assert not (tmp_path / "gpu-bundles.json").exists()
