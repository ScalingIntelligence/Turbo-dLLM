from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dllm_parallel.core.kernels import bundle_manifest
from dllm_parallel.core.kernels.bundle_manifest import (
    BundleEnvironment,
    create_bundle_catalog,
    create_bundle_manifest,
    resolve_bundle_manifest,
    select_bundle,
    validate_bundle_manifest,
)


WHEELS = (
    "bdlm_flash_attn_3-0.1.0-cp312-cp312-linux_x86_64.whl",
    "turbo_dllm-0.1.0-py3-none-any.whl",
    "flash_attn_4-4.0.0b19-cp312-cp312-linux_x86_64.whl",
)


def _environment(**overrides: str) -> BundleEnvironment:
    values = {
        "package_version": "0.1.0",
        "python_abi": "cp312",
        "platform": "linux_x86_64",
        "cuda": "12.8",
        "compute_capability": "9.0",
    }
    values.update(overrides)
    return BundleEnvironment(**values)


def _catalog_manifest() -> dict[str, object]:
    return {
        "format": "dllm.gpu_bundle.v1",
        "package_version": "0.1.0",
        "source_revision": "a" * 40,
        "python_abi": "cp312",
        "platform": "linux_x86_64",
        "cuda": "12.8",
        "architectures": ["9.0"],
        "artifacts": [
            {
                "name": name,
                "url": f"https://example.invalid/{name}",
                "sha256": "0" * 64,
                "size": 1,
            }
            for name in WHEELS
        ],
    }


def test_create_bundle_manifest_hashes_exact_coordinated_wheels(tmp_path: Path) -> None:
    for index, name in enumerate(WHEELS):
        (tmp_path / name).write_bytes(f"wheel-{index}".encode())
    output = tmp_path / "gpu-sm90-cu128.json"

    manifest = create_bundle_manifest(
        directory=tmp_path,
        output=output,
        base_url="https://example.invalid/releases/v0.1.0",
        cuda="12.8",
        architectures="9.0",
        python_abi="cp312",
        platform="linux_x86_64",
        package_version="0.1.0",
        source_revision="a" * 40,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert manifest["format"] == "dllm.gpu_bundle.v1"
    assert [item["name"] for item in manifest["artifacts"]] == sorted(WHEELS)
    for artifact in manifest["artifacts"]:
        payload = (tmp_path / artifact["name"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
        assert artifact["size"] == len(payload)


@pytest.mark.parametrize(
    ("override", "value", "match"),
    (
        ("package_version", "0.2.0", "package version"),
        ("python_abi", "cp311", "Python ABI"),
        ("platform", "linux_aarch64", "platform"),
        ("cuda", "12.6", "CUDA"),
        ("compute_capability", "8.9", "compute capability"),
    ),
)
def test_bundle_manifest_rejects_incompatible_environment(
    tmp_path: Path,
    override: str,
    value: str,
    match: str,
) -> None:
    for index, name in enumerate(WHEELS):
        (tmp_path / name).write_bytes(f"wheel-{index}".encode())
    manifest = create_bundle_manifest(
        directory=tmp_path,
        output=tmp_path / "bundle.json",
        base_url="https://example.invalid/releases/v0.1.0",
        cuda="12.8",
        architectures="9.0",
        python_abi="cp312",
        platform="linux_x86_64",
        package_version="0.1.0",
        source_revision="a" * 40,
    )

    with pytest.raises(RuntimeError, match=match):
        validate_bundle_manifest(
            manifest, environment=_environment(**{override: value})
        )


def test_bundle_manifest_rejects_untrusted_artifact_url(tmp_path: Path) -> None:
    for index, name in enumerate(WHEELS):
        (tmp_path / name).write_bytes(f"wheel-{index}".encode())
    manifest = create_bundle_manifest(
        directory=tmp_path,
        output=tmp_path / "bundle.json",
        base_url="https://example.invalid/releases/v0.1.0",
        cuda="12.8",
        architectures="9.0",
        python_abi="cp312",
        platform="linux_x86_64",
        package_version="0.1.0",
        source_revision="a" * 40,
    )
    manifest["artifacts"][0]["url"] = "http://example.invalid/wheel.whl"

    with pytest.raises(ValueError, match="HTTPS"):
        validate_bundle_manifest(manifest, environment=_environment())


def test_bundle_install_resolves_the_coordinated_gpu_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheels = tuple(tmp_path / name for name in WHEELS)
    for wheel in wheels:
        wheel.write_bytes(b"wheel")
    commands: list[list[str]] = []
    monkeypatch.setattr(bundle_manifest, "read_bundle_manifest", lambda _: {})
    monkeypatch.setattr(
        bundle_manifest,
        "download_bundle",
        lambda manifest, *, output_dir: wheels,
    )
    monkeypatch.setattr(
        bundle_manifest.subprocess,
        "run",
        lambda command, **kwargs: (
            commands.append([str(value) for value in command])
            or SimpleNamespace(returncode=0)
        ),
    )

    bundle_manifest.install_bundle(tmp_path / "bundle.json")

    assert len(commands) == 1
    command = commands[0]
    assert "--no-deps" not in command
    package_wheel = next(
        path for path in wheels if path.name.startswith("turbo_dllm")
    )
    assert f"turbo-dllm[gpu] @ {package_wheel.resolve().as_uri()}" in command
    assert all(str(path) in command for path in wheels if path != package_wheel)


def test_catalog_records_hashes_and_selects_an_exact_host(tmp_path: Path) -> None:
    manifest_path = tmp_path / "gpu-sm90-cu128-cp312.json"
    manifest_path.write_text(
        json.dumps(_catalog_manifest()),
        encoding="utf-8",
    )

    catalog = create_bundle_catalog(
        directory=tmp_path,
        output=tmp_path / "gpu-bundles.json",
        base_url="https://example.invalid/releases/download/v0.1.0",
        release="v0.1.0",
    )

    entry = select_bundle(catalog, environment=_environment())
    assert catalog["format"] == "dllm.gpu_bundle_catalog.v1"
    assert catalog["release"] == "v0.1.0"
    assert entry["manifest"] == manifest_path.name
    assert entry["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_catalog_no_match_explains_host_and_supported_combinations(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "gpu-sm90-cu128-cp312.json"
    manifest_path.write_text(
        json.dumps(_catalog_manifest()),
        encoding="utf-8",
    )
    catalog = create_bundle_catalog(
        directory=tmp_path,
        output=tmp_path / "gpu-bundles.json",
        base_url="https://example.invalid/releases/download/v0.1.0",
        release="v0.1.0",
    )

    with pytest.raises(RuntimeError) as error:
        select_bundle(catalog, environment=_environment(compute_capability="8.9"))

    message = str(error.value)
    assert "compute capability=8.9" in message
    assert "Python ABI=cp312" in message
    assert "CUDA=12.8" in message
    assert "architectures=9.0" in message


def test_catalog_creation_rejects_a_malformed_manifest(tmp_path: Path) -> None:
    manifest = _catalog_manifest()
    manifest["artifacts"] = []
    (tmp_path / "gpu-broken.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly three"):
        create_bundle_catalog(
            directory=tmp_path,
            output=tmp_path / "gpu-bundles.json",
            base_url="https://example.invalid/releases/download/v0.1.0",
            release="v0.1.0",
        )


def test_auto_resolution_verifies_catalog_manifest_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _catalog_manifest()
    manifest_payload = json.dumps(manifest).encode()
    catalog = {
        "format": "dllm.gpu_bundle_catalog.v1",
        "release": "v0.1.0",
        "package_version": "0.1.0",
        "bundles": [
            {
                "manifest": "gpu-sm90.json",
                "url": "https://example.invalid/gpu-sm90.json",
                "sha256": "f" * 64,
                "python_abi": "cp312",
                "platform": "linux_x86_64",
                "cuda": "12.8",
                "architectures": ["9.0"],
            }
        ],
    }
    payloads = {
        "https://example.invalid/releases/download/v0.1.0/gpu-bundles.json": json.dumps(
            catalog
        ).encode(),
        "https://example.invalid/gpu-sm90.json": manifest_payload,
    }
    monkeypatch.setattr(
        bundle_manifest,
        "_read_url",
        lambda url, *, limit: payloads[url],
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        resolve_bundle_manifest(
            "v0.1.0",
            environment=_environment(),
            releases_url="https://example.invalid/releases/download",
        )


def test_auto_resolution_returns_an_exact_validated_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _catalog_manifest()
    manifest_payload = json.dumps(manifest).encode()
    catalog = {
        "format": "dllm.gpu_bundle_catalog.v1",
        "release": "v0.1.0",
        "package_version": "0.1.0",
        "bundles": [
            {
                "manifest": "gpu-sm90.json",
                "url": "https://example.invalid/gpu-sm90.json",
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "python_abi": "cp312",
                "platform": "linux_x86_64",
                "cuda": "12.8",
                "architectures": ["9.0"],
            }
        ],
    }
    payloads = {
        "https://example.invalid/releases/download/v0.1.0/gpu-bundles.json": json.dumps(
            catalog
        ).encode(),
        "https://example.invalid/gpu-sm90.json": manifest_payload,
    }
    monkeypatch.setattr(
        bundle_manifest,
        "_read_url",
        lambda url, *, limit: payloads[url],
    )

    resolved = resolve_bundle_manifest(
        "v0.1.0",
        environment=_environment(),
        releases_url="https://example.invalid/releases/download",
    )

    assert resolved == manifest


def test_auto_resolution_rejects_catalog_manifest_metadata_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _catalog_manifest()
    manifest["architectures"] = ["9.0", "8.9"]
    manifest_payload = json.dumps(manifest).encode()
    catalog = {
        "format": "dllm.gpu_bundle_catalog.v1",
        "release": "v0.1.0",
        "package_version": "0.1.0",
        "bundles": [
            {
                "manifest": "gpu-sm90.json",
                "url": "https://example.invalid/gpu-sm90.json",
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "python_abi": "cp312",
                "platform": "linux_x86_64",
                "cuda": "12.8",
                "architectures": ["9.0"],
            }
        ],
    }
    payloads = {
        "https://example.invalid/releases/download/v0.1.0/gpu-bundles.json": json.dumps(
            catalog
        ).encode(),
        "https://example.invalid/gpu-sm90.json": manifest_payload,
    }
    monkeypatch.setattr(
        bundle_manifest,
        "_read_url",
        lambda url, *, limit: payloads[url],
    )

    with pytest.raises(ValueError, match="architectures"):
        resolve_bundle_manifest(
            "v0.1.0",
            environment=_environment(),
            releases_url="https://example.invalid/releases/download",
        )
