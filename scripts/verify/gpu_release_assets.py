#!/usr/bin/env python3
"""Verify local GPU release assets before creating their catalog and checksums."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from dllm_parallel.core.kernels.bundle_manifest import (
    BundleEnvironment,
    RUNTIME_DISTRIBUTIONS,
    create_bundle_catalog,
    read_bundle_manifest,
    validate_bundle_manifest,
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_assets(
    directory: Path, *, release: str, source_revision: str, base_url: str
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a full lowercase Git commit")
    if not base_url.rstrip("/").endswith(f"/releases/download/{release}"):
        raise ValueError("base URL must point to the requested release")
    manifests = sorted(directory.glob("gpu-*.json"))
    manifests = [path for path in manifests if path.name != "gpu-bundles.json"]
    if not manifests:
        raise ValueError("at least one qualified GPU bundle is required")
    assets: set[Path] = set(manifests)
    for path in manifests:
        manifest = read_bundle_manifest(path)
        if manifest.get("source_revision") != source_revision:
            raise ValueError(f"source revision mismatch: {path.name}")
        environment = BundleEnvironment(
            package_version=release.removeprefix("v"),
            python_abi=str(manifest.get("python_abi", "")),
            platform=str(manifest.get("platform", "")),
            cuda=str(manifest.get("cuda", "")),
            compute_capability=str((manifest.get("architectures") or [""])[0]),
        )
        runtime = manifest.get("runtime_artifacts", [])
        if {item["name"].split("-", 1)[0] for item in runtime} != set(
            RUNTIME_DISTRIBUTIONS
        ):
            raise ValueError(
                "release bundles require prebuilt DeepSpeed and Transformer Engine PyTorch wheels"
            )
        for artifact in validate_bundle_manifest(manifest, environment=environment):
            wheel = directory / str(artifact["name"])
            if artifact["url"] != f"{base_url.rstrip('/')}/{wheel.name}":
                raise ValueError(
                    f"artifact URL does not target this release: {wheel.name}"
                )
            if wheel.is_symlink() or not wheel.is_file():
                raise ValueError(f"missing regular wheel file: {wheel.name}")
            if wheel.stat().st_size != artifact["size"]:
                raise ValueError(f"wheel size mismatch: {wheel.name}")
            if _digest(wheel) != artifact["sha256"]:
                raise ValueError(f"wheel SHA-256 mismatch: {wheel.name}")
            assets.add(wheel)
    unreferenced = set(directory.glob("*.whl")) - assets
    if unreferenced:
        raise ValueError(
            f"unreferenced release wheels: {sorted(p.name for p in unreferenced)}"
        )
    catalog = directory / "gpu-bundles.json"
    create_bundle_catalog(
        directory=directory,
        output=catalog,
        base_url=base_url,
        release=release,
    )
    assets.add(catalog)
    (directory / "SHA256SUMS-gpu").write_text(
        "".join(f"{_digest(path)}  {path.name}\n" for path in sorted(assets)),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    prepare_assets(
        args.directory,
        release=args.release,
        source_revision=args.source_revision,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
