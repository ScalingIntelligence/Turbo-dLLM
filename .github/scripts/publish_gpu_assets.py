"""Resume a GPU asset upload safely, exposing the catalog only after its files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def publish_assets(directory: Path, *, release: str, repository: str) -> None:
    catalog = directory / "gpu-bundles.json"
    if not catalog.is_file() or catalog.is_symlink():
        raise ValueError("a verified GPU bundle catalog is required")
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path != catalog
    )
    files.append(catalog)
    if any(path.is_symlink() for path in files):
        raise ValueError("release assets must be regular files")
    # The tag endpoint only resolves published releases. gh also discovers
    # drafts; resolve its database ID before reading asset digests via the API.
    release_id = int(
        subprocess.check_output(
            [
                "gh",
                "release",
                "view",
                release,
                "--repo",
                repository,
                "--json",
                "databaseId",
                "--jq",
                ".databaseId",
            ],
            text=True,
        ).strip()
    )
    release_data = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repository}/releases/{release_id}"],
            text=True,
        )
    )
    existing = {asset["name"]: asset.get("digest") for asset in release_data["assets"]}
    expected = {path.name: _digest(path) for path in files}
    # Preflight the entire batch before making any change to the release.
    for path in files:
        if path.name in existing and existing[path.name] != expected[path.name]:
            raise ValueError(
                f"existing release asset differs or has no verifiable digest: {path.name}"
            )
    for path in files:
        if path.name in existing:
            print(f"Already verified: {path.name}")
            continue
        subprocess.run(
            ["gh", "release", "upload", release, "--repo", repository, str(path)],
            check=True,
        )
        print(f"Uploaded: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", args.release):
        parser.error("release must be a version tag")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("repository must be owner/name")
    publish_assets(args.directory, release=args.release, repository=args.repository)


if __name__ == "__main__":
    main()
