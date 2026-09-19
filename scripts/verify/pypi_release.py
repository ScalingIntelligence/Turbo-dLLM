#!/usr/bin/env python3
"""Fail before building when a release tag cannot be published to PyPI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class ReleasePreflightError(RuntimeError):
    """A release identity is invalid or already immutable on PyPI."""


def verify_release(
    *,
    project_path: Path,
    tag: str,
    index_url: str = "https://pypi.org/pypi",
) -> str:
    metadata = tomllib.loads(project_path.read_text(encoding="utf-8"))["project"]
    name = str(metadata["name"])
    version = str(metadata["version"])
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ReleasePreflightError(
            f"tag {tag} does not match package version {version} ({expected_tag})"
        )

    endpoint = (
        f"{index_url.rstrip('/')}/{quote(name, safe='')}/{quote(version, safe='')}/json"
    )
    request = Request(endpoint, headers={"User-Agent": "turbo-dllm-release-preflight"})
    try:
        with urlopen(request, timeout=30) as response:
            response.read(1)
    except HTTPError as error:
        if error.code == 404:
            return f"{name} {version} is available for publication"
        raise ReleasePreflightError(
            f"PyPI version check failed with HTTP {error.code}: {endpoint}"
        ) from error
    except URLError as error:
        raise ReleasePreflightError(f"PyPI version check failed: {error.reason}") from error

    raise ReleasePreflightError(
        f"{name} {version} already exists on PyPI and cannot be replaced; bump the version"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a tagged package version is unpublished on PyPI."
    )
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--index-url", default="https://pypi.org/pypi")
    args = parser.parse_args(argv)
    try:
        message = verify_release(
            project_path=args.project,
            tag=args.tag,
            index_url=args.index_url,
        )
    except (KeyError, OSError, ReleasePreflightError, tomllib.TOMLDecodeError) as error:
        print(error, file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
