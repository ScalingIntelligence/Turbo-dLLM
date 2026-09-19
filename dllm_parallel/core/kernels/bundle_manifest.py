"""Create, validate, download, and install coordinated GPU wheel bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as platform_module
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from dllm_parallel import __version__


BUNDLE_FORMAT = "dllm.gpu_bundle.v1"
BUNDLE_CATALOG_FORMAT = "dllm.gpu_bundle_catalog.v1"
DEFAULT_RELEASES_URL = (
    "https://github.com/ScalingIntelligence/Turbo-dLLM/releases/download"
)
REQUIRED_DISTRIBUTIONS = (
    "bdlm_flash_attn_3",
    "turbo_dllm",
    "flash_attn_4",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MAX_MANIFEST_BYTES = 1024 * 1024
_RELEASE_TAG = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+-]*)?)$")


@dataclass(frozen=True)
class BundleEnvironment:
    """Runtime attributes that select one native bundle."""

    package_version: str
    python_abi: str
    platform: str
    cuda: str
    compute_capability: str


def _architectures(value: str | Sequence[str]) -> tuple[str, ...]:
    values = (
        re.split(r"[;,\s]+", value.strip())
        if isinstance(value, str)
        else [str(item).strip() for item in value]
    )
    result = tuple(item for item in values if item)
    if not result or any(not re.fullmatch(r"\d+\.\d+", item) for item in result):
        raise ValueError("architectures must contain values such as 8.9 or 9.0")
    return result


def _distribution_for_wheel(name: str) -> str:
    normalized = name.casefold().replace("-", "_")
    matches = [
        distribution
        for distribution in REQUIRED_DISTRIBUTIONS
        if normalized.startswith(f"{distribution}_")
    ]
    if len(matches) != 1:
        raise ValueError(f"unrecognized coordinated wheel: {name}")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle_manifest(
    *,
    directory: str | Path,
    output: str | Path,
    base_url: str,
    cuda: str,
    architectures: str | Sequence[str],
    python_abi: str,
    platform: str,
    package_version: str,
    source_revision: str,
) -> dict[str, Any]:
    """Write a deterministic manifest for exactly three coordinated wheels."""

    wheel_dir = Path(directory)
    wheels = sorted(wheel_dir.glob("*.whl"), key=lambda path: path.name)
    distributions = [_distribution_for_wheel(path.name) for path in wheels]
    if sorted(distributions) != sorted(REQUIRED_DISTRIBUTIONS):
        raise ValueError(
            "bundle directory must contain exactly one Turbo-dLLM, FA3, and FA4 wheel"
        )
    parsed_base = urlparse(base_url)
    if parsed_base.scheme != "https" or not parsed_base.netloc:
        raise ValueError("base_url must be an HTTPS URL")
    if not _GIT_REVISION.fullmatch(source_revision):
        raise ValueError("source_revision must be a full lowercase Git commit")

    manifest: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "package_version": str(package_version),
        "source_revision": source_revision,
        "python_abi": str(python_abi),
        "platform": str(platform),
        "cuda": str(cuda),
        "architectures": list(_architectures(architectures)),
        "artifacts": [
            {
                "name": path.name,
                "url": f"{base_url.rstrip('/')}/{path.name}",
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in wheels
        ],
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _release_version(release: str) -> str:
    match = _RELEASE_TAG.fullmatch(str(release))
    if match is None:
        raise ValueError("release must be a version tag such as v0.1.0")
    return match.group("version")


def create_bundle_catalog(
    *,
    directory: str | Path,
    output: str | Path,
    base_url: str,
    release: str,
) -> dict[str, Any]:
    """Create the immutable index used for exact automatic bundle selection."""

    package_version = _release_version(release)
    parsed_base = urlparse(base_url)
    if parsed_base.scheme != "https" or not parsed_base.netloc:
        raise ValueError("base_url must be an HTTPS URL")
    manifests: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(Path(directory).glob("gpu-*.json"), key=lambda item: item.name):
        if path.resolve() == Path(output).resolve():
            continue
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, Mapping) and parsed.get("format") == BUNDLE_FORMAT:
            manifests.append((path, parsed))
    if not manifests:
        raise ValueError("bundle catalog requires at least one GPU bundle manifest")

    bundles: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for path, manifest in manifests:
        _validate_bundle_manifest_schema(manifest)
        manifest_version = str(manifest.get("package_version", ""))
        if manifest_version != package_version:
            raise ValueError(
                f"{path.name} package version {manifest_version!r} does not match "
                f"release {release!r}"
            )
        architectures = _architectures(manifest.get("architectures", ()))
        python_abi = str(manifest.get("python_abi", ""))
        platform = str(manifest.get("platform", ""))
        cuda = str(manifest.get("cuda", ""))
        if not python_abi or not platform or not cuda:
            raise ValueError(f"{path.name} is missing bundle compatibility metadata")
        identity = (python_abi, platform, cuda, architectures)
        if identity in identities:
            raise ValueError(f"duplicate bundle compatibility entry: {identity!r}")
        identities.add(identity)
        bundles.append(
            {
                "manifest": path.name,
                "url": f"{base_url.rstrip('/')}/{path.name}",
                "sha256": _sha256(path),
                "python_abi": python_abi,
                "platform": platform,
                "cuda": cuda,
                "architectures": list(architectures),
            }
        )

    catalog: dict[str, Any] = {
        "format": BUNDLE_CATALOG_FORMAT,
        "release": release,
        "package_version": package_version,
        "bundles": bundles,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog


def detect_bundle_environment() -> BundleEnvironment:
    """Inspect the installed Python, CUDA runtime, and first visible GPU."""

    if sys.platform != "linux":
        raise RuntimeError("GPU bundles support Linux only")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a core dependency.
        raise RuntimeError("PyTorch must be installed before a GPU bundle") from exc
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("a visible CUDA GPU and CUDA-enabled PyTorch are required")
    major, minor = torch.cuda.get_device_capability()
    return BundleEnvironment(
        package_version=__version__,
        python_abi=f"cp{sys.version_info.major}{sys.version_info.minor}",
        platform=f"linux_{platform_module.machine().casefold()}",
        cuda=".".join(str(torch.version.cuda).split(".")[:2]),
        compute_capability=f"{major}.{minor}",
    )


def validate_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    environment: BundleEnvironment | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate schema, artifact trust constraints, and host compatibility."""

    architectures, artifacts = _validate_bundle_manifest_schema(manifest)
    host = detect_bundle_environment() if environment is None else environment
    comparisons = (
        ("package version", host.package_version, manifest.get("package_version")),
        ("Python ABI", host.python_abi, manifest.get("python_abi")),
        ("platform", host.platform, manifest.get("platform")),
        ("CUDA", host.cuda, manifest.get("cuda")),
    )
    for label, actual, expected in comparisons:
        if actual != str(expected):
            raise RuntimeError(
                f"GPU bundle {label} mismatch: host={actual!r}, bundle={expected!r}"
            )
    if host.compute_capability not in architectures:
        raise RuntimeError(
            "GPU bundle compute capability mismatch: "
            f"host={host.compute_capability!r}, bundle={architectures!r}"
        )
    return artifacts


def _validate_bundle_manifest_schema(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    """Validate manifest identity and artifacts without probing the local GPU."""

    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("unsupported GPU bundle manifest")
    if not _GIT_REVISION.fullmatch(str(manifest.get("source_revision", ""))):
        raise ValueError("bundle source_revision is invalid")
    architectures = _architectures(manifest.get("architectures", ()))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise ValueError("bundle must describe exactly three wheel artifacts")

    distributions: list[str] = []
    names: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("bundle artifact entries must be objects")
        name = str(artifact.get("name", ""))
        if PurePosixPath(name).name != name or not name.endswith(".whl"):
            raise ValueError(f"invalid wheel name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate wheel name: {name}")
        names.add(name)
        distributions.append(_distribution_for_wheel(name))
        parsed = urlparse(str(artifact.get("url", "")))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"artifact URL must use HTTPS: {name}")
        if PurePosixPath(parsed.path).name != name:
            raise ValueError(f"artifact URL filename does not match {name}")
        if not _SHA256.fullmatch(str(artifact.get("sha256", ""))):
            raise ValueError(f"invalid SHA-256 for {name}")
        if not isinstance(artifact.get("size"), int) or int(artifact["size"]) <= 0:
            raise ValueError(f"invalid artifact size for {name}")
    if sorted(distributions) != sorted(REQUIRED_DISTRIBUTIONS):
        raise ValueError("bundle artifact identities are incomplete")
    return architectures, tuple(artifacts)


def _read_url(url: str, *, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "turbo-dllm"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise RuntimeError(f"download exceeds the {limit}-byte safety limit")
    return payload


def _read_json_payload(source: str | Path) -> bytes:
    value = str(source)
    if value.startswith("https://"):
        return _read_url(value, limit=_MAX_MANIFEST_BYTES)
    if "://" in value:
        raise ValueError("bundle metadata URL must use HTTPS")
    payload = Path(value).read_bytes()
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise RuntimeError(
            f"bundle metadata exceeds the {_MAX_MANIFEST_BYTES}-byte safety limit"
        )
    return payload


def read_bundle_manifest(source: str | Path) -> dict[str, Any]:
    """Read a local manifest or HTTPS manifest URL with a size limit."""

    parsed = json.loads(_read_json_payload(source))
    if not isinstance(parsed, dict):
        raise ValueError("bundle manifest must be a JSON object")
    return parsed


def validate_bundle_catalog(
    catalog: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Validate the release catalog schema and its HTTPS trust boundary."""

    if catalog.get("format") != BUNDLE_CATALOG_FORMAT:
        raise ValueError("unsupported GPU bundle catalog")
    release = str(catalog.get("release", ""))
    version = _release_version(release)
    if str(catalog.get("package_version", "")) != version:
        raise ValueError("bundle catalog release and package version do not match")
    bundles = catalog.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("bundle catalog must describe at least one bundle")
    identities: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for entry in bundles:
        if not isinstance(entry, Mapping):
            raise ValueError("bundle catalog entries must be objects")
        name = str(entry.get("manifest", ""))
        if PurePosixPath(name).name != name or not name.endswith(".json"):
            raise ValueError(f"invalid bundle manifest name: {name!r}")
        parsed_url = urlparse(str(entry.get("url", "")))
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError(f"bundle manifest URL must use HTTPS: {name}")
        if PurePosixPath(parsed_url.path).name != name:
            raise ValueError(f"bundle manifest URL filename does not match {name}")
        if not _SHA256.fullmatch(str(entry.get("sha256", ""))):
            raise ValueError(f"invalid bundle manifest SHA-256: {name}")
        architectures = _architectures(entry.get("architectures", ()))
        python_abi = str(entry.get("python_abi", ""))
        platform = str(entry.get("platform", ""))
        cuda = str(entry.get("cuda", ""))
        if not python_abi or not platform or not cuda:
            raise ValueError(
                f"bundle catalog entry is missing compatibility data: {name}"
            )
        identity = (python_abi, platform, cuda, architectures)
        if identity in identities:
            raise ValueError(f"duplicate bundle compatibility entry: {identity!r}")
        identities.add(identity)
    return tuple(bundles)


def read_bundle_catalog(source: str | Path) -> dict[str, Any]:
    """Read and validate a local or HTTPS release catalog."""

    parsed = json.loads(_read_json_payload(source))
    if not isinstance(parsed, dict):
        raise ValueError("bundle catalog must be a JSON object")
    validate_bundle_catalog(parsed)
    return parsed


def select_bundle(
    catalog: Mapping[str, Any],
    *,
    environment: BundleEnvironment,
) -> Mapping[str, Any]:
    """Select one exact bundle or report the host and all qualified options."""

    bundles = validate_bundle_catalog(catalog)
    matches = [
        entry
        for entry in bundles
        if str(catalog["package_version"]) == environment.package_version
        and str(entry["python_abi"]) == environment.python_abi
        and str(entry["platform"]) == environment.platform
        and str(entry["cuda"]) == environment.cuda
        and environment.compute_capability in _architectures(entry["architectures"])
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError("bundle catalog contains multiple matches for this host")
    supported = "\n".join(
        "  - "
        f"Python ABI={entry['python_abi']}, platform={entry['platform']}, "
        f"CUDA={entry['cuda']}, "
        f"architectures={','.join(_architectures(entry['architectures']))}"
        for entry in bundles
    )
    raise RuntimeError(
        "no compatible GPU bundle for detected environment: "
        f"package={environment.package_version}, Python ABI={environment.python_abi}, "
        f"platform={environment.platform}, CUDA={environment.cuda}, "
        f"compute capability={environment.compute_capability}. "
        f"Available bundles for {catalog['release']}:\n{supported}"
    )


def catalog_url_for_release(
    release: str,
    *,
    releases_url: str = DEFAULT_RELEASES_URL,
) -> str:
    """Return the catalog URL for one validated immutable release tag."""

    _release_version(release)
    parsed = urlparse(releases_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("releases_url must be an HTTPS base URL")
    return f"{releases_url.rstrip('/')}/{release}/gpu-bundles.json"


def resolve_bundle_manifest(
    release: str,
    *,
    environment: BundleEnvironment | None = None,
    releases_url: str = DEFAULT_RELEASES_URL,
) -> dict[str, Any]:
    """Resolve and integrity-check the exact manifest for the detected host."""

    host = detect_bundle_environment() if environment is None else environment
    catalog_url = catalog_url_for_release(release, releases_url=releases_url)
    catalog = read_bundle_catalog(catalog_url)
    if str(catalog["release"]) != release:
        raise ValueError(
            f"bundle catalog release mismatch: requested={release!r}, "
            f"catalog={catalog['release']!r}"
        )
    entry = select_bundle(catalog, environment=host)
    payload = _read_json_payload(str(entry["url"]))
    if hashlib.sha256(payload).hexdigest() != str(entry["sha256"]):
        raise RuntimeError(f"bundle manifest SHA-256 mismatch: {entry['manifest']}")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("bundle manifest must be a JSON object")
    metadata_pairs = (
        ("package_version", catalog["package_version"], parsed.get("package_version")),
        ("python_abi", entry["python_abi"], parsed.get("python_abi")),
        ("platform", entry["platform"], parsed.get("platform")),
        ("cuda", entry["cuda"], parsed.get("cuda")),
    )
    for label, indexed, recorded in metadata_pairs:
        if str(indexed) != str(recorded):
            raise ValueError(f"bundle catalog {label} does not match selected manifest")
    if _architectures(entry["architectures"]) != _architectures(
        parsed.get("architectures", ())
    ):
        raise ValueError("bundle catalog architectures do not match selected manifest")
    validate_bundle_manifest(parsed, environment=host)
    return parsed


def download_bundle(
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    environment: BundleEnvironment | None = None,
) -> tuple[Path, ...]:
    """Download and hash-check all compatible bundle artifacts."""

    artifacts = validate_bundle_manifest(manifest, environment=environment)
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for artifact in artifacts:
        destination = destination_root / str(artifact["name"])
        temporary = destination.with_suffix(destination.suffix + ".part")
        digest = hashlib.sha256()
        size = 0
        request = urllib.request.Request(
            str(artifact["url"]), headers={"User-Agent": "turbo-dllm"}
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("xb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > int(artifact["size"]):
                        raise RuntimeError(
                            f"downloaded size exceeds manifest for {destination.name}"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
            if size != int(artifact["size"]):
                raise RuntimeError(f"downloaded size mismatch for {destination.name}")
            if digest.hexdigest() != artifact["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {destination.name}")
            os.replace(temporary, destination)
            downloaded.append(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return tuple(downloaded)


def _install_parsed_bundle(parsed: Mapping[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="dllm-gpu-bundle-") as directory:
        wheels = download_bundle(parsed, output_dir=directory)
        wheels_by_distribution = {
            _distribution_for_wheel(wheel.name): wheel.resolve() for wheel in wheels
        }
        package_wheel = wheels_by_distribution["turbo_dllm"]
        native_wheels = [
            wheels_by_distribution[distribution]
            for distribution in REQUIRED_DISTRIBUTIONS
            if distribution != "turbo_dllm"
        ]
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"turbo-dllm[gpu] @ {package_wheel.as_uri()}",
                *map(str, native_wheels),
            ],
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"pip failed with exit code {completed.returncode}")


def install_bundle(manifest: str | Path) -> None:
    """Install a compatible, hash-verified bundle and its GPU runtime."""

    _install_parsed_bundle(read_bundle_manifest(manifest))


def install_bundle_for_release(release: str) -> None:
    """Detect the host and install its exact qualified release bundle."""

    _install_parsed_bundle(resolve_bundle_manifest(release))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dllm_parallel.core.kernels.bundle_manifest"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--directory", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--base-url", required=True)
    create.add_argument("--cuda", required=True)
    create.add_argument("--architectures", required=True)
    create.add_argument("--python-abi", required=True)
    create.add_argument("--platform", required=True)
    create.add_argument("--package-version", default=__version__)
    create.add_argument("--source-revision", required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--directory", required=True, type=Path)
    catalog.add_argument("--output", required=True, type=Path)
    catalog.add_argument("--base-url", required=True)
    catalog.add_argument("--release", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "create":
        create_bundle_manifest(
            directory=args.directory,
            output=args.output,
            base_url=args.base_url,
            cuda=args.cuda,
            architectures=args.architectures,
            python_abi=args.python_abi,
            platform=args.platform,
            package_version=args.package_version,
            source_revision=args.source_revision,
        )
    elif args.command == "catalog":
        create_bundle_catalog(
            directory=args.directory,
            output=args.output,
            base_url=args.base_url,
            release=args.release,
        )


if __name__ == "__main__":
    main()


__all__ = (
    "BUNDLE_FORMAT",
    "BUNDLE_CATALOG_FORMAT",
    "BundleEnvironment",
    "DEFAULT_RELEASES_URL",
    "catalog_url_for_release",
    "create_bundle_catalog",
    "create_bundle_manifest",
    "detect_bundle_environment",
    "download_bundle",
    "install_bundle",
    "install_bundle_for_release",
    "read_bundle_catalog",
    "read_bundle_manifest",
    "resolve_bundle_manifest",
    "select_bundle",
    "validate_bundle_catalog",
    "validate_bundle_manifest",
)
