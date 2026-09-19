from __future__ import annotations

import subprocess
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from dllm_parallel.core.profiling.release_gates import run_release_policy_gates


ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str) -> dict[str, Any]:
    path = ROOT / ".github/workflows" / name
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict), path
    return loaded


def _step_names(job: dict[str, Any]) -> set[str]:
    return {
        str(step["name"])
        for step in job.get("steps", ())
        if isinstance(step, dict) and "name" in step
    }


def _used_actions(job: dict[str, Any]) -> set[str]:
    return {
        str(step["uses"])
        for step in job.get("steps", ())
        if isinstance(step, dict) and "uses" in step
    }


def test_release_workflows_have_a_gated_publish_graph() -> None:
    workflows = ROOT / ".github/workflows"
    assert {path.name for path in workflows.glob("*.yml")} == {
        "ci.yml",
        "gpu-validation.yml",
        "pages.yml",
        "release.yml",
    }

    ci = _workflow("ci.yml")
    assert set(ci["on"]) == {"pull_request", "push"}
    assert "release/**" in ci["on"]["push"]["branches"]
    assert set(ci["jobs"]) == {"portable", "checkpoint-resume", "package"}
    assert set(ci["jobs"]["package"]["needs"]) == {"portable", "checkpoint-resume"}

    gpu = _workflow("gpu-validation.yml")
    assert "workflow_call" in gpu["on"]
    qualify = gpu["jobs"]["qualify"]
    assert {"self-hosted", "gpu"} <= set(qualify["runs-on"])
    assert {
        "Build coordinated native wheels",
        "Verify wheel identities and native manifests",
        "GPU smoke",
        "Attention and kernel correctness",
        "Distributed correctness",
        "Checkpoint resume",
        "Performance instrumentation smoke",
    } <= _step_names(qualify)

    release = _workflow("release.yml")
    assert release["permissions"] == {"contents": "read"}
    assert release["on"]["push"]["tags"] == ["v*"]
    assert set(release["jobs"]) == {"portable", "publish-pypi", "github-release"}
    assert release["jobs"]["publish-pypi"]["needs"] == "portable"
    assert release["jobs"]["publish-pypi"]["environment"]["name"] == "pypi"
    assert release["jobs"]["portable"]["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    assert release["jobs"]["publish-pypi"]["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }
    assert release["jobs"]["github-release"]["permissions"] == {
        "contents": "write"
    }
    assert release["jobs"]["github-release"]["needs"] == "publish-pypi"
    assert "pypa/gh-action-pypi-publish@release/v1" in _used_actions(
        release["jobs"]["publish-pypi"]
    )
    assert "actions/attest-build-provenance@v2" in _used_actions(
        release["jobs"]["portable"]
    )


class _PyPIResponseHandler(BaseHTTPRequestHandler):
    status = 404

    def do_GET(self) -> None:  # noqa: N802
        assert self.path == "/pypi/turbo-dllm/0.1.1/json"
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{}')

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _run_pypi_preflight(tmp_path: Path, *, status: int, tag: str = "v0.1.1"):
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "turbo-dllm"\nversion = "0.1.1"\n',
        encoding="utf-8",
    )
    _PyPIResponseHandler.status = status
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PyPIResponseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify/pypi_release.py"),
                "--project",
                str(project),
                "--tag",
                tag,
                "--index-url",
                f"http://127.0.0.1:{server.server_port}/pypi",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_pypi_preflight_accepts_an_unpublished_release(tmp_path: Path) -> None:
    completed = _run_pypi_preflight(tmp_path, status=404)

    assert completed.returncode == 0, completed.stderr
    assert "turbo-dllm 0.1.1 is available for publication" in completed.stdout


def test_pypi_preflight_rejects_an_immutable_existing_release(tmp_path: Path) -> None:
    completed = _run_pypi_preflight(tmp_path, status=200)

    assert completed.returncode == 1
    assert "turbo-dllm 0.1.1 already exists on PyPI" in completed.stderr


def test_pypi_preflight_rejects_a_tag_version_mismatch(tmp_path: Path) -> None:
    completed = _run_pypi_preflight(tmp_path, status=404, tag="v0.2.0")

    assert completed.returncode == 1
    assert "tag v0.2.0 does not match package version 0.1.1" in completed.stderr


def test_gpu_wheel_smoke_uses_a_clean_dependency_resolving_environment() -> None:
    workflow = (ROOT / ".github/workflows/gpu-validation.yml").read_text(
        encoding="utf-8"
    )

    assert "python -m venv" in workflow
    assert "turbo-dllm[gpu,test]" in workflow
    assert "--no-deps" not in workflow
    assert '"$GITHUB_WORKSPACE/tests/unit/attention"' in workflow
    assert '"$GITHUB_WORKSPACE/tests/unit/kernels"' in workflow


def test_portable_wheel_smoke_does_not_duplicate_runtime_dependencies() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "python -m venv --system-site-packages" in workflow
    assert "pip install --force-reinstall --no-deps dist/*.whl" in workflow


def test_ci_qualifies_the_documented_python_range() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.12", "3.14"]' in ci


def test_gpu_qualification_does_not_shadow_installed_wheels() -> None:
    steps = {
        step.get("name"): step
        for step in _workflow("gpu-validation.yml")["jobs"]["qualify"]["steps"]
    }
    for name in (
        "GPU smoke",
        "Attention and kernel correctness",
        "Distributed correctness",
        "Checkpoint resume",
        "Performance instrumentation smoke",
    ):
        assert steps[name]["working-directory"] == "${{ runner.temp }}"
        if name != "GPU smoke":
            assert "-o pythonpath=" in steps[name]["run"]
    assert (
        '"$GITHUB_WORKSPACE/tests/distributed/test_gpu_performance_smoke.py"'
        in (steps["Performance instrumentation smoke"]["run"])
    )


def test_cuda_wheel_snapshot_includes_package_metadata() -> None:
    builder = (ROOT / "scripts/build/build_cuda_wheels.sh").read_text()
    assert "pyproject.toml setup.py README.md LICENSE NOTICE" in builder


def test_cuda_wheel_snapshot_keeps_builder_but_not_generated_artifacts(
    tmp_path,
) -> None:
    source, staged = tmp_path / "source", tmp_path / "staged"
    staged.mkdir()
    paths = (
        "pyproject.toml",
        "setup.py",
        "README.md",
        "LICENSE",
        "NOTICE",
        "dllm_parallel/__init__.py",
        "scripts/build/build_cuda_wheels.sh",
        "third_party/flash-attention/README.md",
        "third_party/flash-attention/hopper/setup.py",
        "third_party/flash-attention/hopper/build/stale.txt",
        "third_party/flash-attention/flash_attn/cute/interface.py",
        "third_party/flash-attention/flash_attn/cute/build/stale.txt",
        "third_party/flash-attention/csrc/cutlass/include/header.h",
    )
    for name in paths:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    builder = (ROOT / "scripts/build/build_cuda_wheels.sh").read_text()
    snapshot = "tar \\\n" + builder.split("\ntar \\\n", 1)[1].split("\n(\n", 1)[0]
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", snapshot],
        check=True,
        env={**os.environ, "ROOT": str(source), "BUILD_ROOT": str(staged)},
    )
    assert (staged / "scripts/build/build_cuda_wheels.sh").is_file()
    assert (staged / "README.md").is_file()
    assert (staged / "LICENSE").is_file()
    assert not list(staged.rglob("stale.txt"))


def test_release_publishes_portable_artifacts() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert 'gh release create "$GITHUB_REF_NAME" release-assets/*' in workflow
    assert "portable-dist" in workflow
    assert "portable-reports" in workflow


def test_release_downloads_and_container_metadata_use_public_repository() -> None:
    from dllm_parallel.core.kernels.bundle_manifest import DEFAULT_RELEASES_URL

    public_repository = "https://github.com/ScalingIntelligence/Turbo-dLLM"
    assert DEFAULT_RELEASES_URL == f"{public_repository}/releases/download"
    container = (ROOT / "containers/cuda/Containerfile").read_text(encoding="utf-8")
    assert f'org.opencontainers.image.source="{public_repository}"' in container


def test_cuda_builder_cleans_pre_rename_distribution_wheels() -> None:
    builder = (ROOT / "scripts/build/build_cuda_wheels.sh").read_text(encoding="utf-8")

    assert '"$OUTPUT_DIR"/dllm_parallel-*.whl' in builder
    assert '"$OUTPUT_DIR"/turbo_dllm-*.whl' in builder


def test_release_repository_has_complete_test_taxonomy() -> None:
    tests = ROOT / "tests"
    assert not (tests / "unit_tests").exists()
    for category in ("unit", "integration", "distributed", "packaging", "release"):
        folder = tests / category
        assert folder.is_dir()
        assert any(folder.rglob("test_*.py")), category


def test_container_builder_rejects_a_mutable_base_before_building(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build/build_container.sh"),
            "--base-image",
            "ubuntu:24.04",
            "--wheel-dir",
            str(wheel_dir),
            "--image",
            "example.invalid/dllm:test",
            "--source-revision",
            "0" * 40,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "immutable @sha256: digest" in completed.stderr


def test_portable_builder_isolated_from_an_ignored_local_build_package(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    script = repository / "scripts" / "build" / "build_portable.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes((ROOT / "scripts/build/build_portable.sh").read_bytes())
    (repository / ".gitignore").write_text("/build/\n/dist/\n")

    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release-test@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        check=True,
    )

    # Reproduce the common maintainer state that used to shadow `python -m build`.
    (repository / "build").mkdir()
    (repository / "build" / "__init__.py").write_text("")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ ${1:-} == -c ]]; then printf '%s\\n' \"$0\"; exit 0; fi\n"
        '[[ $PWD != "$POISON_ROOT" ]] || exit 72\n'
        "while (($#)); do\n"
        "  if [[ $1 == --outdir ]]; then shift; outdir=$1; break; fi\n"
        "  shift\n"
        "done\n"
        'mkdir -p "$outdir"\n'
        'touch "$outdir/turbo_dllm-0.1.0-py3-none-any.whl"\n'
        'touch "$outdir/turbo_dllm-0.1.0.tar.gz"\n'
    )
    fake_python.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(script),
            "--python",
            str(fake_python),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ],
        cwd=repository,
        env={"PATH": "/usr/bin:/bin", "POISON_ROOT": str(repository)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.suffix for path in (tmp_path / "artifacts").iterdir()) == [
        ".gz",
        ".whl",
    ]


def test_portable_source_policy_does_not_require_generated_native_binaries() -> None:
    assert run_release_policy_gates(ROOT, require_native_artifacts=False) == []


def test_dflash_serving_verifiers_expose_help_without_optional_runtimes() -> None:
    for name in ("sglang_dflash2.sh", "vllm_dflash2.sh"):
        script = ROOT / "scripts" / "verify" / name
        completed = subprocess.run(
            ["bash", str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "DFlash2" in completed.stdout or "DFLASH" in completed.stdout
