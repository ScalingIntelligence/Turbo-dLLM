"""Build on CPU, then qualify one immutable CUDA 12.8 / CPython 3.12 bundle."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


SOURCE = Path("/opt/dllm/source")
DIST = Path("/opt/dllm/dist")


def build_wheels() -> None:
    import torch

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"expected CUDA 12.8 PyTorch, got {torch.version.cuda}")
    subprocess.run(
        [
            "bash",
            str(SOURCE / "scripts/build/build_cuda_wheels.sh"),
            "--python",
            sys.executable,
            "--cuda-arch-list",
            "9.0",
            "--output-dir",
            str(DIST),
        ],
        check=True,
        cwd=SOURCE,
    )


def install_wheels() -> None:
    package = next(DIST.glob("turbo_dllm-*.whl"))
    native = sorted(path for path in DIST.glob("*.whl") if path != package)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            f"turbo-dllm[gpu,test] @ {package.as_uri()}",
            *map(str, native),
        ],
        check=True,
        cwd="/tmp",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(package),
            *map(str, native),
        ],
        check=True,
        cwd="/tmp",
    )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


def cache_smoke_model() -> None:
    from huggingface_hub import snapshot_download
    import yaml

    from dllm_parallel.recipes import recipe_text

    snapshot_download(
        yaml.safe_load(recipe_text("smoke/cuda-fast-dllm-v2"))["model"]["id"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        max_workers=4,
    )


def qualify(release: str, revision: str, repository: str) -> list[str]:
    import modal
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError("qualification requires exactly two GPUs")
    if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(2)):
        raise RuntimeError("qualification requires SM90 GPUs")
    reports = Path("/tmp/qualification")
    reports.mkdir()
    python = sys.executable
    subprocess.run(["dllm", "doctor", "--training"], check=True, cwd="/tmp")
    smoke = reports / "smoke.yaml"
    subprocess.run(
        ["dllm", "recipe", "copy", "smoke/cuda-fast-dllm-v2", str(smoke)],
        check=True,
        cwd="/tmp",
    )
    subprocess.run(
        ["bash", str(SOURCE / "scripts/verify/gpu_smoke.sh"), "--config", str(smoke)],
        check=True,
        cwd="/tmp",
    )
    suites = {
        "attention-kernels": [
            str(SOURCE / "tests/unit/attention"),
            str(SOURCE / "tests/unit/kernels"),
        ],
        "distributed": ["-m", "distributed", str(SOURCE / "tests/distributed")],
        "checkpoint-resume": [
            "-m",
            "checkpoint_resume",
            str(SOURCE / "tests/integration"),
        ],
        "performance": [
            "-m",
            "performance",
            str(SOURCE / "tests/distributed/test_gpu_performance_smoke.py"),
        ],
    }
    for name, arguments in suites.items():
        subprocess.run(
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-o",
                "pythonpath=",
                f"--junitxml={reports / (name + '.xml')}",
                *arguments,
            ],
            check=True,
            cwd="/tmp",
        )
    base_url = f"https://github.com/{repository}/releases/download/{release}"
    subprocess.run(
        [
            python,
            "-m",
            "dllm_parallel.core.kernels.bundle_manifest",
            "create",
            "--directory",
            str(DIST),
            "--output",
            str(DIST / "gpu-sm90-cu128-cp312.json"),
            "--base-url",
            base_url,
            "--cuda",
            "12.8",
            "--architectures",
            "9.0",
            "--python-abi",
            "cp312",
            "--platform",
            "linux_x86_64",
            "--package-version",
            release.removeprefix("v"),
            "--source-revision",
            revision,
        ],
        check=True,
        cwd="/tmp",
    )
    (DIST / "qualification-sm90-cu128-cp312.json").write_text(
        json.dumps(
            {
                "release": release,
                "source_revision": revision,
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "devices": [torch.cuda.get_device_name(i) for i in range(2)],
                "suites": list(suites),
                "doctor": "passed",
                "smoke": "passed",
            },
            indent=2,
        )
        + "\n"
    )
    destination = Path("/release-assets") / revision
    destination.mkdir(parents=True, exist_ok=True)
    files = [*DIST.iterdir(), *reports.glob("*.xml")]
    for path in files:
        shutil.copy2(path, destination / path.name)
    modal.Volume.from_name("turbo-dllm-release-assets").commit()
    return [path.name for path in files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repository", default="ScalingIntelligence/Turbo-dLLM")
    parser.add_argument("--output-dir", type=Path, default=Path("dist-gpu"))
    args = parser.parse_args()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", args.release):
        parser.error("release must be a version tag, for example v0.1.1")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full lowercase Git commit")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("repository must be owner/name")
    import modal

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "clang", "ninja-build")
        .pip_install(
            "build==1.5.0", "ninja==1.13.0", "setuptools", "wheel", "packaging"
        )
        .pip_install(
            "torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128"
        )
        .env(
            {
                "MAX_JOBS": "4",
                "NVCC_THREADS": "2",
                "CUDA_HOME": "/usr/local/cuda",
                "TORCH_CUDA_ARCH_LIST": "9.0",
                "NVTE_CUDA_ARCHS": "90",
            }
        )
        .pip_install("setuptools==80.9.0")
        .run_commands(
            f"git clone --branch {args.release} --depth 1 https://github.com/{args.repository}.git {SOURCE}",
            f"test $(git -C {SOURCE} rev-parse HEAD) = {args.revision}",
            f"python -m pip install --no-build-isolation '{SOURCE}[test,kernel-build]'",
        )
        .run_function(build_wheels, cpu=8, memory=32768, timeout=7200)
        .env(
            {
                "CPATH": "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/include",
                "CUDNN_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cudnn",
                "LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib",
            }
        )
        .run_function(install_wheels, cpu=8, memory=32768, timeout=3600)
        .run_function(cache_smoke_model, cpu=2, memory=4096, timeout=1200)
    )
    volume = modal.Volume.from_name("turbo-dllm-release-assets", create_if_missing=True)
    app = modal.App("turbo-dllm-gpu-release")
    qualified = app.function(
        image=image,
        gpu="H100:2",
        cpu=8,
        memory=32768,
        timeout=1800,
        retries=0,
        volumes={"/release-assets": volume},
    )(qualify)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with modal.enable_output(), app.run():
        names = qualified.remote(args.release, args.revision, args.repository)
        for name in names:
            with (args.output_dir / name).open("wb") as stream:
                for chunk in volume.read_file(f"/{args.revision}/{name}"):
                    stream.write(chunk)
    print(f"Qualified GPU assets downloaded to {args.output_dir}")


if __name__ == "__main__":
    main()
