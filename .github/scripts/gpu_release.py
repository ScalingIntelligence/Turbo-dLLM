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
ATTENTION = Path("/opt/dllm/attention")


def build_wheels() -> None:
    import torch

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"expected CUDA 12.8 PyTorch, got {torch.version.cuda}")
    attention = ATTENTION
    extra = ["--attention-wheels-dir", str(attention)] if attention.is_dir() else []
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
            *extra,
        ],
        check=True,
        cwd=SOURCE,
    )

    if attention.is_dir():
        shutil.copy2(
            attention / "attention-provenance.json", DIST / "attention-provenance.json"
        )


def validate_cached_attention(wheels: list[Path], revision: str) -> None:
    import hashlib
    import sysconfig
    import zipfile
    import torch

    fa4 = next(
        (wheel for wheel in wheels if wheel.name.startswith("flash_attn_4-")), None
    )
    if fa4 is not None and f"+bdlm.{revision[:12]}-" not in fa4.name:
        raise RuntimeError("cached FA4 source revision mismatch")
    fa3 = next(
        (wheel for wheel in wheels if wheel.name.startswith("bdlm_flash_attn_3-")), None
    )
    if fa3 is None:
        return
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    from packaging.requirements import Requirement

    pins = {
        Requirement(item).name: str(Requirement(item).specifier)
        for item in tomllib.loads((SOURCE / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]["gpu"]
    }
    with zipfile.ZipFile(fa3) as archive:
        metadata = json.loads(archive.read("flash_attn_3/build_metadata.json"))
        if metadata["source"]["revision"] != revision:
            raise RuntimeError("cached FA3 source revision mismatch")
        build = metadata["build"]
        if (
            build["torch"] != str(torch.__version__)
            or build["torch_cuda"] != torch.version.cuda
            or build["python_extension_suffix"]
            != sysconfig.get_config_var("EXT_SUFFIX")
            or build["cxx11_abi"] != bool(torch._C._GLIBCXX_USE_CXX11_ABI)
            or build["cuda_architectures"] != ["sm_90a"]
        ):
            raise RuntimeError("cached attention ABI mismatch")
        binary = metadata["binary"]
        if hashlib.sha256(archive.read(binary["path"])).hexdigest() != binary["sha256"]:
            raise RuntimeError("cached attention binary checksum mismatch")
        manifests = [
            name
            for name in archive.namelist()
            if name.startswith("flash_attn_3/bdlm_splitd/_artifacts/")
            and name.endswith("/manifest.json")
        ]
        if len(manifests) != 1:
            raise RuntimeError("cached Split-D manifest mismatch")
        splitd = json.loads(archive.read(manifests[0]))
        if (
            splitd.get("complete") is not True
            or "==" + splitd.get("tvm_ffi", "") != pins["apache-tvm-ffi"]
            or "==" + splitd.get("cutlass", "") != pins["nvidia-cutlass-dsl"]
            or splitd.get("torch") != str(torch.__version__)
            or splitd.get("torch_cuda") != str(torch.version.cuda)
        ):
            raise RuntimeError("cached Split-D ABI or completeness mismatch")


def prepare_source(release: str, revision: str, repository: str) -> None:
    """Reuse attention only inside the same immutable toolchain image."""
    import hashlib
    import os

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(SOURCE), *arguments], text=True
        ).strip()

    def profile(text: str) -> list[str]:
        return re.findall(r"(?:BUILD_TARGET|FLASH_ATTENTION_[A-Z0-9_]+)=[^\s\\]+", text)

    shutil.rmtree(ATTENTION, ignore_errors=True)
    old_revision = git("rev-parse", "HEAD")
    paths = (
        "third_party/flash-attention/hopper",
        "third_party/flash-attention/flash_attn/cute",
        "third_party/flash-attention/csrc/cutlass/include",
    )
    old_trees = {path: git("rev-parse", f"HEAD:{path}") for path in paths}
    old_profile = profile((SOURCE / "scripts/build/build_cuda_wheels.sh").read_text())
    git("fetch", "--no-tags", "--depth=1", "origin", f"refs/tags/{release}")
    if git("rev-parse", "FETCH_HEAD^{commit}") != revision:
        raise RuntimeError("release tag revision mismatch")
    git("checkout", "--detach", revision)
    current_trees = {path: git("rev-parse", f"HEAD:{path}") for path in paths}
    profile_matches = old_profile == profile(
        (SOURCE / "scripts/build/build_cuda_wheels.sh").read_text()
    )
    shared_matches = old_trees[paths[2]] == current_trees[paths[2]]
    reuse_fa3 = (
        shared_matches
        and profile_matches
        and old_trees[paths[0]] == current_trees[paths[0]]
    )
    reuse_fa4 = shared_matches and old_trees[paths[1]] == current_trees[paths[1]]
    wheels = [
        wheel
        for pattern, reusable in (
            ("bdlm_flash_attn_3-*.whl", reuse_fa3),
            ("flash_attn_4-*.whl", reuse_fa4),
        )
        if reusable
        for wheel in DIST.glob(pattern)
    ]
    if not wheels:
        print("Attention inputs changed; rebuilding all wheels")
        shutil.rmtree(DIST)
        return
    import torch
    import sysconfig

    cache = ATTENTION
    cache.mkdir()
    validate_cached_attention(wheels, old_revision)
    records = []
    for wheel in wheels:
        shutil.copy2(wheel, cache / wheel.name)
        records.append(
            {
                "name": wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "source_revision": old_revision,
            }
        )
    (cache / "attention-provenance.json").write_text(
        json.dumps(
            {
                "immutable_build_image": os.environ["DLLM_ATTENTION_IMAGE"],
                "source_trees": old_trees,
                "build_profile": old_profile,
                "python_abi": sysconfig.get_config_var("SOABI"),
                "torch": str(torch.__version__),
                "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                "cuda": subprocess.check_output(["nvcc", "--version"], text=True),
                "architectures": "9.0",
                "components": records,
            },
            indent=2,
        )
        + "\n"
    )
    shutil.rmtree(DIST)


def build_runtime_wheels() -> None:
    """Compile the two sdist-only dependencies against the pinned Torch ABI."""
    import os
    import tomllib
    from packaging.requirements import Requirement

    requirements = tomllib.loads((SOURCE / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["gpu"]
    wanted = {"deepspeed", "transformer-engine"}
    selected = [
        "transformer-engine-torch" + str(Requirement(item).specifier)
        if Requirement(item).name == "transformer-engine"
        else item
        for item in requirements
        if Requirement(item).name in wanted
    ]
    if len(selected) != 2:
        raise RuntimeError("expected two pinned prebuilt runtime dependencies")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(DIST),
            *selected,
        ],
        check=True,
        env={**os.environ, "DS_BUILD_OPS": "0", "NVTE_PYTORCH_FORCE_BUILD": "TRUE"},
        cwd="/tmp",
    )


def install_wheels() -> None:
    """Resolve every runtime dependency as a wheel in a new isolated venv."""
    import os

    # Modal injects SDK dependencies through PYTHONPATH; exclude them from
    # every subprocess that qualifies the independently installed runtime.
    os.environ.pop("PYTHONPATH", None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    python = "/opt/dllm/clean/bin/python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--clear", "/opt/dllm/clean"], check=True
    )
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "torch==2.10.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        check=True,
    )
    package = next(DIST.glob("turbo_dllm-*.whl"))
    native = sorted(path for path in DIST.glob("*.whl") if path != package)
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            f"turbo-dllm[gpu] @ {package.as_uri()}",
            *map(str, native),
        ],
        check=True,
        cwd="/tmp",
    )
    subprocess.run(
        [python, "-m", "pip", "install", "--only-binary=:all:", "pytest>=8"], check=True
    )
    subprocess.run(
        [
            "bash",
            str(SOURCE / "scripts/build/build_portable.sh"),
            "--python",
            sys.executable,
            "--output-dir",
            "/opt/dllm/portable",
        ],
        check=True,
        cwd="/tmp",
    )
    portable = next(Path("/opt/dllm/portable").glob("*.whl"))
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(portable),
        ],
        check=True,
    )
    subprocess.run([python, "-m", "pip", "check"], check=True)
    # Catch native runtime loading failures on CPU before allocating GPUs.
    subprocess.run(
        [
            python,
            "-c",
            "from dllm_parallel.core.parallel.transformer_engine import load_transformer_engine; load_transformer_engine(); from flash_attn_3.bdlm_splitd import verify_splitd_artifacts; verify_splitd_artifacts()",
        ],
        check=True,
        cwd="/tmp",
    )


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
    import os

    # Modal injects SDK dependencies through PYTHONPATH; exclude them from
    # every subprocess that qualifies the independently installed runtime.
    os.environ.pop("PYTHONPATH", None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    python = "/opt/dllm/clean/bin/python"
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
    subprocess.run(
        [
            python,
            str(SOURCE / ".github/scripts/verify_bundle_install.py"),
            str(DIST / "gpu-sm90-cu128-cp312.json"),
        ],
        check=True,
        cwd="/tmp",
    )
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
                "--maxfail=1",
                "-o",
                "pythonpath=",
                f"--junitxml={reports / (name + '.xml')}",
                *arguments,
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
                "binary_only_clean_environment": "passed",
                "portable_to_native_installer": "passed",
                "build_image": __import__("os").environ.get("DLLM_ATTENTION_IMAGE"),
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
    parser.add_argument("--attention-image", default="")
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

    if args.attention_image and not re.fullmatch(
        r"im-[A-Za-z0-9]+", args.attention_image
    ):
        parser.error("attention image must be an immutable Modal image ID")
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
    )
    if args.attention_image:
        image = (
            modal.Image.from_id(args.attention_image)
            .env({"DLLM_ATTENTION_IMAGE": args.attention_image})
            .run_function(
                prepare_source,
                args=(args.release, args.revision, args.repository),
                cpu=2,
                memory=4096,
                timeout=300,
            )
        )
    image = (
        image.run_function(build_wheels, cpu=8, memory=32768, timeout=7200)
        .env(
            {
                "CPATH": "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/include",
                "CUDNN_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cudnn",
                "LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib",
            }
        )
        .pip_install(
            "transformer-engine[core-cu12]==2.13.0", "cmake==4.0.3", "pybind11==3.0.1"
        )
        .run_function(build_runtime_wheels, cpu=8, memory=32768, timeout=3600)
        .run_function(install_wheels, cpu=8, memory=32768, timeout=3600)
        .env(
            {
                "PATH": "/opt/dllm/clean/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
            }
        )
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
