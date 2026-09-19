from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import dllm_parallel


ROOT = Path(__file__).resolve().parents[2]


def test_project_metadata_is_complete_for_public_indexes() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project["name"] == "turbo-dllm"
    assert project["readme"] == "README.md"
    assert project["authors"]
    assert project["urls"]["Source"].startswith("https://github.com/")
    assert project["requires-python"] == ">=3.10"
    assert "Programming Language :: Python :: 3.10" in project["classifiers"]
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]
    assert "Programming Language :: Python :: 3.14" in project["classifiers"]
    assert not any(item.startswith("License ::") for item in project["classifiers"])
    assert project["scripts"]["dllm"] == "dllm_parallel.cli:main"
    assert project["urls"] == {
        "Documentation": "https://github.com/ScalingIntelligence/Turbo-dLLM/tree/main/docs",
        "Issues": "https://github.com/ScalingIntelligence/Turbo-dLLM/issues",
        "Source": "https://github.com/ScalingIntelligence/Turbo-dLLM",
    }
    assert project["optional-dependencies"]["sglang"] == [
        "sglang==0.5.20; sys_platform == 'linux' and python_version < '3.14'",
        "cuda-tile==1.6.0rc5; sys_platform == 'linux' and python_version < '3.14'",
        "flash-attn-4==4.0.0b19; sys_platform == 'linux' and python_version < '3.14'",
    ]
    assert project["optional-dependencies"]["vllm"] == [
        "vllm==0.29.0; sys_platform == 'linux' and python_version < '3.14'"
    ]
    assert project["optional-dependencies"]["capture"] == [
        "specforge==0.2.0; sys_platform == 'linux' and python_version >= '3.11' and python_version < '3.14'",
        "flash-attn-4==4.0.0b15; sys_platform == 'linux' and python_version >= '3.11' and python_version < '3.14'",
    ]


def test_portable_metadata_offers_coordinated_gpu_and_model_extras() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    extras = project["optional-dependencies"]

    assert "production" not in extras
    assert {
        "torch==2.10.0",
        "transformers==5.13.0",
        "accelerate==1.14.0",
        "deepspeed==0.19.2",
        "transformer-engine[core_cu12,pytorch]==2.13.0",
        "cuda-bindings==12.9.4",
        "cuda-python==12.9.4",
        "apache-tvm-ffi==0.1.12",
        "nvidia-cutlass-dsl==4.5.2",
        "nvidia-cutlass-dsl-libs-base==4.5.2",
        "torch-c-dlpack-ext==0.1.5",
        "quack-kernels==0.5.0",
    }.issubset(extras["gpu"])
    assert "transformer-engine[core_cu12,pytorch]==2.13.0" in extras["qwen3_8"]
    assert "deepep" not in extras
    dependencies = [
        *project.get("dependencies", []),
        *(item for values in extras.values() for item in values),
    ]
    assert not any(" @ " in item or "git+" in item for item in dependencies)


def test_serving_extras_are_isolated_from_training_environments() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    conflicts = metadata["tool"]["uv"]["conflicts"]
    pairs = {frozenset(entry["extra"] for entry in conflict) for conflict in conflicts}

    for serving in ("sglang", "vllm"):
        for training in ("gpu", "qwen3_8", "kernel-build"):
            assert frozenset((serving, training)) in pairs
    assert frozenset(("sglang", "vllm")) in pairs
    for environment in ("sglang", "vllm", "gpu", "qwen3_8", "kernel-build"):
        assert frozenset(("capture", environment)) in pairs


def test_deepep_source_build_is_exactly_pinned_outside_project_metadata() -> None:
    constraint = ROOT / "constraints" / "deepep-source.txt"

    assert constraint.read_text(encoding="utf-8").splitlines() == [
        "deep-ep @ git+https://github.com/deepseek-ai/DeepEP.git@"
        "dd758caf451848bd150e1046af3d0a73e5fff38d"
    ]


def test_development_extra_can_build_portable_artifacts() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert "build==1.5.0" in project["optional-dependencies"]["dev"]


def test_data_extra_owns_optional_huggingface_and_parquet_frontends() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project["optional-dependencies"]["data"] == ["datasets==4.8.5"]


def test_package_exposes_pep440_version() -> None:
    assert dllm_parallel.__version__ == "0.1.1"


def test_documented_install_paths_are_portable_and_bundle_aware() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("# Turbo-dLLM\n")
    assert "python -m pip install turbo-dllm" in readme
    assert "python3 -m venv .venv" in readme
    assert "source .venv/bin/activate" in readme
    assert "\npip install" not in readme
    assert "pip install dllm-parallel" not in readme
    assert "dllm bundle install --release v0.1.1 --auto" in readme
    assert "dllm launch" in readme
    assert "install_production.sh" not in readme


def test_source_manifest_has_all_license_and_build_boundaries() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required_entries = (
        "constraints/gpu-cu128.txt",
        "constraints/deepep-source.txt",
        "recursive-include scripts *.sh",
        "recursive-include containers *",
        "recursive-include docs *.md",
        "third_party/flash-attention/LICENSE",
        "third_party/flash-attention/PROVENANCE.md",
        "third_party/flash-attention/hopper",
        "third_party/flash-attention/csrc/cutlass/include",
    )

    assert [entry for entry in required_entries if entry not in manifest] == []
    assert "scripts/install.sh" not in manifest


def test_generated_distribution_directory_is_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/dist/" in ignored
