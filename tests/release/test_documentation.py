from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def test_governance_and_documentation_sections_exist() -> None:
    required = (
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "docs/getting-started/installation.md",
        "docs/getting-started/dflash2-training-and-serving.md",
        "docs/concepts/architecture.md",
        "docs/configuration/run-spec.md",
        "docs/configuration/data-preparation.md",
        "docs/models/supported.md",
        "docs/parallelism/topologies.md",
        "docs/operations/checkpointing.md",
        "docs/api/index.md",
        "docs/development/testing.md",
        "docs/release-notes/v0.1.1.md",
    )
    assert [path for path in required if not (ROOT / path).is_file()] == []


def test_internal_markdown_links_resolve() -> None:
    broken: list[str] = []
    for source in (ROOT / "README.md", *(ROOT / "docs").rglob("*.md")):
        for target in LINK.findall(source.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            destination = (source.parent / target.split("#", 1)[0]).resolve()
            if not destination.exists():
                broken.append(f"{source.relative_to(ROOT)} -> {target}")
    assert broken == []


def test_public_docs_have_no_machine_or_removed_layout_paths() -> None:
    forbidden = (
        "/users/",
        "/persistent/",
        "/workspace/",
        "recipes/prod",
        "scripts/release",
        "release branch",
        "oss branch",
        "coderforge",
        "openhands",
        "swe-bench",
        "dllm_baseline",
    )
    offenders: list[str] = []
    for path in (ROOT / "README.md", *(ROOT / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8").casefold()
        if any(value in text for value in forbidden):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_readme_is_concise_and_user_facing() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert len(readme.split()) < 500
    assert "## Repository boundaries" not in readme
    for command in (
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "python -m pip install turbo-dllm",
        "dllm init",
        "dllm data prepare",
        "dllm doctor --config",
        "dllm launch",
    ):
        assert command in readme


def test_installation_uses_an_isolated_path_safe_environment() -> None:
    installation = (ROOT / "docs/getting-started/installation.md").read_text(
        encoding="utf-8"
    )

    assert "Python 3.10 through 3.14" in installation
    assert "python3 -m venv .venv" in installation
    assert "source .venv/bin/activate" in installation
    assert "python -m pip install turbo-dllm" in installation
    assert "command not found: pip" in installation
    assert "No matching distribution found" in installation
    assert "\npip install" not in installation


def test_data_preparation_docs_cover_installed_frontend_and_boundaries() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/configuration/data-preparation.md").read_text(
        encoding="utf-8"
    )

    for command in (
        "dllm data prepare",
        "dllm data inspect",
        "dllm data validate",
        "dllm data stats",
    ):
        assert command in readme or command in guide
    for source_type in ("huggingface", "jsonl", "parquet", "text", "pretokenized"):
        assert f"type: {source_type}" in guide
    for record_type in ("messages", "prompt_completion", "pretokenized"):
        assert f"type: {record_type}" in guide
    assert "offline" in guide.lower()
    assert "GPU training loop" in guide
    assert "dataset-specific" in guide.lower()


def test_installed_gpu_and_distributed_workflows_are_documented() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    installation = (ROOT / "docs/getting-started/installation.md").read_text(
        encoding="utf-8"
    )
    topologies = (ROOT / "docs/parallelism/topologies.md").read_text(encoding="utf-8")

    assert "dllm bundle install --release v0.1.1 --auto" in readme
    assert "dllm bundle install --release v0.1.1 --auto" in installation
    assert "dllm launch" in readme
    assert "dllm launch" in topologies
    assert "scripts/launch/torchrun.sh" not in topologies


def test_documented_config_validation_uses_supported_cli_arguments() -> None:
    for relative in (
        "docs/configuration/run-spec.md",
        "docs/configuration/recipes.md",
        "docs/getting-started/quickstart.md",
        "docs/getting-started/dflash2-training-and-serving.md",
    ):
        contents = (ROOT / relative).read_text(encoding="utf-8")
        assert "dllm config validate" in contents
        assert "--seq-len" not in contents
        assert "--target-feature-path" not in contents
        assert "--save-checkpoint-dir" not in contents


def test_dflash2_guide_is_one_complete_installed_workflow() -> None:
    guide = (ROOT / "docs/getting-started/dflash2-training-and-serving.md").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for step in range(1, 8):
        assert f"## {step}." in guide
    for command in (
        "dllm data prepare",
        "dllm dflash prepare-features",
        'python -m pip install "turbo-dllm[capture]"',
        "dllm recipe copy runs/dflash2-qwen3-8-27b-1m",
        "dllm config validate",
        "dllm launch",
        "dllm dflash export",
        'python -m pip install "turbo-dllm[vllm]"',
        "dllm dflash serve-vllm",
        'python -m pip install "turbo-dllm[sglang]"',
        "dllm dflash serve-sglang",
    ):
        assert command in guide
    assert "dflash2-training-and-serving.md" in readme


def test_repository_run_recipe_docs_are_not_profiling_claims() -> None:
    guide = (ROOT / "docs/configuration/recipes.md").read_text(encoding="utf-8")

    for name in (
        "dflash2-qwen3-8-27b-1m",
        "dflash2-muse-glimmer-30b-1m",
        "diffusiongemma-26b-sft-256k",
        "qwen3-8-27b-fast-dllm-v2-256k",
    ):
        assert name in guide
    assert "performance recipes" not in guide.lower()
    assert "tokens/s" not in guide
    assert "speedup" not in guide.lower()
