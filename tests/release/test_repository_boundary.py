from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_research_integrations_are_outside_the_release_repository() -> None:
    forbidden = (
        "dllm_baseline",
        "tools/modal",
        "scripts/laude",
        "scripts/data",
        "dllm_parallel/data/coderforge.py",
        "dllm_parallel/data/coderforge_terminal.py",
        "dllm_parallel/data/dflash_feature_capture.py",
        "dllm_parallel/data/openhands_rollouts.py",
        "dllm_parallel/data/swebench_trajectories.py",
        "scripts/dllm_baseline_train.sh",
    )

    present = [relative for relative in forbidden if (ROOT / relative).exists()]

    assert present == []


def test_package_metadata_has_no_dataset_or_provider_entrypoints() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8").casefold()
    forbidden = ("prepare-coderforge", "modal", "dllm_baseline")

    assert [name for name in forbidden if name in metadata] == []
