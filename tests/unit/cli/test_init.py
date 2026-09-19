from __future__ import annotations

from pathlib import Path

import pytest

from dllm_parallel import __version__
from dllm_parallel.cli.main import run
from dllm_parallel.data.schemas import PreparationSpec
from dllm_parallel.training.run_spec import load_run_spec


def test_init_creates_a_valid_portable_training_project(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "my run"

    assert run(["init", str(project)]) == 0

    assert sorted(path.name for path in project.iterdir()) == [
        ".gitignore",
        "README.md",
        "data",
        "prepare.yaml",
        "train.yaml",
    ]
    assert project.joinpath("data").is_dir()

    preparation = PreparationSpec.from_path(project / "prepare.yaml")
    assert preparation.source.type == "jsonl"
    assert preparation.source.path == str((project / "data/train.jsonl").resolve())
    assert preparation.output.path == str((project / "data/prepared").resolve())
    assert preparation.packing.maximum_length == 2048

    run_spec = load_run_spec(project / "train.yaml")
    assert run_spec.model.id == "Qwen/Qwen3-8B"
    assert run_spec.objective.name == "fast_dllm_v2"
    assert run_spec.objective.block_size == 32
    assert run_spec.data.input_mode == "dataset"
    assert run_spec.data.dataset_path == "data/prepared"
    assert run_spec.topology.context_parallel_size == 1
    assert run_spec.topology.block_parallel_size == 1
    assert run_spec.optimizer.backend == "torch_adamw"

    readme = project.joinpath("README.md").read_text(encoding="utf-8")
    assert '"text": "Your first training example."' in readme
    assert "dllm data prepare --config prepare.yaml" in readme
    assert "dllm data validate data/prepared --config train.yaml" in readme
    assert "dllm launch --config train.yaml --dry-run" in readme
    assert "dllm launch --config train.yaml --nproc-per-node 1" in readme
    assert f"dllm bundle install --release v{__version__} --auto" in readme

    gitignore = project.joinpath(".gitignore").read_text(encoding="utf-8")
    assert "/data/" in gitignore
    assert "/checkpoints/" in gitignore
    assert "/runs/" in gitignore

    assert project.joinpath("prepare.yaml").is_relative_to(project)
    assert str(tmp_path) not in project.joinpath("prepare.yaml").read_text(
        encoding="utf-8"
    )
    assert str(tmp_path) not in project.joinpath("train.yaml").read_text(
        encoding="utf-8"
    )

    output = capsys.readouterr().out
    assert f"Created Turbo-dLLM project at {project.resolve()}" in output
    assert "cd " in output
    assert "dllm data prepare --config prepare.yaml" in output


def test_init_refuses_to_overwrite_any_generated_file(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    existing = project / "train.yaml"
    existing.write_text("owned-by-user\n", encoding="utf-8")

    assert run(["init", str(project)]) == 2

    assert existing.read_text(encoding="utf-8") == "owned-by-user\n"
    assert not (project / "prepare.yaml").exists()
    assert not (project / "README.md").exists()
    error = capsys.readouterr().err
    assert "refusing to overwrite" in error
    assert "train.yaml" in error
    assert "Traceback" not in error


def test_init_can_add_scaffold_to_an_existing_directory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    notes = project / "notes.txt"
    notes.write_text("keep me\n", encoding="utf-8")

    assert run(["init", str(project)]) == 0

    assert notes.read_text(encoding="utf-8") == "keep me\n"
    assert (project / "train.yaml").is_file()


def test_generated_project_prepares_validates_and_dry_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 0
        mask_token_id = 8
        chat_template = ""

        def __len__(self) -> int:
            return 10

        def get_vocab(self) -> dict[str, int]:
            return {str(index): index for index in range(10)}

        def encode(
            self,
            text: str,
            *,
            add_special_tokens: bool = False,
        ) -> list[int]:
            assert not add_special_tokens
            return [ord(character) % 7 + 1 for character in text]

    project = tmp_path / "project"
    assert run(["init", str(project)]) == 0
    capsys.readouterr()
    project.joinpath("data/train.jsonl").write_text(
        '{"text":"A generic training document."}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dllm_parallel.data.prepare.load_tokenizer",
        lambda spec: Tokenizer(),
    )
    monkeypatch.chdir(project)

    assert run(["data", "prepare", "--config", "prepare.yaml"]) == 0
    capsys.readouterr()
    assert (
        run(
            [
                "data",
                "validate",
                "data/prepared",
                "--config",
                "train.yaml",
            ]
        )
        == 0
    )
    assert "compatible" in capsys.readouterr().out
    assert run(["launch", "--config", "train.yaml", "--dry-run"]) == 0
    assert "dllm_parallel.training" in capsys.readouterr().out


def test_root_help_advertises_project_initialization(capsys) -> None:
    assert run(["--help"]) == 0

    output = capsys.readouterr().out
    assert "init" in output
    assert "Turbo-dLLM" in output
