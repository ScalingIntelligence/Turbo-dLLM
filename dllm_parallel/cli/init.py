# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Create a portable, editable Turbo-dLLM starter project."""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from dllm_parallel import __version__
from dllm_parallel.data.schemas import PreparationSpec
from dllm_parallel.training.run_spec import RunSpec


_PREPARE_YAML = """\
# Compile ordinary JSONL text records into the optimized training format.
format: dllm.data.prepare.v1
source:
  type: jsonl
  path: data/train.jsonl
records:
  type: text
  text_field: text
tokenizer:
  model: Qwen/Qwen3-8B
  trust_remote_code: false
  use_fast: true
  add_eos: true
supervision:
  policy: full
packing:
  maximum_length: 2048
  alignment: 32
  overflow: split
  alignment_policy: truncate_right
output:
  path: data/prepared
  format: auto
  overwrite: false
"""


_TRAIN_YAML = """\
# A conservative one-GPU starter. Scale topology and optimizer deliberately.
launch:
  recipe_kind: smoke
  seed: 2026
model:
  id: Qwen/Qwen3-8B
  family: causal_lm
  seq_len: 2048
  mask_token_id: 151665
  dtype: bf16
objective:
  name: fast_dllm_v2
  block_size: 32
  noise_schedule: linear_mask
  loss_weighting: unit
  noise_schedule_epsilon: 0.001
data:
  input_mode: dataset
  dataset_path: data/prepared
  shuffle: true
training:
  batch_size: 1
  steps: 10
  gradient_accumulation_steps: 1
  activation_checkpointing: true
topology:
  context_parallel_size: 1
  block_parallel_size: 1
  tensor_parallel_size: 1
  expert_parallel_size: 1
  sequence_parallel: false
optimizer:
  backend: torch_adamw
  lr: 0.000001
kernel:
  runtime_jit: false
checkpointing:
  save_checkpoint_dir: checkpoints
  save_checkpoint_interval: 10
  keep_last_n: 2
  save_final: true
logging:
  wandb: false
"""


_README_TEMPLATE = """\
# Turbo-dLLM starter project

This project prepares ordinary text records offline, validates the resulting
artifact, and launches the unchanged optimized Turbo-dLLM training runtime.
Commands below assume the current directory is this project.

## 1. Install

```bash
python -m pip install 'turbo-dllm[data]'
dllm doctor
```

GPU training also needs the exact bundle published for your release and host:

```bash
dllm bundle install --release __RELEASE_TAG__ --auto
```

## 2. Add data

Create `data/train.jsonl` with one JSON object per line. The starter
configuration reads the `text` field:

```json
{"text": "Your first training example."}
{"text": "Another document to include in the corpus."}
```

JSONL, text, Parquet, Hugging Face datasets, messages, prompt/completion, and
pretokenized inputs are supported. Change `prepare.yaml` for another source or
record schema; keep tokenization and formatting outside the training hot path.

## 3. Prepare and validate

```bash
dllm data prepare --config prepare.yaml
dllm data inspect data/prepared
dllm data validate data/prepared --config train.yaml
dllm config validate --config train.yaml
dllm doctor --config train.yaml
```

Preparation is deterministic and refuses to overwrite an existing artifact.
Use `dllm data prepare --config prepare.yaml --overwrite` only when replacement
is intentional.

## 4. Preview and launch

```bash
dllm launch --config train.yaml --dry-run
dllm launch --config train.yaml --nproc-per-node 1
```

The default is a short single-GPU Fast-dLLM v2 run with the generic causal-LM
adapter. Before a real run, choose the model and immutable revision, sequence
and block lengths, parallel topology, optimizer, step count, and checkpoint
policy appropriate for your hardware. For distributed training, set the
topology in `train.yaml` and pass the matching `--nproc-per-node`; multi-node
rendezvous options are available from `dllm launch --help`.

See the Turbo-dLLM data-preparation and RunSpec documentation for all fields.
"""


_GITIGNORE = """\
# Local data and generated artifacts
/data/
/checkpoints/
/runs/
/.cache/
"""


def _project_files() -> Mapping[Path, str]:
    return {
        Path("prepare.yaml"): _PREPARE_YAML,
        Path("train.yaml"): _TRAIN_YAML,
        Path("README.md"): _README_TEMPLATE.replace(
            "__RELEASE_TAG__", f"v{__version__}"
        ),
        Path(".gitignore"): _GITIGNORE,
    }


def _validated_templates(destination: Path) -> None:
    prepare = yaml.safe_load(_PREPARE_YAML)
    train = yaml.safe_load(_TRAIN_YAML)
    if not isinstance(prepare, Mapping) or not isinstance(train, Mapping):
        raise RuntimeError("built-in project templates are not YAML mappings")
    PreparationSpec.from_mapping(prepare, base_dir=destination)
    RunSpec.from_mapping(train)


def initialize_project(destination: str | Path) -> tuple[Path, ...]:
    """Create a starter project without overwriting existing user files."""

    root = Path(destination).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"project destination is not a directory: {root}")

    files = _project_files()
    targets = tuple(root / relative for relative in files)
    conflicts = tuple(path for path in targets if path.exists())
    if conflicts:
        names = ", ".join(str(path.relative_to(root)) for path in conflicts)
        raise ValueError(f"refusing to overwrite existing project file(s): {names}")
    data_dir = root / "data"
    if data_dir.exists() and not data_dir.is_dir():
        raise ValueError(f"project data path is not a directory: {data_dir}")

    _validated_templates(root)

    created_files: list[Path] = []
    created_directories: list[Path] = []
    try:
        if not root.exists():
            root.mkdir(parents=True)
            created_directories.append(root)
        if not data_dir.exists():
            data_dir.mkdir()
            created_directories.append(data_dir)
        for relative, content in files.items():
            target = root / relative
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            created_files.append(target)
        # Parse the actual bytes written to catch future template or I/O regressions.
        PreparationSpec.from_path(root / "prepare.yaml")
        RunSpec.from_mapping(
            yaml.safe_load((root / "train.yaml").read_text(encoding="utf-8"))
        )
    except BaseException:
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except OSError:
                pass
        raise

    return targets


def run_init(argv: Sequence[str]) -> int:
    """Execute ``dllm init``."""

    parser = argparse.ArgumentParser(
        prog="dllm init",
        description="Create a portable Turbo-dLLM training project.",
    )
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(list(argv))

    destination = args.destination.expanduser().resolve()
    initialize_project(destination)
    print(f"Created Turbo-dLLM project at {destination}")
    print("Next:")
    print(f"  cd {shlex.quote(str(destination))}")
    print("  # Add JSONL records to data/train.jsonl")
    print("  dllm data prepare --config prepare.yaml")
    print("  dllm data validate data/prepared --config train.yaml")
    print("  dllm launch --config train.yaml --dry-run")
    return 0


__all__ = ("initialize_project", "run_init")
