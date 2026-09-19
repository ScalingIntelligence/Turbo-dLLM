# Contributing

## Before opening a change

Open an issue for changes to public APIs, configuration semantics, checkpoint
formats, supported platforms, or native artifact identity. Small fixes and
documentation improvements can go directly to a pull request.

Do not add provider-specific launchers, branded dataset pipelines, local paths,
or training behavior to repository scripts. Reusable behavior belongs in the
package; scripts only orchestrate it.

## Development setup

```bash
git clone https://github.com/ScalingIntelligence/Turbo-dLLM.git
cd Turbo-dLLM
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
scripts/verify/cpu.sh --python .venv/bin/python
```

Use a separate Git worktree for changes that could disrupt active training
runs. Never commit generated native binaries, credentials, model weights,
datasets, run logs, or cache directories.

## Required checks

```bash
ruff check dllm_parallel tests
python -m compileall -q dllm_parallel tests
python -m pytest -q tests/unit tests/packaging tests/release
uv lock --check
```

New behavior requires a focused failing test before implementation. Native and
distributed changes also require the relevant self-hosted GPU workflow. State
the hardware, CUDA/PyTorch versions, commands, and observed result in the pull
request; do not generalize beyond the tested configuration.

## Compatibility rules

- Preserve objective math, loss scaling, collectives, and checkpoint semantics
  unless the change explicitly proposes a versioned compatibility break.
- Keep package imports CPU-safe and avoid heavyweight eager imports.
- Add public names deliberately and document their stability.
- Fail closed for unsupported topology, ABI, CUDA architecture, or artifact
  combinations.
- Update the changelog and release notes for user-visible behavior.
