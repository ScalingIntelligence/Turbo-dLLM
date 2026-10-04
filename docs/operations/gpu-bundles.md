# GPU bundles

Turbo-dLLM publishes GPU bundles for supported combinations of Python, CUDA,
Linux, and GPU architecture.

The first qualified target is CPython 3.12 on Ubuntu 24.04 x86_64, CUDA 12.8
PyTorch, and SM90 (H100/H200). Other Python, CUDA, and GPU combinations
require their own qualified bundle or an explicit source build. The CUDA
version printed by `nvidia-smi` is the driver limit; bundle selection uses
`torch.version.cuda`.

Install the matching bundle automatically:

```bash
dllm bundle install --release v0.1.6 --auto
```

The installer verifies every downloaded wheel and runs `dllm doctor
--training`. It does not substitute a different CUDA version or compile missing
kernels automatically.

An HTTP 404 for `gpu-bundles.json` means the requested release has no
published catalog. The portable `py3-none-any.whl` does not contain native
GPU kernels, and installing the `gpu` extra alone does not supply them.
Check the release assets or follow [native development](../development/native.md).

Version v0.1.1 published only portable artifacts. Upgrade the installer before
selecting the v0.1.6 bundle:

```bash
python -m pip install --upgrade "turbo-dllm==0.1.6"
dllm bundle install --release v0.1.6 --auto
```

The v0.1.6 installer explicitly replaces same-version portable and native
wheels after resolving the GPU runtime dependencies. This avoids pip
retaining an already-installed portable wheel.

For a reviewed local manifest or private HTTPS mirror:

```bash
dllm bundle install --manifest ./gpu-sm90-cu128.json
```

Run a configuration-specific check before training:

```bash
dllm doctor --config ./train.yaml
```

Qwen3.8 also needs `turbo-dllm[qwen3_8]`. DiffusionGemma expert parallelism
needs a compatible DeepEP installation; the tested source revision is recorded
in `constraints/deepep-source.txt`.

For native development and runtime JIT requirements, see
[native development](../development/native.md).

## Publishing and repairing bundles

Maintainers configure the repository Actions secrets `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`. The release workflow builds native wheels on CPU,
then qualifies them on two SM90 GPUs. GPU qualification must succeed before
PyPI publication. The GitHub release stays a draft until every asset is
uploaded successfully. No self-hosted runner is needed.

For v0.1.6 or newer tags with the runtime-wheel manifest format, repair an
unpublished release after an orchestration failure by running
**Release** from the `main` branch with its immutable version tag. The
portable artifacts and native wheels are built from that tag, even when the
workflow repair is newer. The `pypi` environment allows `v*` tags. A repair
from `main` also requires that environment to allow the `main` branch during
the repair. Published PyPI versions cannot be overwritten.

For a compatible v0.1.6 or newer release missing GPU assets, run
**GPU validation** from Actions with its version tag. The older v0.1.1
installer and manifest do not support the new runtime wheels; upgrade
instead of backfilling that portable-only release. Validation builds the
exact tagged commit without
republishing PyPI. Successful runs upload the wheels, compatibility
manifest, qualification reports, checksums, and SBOMs; `gpu-bundles.json`
is uploaded last so automatic installation cannot select an unfinished
upload. Identical existing assets are skipped; mismatched assets stop
publication and are never overwritten. If an upload is interrupted, rerun
the failed publishing job to reuse the qualified workflow artifacts.

Before publication, `scripts/verify/gpu_release_assets.py` checks the source
revision, package version, release URLs, wheel sizes, and SHA-256 hashes.
The GPU checks load installed wheels outside the checkout and run the
training smoke, attention/kernel correctness, distributed correctness,
checkpoint resume, and performance instrumentation suites. Compilation
uses a cached CUDA development image, and smoke model downloads run on CPU
before allocating GPUs. GPU allocation is limited to the
qualification step, with no automatic retries.

The release builder also supplies Transformer Engine PyTorch and DeepSpeed
wheels. Installation resolves GPU dependencies with `--only-binary=:all:` in
a clean Python environment; a missing binary dependency stops installation
instead of silently compiling against a different PyTorch ABI. DeepSpeed
optional custom operations are not precompiled by this bundle.

The qualified build uses Ubuntu 24.04, CPython 3.12, PyTorch 2.10.0 and CUDA
12.8 on SM90. Other Linux baselines require independent compatibility checks.
An optional `GPU_ATTENTION_IMAGE` repository variable selects an immutable
Modal build image containing previously built attention wheels. Reuse requires
identical vendor source trees, build flags and wheel ABI metadata; the image
preserves the entire original compiler toolchain. `attention-provenance.json`
records the cached component revisions and hashes separately from the release
revision. Changed source inputs trigger a complete rebuild.
