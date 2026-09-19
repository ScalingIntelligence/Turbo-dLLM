# Installation

## Install Turbo-dLLM

Turbo-dLLM supports Python 3.10 through 3.14. Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install turbo-dllm
dllm doctor
```

Use `python -m pip` instead of `pip` to avoid `command not found: pip` when
Python's scripts directory is not on `PATH`.

If you see `No matching distribution found`, check the available releases on
[PyPI](https://pypi.org/project/turbo-dllm/) and confirm that your environment
uses a supported Python version.

## Install GPU support

Install the bundle matching your Python, CUDA version, and GPU:

```bash
dllm bundle install --release v0.1.1 --auto
```

The command selects and verifies the correct native wheels. If no supported
bundle matches your system, it reports the detected environment and available
options.

Check a training configuration before launching it:

```bash
dllm doctor --config ./train.yaml
```

See [GPU bundles](../operations/gpu-bundles.md) for custom manifests and
offline installation.

## Optional model support

Qwen3.8 training requires its additional runtime:

```bash
python -m pip install "turbo-dllm[qwen3_8]==0.1.1"
```

DiffusionGemma expert parallelism requires DeepEP. Install the tested revision
after PyTorch, CUDA, and NVSHMEM are available:

```bash
python -m pip install --no-build-isolation \
  'deep-ep @ git+https://github.com/deepseek-ai/DeepEP.git@dd758caf451848bd150e1046af3d0a73e5fff38d'
```

## Install DFlash2 serving

Use a separate Linux environment with Python 3.10 through 3.13 for either
serving runtime:

```bash
python -m pip install "turbo-dllm[vllm]"   # vLLM
python -m pip install "turbo-dllm[sglang]" # SGLang
```

Follow the [DFlash2 guide](dflash2-training-and-serving.md) to train, export,
and serve a draft.

## Install from source

```bash
git clone https://github.com/ScalingIntelligence/Turbo-dLLM.git
cd Turbo-dLLM
python -m pip install -e '.[dev]'
dllm doctor
```

Native source builds are covered in [native development](../development/native.md).
