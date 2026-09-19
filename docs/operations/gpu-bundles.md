# GPU bundles

Turbo-dLLM publishes GPU bundles for supported combinations of Python, CUDA,
Linux, and GPU architecture.

Install the matching bundle automatically:

```bash
dllm bundle install --release v0.1.1 --auto
```

The installer verifies every downloaded wheel and runs `dllm doctor
--training`. It does not substitute a different CUDA version or compile missing
kernels automatically.

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
