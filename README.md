# Turbo-dLLM

Turbo-dLLM is a highly optimized distributed training library for diffusion
language models and diffusion-based speculative decoders. It includes the
official implementation of context-sharded block parallelism for scaling
training to large contexts, plus typed configuration, prepared-data runtimes,
checkpointing, and optimized CUDA kernels.

Paper: [Block Parallelism For Efficient Distributed Long-Context Diffusion
Language Model Training](https://arxiv.org/abs/2609.19242)

## Install

Install the portable package for configuration, data preparation, APIs, and
CPU-safe validation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install turbo-dllm
dllm doctor
```

GPU training uses a bundle matched to the host's Python, CUDA, and GPU
architecture.

The qualified target is Python 3.12, CUDA 12.8, Linux x86_64, and SM90
(H100/H200). Install a CUDA 12.8 PyTorch runtime before selecting the bundle:

```bash
dllm bundle install --release v0.1.2 --auto
```

The installer downloads an exact supported bundle and verifies its native
artifacts. See [installation](docs/getting-started/installation.md) for source
installs, offline mirrors, and model-specific extras.

## Start training

Create a starter project:

```bash
dllm init ./my-run
cd ./my-run
```

Add JSONL records such as `{"text": "A training document."}` to
`data/train.jsonl`, then prepare and launch:

```bash
dllm data prepare --config prepare.yaml
dllm data validate data/prepared --config train.yaml
dllm launch --config train.yaml --dry-run
dllm doctor --config train.yaml
dllm launch --config train.yaml --nproc-per-node 1
```

The launcher validates the config and runtime before starting workers. For
distributed training, change the topology in `train.yaml` and set the matching
process count:

```bash
dllm launch --config train.yaml --nproc-per-node 8
```

For speculative training and deployment, follow the
[DFlash2 train-to-serve guide](docs/getting-started/dflash2-training-and-serving.md).

## Data and recipes

Prepare Hugging Face datasets, JSONL, Parquet, text, or token IDs with text,
chat, prompt/completion, and token-level supervision. Tokenization runs
outside the GPU training loop.

```bash
python -m pip install 'turbo-dllm[data]'
dllm data prepare --config prepare.yaml
dllm data inspect data/prepared
dllm data stats data/prepared
```

Explore packaged recipes:

```bash
dllm recipe list
dllm recipe show smoke/cuda-fast-dllm-v2
dllm recipe copy examples/fast-dllm-v2-qwen3 ./run.yaml
dllm config validate --config ./run.yaml
```

## Supported training

- Models: generic causal LMs, DFlash, DiffusionGemma, Nemotron Labs Diffusion,
  and Qwen3.8.
- Objectives: standard block diffusion, Fast-dLLM v2, DFlash distillation, and
  DiffusionGemma native SFT.
- Parallelism: data, context, block, tensor, sequence, FSDP, and supported
  DiffusionGemma expert parallelism.
- Operations: deterministic data artifacts, checkpoint/resume, profiling, and
  optional W&B logging.

Unsupported combinations fail during validation instead of silently falling
back.

## Documentation

- [Quickstart](docs/getting-started/quickstart.md)
- [Train and serve DFlash2](docs/getting-started/dflash2-training-and-serving.md)
- [Data preparation](docs/configuration/data-preparation.md)
- [RunSpec configuration](docs/configuration/run-spec.md)
- [Supported models](docs/models/supported.md)
- [Parallelism](docs/parallelism/topologies.md)
- [GPU bundles](docs/operations/gpu-bundles.md)
- [API](docs/api/index.md)
- [Contributing](CONTRIBUTING.md) and [security](SECURITY.md)

## Citation

If you use Turbo-dLLM in your research, please cite:

```bibtex
@misc{suresh2026blockparallelismefficientdistributed,
  title = {Block Parallelism For Efficient Distributed Long-Context Diffusion Language Model Training},
  author = {Tarun Suresh and Pranshu Chaturvedi and Hangoo Kang and Parth Shroff and Ishan S. Khare and Hermann Kumbong and Azalia Mirhoseini},
  year = {2026},
  eprint = {2609.19242},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2609.19242}
}
```

## License

First-party code is Apache-2.0. Vendored components retain their upstream
licenses; see [NOTICE](NOTICE) and
[FlashAttention provenance](third_party/flash-attention/PROVENANCE.md).
