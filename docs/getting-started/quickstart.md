# Quickstart

For a real dataset, generate an editable starter project:

```bash
dllm init ./my-run
cd ./my-run
```

Add records such as `{"text": "A training document."}` to
`data/train.jsonl`, then prepare, validate, preview, and launch:

```bash
dllm data prepare --config prepare.yaml
dllm data validate data/prepared --config train.yaml
dllm launch --config train.yaml --dry-run
dllm doctor --config train.yaml
dllm launch --config train.yaml --nproc-per-node 1
```

Edit training settings in `train.yaml` and data settings in `prepare.yaml`.
`dllm init` does not download data or model weights.

List the included smoke tests and examples:

```bash
dllm recipe list
dllm recipe show smoke/cpu-config
dllm config validate --recipe smoke/cpu-config
```

Copy and edit an example:

```bash
dllm recipe copy examples/fast-dllm-v2-qwen3 ./qwen3.yaml
dllm config validate --config ./qwen3.yaml
```

Change the sequence length, data path, and topology in `qwen3.yaml` before
launching it.

After installing a compatible GPU bundle, launch with:

```bash
dllm launch --config ./qwen3.yaml --nproc-per-node 8
```

The launcher checks the configuration, GPU runtime, and prepared data before
starting workers.

To train a DFlash2 draft and deploy it with vLLM or SGLang, use the
[step-by-step DFlash2 guide](dflash2-training-and-serving.md).
