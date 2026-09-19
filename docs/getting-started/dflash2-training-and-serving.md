# Train and serve DFlash2

This guide trains the Qwen3.8 27B DFlash2 draft at a 1M-token context and
exports it for vLLM or SGLang. Your dataset stays outside Turbo-dLLM.

The example needs eight H100 80GB GPUs for training. Feature capture also uses
eight GPUs here, but it can run as a separate job.

## 1. Install the training environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "turbo-dllm[data]"
dllm bundle install --release v0.1.1 --auto
```

## 2. Prepare your dataset

Create `train.jsonl` with one conversation per line:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Create `prepare-qwen.yaml`:

```yaml
format: dllm.data.prepare.v1
source:
  type: jsonl
  path: ./train.jsonl
records:
  type: messages
tokenizer:
  model: Qwen/Qwen3.8-27B
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
  add_eos: true
supervision:
  policy: assistant_only
packing:
  maximum_length: 1048576
  alignment: 1
  overflow: reject
output:
  path: ./data/prepared
  format: indexed
```

Prepare and check the artifact:

```bash
dllm data prepare --config prepare-qwen.yaml
dllm data validate ./data/prepared
```

Records may be shorter than 1M tokens. The compact feature format stores only
real verifier features and reconstructs padding when training reads a batch.

## 3. Capture verifier features

DFlash2 trains from frozen verifier features. Capture them in a separate
environment so SpecForge and the training bundle do not compete for CUDA
dependencies. Use Linux with Python 3.11 through 3.13. The
`dllm dflash prepare-features` command runs under
`torchrun` so the verifier can use tensor parallelism:

```bash
python3 -m venv .capture-venv
source .capture-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "turbo-dllm[capture]"

torchrun --standalone --nproc-per-node=8 -m dllm_parallel.cli \
  dflash prepare-features \
  --source ./data/prepared \
  --output ./data/dflash-features \
  --draft-model incoai/Qwen3.8-27B-DFlash2 \
  --draft-revision dedf8df68adfb1afeaf7b7480c0a0243108177b4 \
  --verifier-model Qwen/Qwen3.8-27B \
  --verifier-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --sequence-length 1048576 \
  --tensor-parallel-size 8
```

The output directory must be visible to every capture rank and to the training
job. The command validates hashes, layer IDs, token alignment, and feature
width before publishing the final manifest.

## 4. Start training

Return to the training environment and copy the editable run configuration:

```bash
source .venv/bin/activate
dllm recipe copy runs/dflash2-qwen3-8-27b-1m ./qwen-dflash2.yaml
```

In `qwen-dflash2.yaml`, set:

```yaml
data:
  target_feature_path: ./data/dflash-features
checkpointing:
  save_checkpoint_dir: ./checkpoints/qwen-dflash2
```

Validate the edited configuration and preview the exact launch:

```bash
dllm config validate --config ./qwen-dflash2.yaml
dllm launch --config ./qwen-dflash2.yaml --nproc-per-node 8 --dry-run
```

Start the run by removing `--dry-run`:

```bash
dllm launch --config ./qwen-dflash2.yaml --nproc-per-node 8
```

The configuration defaults to 1,000 optimizer steps and checkpoints every 100
steps. Edit the copied YAML for your training schedule, learning rate, and
checkpoint policy.

## 5. Export the trained draft

```bash
dllm dflash export \
  --checkpoint ./checkpoints/qwen-dflash2 \
  --output ./exports/qwen-dflash2 \
  --model-id incoai/Qwen3.8-27B-DFlash2 \
  --model-revision dedf8df68adfb1afeaf7b7480c0a0243108177b4 \
  --block-size 8
```

Turbo-dLLM downloads the exact pinned base draft when `--base-model` is
omitted, merges the trained tensors, and validates the serving artifact.

## 6. Serve with vLLM

Use a clean serving environment. Native vLLM DFlash2 currently supports the
Qwen3 export contract.

```bash
python3 -m venv .vllm-venv
source .vllm-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "turbo-dllm[vllm]"

dllm dflash serve-vllm \
  --target Qwen/Qwen3.8-27B \
  --draft ./exports/qwen-dflash2 \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 8000
```

The server exposes the OpenAI-compatible API at `http://localhost:8000/v1`.

## 7. Serve with SGLang

Use this instead of the vLLM environment:

```bash
python3 -m venv .sglang-venv
source .sglang-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "turbo-dllm[sglang]"

dllm dflash serve-sglang \
  --target Qwen/Qwen3.8-27B \
  --draft ./exports/qwen-dflash2 \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 30000
```

The server exposes the OpenAI-compatible API at `http://localhost:30000/v1`.

For Muse-Glimmer, copy `runs/dflash2-muse-glimmer-30b-1m`, use its pinned
draft and verifier IDs during capture and export, and serve the export with
SGLang. Its 1M setting reproduces the paper's systems configuration and
exceeds the verifier checkpoint's declared 128K generation context.
