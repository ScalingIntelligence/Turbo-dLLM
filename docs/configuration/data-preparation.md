# Data preparation

`Turbo-dLLM` compiles records offline into memory-mapped artifacts before
GPU training starts. Tokenization, chat formatting, checksumming, truncation,
and alignment therefore add no work to the GPU training loop.

Install the optional Hugging Face/Parquet source support with:

```bash
python -m pip install 'turbo-dllm[data]'
```

Local text, JSONL, and token-file preparation uses the portable package and
does not require the optional dependency.

## Commands

`dllm init PROJECT` generates a commented JSONL preparation configuration and
a matching training configuration. It is the shortest path for a new run;
the complete schema below is available when a different source or supervision
policy is needed.

```bash
dllm data prepare --config prepare.yaml
dllm data inspect /data/prepared
dllm data validate /data/prepared
dllm data validate /data/prepared --config train.yaml
dllm data stats /data/prepared
```

`inspect` reads manifest metadata without hashing large payloads. `validate`
checks the artifact schema, byte counts, record offsets, supervision masks,
and SHA-256 of every payload. With `--config`, it also checks compatibility
with the training `RunSpec`. Every command accepts `--json`; `prepare` accepts
`--output` and explicit `--overwrite` overrides.

## Complete configuration shape

```yaml
format: dllm.data.prepare.v1

source:
  type: huggingface
  id: organization/dataset
  name: optional-subset
  revision: immutable-source-revision
  split: train
  streaming: false
  loader_kwargs: {}

records:
  type: messages
  messages_field: messages
  role_field: role
  content_field: content
  sample_id_field: id
  group_id_field: conversation_id

tokenizer:
  model: organization/model
  revision: immutable-tokenizer-revision
  trust_remote_code: false
  use_fast: true
  add_eos: true
  kwargs: {}
  chat_template_kwargs: {}

supervision:
  policy: assistant_only

packing:
  maximum_length: 131072
  alignment: 256
  overflow: reject
  alignment_policy: truncate_left
  minimum_length: 256
  pad_token_id: null
  separator_token_id: null

output:
  path: /data/prepared
  format: auto
  overwrite: false
```

Unknown fields and invalid combinations are rejected. Relative local source
and output paths are resolved relative to the preparation config file.

## Sources

### Hugging Face

```yaml
source:
  type: huggingface
  id: organization/dataset
  name: optional-configuration
  revision: immutable-revision
  split: train
  streaming: true
```

The adapter delegates explicit options to `datasets.load_dataset`. Authentication
values may be supplied through the normal Hugging Face environment or
`loader_kwargs`; keys containing `token`, `key`, `password`, `secret`, or
`credential` are redacted from artifact provenance.

### JSONL

```yaml
source:
  type: jsonl
  paths: [train-000.jsonl.gz, train-001.jsonl]
```

Plain JSONL and gzip-compressed JSONL are supported. Blank lines are ignored,
and every non-empty line must be a JSON object. File paths and globs are sorted
for deterministic traversal.

### Parquet

```yaml
source:
  type: parquet
  path: shards/*.parquet
  split: train
```

Parquet loading uses the optional Hugging Face Datasets dependency and supports
the same explicit `loader_kwargs` escape hatch.

### Text

```yaml
source:
  type: text
  path: corpus/*.txt
  text_mode: document
  encoding: utf-8
```

`text_mode` is `document` or `line`. Document mode treats each file as one
record. Line mode yields each non-empty line independently.

### Pretokenized files

```yaml
source:
  type: pretokenized
  path: tokens.i32
```

Supported inputs are `.i32`, `.int32`, `.i64`, `.int64`, `.bin`, `.tokens`,
`.npy`, `.pt`, `.json`, `.jsonl`, and `.jsonl.gz`. One-dimensional arrays are
one record; two-dimensional arrays yield one record per row. JSON records can
carry loss masks and sample/group IDs. Raw binary and NumPy inputs are memory
mapped and flow through preparation as numeric views; the writer converts only
bounded segments, avoiding a Python-object copy of the token corpus.

## Record schemas

### Text records

```yaml
records:
  type: text
  text_field: text
```

Dot-separated field paths such as `document.body` are supported.

### Message records

```yaml
records:
  type: messages
  messages_field: messages
  role_field: role
  content_field: content
supervision:
  policy: assistant_only
```

Additional message keys are retained for tool-aware chat templates.
Assistant-only supervision requires `tokenizer.apply_chat_template` to return
an assistant token mask. Preparation fails if the template cannot identify
assistant tokens; it never guesses text boundaries.

### Prompt/completion records

```yaml
records:
  type: prompt_completion
  prompt_field: prompt
  completion_field: completion
supervision:
  policy: completion_only
```

Prompt and completion may both be strings or both be message sequences. String
pairs are tokenized separately and concatenated. Message pairs use the model's
chat template: the prompt is rendered with a generation prompt, the combined
conversation is rendered without one, and preparation requires the first token
sequence to be an exact prefix of the second. It fails instead of guessing when
a template does not preserve that boundary. Only the completion and an
explicitly added EOS token are supervised.

```json
{
  "prompt": [{"role": "user", "content": "Explain tensor parallelism."}],
  "completion": [{"role": "assistant", "content": "Tensor parallelism..."}]
}
```

### Pretokenized records

```yaml
records:
  type: pretokenized
  input_ids_field: input_ids
  labels_field: labels
  loss_mask_field: loss_mask
  assistant_mask_field: assistant_masks
  completion_mask_field: completion_mask
supervision:
  policy: provided
```

For `provided`, `loss_mask` takes precedence and `labels != -100` is the
fallback. `assistant_only` and `completion_only` use their named mask fields.
`full` ignores input masks and builds a continuous full-sequence stream.

## Length and packing policies

`maximum_length` bounds each source record. The default `overflow: reject`
prevents accidental data loss. Explicit `truncate_left` and `truncate_right`
remain available. Full-supervision packed output also supports `split`, which
preserves every token and treats oversized records as logical chunks without
inserting separators between those chunks. Separators remain source-document
boundaries only.

Supervised artifacts must also satisfy `alignment`; `alignment_policy` chooses
`reject` or explicit left/right truncation. The preparation frontend rejects
padding because the v1 artifact has no valid-token mask consumed by attention.
The low-level packing API retains padding helpers for compatibility, but
prepared artifacts cannot silently make padding attention-visible. No dynamic
padding or implicit sequence transformation occurs during training.

For packed full-sequence corpora, `separator_token_id` inserts an explicit
document separator. Usually this should be the tokenizer EOS ID. Without it,
adjacent documents are intentionally concatenated directly.

## Artifact formats

`output.format: auto` selects:

- `dllm_parallel.packed_tokens` version 1 for `supervision.policy: full`.
- `dllm_parallel.indexed_supervised_tokens` version 1 for assistant-only,
  completion-only, or provided token masks.

Both are directories with `manifest.json`, binary int32 tokens, hashes,
preparation/source provenance, counts, and transformation statistics. Indexed
artifacts additionally contain int64 record offsets and uint8 loss masks.
Manifests declare sequence layout, supervision shape, padding behavior, and the
exact sampling policy. Tokenizer vocabulary, chat template, EOS, padding, and
diffusion mask-token identity are recorded and checked by the existing training
runtime. Checkpoint resume also requires the exact prepared-artifact
fingerprint. These checks happen during preparation, explicit validation,
startup, or resume—not in batch production.

Use the artifact directory directly in a training config:

```yaml
launch:
  recipe_kind: prod
data:
  input_mode: dataset
  dataset_path: /data/prepared
```

Existing flat token files and indexed-supervised v1 directories remain
supported.

## Extension API and boundaries

Python applications can use `register_source()` and `register_formatter()` for
storage systems or record schemas not covered above. A source yields raw
records; a formatter returns `FormattedRecord`. The standard tokenization,
packing, checksumming, and runtime handoff remain shared.

Dataset-specific acquisition, filtering, decontamination, agent execution,
evaluation, credentials, and research campaign policy do not belong in the
installed package. Integrations should normalize their records through this
generic boundary.
