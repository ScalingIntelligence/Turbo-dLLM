# RunSpec configuration

A run is described by YAML sections for `launch`, `model`, `objective`, `data`,
`training`, `evaluation`, `adapter`, `topology`, `optimizer`, `scheduler`,
`checkpointing`, `profiler`, `logging`, `debug`, and `kernel`.

Copy a recipe, edit its YAML, and validate the complete configuration:

```bash
dllm recipe copy examples/fast-dllm-v2-qwen3 ./run.yaml
dllm config validate --config ./run.yaml
```

Validation prints the fully resolved spec and exits before CUDA initialization.
Unknown keys, positional frontends, unsupported values, and invalid topology
combinations fail closed. Keep dataset paths, checkpoint roots, logging
destinations, and rendezvous configuration in deployment-owned copies rather
than shipped recipes.

For real training, set `data.input_mode: dataset` and point `data.dataset_path`
at a prepared artifact directory or a supported legacy flat-token input. The
[data preparation guide](data-preparation.md) describes generic Hugging Face,
JSONL, Parquet, text, conversational, prompt/completion, and pretokenized
frontends. Preparation is offline and does not alter the training hot path.

`training.batch_size` is the rank-local microbatch size. Global batch size is
`training.batch_size * training.gradient_accumulation_steps * data_parallel_size`.
Indexed variable-length data is bucketed into same-length global batches and
must be aligned to `objective.block_size`.
