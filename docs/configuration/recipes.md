# Recipes

Turbo-dLLM ships three kinds of editable configuration:

- `smoke/` checks an installation with a short synthetic run.
- `examples/` demonstrates one model or parallelism feature.
- `runs/` provides complete, dataset-independent training configurations.

List, inspect, or copy any installed recipe:

```bash
dllm recipe list
dllm recipe show runs/dflash2-qwen3-8-27b-1m
dllm recipe copy runs/dflash2-qwen3-8-27b-1m ./run.yaml
dllm config validate --config ./run.yaml
```

The four large-run configurations preserve the model, objective, context,
topology, and memory-sensitive settings used by the paper:

| Recipe | Hardware | Context |
|---|---:|---:|
| `dflash2-qwen3-8-27b-1m` | 8 H100 80GB | 1M |
| `dflash2-muse-glimmer-30b-1m` | 8 H100 80GB | 1M |
| `diffusiongemma-26b-sft-256k` | 16 H200 | 256K |
| `qwen3-8-27b-fast-dllm-v2-256k` | 16 H200 | 256K |

These are training templates, not bundled datasets or result claims. After
copying one, set its dataset or DFlash2 feature path, training duration, and
checkpoint destination. The same four source files live under
`benchmarks/recipes/` for repository-based paper workflows.

For Python callers,
[`inspect_recipes.py`](../../examples/inspect_recipes.py) and
[`validate_recipe.py`](../../examples/validate_recipe.py) demonstrate catalog
discovery and typed `RunSpec` loading.
