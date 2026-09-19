# Python API

Stable root exports include objective/parallel specifications, `RunSpec`,
`load_run_spec`, and `train`. Importing the package does not eagerly import
Torch, Transformers, DeepSpeed, or model weights.

```python
from dllm_parallel import RunSpec, load_run_spec, train
```

Generic data interfaces are documented in [data API](data.md). Lower-level
`dllm_parallel.core` modules are public for advanced integrations but may grow
more quickly than the facade. Native extension modules under `core._C` are
artifact internals and not a stable Python API.
