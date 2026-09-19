# Architecture

The installed library has three layers:

1. `dllm_parallel.core` owns models, objectives, schedules, attention,
   parallelism, kernels, checkpoint formats, data runtime implementations, and
   profiling primitives.
2. `dllm_parallel.training` owns the typed `RunSpec`, trainer lifecycle,
   optimization, metrics, checkpoint coordination, and the existing execution
   entry point.
3. Public facades (`dllm_parallel`, `dllm_parallel.data`, and
   `dllm_parallel.cli`) expose stable, low-friction interfaces without moving
   or rewriting hot paths.

Packaged recipes are immutable resources. Repository scripts orchestrate
installed behavior and are never imported by the library.
