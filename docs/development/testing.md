# Testing

The portable gate is:

```bash
ruff check dllm_parallel tests
python -m compileall -q dllm_parallel tests
python -m pytest -q tests/unit tests/packaging tests/release
uv lock --check
```

Integration, distributed, and performance tests are separate because they have
different dependency and hardware contracts. GPU results must name the exact
wheel hashes and runner hardware. A skipped CUDA test is not a passed GPU gate.

Release tests inspect repository boundaries, metadata, third-party provenance,
workflow policy, and built artifact contents. Tests should assert observable
behavior rather than duplicated implementation details.
