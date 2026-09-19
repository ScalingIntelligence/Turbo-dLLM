# Data API

```python
from dllm_parallel.data import (
    DataBatch,
    DataRuntime,
    IndexedDataset,
    PackedTokenDataset,
    build_data_runtime,
    inspect_artifact,
    prepare_dataset,
    register_formatter,
    register_source,
    validate_artifact,
    validate_artifact_for_run,
)
```

`DataRuntime` supplies restartable batches plus serializable state and logging
metadata. `PackedTokenDataset` reads generic packed token streams;
`IndexedDataset` reads the indexed supervised-token format. The shorter names
are aliases, so checkpoint/data behavior remains identical to the existing core
implementations.

This facade is the stable data API. Detailed configuration types and artifact
constants remain available from `dllm_parallel.data.schemas` and
`dllm_parallel.data.indexed` for advanced integrations without expanding the
top-level compatibility contract.

`PreparationSpec.from_path()` loads a strict YAML or JSON preparation
configuration. `prepare_dataset()` accepts that spec, a mapping, or a config
path and returns a `PreparationResult` containing the artifact path, format,
counts, and deterministic dataset fingerprint. `inspect_artifact()` reads only
the manifest; `validate_artifact()` additionally checks every payload size and
SHA-256 digest. `validate_artifact_for_run()` checks indexed block alignment and
maximum length against a resolved `RunSpec`.

Custom integrations can register a raw source or normalized record formatter:

```python
from dllm_parallel.data import register_formatter, register_source
from dllm_parallel.data.formatting import FormattedRecord

register_source("warehouse", lambda spec: iter(fetch_rows(spec.id)))
register_formatter(
    "question-answer",
    lambda row, spec: FormattedRecord(
        kind="prompt_completion",
        prompt=row["question"],
        completion=row["answer"],
    ),
)
```

Direct registration is process-local and explicit; accidental replacement is
rejected. Installable integrations can instead expose callable Python entry
points:

```toml
[project.entry-points."dllm_parallel.data_sources.v1"]
warehouse = "my_dllm_plugin:iter_warehouse"

[project.entry-points."dllm_parallel.data_formatters.v1"]
question-answer = "my_dllm_plugin:format_question_answer"
```

The versioned groups are discovered once and only when an unknown source or
formatter is resolved. Names are normalized, handlers must be callable, and a
plugin cannot replace an existing registration. Dataset acquisition, branded
schemas, policy-specific filtering, agent trajectory conversion, and offline
feature-generation pipelines remain outside the package.
