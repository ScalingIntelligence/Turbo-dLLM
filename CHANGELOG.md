# Changelog

## Unreleased

## [0.1.1] - 2026-09-19

- Add a generic offline preparation frontend for Hugging Face, JSONL, Parquet,
  text, and pretokenized datasets.
- Add versioned, checksummed packed and indexed artifact writers plus
  inspection, validation, statistics, and RunSpec compatibility commands.
- Add exact automatic GPU bundle discovery through an attested release catalog.
- Add an installed local, multi-GPU, and multi-node `dllm launch` command.
- Add `dllm init` for a validated, non-destructive data-preparation and training
  starter project that works from an installed wheel.
- Add config-aware runtime diagnostics that verify CUDA, native artifact
  identity, optional runtimes, and data paths before workers start.
- Add an end-to-end DFlash2 workflow for generic prepared datasets, offline
  verifier-feature capture, training, export, and native vLLM or SGLang serving.
- Add editable run recipes matching the paper-sensitive Qwen3.8, Muse-Glimmer,
  and DiffusionGemma configurations without embedding a dataset.
- Publish corrected PyPI metadata, repository links, and serving extras.

This project follows Semantic Versioning. Release dates use ISO 8601.

[Unreleased]: https://github.com/ScalingIntelligence/Turbo-dLLM/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ScalingIntelligence/Turbo-dLLM/releases/tag/v0.1.1
