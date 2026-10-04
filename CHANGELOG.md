# Changelog

## Unreleased

## [0.1.6] - 2026-10-04

- Publish a qualified SM90 / CUDA 12.8 / CPython 3.12 GPU bundle, including
  binary Transformer Engine and DeepSpeed runtime dependencies.
- Replace an installed portable wheel during same-version GPU installation.
- Load packaged cuDNN automatically and verify Split-D build/runtime ABI pins.
- Correct FA4 CUDA stream ordering and reverse sparse-mask backward metadata.
- Gate publication on clean installation, training, GPU correctness, distributed
  execution, checkpoint resume, and performance instrumentation.
- Include verified checksums, source provenance, SBOMs, and build attestations.

Versions 0.1.2–0.1.5 remain immutable, unpublished qualification attempts.

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

[Unreleased]: https://github.com/ScalingIntelligence/Turbo-dLLM/compare/v0.1.6...HEAD
[0.1.1]: https://github.com/ScalingIntelligence/Turbo-dLLM/releases/tag/v0.1.1

[0.1.6]: https://github.com/ScalingIntelligence/Turbo-dLLM/releases/tag/v0.1.6
