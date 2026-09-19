# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DiffusionGemma backbone (Gemma4-MoE block diffusion): metadata + executor."""

from dllm_parallel.core.models.backbones.diffusiongemma.metadata import (
    build_schedule,
    summarize_config,
    validate_parallel_spec,
)

_MODEL_EXPORTS = {
    "DiffusionGemmaBackboneExecutor",
    "DiffusionGemmaExpertParallelExperts",
    "build_executor",
    "build_packed_block_diffusion_model",
}

_ARTIFACT_EXPORTS = {
    "DiffusionGemmaMergeResult",
    "merge_diffusion_gemma_lora_checkpoint",
    "validate_merged_diffusion_gemma_artifact",
}

__all__ = [
    "build_schedule",
    "summarize_config",
    "validate_parallel_spec",
    "DiffusionGemmaBackboneExecutor",
    "DiffusionGemmaExpertParallelExperts",
    "build_executor",
    "build_packed_block_diffusion_model",
    "DiffusionGemmaMergeResult",
    "merge_diffusion_gemma_lora_checkpoint",
    "validate_merged_diffusion_gemma_artifact",
]


def __getattr__(name: str):
    if name in _ARTIFACT_EXPORTS:
        from dllm_parallel.core.models.backbones.diffusiongemma import export as module

        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _MODEL_EXPORTS:
        if name == "DiffusionGemmaExpertParallelExperts":
            from dllm_parallel.core.models.backbones.diffusiongemma import (
                expert_parallel as module,
            )
        else:
            from dllm_parallel.core.models.backbones.diffusiongemma import model as module

        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
