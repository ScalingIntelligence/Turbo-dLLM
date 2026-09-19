"""Fail-closed model-family and training-objective compatibility."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


SUPPORTED_OBJECTIVES_BY_FAMILY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "causal_lm": ("fast_dllm_v2",),
        "dflash": ("dflash_distillation",),
        "diffusion_gemma": (
            "diffusiongemma_native_sft",
            "standard_block_diffusion",
        ),
        "nemotron_labs_diffusion": ("standard_block_diffusion",),
        "qwen3_8": ("fast_dllm_v2",),
    }
)


def supported_objectives_for_family(family: str) -> tuple[str, ...]:
    """Return the immutable objective allowlist for a resolved model family."""

    try:
        return SUPPORTED_OBJECTIVES_BY_FAMILY[str(family)]
    except KeyError as exc:
        raise ValueError(f"unsupported model family {family!r}") from exc


def validate_objective_for_family(*, family: str, objective: str) -> None:
    """Reject an objective that is not implemented by the resolved executor."""

    supported = supported_objectives_for_family(family)
    if str(objective) not in supported:
        choices = ", ".join(supported)
        raise ValueError(
            f"objective {objective!r} is not supported by model family {family!r}; "
            f"supported objectives: {choices}"
        )


__all__ = (
    "SUPPORTED_OBJECTIVES_BY_FAMILY",
    "supported_objectives_for_family",
    "validate_objective_for_family",
)
