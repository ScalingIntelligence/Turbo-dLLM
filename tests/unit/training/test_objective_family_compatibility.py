from __future__ import annotations

from types import SimpleNamespace

import pytest

from dllm_parallel.core.models.backbones.diffusiongemma.model import (
    DiffusionGemmaBackboneExecutor,
)
from dllm_parallel.core.models.backbones.nemotron.model import (
    NemotronLabsDiffusionBackboneExecutor,
)
from dllm_parallel.training.block_diffusion_trainer import (
    _validate_objective_family,
)


OBJECTIVES_BY_FAMILY = {
    "causal_lm": {"fast_dllm_v2"},
    "dflash": {"dflash_distillation"},
    "diffusion_gemma": {
        "diffusiongemma_native_sft",
        "standard_block_diffusion",
    },
    "nemotron_labs_diffusion": {"standard_block_diffusion"},
    "qwen3_8": {"fast_dllm_v2"},
}
ALL_OBJECTIVES = {
    "dflash_distillation",
    "diffusiongemma_native_sft",
    "fast_dllm_v2",
    "standard_block_diffusion",
    "uno_distillation",
}


@pytest.mark.parametrize(
    ("family", "objective"),
    [
        (family, objective)
        for family, supported in OBJECTIVES_BY_FAMILY.items()
        for objective in sorted(supported)
    ],
)
def test_resolved_model_family_accepts_every_supported_objective(
    family: str,
    objective: str,
) -> None:
    _validate_objective_family(SimpleNamespace(name=objective), family=family)


@pytest.mark.parametrize(
    ("family", "objective"),
    [
        (family, objective)
        for family, supported in OBJECTIVES_BY_FAMILY.items()
        for objective in sorted(ALL_OBJECTIVES - supported)
    ],
)
def test_resolved_model_family_rejects_every_unsupported_objective(
    family: str,
    objective: str,
) -> None:
    with pytest.raises(ValueError, match=f"model family {family!r}"):
        _validate_objective_family(SimpleNamespace(name=objective), family=family)


def _diffusion_spec(objective: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(family="auto"),
        objective=SimpleNamespace(name=objective, block_size=256),
        kernel=SimpleNamespace(cp_bp_attention_policy="production"),
        topology=SimpleNamespace(
            sequence_parallel=False,
            tensor_parallel_size=1,
            context_parallel_size=1,
            block_parallel_size=1,
            expert_parallel_size=1,
        ),
        optimizer=SimpleNamespace(backend="auto"),
        adapter=SimpleNamespace(type="none", targets=()),
    )


def test_diffusiongemma_executor_does_not_default_fast_dllm_to_standard() -> None:
    with pytest.raises(ValueError, match="model family 'diffusion_gemma'"):
        DiffusionGemmaBackboneExecutor().validate_run_spec(
            _diffusion_spec("fast_dllm_v2")
        )


def test_nemotron_executor_does_not_default_fast_dllm_to_standard() -> None:
    with pytest.raises(ValueError, match="model family 'nemotron_labs_diffusion'"):
        NemotronLabsDiffusionBackboneExecutor().validate_run_spec(
            _diffusion_spec("fast_dllm_v2")
        )
