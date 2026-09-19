from __future__ import annotations

from types import SimpleNamespace

import pytest

from dllm_parallel.core.models.backbones.diffusiongemma.model import (
    DiffusionGemmaBackboneExecutor,
)
from dllm_parallel.core.models.backbones.nemotron.model import (
    NemotronLabsDiffusionBackboneExecutor,
)
from dllm_parallel.training import block_diffusion_trainer
from dllm_parallel.training.block_diffusion_trainer import (
    _load_training_model_config,
    _resolve_model_family,
)
from dllm_parallel.training.run_spec import ModelRunSpec, RunSpec


@pytest.mark.parametrize(
    "family",
    ("diffusion_gemma", "nemotron_labs_diffusion"),
)
def test_explicit_registered_hf_family_survives_run_spec_resolution(
    family: str,
) -> None:
    spec = RunSpec.from_mapping({"model": {"family": family}})

    assert _resolve_model_family(spec) == family


@pytest.mark.parametrize(
    "family",
    ("diffusion_gemma", "nemotron_labs_diffusion"),
)
def test_explicit_registered_hf_family_reaches_the_config_loader(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(model_type=family)
    model_family_spec = SimpleNamespace(family=family)

    def load_hf_config(
        model_id: str,
        *,
        trust_remote_code: bool,
        revision: str | None,
        family: str | None,
    ) -> SimpleNamespace:
        if family not in {"diffusion_gemma", "nemotron_labs_diffusion"}:
            raise AssertionError(f"explicit family was discarded: {family!r}")
        return SimpleNamespace(config=config, spec=model_family_spec)

    class Executor:
        def metadata(self, loaded_config, *, model_id: str):
            assert loaded_config is config
            return model_family_spec

    monkeypatch.setattr(block_diffusion_trainer, "load_hf_config", load_hf_config)
    monkeypatch.setattr(
        block_diffusion_trainer,
        "executor_for_family",
        lambda requested: Executor(),
    )
    spec = RunSpec(model=ModelRunSpec(family=family))

    loaded = _load_training_model_config(spec, family=family)

    assert loaded.family == family
    assert loaded.config is config
    assert loaded.spec is model_family_spec
    assert loaded.hf is True


@pytest.mark.parametrize(
    ("family", "executor_type"),
    (
        ("diffusion_gemma", DiffusionGemmaBackboneExecutor),
        ("nemotron_labs_diffusion", NemotronLabsDiffusionBackboneExecutor),
    ),
)
def test_executor_accepts_its_explicit_public_family(
    family: str,
    executor_type: type,
) -> None:
    spec = RunSpec.from_mapping({"model": {"family": family}})

    executor_type().validate_run_spec(spec)
