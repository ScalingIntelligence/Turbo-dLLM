from __future__ import annotations

import pytest

from dllm_parallel.core.models.registry import (
    SUPPORTED_OPTIMIZER_BACKENDS,
    describe_supported_configs,
    executor_for_family,
    supported_families,
    supported_packed_families,
    infer_family,
)


def test_supported_families_includes_known_backbones() -> None:
    families = supported_families()
    assert "causal_lm" in families
    assert "nemotron_labs_diffusion" in families
    assert "bd3lm" not in families
    assert "muse_glimmer" not in families


def test_packed_families_are_a_subset_of_families() -> None:
    assert set(supported_packed_families()) <= set(supported_families())
    assert "nemotron_labs_diffusion" in supported_packed_families()
    assert "causal_lm" in supported_packed_families()


def test_executor_registry_exposes_supported_training_backbones() -> None:
    assert (
        executor_for_family("nemotron_labs_diffusion").family
        == "nemotron_labs_diffusion"
    )
    assert executor_for_family("causal_lm").family == "causal_lm"


@pytest.mark.parametrize("family", ["bd3lm", "muse_glimmer"])
def test_removed_executor_families_fail_closed(family: str) -> None:
    with pytest.raises(ValueError, match="does not expose a BackboneExecutor"):
        executor_for_family(family)


def test_registry_infers_standard_hf_causal_lm() -> None:
    assert (
        infer_family(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
            }
        )
        == "causal_lm"
    )


def test_qwen38_requires_explicit_public_family_marker() -> None:
    config = {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
    }
    assert infer_family(config) == "causal_lm"
    config["dllm_model_family"] = "qwen3_8"
    assert infer_family(config) == "qwen3_8"


def test_describe_supported_configs_shape() -> None:
    described = describe_supported_configs()
    assert described["optimizer_backends"] == list(SUPPORTED_OPTIMIZER_BACKENDS)
    assert described["objectives_by_family"] == {
        "causal_lm": ["fast_dllm_v2"],
        "dflash": ["dflash_distillation"],
        "diffusion_gemma": [
            "diffusiongemma_native_sft",
            "standard_block_diffusion",
        ],
        "nemotron_labs_diffusion": ["standard_block_diffusion"],
        "qwen3_8": ["fast_dllm_v2"],
    }
    axes = described["parallel_axes"]
    assert axes["context_parallel"] == "supported"
    assert axes["block_parallel"] == "supported"
    assert axes["tensor_parallel"] == "supported"
    assert axes["pipeline_parallel"].startswith("unsupported")
    assert axes["expert_parallel"].startswith("supported")
    assert "cp_bp_layout_rule" in described
