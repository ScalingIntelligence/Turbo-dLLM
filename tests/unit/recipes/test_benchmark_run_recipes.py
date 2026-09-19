from __future__ import annotations

from pathlib import Path

import yaml

from dllm_parallel.training.run_spec import load_run_spec


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RECIPE_ROOT = REPOSITORY_ROOT / "benchmarks" / "recipes"
EXPECTED_RECIPES = {
    "dflash2_muse_glimmer_30b_1m.yaml",
    "dflash2_qwen3_8_27b_1m.yaml",
    "diffusiongemma_26b_sft_256k.yaml",
    "qwen3_8_27b_fast_dllm_v2_256k.yaml",
}


def _load(name: str):
    path = RECIPE_ROOT / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw, load_run_spec(path)


def test_repository_run_recipe_inventory_is_small_and_explicit() -> None:
    assert {path.name for path in RECIPE_ROOT.glob("*.yaml")} == EXPECTED_RECIPES


def test_repository_run_recipes_are_real_runs_not_profiles() -> None:
    for name in EXPECTED_RECIPES:
        raw, spec = _load(name)
        assert spec.launch.recipe_kind == "prod"
        assert spec.training.steps >= 100
        assert spec.profiler.warmup_steps == 0
        assert spec.profiler.phase_timing is False
        assert "profiler" not in raw
        assert spec.kernel.runtime_jit is False
        assert spec.checkpointing.save_checkpoint_dir
        assert spec.checkpointing.save_checkpoint_interval > 0
        assert spec.checkpointing.auto_resume is True
        assert spec.checkpointing.save_final is True
        assert spec.logging.wandb is False


def test_qwen38_dflash2_1m_matches_paper_runtime_configuration() -> None:
    _, spec = _load("dflash2_qwen3_8_27b_1m.yaml")

    assert spec.model.id == "incoai/Qwen3.8-27B-DFlash2"
    assert spec.model.revision == "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
    assert spec.model.verifier_id == "Qwen/Qwen3.8-27B"
    assert spec.model.verifier_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert spec.model.target_layer_ids == (6, 20, 34, 48, 62)
    assert spec.model.seq_len == 1_048_576
    assert spec.objective.name == "dflash_distillation"
    assert spec.objective.block_size == 8
    assert spec.objective.max_anchors == 2_048
    assert spec.objective.anchor_sampling == "locality"
    assert spec.objective.dflash_loss == "dpace"
    assert spec.objective.dflash_vocab_block_size == 32_768
    assert spec.data.input_mode == "dataset"
    assert spec.data.target_features == "offline"
    assert spec.data.target_feature_path == "/data/dflash-features"
    assert spec.topology.context_parallel_size == 2
    assert spec.topology.block_parallel_size == 2
    assert spec.optimizer.backend == "torch_fused_adamw"
    assert spec.training.activation_checkpointing is False


def test_muse_glimmer_dflash2_1m_matches_paper_runtime_configuration() -> None:
    _, spec = _load("dflash2_muse_glimmer_30b_1m.yaml")

    assert spec.model.id == "incoai/Muse-Glimmer-30B-DFlash2"
    assert spec.model.revision == "8336acb8dc9b8bf9c25f12d7785ee6df26703119"
    assert spec.model.verifier_id == "meta-models/Muse-Glimmer-30B"
    assert spec.model.verifier_revision == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    assert spec.model.target_layer_ids == (2, 14, 26, 38, 50)
    assert spec.model.seq_len == 1_048_576
    assert spec.objective.name == "dflash_distillation"
    assert spec.objective.block_size == 16
    assert spec.objective.max_anchors == 512
    assert spec.objective.anchor_sampling == "locality"
    assert spec.objective.dflash_loss == "dpace"
    assert spec.objective.dflash_vocab_block_size == 16_384
    assert spec.data.target_features == "offline"
    assert spec.topology.context_parallel_size == 4
    assert spec.topology.block_parallel_size == 4
    assert spec.optimizer.backend == "torch_fused_adamw"
    assert spec.training.activation_checkpointing is False


def test_diffusiongemma_sft_256k_matches_paper_runtime_configuration() -> None:
    _, spec = _load("diffusiongemma_26b_sft_256k.yaml")

    assert spec.model.id == "google/diffusiongemma-26B-A4B-it"
    assert spec.model.revision == "f7f5b7f5fa82ffc52addd066915886d497f5517b"
    assert spec.model.seq_len == 262_144
    assert spec.objective.name == "diffusiongemma_native_sft"
    assert spec.objective.block_size == 256
    assert spec.objective.loss_weighting == "unit"
    assert spec.data.input_mode == "dataset"
    assert spec.data.dataset_path == "/data/prepared"
    assert spec.topology.tensor_parallel_size == 1
    assert spec.topology.expert_parallel_size == 2
    assert spec.topology.context_parallel_size == 8
    assert spec.topology.block_parallel_size == 8
    assert spec.topology.placement_policy == "inter_node_cp"
    assert spec.optimizer.backend == "deepspeed_zero2"
    assert spec.training.activation_checkpointing is True


def test_qwen38_fast_dllm_v2_256k_matches_paper_runtime_configuration() -> None:
    _, spec = _load("qwen3_8_27b_fast_dllm_v2_256k.yaml")

    assert spec.model.id == "Qwen/Qwen3.8-27B"
    assert spec.model.revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert spec.model.family == "qwen3_8"
    assert spec.model.seq_len == 262_144
    assert spec.objective.name == "fast_dllm_v2"
    assert spec.objective.block_size == 256
    assert spec.objective.noise_schedule == "linear_mask"
    assert spec.objective.loss_weighting == "unit"
    assert spec.data.input_mode == "dataset"
    assert spec.data.dataset_path == "/data/prepared"
    assert spec.topology.tensor_parallel_size == 2
    assert spec.topology.sequence_parallel is True
    assert spec.topology.context_parallel_size == 8
    assert spec.topology.block_parallel_size == 8
    assert spec.topology.placement_policy == "inter_node_cp"
    assert spec.optimizer.backend == "deepspeed_zero2"
    assert spec.training.activation_checkpointing is True
