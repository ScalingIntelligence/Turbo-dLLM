from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dllm_parallel import recipes
from dllm_parallel.core.models.registry import supported_families
from dllm_parallel.training.run_spec import load_run_spec


def test_catalog_lists_shipped_smoke_example_and_run_recipes() -> None:
    catalog = recipes.list_recipes()

    assert catalog
    assert {item.kind for item in catalog} == {"examples", "runs", "smoke"}
    assert "smoke/cpu-config" in {item.name for item in catalog}
    assert "runs/dflash2-qwen3-8-27b-1m" in {item.name for item in catalog}
    assert all("production" not in item.name for item in catalog)
    assert {item.model_family for item in catalog} <= set(supported_families())


def test_every_catalog_entry_is_packaged_and_valid(tmp_path: Path) -> None:
    for item in recipes.list_recipes():
        text = recipes.recipe_text(item.name)
        assert yaml.safe_load(text)["launch"]["recipe_kind"] in {
            "prod",
            "profile",
            "smoke",
        }
        path = recipes.copy_recipe(item.name, tmp_path / f"{item.name}.yaml")
        assert load_run_spec(path).model.family


@pytest.mark.parametrize("name", ("../manifest", "/tmp/config", "smoke/missing"))
def test_recipe_lookup_rejects_unsafe_or_unknown_names(name: str) -> None:
    with pytest.raises((KeyError, ValueError)):
        recipes.recipe_text(name)


def test_copy_recipe_refuses_to_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "recipe.yaml"
    destination.write_text("owned by user\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        recipes.copy_recipe("smoke/cpu-config", destination)


def test_packaged_run_recipes_match_repository_run_configs() -> None:
    root = Path(__file__).resolve().parents[3]
    names = {
        "dflash2-muse-glimmer-30b-1m": "dflash2_muse_glimmer_30b_1m.yaml",
        "dflash2-qwen3-8-27b-1m": "dflash2_qwen3_8_27b_1m.yaml",
        "diffusiongemma-26b-sft-256k": "diffusiongemma_26b_sft_256k.yaml",
        "qwen3-8-27b-fast-dllm-v2-256k": "qwen3_8_27b_fast_dllm_v2_256k.yaml",
    }
    for packaged, repository_file in names.items():
        assert recipes.recipe_text(f"runs/{packaged}") == (
            root / "benchmarks" / "recipes" / repository_file
        ).read_text(encoding="utf-8")
