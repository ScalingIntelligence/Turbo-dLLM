from __future__ import annotations

from pathlib import Path

import pytest

from dllm_parallel.core.models.registry import (
    executor_for_family,
    supported_families,
    supported_packed_families,
)
from dllm_parallel.recipes import list_recipes, recipe_text


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REMOVED_MODEL_FAMILIES = {"bd3lm", "muse_glimmer"}


def test_removed_model_families_are_absent_from_public_registry() -> None:
    assert REMOVED_MODEL_FAMILIES.isdisjoint(supported_families())
    assert REMOVED_MODEL_FAMILIES.isdisjoint(supported_packed_families())
    for family in REMOVED_MODEL_FAMILIES:
        with pytest.raises(ValueError, match="does not expose a BackboneExecutor"):
            executor_for_family(family)


def test_removed_model_families_have_no_packaged_implementation() -> None:
    backbones = REPOSITORY_ROOT / "dllm_parallel/core/models/backbones"
    assert not (backbones / "bd3lm").exists()
    assert not (backbones / "muse_glimmer").exists()


def test_removed_model_families_have_no_packaged_recipe() -> None:
    for recipe in list_recipes():
        assert recipe.model_family not in REMOVED_MODEL_FAMILIES
        contents = recipe_text(recipe.name).lower()
        assert "family: bd3lm" not in contents
        assert "family: muse_glimmer" not in contents
