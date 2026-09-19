"""Packaged smoke and example recipe catalog."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath

import yaml


@dataclass(frozen=True)
class RecipeInfo:
    """Metadata for one packaged recipe."""

    name: str
    kind: str
    summary: str
    hardware: str
    model_family: str


def _manifest() -> dict[str, object]:
    resource = files(__package__).joinpath("manifest.yaml")
    parsed = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or parsed.get("format") != "dllm.recipe_catalog.v1":
        raise RuntimeError("invalid packaged recipe manifest")
    return parsed


def list_recipes() -> tuple[RecipeInfo, ...]:
    """Return every packaged recipe in stable name order."""

    entries = _manifest().get("recipes")
    if not isinstance(entries, dict):
        raise RuntimeError("packaged recipe manifest has no recipe table")
    return tuple(
        RecipeInfo(
            name=str(name),
            kind=str(metadata["kind"]),
            summary=str(metadata["summary"]),
            hardware=str(metadata["hardware"]),
            model_family=str(metadata["model_family"]),
        )
        for name, metadata in sorted(entries.items())
        if isinstance(metadata, dict)
    )


def _recipe_relative_path(name: str) -> PurePosixPath:
    candidate = PurePosixPath(str(name))
    if candidate.is_absolute() or len(candidate.parts) != 2 or ".." in candidate.parts:
        raise ValueError("recipe name must be KIND/NAME")
    catalog = {item.name: item for item in list_recipes()}
    if name not in catalog:
        raise KeyError(f"unknown packaged recipe: {name}")
    return PurePosixPath(f"{name}.yaml")


def recipe_text(name: str) -> str:
    """Read a packaged recipe by logical name."""

    relative = _recipe_relative_path(name)
    return files(__package__).joinpath(*relative.parts).read_text(encoding="utf-8")


def copy_recipe(name: str, destination: str | Path) -> Path:
    """Copy a packaged recipe without overwriting an existing file."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(recipe_text(name))
    return output


__all__ = ("RecipeInfo", "copy_recipe", "list_recipes", "recipe_text")
