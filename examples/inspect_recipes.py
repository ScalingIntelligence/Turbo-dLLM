"""List packaged recipes without requiring a source checkout."""

from dllm_parallel.recipes import list_recipes


for recipe in list_recipes():
    print(f"{recipe.name}: {recipe.summary} ({recipe.hardware})")
