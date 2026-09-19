"""Copy and load a packaged RunSpec for programmatic inspection."""

from pathlib import Path
from tempfile import TemporaryDirectory

from dllm_parallel import load_run_spec
from dllm_parallel.recipes import copy_recipe


with TemporaryDirectory() as directory:
    path = copy_recipe("smoke/cpu-config", Path(directory) / "recipe.yaml")
    spec = load_run_spec(path)
    print(spec.model.family, spec.model.seq_len)
