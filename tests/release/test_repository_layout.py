from __future__ import annotations

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_scripts_have_only_maintainer_orchestration_categories() -> None:
    scripts = ROOT / "scripts"
    assert {path.name for path in scripts.iterdir()} == {
        "build",
        "install",
        "launch",
        "verify",
    }
    assert not (scripts / "release").exists()


def test_release_has_no_baseline_comparison_module() -> None:
    assert not (ROOT / "dllm_parallel/core/profiling/gates.py").exists()
    assert not (ROOT / "tests/unit/profiling/test_profile_gates.py").exists()


def test_shell_entrypoints_are_syntactically_valid_and_documented() -> None:
    shell_scripts = sorted((ROOT / "scripts").glob("*/*.sh"))
    assert shell_scripts
    for script in shell_scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
        completed = subprocess.run(
            ["bash", str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (script, completed.stderr)
        assert "Usage:" in completed.stdout, script


def test_scripts_contain_no_provider_or_dataset_integration_names() -> None:
    forbidden = ("modal", "coderforge", "openhands", "swe-bench", "laude")
    for path in (ROOT / "scripts").glob("**/*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore").casefold()
            assert [term for term in forbidden if term in text] == [], path


def test_checkout_launch_wrapper_has_no_training_or_environment_semantics() -> None:
    wrapper = (ROOT / "scripts/launch/torchrun.sh").read_text(encoding="utf-8")

    assert "-m dllm_parallel.cli launch" in wrapper
    assert "torch.distributed.run" not in wrapper
    assert "PYTHONPATH" not in wrapper
    assert "CUDA_HOME" not in wrapper


def test_internal_import_targets_exist_in_the_release_package() -> None:
    package = ROOT / "dllm_parallel"
    missing: list[tuple[Path, int, str]] = []

    for source in sorted(package.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if node.level or not module.startswith("dllm_parallel."):
                continue
            target = ROOT.joinpath(*module.split("."))
            if target.with_suffix(".py").is_file() or (
                target / "__init__.py"
            ).is_file():
                continue
            missing.append((source.relative_to(ROOT), node.lineno, module))

    assert missing == []
