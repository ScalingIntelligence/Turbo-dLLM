"""Thin command adapters around the stable package APIs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from collections.abc import Callable, Sequence
from importlib import metadata, resources
from pathlib import Path

from dllm_parallel.recipes import copy_recipe, list_recipes, recipe_text


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dllm",
        description="Configure, inspect, and launch Turbo-dLLM training.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "bundle",
            "config",
            "data",
            "doctor",
            "dflash",
            "init",
            "launch",
            "recipe",
            "train",
        ),
    )
    return parser


def _version() -> str:
    try:
        return metadata.version("turbo-dllm")
    except metadata.PackageNotFoundError:
        return "0.1.1+source"


def _doctor(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="dllm doctor")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--training",
        action="store_true",
        help="Verify the complete common GPU training bundle.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--recipe", help="Verify requirements for a packaged recipe.")
    source.add_argument("--config", type=Path, help="Verify requirements for a RunSpec.")
    args = parser.parse_args(argv)

    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = None
        cuda_available = False
        cuda_version = None

    try:
        native_manifest = resources.files("dllm_parallel.core._C").joinpath(
            "native_kernels.json"
        )
        native_kernel_status = "installed" if native_manifest.is_file() else "portable"
    except (ImportError, ModuleNotFoundError):
        native_kernel_status = "portable"

    def installed_version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    from dllm_parallel.core import diagnostics

    spec = None
    if args.recipe or args.config:
        from dllm_parallel.cli.launch import _validate_config

        if args.recipe:
            with tempfile.TemporaryDirectory(prefix="dllm-doctor-") as directory:
                config = Path(directory) / "recipe.yaml"
                config.write_text(recipe_text(args.recipe), encoding="utf-8")
                spec = _validate_config(config, ())
        else:
            spec = _validate_config(args.config, ())
    if args.training or spec is not None:
        readiness = diagnostics.inspect_training_runtime(
            spec,
            distributed=False,
            full_bundle=bool(args.training),
        )
    else:
        readiness = diagnostics.inspect_portable_runtime()

    report = {
        "package": "turbo-dllm",
        "version": _version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_runtime": cuda_version,
        "native_artifacts": {
            "bdlm-flash-attn-3": installed_version("bdlm-flash-attn-3"),
            "dllm-native-kernels": native_kernel_status,
            "flash-attn-4": installed_version("flash-attn-4"),
        },
        "gpu_runtime": {
            distribution: installed_version(distribution)
            for distribution in (
                "apache-tvm-ffi",
                "cuda-bindings",
                "cuda-python",
                "deepspeed",
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-base",
                "quack-kernels",
                "torch-c-dlpack-ext",
                "transformer-engine",
            )
        },
        "model_runtimes": {
            distribution: installed_version(distribution)
            for distribution in (
                "deep-ep",
                "flash-linear-attention",
                "tilelang",
            )
        },
        "readiness": readiness.to_dict(),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Turbo-dLLM {report['version']}")
        print(f"Python {report['python']} on {report['platform']}")
        print(
            f"PyTorch {report['torch'] or 'not installed'}; "
            f"CUDA {report['cuda_runtime'] or 'unavailable'}"
        )
        print(f"Native artifacts: {native_kernel_status}")
        for check in readiness.checks:
            print(f"[{check.status}] {check.name}: {check.detail}")
            if check.remediation:
                print(f"  fix: {check.remediation}")
        print(f"Ready: {'yes' if readiness.ready else 'no'}")
    return 0 if readiness.ready else 1


def _recipe(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="dllm recipe")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list", help="List packaged recipes.")
    show = subparsers.add_parser("show", help="Print one packaged recipe.")
    show.add_argument("name")
    copy = subparsers.add_parser("copy", help="Copy a recipe for editing.")
    copy.add_argument("name")
    copy.add_argument("destination", type=Path)
    args = parser.parse_args(argv)

    if args.action == "list":
        for item in list_recipes():
            print(f"{item.name}\t{item.hardware}\t{item.summary}")
    elif args.action == "show":
        print(recipe_text(args.name), end="")
    else:
        output = copy_recipe(args.name, args.destination)
        print(output)
    return 0


def _bundle(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="dllm bundle")
    subparsers = parser.add_subparsers(dest="action", required=True)
    install = subparsers.add_parser(
        "install", help="Install an exact hash-verified GPU wheel bundle."
    )
    source = install.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--release")
    install.add_argument(
        "--auto",
        action="store_true",
        help="Detect the host and select an exact bundle from the release catalog.",
    )
    args = parser.parse_args(argv)

    from dllm_parallel.core.kernels.bundle_manifest import (
        install_bundle,
        install_bundle_for_release,
    )

    if args.manifest:
        if args.auto:
            raise ValueError("--auto is only valid with --release")
        install_bundle(args.manifest)
    else:
        if not args.auto:
            raise ValueError("--release requires --auto for explicit host detection")
        install_bundle_for_release(args.release)
    print("GPU bundle installed.")
    return _doctor(("--training",))


def _materialized_recipe(name: str, operation: Callable[[str], None]) -> None:
    with tempfile.TemporaryDirectory(prefix="dllm-recipe-") as directory:
        path = Path(directory) / "recipe.yaml"
        path.write_text(recipe_text(name), encoding="utf-8")
        operation(str(path))


def _config(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="dllm config")
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser("validate", help="Resolve and validate a RunSpec.")
    source = validate.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe")
    source.add_argument("--config", type=Path)
    args, overrides = parser.parse_known_args(argv)

    from dllm_parallel.training.entrypoint import main as training_main

    def validate_path(path: str) -> None:
        training_main(["--config", path, *overrides, "--validate-only"])

    if args.recipe:
        _materialized_recipe(args.recipe, validate_path)
    else:
        validate_path(str(args.config))
    return 0


def _train(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="dllm train")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe")
    source.add_argument("--config", type=Path)
    args, overrides = parser.parse_known_args(argv)

    from dllm_parallel.training.entrypoint import main as training_main

    def train_path(path: str) -> None:
        training_main(["--config", path, *overrides])

    if args.recipe:
        _materialized_recipe(args.recipe, train_path)
    else:
        train_path(str(args.config))
    return 0


def _launch(argv: Sequence[str]) -> int:
    from dllm_parallel.cli.launch import run_launch

    return run_launch(argv)


def _init(argv: Sequence[str]) -> int:
    from dllm_parallel.cli.init import run_init

    return run_init(argv)


def run(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _root_parser().print_help()
        return 0

    command, remainder = arguments[0], arguments[1:]
    handlers = {
        "bundle": _bundle,
        "config": _config,
        "data": _data,
        "doctor": _doctor,
        "dflash": _dflash,
        "init": _init,
        "launch": _launch,
        "recipe": _recipe,
        "train": _train,
    }
    handler = handlers.get(command)
    if handler is None:
        print(f"dllm: unknown command: {command}", file=sys.stderr)
        return 2
    try:
        return handler(remainder)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"dllm: {exc}", file=sys.stderr)
        return 2


def _data(argv: Sequence[str]) -> int:
    from dllm_parallel.cli.data import run_data

    return run_data(argv)


def _dflash(argv: Sequence[str]) -> int:
    from dllm_parallel.cli.dflash import run_dflash

    return run_dflash(argv)


def main() -> None:
    raise SystemExit(run())


__all__ = ("main", "run")
