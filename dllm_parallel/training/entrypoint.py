# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Single production training entrypoint for DLLM backbones."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--print-supported-configs" in args:
        _print_supported_configs()
        return
    if args and args[0] in {"-h", "--help"}:
        _print_help()
        raise SystemExit(0)
    _validate_canonical_args(args)
    if "--validate-only" in args:
        _print_resolved_spec(args)
        return
    spec = _prepare_launch_environment(args)
    _preflight_training_runtime(spec)
    _run_training(args)


def _validate_canonical_args(args: Sequence[str]) -> None:
    from dllm_parallel.training.run_spec import build_run_spec_arg_parser

    parser = build_run_spec_arg_parser(add_help=False)
    try:
        parsed, unknown = parser.parse_known_args(list(args))
    except SystemExit as exc:
        raise ValueError("invalid dllm-train arguments") from exc
    if unknown:
        if any("=" in arg and not arg.startswith("--") for arg in unknown):
            raise ValueError(
                "RunSpec overrides must use canonical --flag form; unsupported "
                "argument(s): "
                + ", ".join(repr(arg) for arg in unknown)
            )
        raise ValueError(
            "unknown positional argument(s) or unsupported argument(s): "
            + ", ".join(repr(arg) for arg in unknown)
        )
    if not getattr(parsed, "config", None) and not bool(
        getattr(parsed, "print_supported_configs", False)
    ):
        raise ValueError("dllm-train requires --config with a canonical RunSpec recipe")


def _run_training(args: list[str]) -> None:
    from dllm_parallel.training.block_diffusion_trainer import main as train_main
    from torch.distributed.elastic.multiprocessing.errors import record

    record(train_main)(args)


def _prepare_launch_environment(args: Sequence[str]):
    from dllm_parallel.core.parallel.preflight import (
        prepare_parallel_environment,
    )
    from dllm_parallel.training.run_spec import preflight_run_spec_from_argv

    if "--print-supported-configs" in args:
        return
    spec = preflight_run_spec_from_argv(list(args))
    prepare_parallel_environment(
        sequence_parallel=bool(spec.topology.sequence_parallel),
        context_parallel_size=int(spec.topology.context_parallel_size),
        block_parallel_size=int(spec.topology.block_parallel_size),
        tensor_parallel_size=int(spec.topology.tensor_parallel_size),
        expert_parallel_size=int(spec.topology.expert_parallel_size),
        tensor_parallel_overlap=bool(spec.topology.tensor_parallel_overlap),
    )
    return spec


def _preflight_training_runtime(spec) -> None:
    """Guard direct entrypoints while avoiding duplicate launcher work."""

    if os.environ.get("DLLM_RUNTIME_PREFLIGHT", "auto").casefold() == "off":
        return
    if os.environ.get("DLLM_RUNTIME_PREFLIGHT_DONE") == "1":
        return
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    from dllm_parallel.core.diagnostics import inspect_training_runtime

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    inspect_training_runtime(
        spec,
        distributed=distributed,
        local_processes=local_processes,
    ).require_ready()
    os.environ["DLLM_RUNTIME_PREFLIGHT_DONE"] = "1"


def _print_supported_configs() -> None:
    import json

    from dllm_parallel.core.models.registry import describe_supported_configs

    print(json.dumps(describe_supported_configs(), indent=2, sort_keys=True))


def _print_resolved_spec(args: Sequence[str]) -> None:
    import json

    from dllm_parallel.training.run_spec import preflight_run_spec_from_argv

    spec = preflight_run_spec_from_argv(list(args))
    print(
        json.dumps(
            {
                "event": "validated_run_spec",
                "run_spec": spec.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _print_help() -> None:
    from dllm_parallel.training.run_spec import build_run_spec_arg_parser

    build_run_spec_arg_parser(
        description="Run production block-diffusion training from a typed RunSpec."
    ).print_help()


__all__ = ["main"]
