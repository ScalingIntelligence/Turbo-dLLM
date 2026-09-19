# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Command adapters for generic offline dataset preparation."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dllm_parallel.data.indexed import (
    inspect_artifact,
    validate_artifact,
    validate_artifact_for_run,
)
from dllm_parallel.data.prepare import prepare_dataset
from dllm_parallel.data.schemas import PreparationSpec


def run_data(argv: Sequence[str]) -> int:
    """Execute a ``dllm data`` subcommand."""

    parser = argparse.ArgumentParser(
        prog="dllm data",
        description="Prepare and verify optimized training datasets.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="Compile a configured dataset into a training artifact."
    )
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--json", action="store_true")
    inspect = subparsers.add_parser(
        "inspect", help="Inspect an artifact manifest without hashing payloads."
    )
    inspect.add_argument("artifact", type=Path)
    inspect.add_argument("--json", action="store_true")
    validate = subparsers.add_parser(
        "validate", help="Verify schema, payload sizes, and SHA-256 checksums."
    )
    validate.add_argument("artifact", type=Path)
    validate.add_argument(
        "--config", type=Path, help="Also validate against a RunSpec."
    )
    validate.add_argument("--no-checksums", action="store_true")
    validate.add_argument("--json", action="store_true")
    stats = subparsers.add_parser("stats", help="Print recorded artifact statistics.")
    stats.add_argument("artifact", type=Path)
    stats.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv))

    if args.action == "prepare":
        spec = PreparationSpec.from_path(args.config)
        if args.output is not None or args.overwrite:
            output = dataclasses.replace(
                spec.output,
                path=(
                    str(args.output.expanduser().resolve())
                    if args.output is not None
                    else spec.output.path
                ),
                overwrite=bool(args.overwrite or spec.output.overwrite),
            )
            spec = dataclasses.replace(spec, output=output)
        result = prepare_dataset(spec)
        payload = {
            "artifact_path": str(result.artifact_path),
            "training_path": str(result.training_path),
            "format": result.format,
            "sample_count": result.sample_count,
            "token_count": result.token_count,
            "dataset_fingerprint": result.dataset_fingerprint,
        }
        _print(payload, json_output=args.json)
        return 0
    if args.action == "inspect":
        _print(inspect_artifact(args.artifact), json_output=args.json)
        return 0
    if args.action == "stats":
        payload = inspect_artifact(args.artifact).get("stats") or {}
        _print(payload, json_output=args.json)
        return 0
    manifest = validate_artifact(
        args.artifact,
        verify_checksums=not args.no_checksums,
    )
    payload: dict[str, Any] = {
        "valid": True,
        "path": str(args.artifact.expanduser().resolve()),
        "format": manifest["format"],
        "dataset_fingerprint": manifest["metadata"]["dataset_fingerprint"],
        "checksums_verified": not args.no_checksums,
    }
    if args.config is not None:
        from dllm_parallel.training.run_spec import load_run_spec

        payload["run"] = validate_artifact_for_run(
            args.artifact,
            load_run_spec(args.config),
            manifest=manifest,
        )
    _print(payload, json_output=args.json)
    return 0


def _print(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


__all__ = ("run_data",)
