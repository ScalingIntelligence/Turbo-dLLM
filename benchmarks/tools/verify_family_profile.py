#!/usr/bin/env python3
"""Validate a matched CP versus fused BP+CP model-family smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _optional_last_event(path: Path, event: str) -> dict[str, Any] | None:
    payload = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("event") == event:
            payload = candidate
    return payload


def _last_event(path: Path, event: str) -> dict[str, Any]:
    payload = _optional_last_event(path, event)
    if payload is None:
        raise RuntimeError(f"{path} does not contain a {event!r} event")
    return payload


def _run_identity(path: Path) -> dict[str, Any]:
    resolved = _optional_last_event(path, "resolved_run_spec")
    if resolved is not None:
        spec = resolved.get("spec") or {}
        return {
            "model_id": (spec.get("model") or {}).get("id"),
            "revision": (spec.get("model") or {}).get("revision"),
            "runtime_jit": bool((spec.get("kernel") or {}).get("runtime_jit")),
        }

    context = _optional_last_event(path, "run_context")
    if context is not None:
        spec = context.get("resolved_spec") or {}
        return {
            "model_id": (spec.get("model") or {}).get("id"),
            "revision": (spec.get("model") or {}).get("revision"),
            "runtime_jit": bool(
                (context.get("kernel_metadata") or {}).get("runtime_jit")
            ),
        }

    loading = _last_event(path, "loading_model")
    return {
        "model_id": loading.get("model_id"),
        "revision": loading.get("revision"),
        "runtime_jit": bool(loading.get("kernel_runtime_jit")),
    }


def _resolved_spec(path: Path) -> dict[str, Any]:
    event = _last_event(path, "resolved_run_spec")
    spec = event.get("spec")
    if not isinstance(spec, dict):
        raise RuntimeError(f"{path} has an invalid resolved run specification")
    return spec


def _normalized_paired_spec(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(spec, sort_keys=True))
    topology = normalized.get("topology")
    if isinstance(topology, dict):
        topology.pop("block_parallel_size", None)
    return normalized


def _spec_fingerprint(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _first_mismatch(left: Any, right: Any, path: str = "spec") -> str:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left:
                return f"{child}: missing from CP"
            if key not in right:
                return f"{child}: missing from fused"
            if left[key] != right[key]:
                return _first_mismatch(left[key], right[key], child)
    return f"{path}: CP={left!r}, fused={right!r}"


def _statuses(root: Path) -> dict[str, str]:
    path = root / "status.csv"
    if not path.is_file():
        raise RuntimeError(f"missing profile status file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["mode"]: row["status"] for row in csv.DictReader(handle)}


def _mean_rank_loss(summary: dict[str, Any]) -> float:
    losses = [float(row["avg_loss"]) for row in summary.get("rank_metrics", ())]
    if not losses or any(not math.isfinite(loss) for loss in losses):
        raise RuntimeError("profile summary has missing or nonfinite rank losses")
    return sum(losses) / len(losses)


def validate_profile_pair(
    root: str | Path,
    *,
    expected_model: str,
    expected_revision: str | None = None,
    context_parallel_size: int,
    block_parallel_size: int,
    loss_atol: float = 5e-2,
    loss_rtol: float = 5e-3,
) -> dict[str, Any]:
    root = Path(root)
    statuses = _statuses(root)
    summaries: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    specs: dict[str, dict[str, Any]] = {}
    for mode in ("cp", "fused"):
        if statuses.get(mode) != "ok":
            raise RuntimeError(f"{mode} profile did not complete successfully")
        log_path = root / mode / "train.log"
        summaries[mode] = _last_event(log_path, "summary")
        identities[mode] = _run_identity(log_path)
        specs[mode] = _resolved_spec(log_path)

    for mode, summary in summaries.items():
        if summary.get("model_id") != expected_model:
            raise RuntimeError(
                f"{mode} used model {summary.get('model_id')!r}, expected "
                f"{expected_model!r}"
            )
        if identities[mode].get("model_id") not in (None, expected_model):
            raise RuntimeError(
                f"{mode} resolved model {identities[mode].get('model_id')!r}, "
                f"expected {expected_model!r}"
            )
        if (
            expected_revision is not None
            and identities[mode].get("revision") != expected_revision
        ):
            raise RuntimeError(
                f"{mode} used revision {identities[mode].get('revision')!r}, "
                f"expected {expected_revision!r}"
            )
        expected_bp = 1 if mode == "cp" else int(block_parallel_size)
        actual_topology = (
            int(summary.get("parallel_context_parallel_size", 0)),
            int(summary.get("parallel_block_parallel_size", 0)),
        )
        if actual_topology != (int(context_parallel_size), expected_bp):
            raise RuntimeError(
                f"{mode} used CP/BP {actual_topology}, expected "
                f"({context_parallel_size}, {expected_bp})"
            )
        policy = summary.get("cp_bp_policy") or {}
        if policy.get("attention_policy") != "production":
            raise RuntimeError(f"{mode} did not use the production attention policy")
        if identities[mode]["runtime_jit"]:
            raise RuntimeError(f"{mode} unexpectedly enabled runtime kernel JIT")

    normalized_specs = {
        mode: _normalized_paired_spec(spec) for mode, spec in specs.items()
    }
    if normalized_specs["cp"] != normalized_specs["fused"]:
        mismatch = _first_mismatch(
            normalized_specs["cp"],
            normalized_specs["fused"],
        )
        raise RuntimeError(
            "CP and fused run specifications differ outside the allowed "
            f"block_parallel_size field: {mismatch}"
        )
    spec_fingerprint = _spec_fingerprint(normalized_specs["cp"])

    cp_loss = _mean_rank_loss(summaries["cp"])
    fused_loss = _mean_rank_loss(summaries["fused"])
    allowed_delta = float(loss_atol) + float(loss_rtol) * abs(cp_loss)
    loss_delta = abs(fused_loss - cp_loss)
    if loss_delta > allowed_delta:
        raise RuntimeError(
            "CP and fused BP+CP losses differ beyond the preregistered "
            f"tolerance: cp={cp_loss:.8f}, fused={fused_loss:.8f}, "
            f"delta={loss_delta:.8f}, allowed={allowed_delta:.8f}"
        )

    result = {
        "schema_version": 1,
        "status": "passed",
        "model_id": expected_model,
        "revision": expected_revision,
        "context_parallel_size": int(context_parallel_size),
        "block_parallel_size": int(block_parallel_size),
        "cp_loss": cp_loss,
        "fused_loss": fused_loss,
        "loss_delta": loss_delta,
        "loss_tolerance": allowed_delta,
        "runtime_jit": False,
        "attention_policy": "production",
        "matched_run_spec_sha256": spec_fingerprint,
    }
    output_path = root / "family_validation.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--context-parallel-size", required=True, type=int)
    parser.add_argument("--block-parallel-size", required=True, type=int)
    parser.add_argument("--loss-atol", type=float, default=5e-2)
    parser.add_argument("--loss-rtol", type=float, default=5e-3)
    args = parser.parse_args()
    result = validate_profile_pair(
        args.root,
        expected_model=args.expected_model,
        expected_revision=args.expected_revision,
        context_parallel_size=args.context_parallel_size,
        block_parallel_size=args.block_parallel_size,
        loss_atol=args.loss_atol,
        loss_rtol=args.loss_rtol,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
