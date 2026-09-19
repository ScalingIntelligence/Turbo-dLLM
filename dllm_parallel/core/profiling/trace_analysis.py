# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Overlap-aware analysis for Kineto and Nsight Systems GPU timelines.

Kineto Chrome traces provide portable CUPTI capture. Nsight's ``.nsys-rep``
provides the native-cluster artifact and is read through its exported SQLite
form. Both paths produce the same compact breakdown without depending on
console formatting from a particular profiler release.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import statistics
from bisect import bisect_right
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dllm_parallel.core.profiling.perf import compute_breakdown, is_comm_kernel


_KERNEL_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
)
_NCCL_KERNEL_RE = re.compile(r"nccl|msccl|nvshmem", re.IGNORECASE)
_ATTENTION_KERNEL_RE = re.compile(
    r"bdlm_splitd|flash[_]?attn|flashattention|flex_attention|fmha|attention",
    re.IGNORECASE,
)
_ATTENTION_BACKWARD_RE = re.compile(
    r"bwd|backward|dkdv|(?:^|_)dq(?:_|$)",
    re.IGNORECASE,
)
_MOE_TRANSPORT_RE = re.compile(r"deep[_]?ep|deepep", re.IGNORECASE)
_MOE_GROUPED_GEMM_RE = re.compile(
    r"GroupProblemShape|grouped[_ ]?mm|grouped.?gemm",
    re.IGNORECASE,
)
_DENSE_GEMM_RE = re.compile(r"nvjet|gemm|cutlass", re.IGNORECASE)
_COMMUNICATION_RE = re.compile(
    r"^domain=(?P<domain>[a-z0-9_]+);"
    r"phase=(?P<phase>[a-z0-9_]+);"
    r"collective=(?P<collective>[a-z0-9_]+);"
    r"input_bytes=(?P<input_bytes>[0-9]+);"
    r"logical_bytes=(?P<logical_bytes>[0-9]+)$"
)


def _kernel_category(name: str, *, is_comm: bool) -> str:
    """Classify kernels whose native names identify their execution stage."""

    if is_comm:
        return "collective"
    if _ATTENTION_KERNEL_RE.search(name):
        direction = "backward" if _ATTENTION_BACKWARD_RE.search(name) else "forward"
        return f"attention.{direction}"
    if _MOE_TRANSPORT_RE.search(name):
        return "moe.transport"
    if _MOE_GROUPED_GEMM_RE.search(name):
        return "moe.expert_gemm"
    if _DENSE_GEMM_RE.search(name):
        return "dense_or_projection_gemm"
    return "other"


def _kernel_operator_override(name: str, *, is_comm: bool) -> str | None:
    category = _kernel_category(name, is_comm=is_comm)
    if category.startswith("attention."):
        return "attention.kernel"
    if category.startswith("moe."):
        return "feed_forward.moe.kernel"
    return None


def _annotation_lookup(
    events: list[dict[str, Any]],
    prefix: str,
) -> tuple[list[float], list[tuple[float, float, str]]]:
    intervals = sorted(
        (
            float(event["ts"]),
            float(event["ts"]) + float(event.get("dur", 0.0)),
            str(event["name"]).removeprefix(prefix),
        )
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith(prefix)
    )
    return [item[0] for item in intervals], intervals


def _annotation_at(
    timestamp: float,
    starts: list[float],
    intervals: list[tuple[float, float, str]],
) -> str | None:
    if not intervals:
        return None
    index = bisect_right(starts, timestamp) - 1
    # Phase and operator ranges are normally disjoint. The bounded reverse
    # scan also handles a small number of nested profiler annotations.
    for candidate in range(index, max(-1, index - 32), -1):
        start, stop, name = intervals[candidate]
        if start <= timestamp < stop:
            return name
    return None


def _communication_metadata(label: str) -> dict[str, str | int] | None:
    match = _COMMUNICATION_RE.fullmatch(label)
    if match is None:
        return None
    return {
        "domain": match.group("domain"),
        "phase": match.group("phase"),
        "collective": match.group("collective"),
        "input_bytes": int(match.group("input_bytes")),
        "logical_bytes": int(match.group("logical_bytes")),
    }


def _communication_key(metadata: dict[str, str | int]) -> str:
    return ".".join(str(metadata[field]) for field in ("domain", "phase", "collective"))


def _comm_with_global_compute(
    communication: dict[tuple[int, int], list[tuple[float, float, str, bool]]],
    compute: dict[tuple[int, int], list[tuple[float, float, str, bool]]],
) -> dict[tuple[int, int], list[tuple[float, float, str, bool]]]:
    """Combine selected communication with all compute for exposed-time math."""

    combined: dict[tuple[int, int], list[tuple[float, float, str, bool]]] = {}
    for key in set(communication) | set(compute):
        intervals = [*compute.get(key, ()), *communication.get(key, ())]
        if intervals:
            combined[key] = intervals
    return combined


def _sequence_number(event: dict[str, Any]) -> int | None:
    args = event.get("args", {})
    value = args.get("Sequence number", args.get("Sequence Number"))
    if value is None:
        return None
    try:
        sequence = int(value)
    except (TypeError, ValueError):
        return None
    return sequence if sequence >= 0 else None


def _group_payload(
    grouped: dict[tuple[int, int], list[tuple[float, float, str, bool]]],
) -> dict[str, Any]:
    devices, aggregate = _breakdown_payload(grouped)
    return {"devices": devices, "aggregate": aggregate}


def _breakdown_payload(
    grouped: dict[tuple[int, int], list[tuple[float, float, str, bool]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for (owner_id, device_id), intervals in sorted(grouped.items()):
        breakdown = compute_breakdown(intervals)
        if breakdown is None:
            continue
        payload = asdict(breakdown)
        payload.update(
            {
                "process": owner_id,
                "device": device_id,
                "overlap_ms": breakdown.comm_ms - breakdown.exposed_comm_ms,
                "comm_hidden_fraction": (
                    1.0 - breakdown.exposed_comm_ms / breakdown.comm_ms
                    if breakdown.comm_ms > 0.0
                    else 0.0
                ),
                "comm_kernel_count": sum(
                    1 for _, _, _, is_comm in intervals if is_comm
                ),
                "compute_kernel_count": sum(
                    1 for _, _, _, is_comm in intervals if not is_comm
                ),
            }
        )
        devices.append(payload)

    if not devices:
        raise ValueError("trace has no positive-duration CUDA kernels")
    metric_names = (
        "window_ms",
        "busy_ms",
        "compute_ms",
        "comm_ms",
        "exposed_comm_ms",
        "overlap_ms",
        "idle_ms",
        "comm_hidden_fraction",
    )
    aggregate = {
        f"mean_{name}": statistics.mean(float(item[name]) for item in devices)
        for name in metric_names
    }
    aggregate.update(
        {
            "gpu_count": len(devices),
            "max_window_ms": max(float(item["window_ms"]) for item in devices),
            "total_kernel_count": sum(int(item["n_device_events"]) for item in devices),
            "total_comm_kernel_count": sum(
                int(item["comm_kernel_count"]) for item in devices
            ),
        }
    )
    return devices, aggregate


def analyze_nsys_sqlite(path: str | Path) -> dict[str, Any]:
    """Return per-GPU and aggregate kernel/communication timing for a report."""

    database = Path(path)
    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        kernel_table = next((name for name in _KERNEL_TABLES if name in tables), None)
        if kernel_table is None:
            raise ValueError(f"Nsight report has no CUDA kernel table: {database}")
        columns = {
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{kernel_table}")')
        }
        owner_column = "globalPid" if "globalPid" in columns else "contextId"
        name_column = "shortName" if "shortName" in columns else "demangledName"
        rows = connection.execute(
            f'SELECT k.start, k.end, k.deviceId, k."{owner_column}", s.value '
            f'FROM "{kernel_table}" AS k '
            f'JOIN StringIds AS s ON k."{name_column}" = s.id '
            "ORDER BY k.start"
        ).fetchall()

    grouped: dict[tuple[int, int], list[tuple[float, float, str, bool]]] = {}
    for start_ns, end_ns, device_id, owner_id, name in rows:
        start_ms = float(start_ns) / 1.0e6
        end_ms = float(end_ns) / 1.0e6
        if end_ms <= start_ms:
            continue
        key = (int(owner_id or 0), int(device_id))
        kernel_name = str(name)
        grouped.setdefault(key, []).append(
            (start_ms, end_ms, kernel_name, bool(_NCCL_KERNEL_RE.search(kernel_name)))
        )

    try:
        devices, aggregate = _breakdown_payload(grouped)
    except ValueError:
        raise ValueError(
            f"Nsight report has no positive-duration CUDA kernels: {database}"
        ) from None
    return {
        "source": str(database),
        "devices": devices,
        "aggregate": aggregate,
    }


def analyze_kineto_trace(path: str | Path) -> dict[str, Any]:
    """Return overlap-aware overall and phase timing from a Chrome trace."""

    trace_path = Path(path)
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    events = trace.get("traceEvents", ())
    phase_starts, phase_intervals = _annotation_lookup(events, "dllm.phase.")
    operator_starts, operator_intervals = _annotation_lookup(
        events,
        "dllm.operator.",
    )
    communication_starts, communication_intervals = _annotation_lookup(
        events,
        "dllm.communication.",
    )

    communication_records: dict[str, dict[str, Any]] = {}
    for start, _, label in communication_intervals:
        metadata = _communication_metadata(label)
        if metadata is None:
            continue
        execution_phase = _annotation_at(start, phase_starts, phase_intervals)
        if execution_phase is not None:
            metadata["phase"] = execution_phase
        key = _communication_key(metadata)
        record = communication_records.setdefault(
            key,
            {
                "domain": metadata["domain"],
                "phase": metadata["phase"],
                "collective": metadata["collective"],
                "calls": 0,
                "input_bytes": 0,
                "logical_bytes": 0,
            },
        )
        record["calls"] += 1
        record["input_bytes"] += int(metadata["input_bytes"])
        record["logical_bytes"] += int(metadata["logical_bytes"])

    external_phase: dict[int, str] = {}
    external_operator: dict[int, str] = {}
    external_communication: dict[int, str] = {}
    sequence_operators: dict[int, set[str]] = {}
    for event in events:
        if event.get("ph") != "X" or str(event.get("cat", "")).startswith("gpu"):
            continue
        external_id = event.get("args", {}).get("External id")
        timestamp = event.get("ts")
        if external_id is None or timestamp is None:
            continue
        event_timestamp = float(timestamp)
        phase = _annotation_at(event_timestamp, phase_starts, phase_intervals)
        if phase is not None:
            external_phase[int(external_id)] = phase
        operator = _annotation_at(
            event_timestamp,
            operator_starts,
            operator_intervals,
        )
        if operator is not None:
            external_operator[int(external_id)] = operator
            sequence = _sequence_number(event)
            if sequence is not None:
                sequence_operators.setdefault(sequence, set()).add(operator)
        communication = _annotation_at(
            event_timestamp,
            communication_starts,
            communication_intervals,
        )
        if communication is not None:
            metadata = _communication_metadata(communication)
            if metadata is not None:
                if phase is not None:
                    metadata["phase"] = phase
                external_communication[int(external_id)] = _communication_key(metadata)

    # Autograd CPU events execute outside their original forward annotation.
    # Kineto preserves their forward sequence number, which lets us attribute
    # attention and feed-forward backward kernels without naming kernels.
    for event in events:
        if event.get("ph") != "X" or str(event.get("cat", "")).startswith("gpu"):
            continue
        external_id = event.get("args", {}).get("External id")
        if external_id is None or int(external_id) in external_operator:
            continue
        sequence = _sequence_number(event)
        operators = sequence_operators.get(sequence) if sequence is not None else None
        if operators is not None and len(operators) == 1:
            external_operator[int(external_id)] = next(iter(operators))

    grouped: dict[tuple[int, int], list[tuple[float, float, str, bool]]] = {}
    grouped_by_phase: dict[
        str, dict[tuple[int, int], list[tuple[float, float, str, bool]]]
    ] = {}
    grouped_by_operator: dict[
        str, dict[tuple[int, int], list[tuple[float, float, str, bool]]]
    ] = {}
    grouped_by_operator_phase: dict[
        tuple[str, str],
        dict[tuple[int, int], list[tuple[float, float, str, bool]]],
    ] = {}
    grouped_by_operator_family: dict[
        str, dict[tuple[int, int], list[tuple[float, float, str, bool]]]
    ] = {}
    grouped_by_operator_family_phase: dict[
        tuple[str, str],
        dict[tuple[int, int], list[tuple[float, float, str, bool]]],
    ] = {}
    grouped_by_kernel_category: dict[
        str, dict[tuple[int, int], list[tuple[float, float, str, bool]]]
    ] = {}
    grouped_by_kernel_category_phase: dict[
        tuple[str, str],
        dict[tuple[int, int], list[tuple[float, float, str, bool]]],
    ] = {}
    grouped_compute: dict[tuple[int, int], list[tuple[float, float, str, bool]]] = {}
    grouped_other_communication: dict[
        tuple[int, int], list[tuple[float, float, str, bool]]
    ] = {}
    grouped_by_communication: dict[
        str, dict[tuple[int, int], list[tuple[float, float, str, bool]]]
    ] = {}
    grouped_by_communication_domain: dict[
        str, dict[tuple[int, int], list[tuple[float, float, str, bool]]]
    ] = {}
    attributed: dict[tuple[int, int], list[tuple[float, float, str, bool]]] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue
        duration_us = float(event.get("dur", 0.0))
        if duration_us <= 0.0:
            continue
        args = event.get("args", {})
        start_ms = float(event["ts"]) / 1.0e3
        end_ms = start_ms + duration_us / 1.0e3
        name = str(event.get("name", ""))
        key = (int(event.get("pid", 0)), int(args.get("device", event.get("pid", 0))))
        is_comm = is_comm_kernel(name)
        interval = (start_ms, end_ms, name, is_comm)
        grouped.setdefault(key, []).append(interval)
        if not is_comm:
            grouped_compute.setdefault(key, []).append(interval)
        external_id = args.get("External id")
        phase_name = (
            external_phase.get(int(external_id)) if external_id is not None else None
        )
        if phase_name is not None:
            grouped_by_phase.setdefault(phase_name, {}).setdefault(key, []).append(
                interval
            )
        correlated_operator = (
            external_operator.get(int(external_id)) if external_id is not None else None
        )
        inferred_operator = _kernel_operator_override(name, is_comm=is_comm)
        if inferred_operator is not None and (
            correlated_operator is None
            or correlated_operator.split(".", 1)[0]
            != inferred_operator.split(".", 1)[0]
        ):
            operator_name = inferred_operator
        else:
            operator_name = correlated_operator
        if operator_name is not None:
            family = operator_name.split(".", 1)[0]
            grouped_by_operator.setdefault(operator_name, {}).setdefault(
                key, []
            ).append(interval)
            grouped_by_operator_family.setdefault(family, {}).setdefault(
                key, []
            ).append(interval)
            attributed.setdefault(key, []).append(interval)
            if phase_name is not None:
                grouped_by_operator_phase.setdefault(
                    (operator_name, phase_name), {}
                ).setdefault(key, []).append(interval)
                grouped_by_operator_family_phase.setdefault(
                    (family, phase_name), {}
                ).setdefault(key, []).append(interval)
        if event.get("cat") == "kernel":
            category = _kernel_category(name, is_comm=is_comm)
            grouped_by_kernel_category.setdefault(category, {}).setdefault(
                key, []
            ).append(interval)
            if phase_name is not None:
                grouped_by_kernel_category_phase.setdefault(
                    (category, phase_name), {}
                ).setdefault(key, []).append(interval)
        if is_comm:
            communication_key = (
                external_communication.get(int(external_id))
                if external_id is not None
                else None
            )
            record = (
                communication_records.get(communication_key)
                if communication_key is not None
                else None
            )
            if record is None:
                grouped_other_communication.setdefault(key, []).append(interval)
            else:
                assert communication_key is not None
                grouped_by_communication.setdefault(communication_key, {}).setdefault(
                    key, []
                ).append(interval)
                grouped_by_communication_domain.setdefault(
                    str(record["domain"]), {}
                ).setdefault(key, []).append(interval)

    devices, aggregate = _breakdown_payload(grouped)
    phase_payload: dict[str, dict[str, Any]] = {}
    for phase_name, phase_grouped in sorted(grouped_by_phase.items()):
        phase_payload[phase_name] = _group_payload(phase_grouped)

    def operator_payload(
        groups: dict[str, dict[tuple[int, int], list[tuple[float, float, str, bool]]]],
        phase_groups: dict[
            tuple[str, str],
            dict[tuple[int, int], list[tuple[float, float, str, bool]]],
        ],
    ) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for operator_name, operator_grouped in sorted(groups.items()):
            item = _group_payload(operator_grouped)
            item["phases"] = {
                phase_name: _group_payload(phase_grouped)
                for (candidate, phase_name), phase_grouped in sorted(
                    phase_groups.items()
                )
                if candidate == operator_name
            }
            payload[operator_name] = item
        return payload

    operators = operator_payload(grouped_by_operator, grouped_by_operator_phase)
    operator_families = operator_payload(
        grouped_by_operator_family,
        grouped_by_operator_family_phase,
    )
    kernel_categories = operator_payload(
        grouped_by_kernel_category,
        grouped_by_kernel_category_phase,
    )
    communication_collectives: dict[str, dict[str, Any]] = {}
    for communication_key, record in sorted(communication_records.items()):
        payload = dict(record)
        selected = grouped_by_communication.get(communication_key)
        if selected:
            payload.update(
                _group_payload(_comm_with_global_compute(selected, grouped_compute))
            )
        communication_collectives[communication_key] = payload

    communication_domains: dict[str, dict[str, Any]] = {}
    for domain in sorted(
        {str(record["domain"]) for record in communication_records.values()}
        | set(grouped_by_communication_domain)
    ):
        records = [
            record
            for record in communication_records.values()
            if record["domain"] == domain
        ]
        payload: dict[str, Any] = {
            "calls": sum(int(record["calls"]) for record in records),
            "input_bytes": sum(int(record["input_bytes"]) for record in records),
            "logical_bytes": sum(int(record["logical_bytes"]) for record in records),
        }
        selected = grouped_by_communication_domain.get(domain)
        if selected:
            payload.update(
                _group_payload(_comm_with_global_compute(selected, grouped_compute))
            )
        communication_domains[domain] = payload
    if grouped_other_communication:
        communication_domains["other"] = {
            "calls": None,
            "input_bytes": None,
            "logical_bytes": None,
            **_group_payload(
                _comm_with_global_compute(
                    grouped_other_communication,
                    grouped_compute,
                )
            ),
        }
    attribution: dict[str, Any] = {}
    if attributed:
        attribution = _group_payload(attributed)
        total_compute = float(aggregate["mean_compute_ms"])
        attributed_compute = float(attribution["aggregate"]["mean_compute_ms"])
        attribution["aggregate"]["compute_coverage"] = (
            attributed_compute / total_compute if total_compute > 0.0 else 0.0
        )
    return {
        "source": str(trace_path),
        "devices": devices,
        "aggregate": aggregate,
        "phases": phase_payload,
        "operators": operators,
        "operator_families": operator_families,
        "kernel_categories": kernel_categories,
        "communication_collectives": communication_collectives,
        "communication_domains": communication_domains,
        "operator_attribution": attribution,
    }


def write_nsys_summary(
    sqlite_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    reports = [analyze_nsys_sqlite(path) for path in sqlite_paths]
    result = {"reports": reports}
    Path(output_path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
