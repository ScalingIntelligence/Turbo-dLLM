#!/usr/bin/env python3
"""Summarize Kineto or Nsight Systems reports under one profile root."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dllm_parallel.core.profiling.trace_analysis import (
    analyze_kineto_trace,
    write_nsys_summary,
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    reports = sorted(args.run_root.glob("*/system_trace/*.nsys-rep"))
    kineto_reports = sorted(args.run_root.glob("*/system_trace/rank_*.trace.json.gz"))
    if not reports and not kineto_reports:
        raise FileNotFoundError(f"no system-trace reports under {args.run_root}")

    by_mode: dict[Path, list[Path]] = {}
    for report in reports:
        trace_dir = report.parent
        sqlite_path = report.with_suffix(".sqlite")
        _run(
            [
                "nsys",
                "export",
                "--type",
                "sqlite",
                "--lazy",
                "false",
                "--force-overwrite",
                "true",
                "--output",
                str(sqlite_path),
                str(report),
            ]
        )
        by_mode.setdefault(trace_dir.parent, []).append(sqlite_path)

    mode_summaries: dict[str, dict[str, object]] = {}
    kineto_by_mode: dict[Path, list[Path]] = {}
    for report in kineto_reports:
        kineto_by_mode.setdefault(report.parent.parent, []).append(report)
    for mode_dir, breakdown_paths in kineto_by_mode.items():
        mode_summaries[mode_dir.name] = {
            "reports": [analyze_kineto_trace(path) for path in breakdown_paths]
        }
    for mode_dir, sqlite_paths in by_mode.items():
        trace_dir = mode_dir / "system_trace"
        summary = write_nsys_summary(sqlite_paths, trace_dir / "gpu_breakdown.json")
        if mode_dir.name in mode_summaries:
            raise ValueError(
                f"mode {mode_dir.name} contains both Kineto and Nsight traces"
            )
        mode_summaries[mode_dir.name] = summary
        _run(
            [
                "nsys",
                "recipe",
                "nccl_sum",
                "--gpu",
                "--dir",
                str(trace_dir),
                "--output",
                str(trace_dir / "nccl_sum"),
                "--force-overwrite",
            ]
        )
        _run(
            [
                "nsys",
                "recipe",
                "nvtx_gpu_proj_sum",
                "--dir",
                str(trace_dir),
                "--output",
                str(trace_dir / "nvtx_gpu_proj_sum"),
                "--force-overwrite",
            ]
        )
        print(trace_dir / "gpu_breakdown.json")

    lines = [
        "# System Trace Summary",
        "",
        "All durations are per-GPU means over the bounded CUPTI capture window.",
        "Overlapping kernels are unioned before durations are computed.",
        "",
        "| Mode | GPUs | Window ms | Compute ms | NCCL ms | Overlap ms | Exposed NCCL ms | Idle ms | NCCL hidden |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("cp", "bp", "fused"):
        summary = mode_summaries.get(mode)
        if summary is None:
            continue
        reports = summary["reports"]
        aggregates = [report["aggregate"] for report in reports]
        gpu_count = sum(int(item["gpu_count"]) for item in aggregates)

        def weighted(name: str) -> float:
            return (
                sum(float(item[name]) * int(item["gpu_count"]) for item in aggregates)
                / gpu_count
            )

        mean_comm = weighted("mean_comm_ms")
        mean_exposed_comm = weighted("mean_exposed_comm_ms")
        hidden_fraction = (
            1.0 - mean_exposed_comm / mean_comm if mean_comm > 0.0 else 0.0
        )

        lines.append(
            f"| {mode} | {gpu_count} | {weighted('mean_window_ms'):.2f} | "
            f"{weighted('mean_compute_ms'):.2f} | {mean_comm:.2f} | "
            f"{weighted('mean_overlap_ms'):.2f} | "
            f"{mean_exposed_comm:.2f} | "
            f"{weighted('mean_idle_ms'):.2f} | "
            f"{100.0 * hidden_fraction:.1f}% |"
        )
    communication_domains = sorted(
        {
            domain
            for summary in mode_summaries.values()
            for report in summary["reports"]
            for domain in report.get("communication_domains", {})
        }
    )
    if communication_domains:
        lines.extend(
            [
                "",
                "## Communication Domain Breakdown",
                "",
                "Attention rows contain collectives explicitly annotated by the "
                "attention transport. Other rows contain all remaining NCCL "
                "kernels, including DP/ZeRO, TP, EP, optimizer, and unattributed "
                "communication. Byte counts are taken from runtime tensors and "
                "represent logical off-rank traffic rather than physical link bytes. "
                "Timing is unioned within each row; rows should not be added because "
                "different communication domains can overlap.",
                "",
                "| Mode | Domain | NCCL ms | Exposed NCCL ms | Calls/GPU | Input GiB/GPU | Logical GiB/GPU |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ("cp", "bp", "fused"):
            summary = mode_summaries.get(mode)
            if summary is None:
                continue
            for domain in communication_domains:
                entries = [
                    report["communication_domains"][domain]
                    for report in summary["reports"]
                    if domain in report.get("communication_domains", {})
                ]
                if not entries:
                    continue
                timed = [entry for entry in entries if "aggregate" in entry]

                def domain_timing(name: str) -> float:
                    if not timed:
                        return 0.0
                    return (
                        sum(
                            float(entry["aggregate"][name])
                            * int(entry["aggregate"]["gpu_count"])
                            for entry in timed
                        )
                        / total_gpus
                    )

                total_gpus = sum(
                    int(report["aggregate"]["gpu_count"])
                    for report in summary["reports"]
                )
                call_values = [
                    int(entry["calls"])
                    for entry in entries
                    if entry.get("calls") is not None
                ]
                calls_text = (
                    f"{sum(call_values) / total_gpus:.1f}" if call_values else "--"
                )
                input_values = [
                    int(entry["input_bytes"])
                    for entry in entries
                    if entry.get("input_bytes") is not None
                ]
                logical_values = [
                    int(entry["logical_bytes"])
                    for entry in entries
                    if entry.get("logical_bytes") is not None
                ]
                input_text = (
                    f"{sum(input_values) / total_gpus / 2**30:.3f}"
                    if input_values
                    else "--"
                )
                logical_text = (
                    f"{sum(logical_values) / total_gpus / 2**30:.3f}"
                    if logical_values
                    else "--"
                )
                lines.append(
                    f"| {mode} | {domain} | "
                    f"{domain_timing('mean_comm_ms'):.2f} | "
                    f"{domain_timing('mean_exposed_comm_ms'):.2f} | "
                    f"{calls_text} | {input_text} | {logical_text} |"
                )

    communication_collectives = sorted(
        {
            name
            for summary in mode_summaries.values()
            for report in summary["reports"]
            for name in report.get("communication_collectives", {})
            if report["communication_collectives"][name].get("domain") == "attention"
        }
    )
    if communication_collectives:
        lines.extend(
            [
                "",
                "## Attention Communication by Collective",
                "",
                "| Mode | Phase | Collective | Calls/GPU | Input GiB/GPU | Logical GiB/GPU | NCCL ms | Exposed NCCL ms |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ("cp", "bp", "fused"):
            summary = mode_summaries.get(mode)
            if summary is None:
                continue
            total_gpus = sum(
                int(report["aggregate"]["gpu_count"]) for report in summary["reports"]
            )
            for name in communication_collectives:
                entries = [
                    report["communication_collectives"][name]
                    for report in summary["reports"]
                    if name in report.get("communication_collectives", {})
                ]
                if not entries:
                    continue
                timed = [entry for entry in entries if "aggregate" in entry]

                def collective_timing(metric: str) -> float:
                    if not timed:
                        return 0.0
                    return (
                        sum(
                            float(entry["aggregate"][metric])
                            * int(entry["aggregate"]["gpu_count"])
                            for entry in timed
                        )
                        / total_gpus
                    )

                exemplar = entries[0]
                lines.append(
                    f"| {mode} | {exemplar['phase']} | "
                    f"{exemplar['collective']} | "
                    f"{sum(int(entry['calls']) for entry in entries) / total_gpus:.1f} | "
                    f"{sum(int(entry['input_bytes']) for entry in entries) / total_gpus / 2**30:.3f} | "
                    f"{sum(int(entry['logical_bytes']) for entry in entries) / total_gpus / 2**30:.3f} | "
                    f"{collective_timing('mean_comm_ms'):.2f} | "
                    f"{collective_timing('mean_exposed_comm_ms'):.2f} |"
                )
    phase_names = sorted(
        {
            phase
            for summary in mode_summaries.values()
            for report in summary["reports"]
            for phase in report.get("phases", {})
        }
    )
    if phase_names:
        lines.extend(
            [
                "",
                "## GPU Phase Breakdown",
                "",
                "Phase values are per-rank means of overlap-aware GPU intervals.",
                "",
                "| Mode | Phase | GPU span ms | Compute ms | NCCL ms | Exposed NCCL ms | Idle ms |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ("cp", "bp", "fused"):
            summary = mode_summaries.get(mode)
            if summary is None:
                continue
            for phase_name in phase_names:
                aggregates = [
                    report["phases"][phase_name]["aggregate"]
                    for report in summary["reports"]
                    if phase_name in report.get("phases", {})
                ]
                if not aggregates:
                    continue
                gpu_count = sum(int(item["gpu_count"]) for item in aggregates)

                def phase_weighted(name: str) -> float:
                    return (
                        sum(
                            float(item[name]) * int(item["gpu_count"])
                            for item in aggregates
                        )
                        / gpu_count
                    )

                lines.append(
                    f"| {mode} | {phase_name} | "
                    f"{phase_weighted('mean_window_ms'):.2f} | "
                    f"{phase_weighted('mean_compute_ms'):.2f} | "
                    f"{phase_weighted('mean_comm_ms'):.2f} | "
                    f"{phase_weighted('mean_exposed_comm_ms'):.2f} | "
                    f"{phase_weighted('mean_idle_ms'):.2f} |"
                )
    kernel_categories = sorted(
        {
            category
            for summary in mode_summaries.values()
            for report in summary["reports"]
            for category in report.get("kernel_categories", {})
        }
    )
    if kernel_categories:
        lines.extend(
            [
                "",
                "## Native Kernel Breakdown",
                "",
                "Kernel categories use unambiguous native CUDA kernel names and "
                "remain reliable when activation-checkpoint replay reuses autograd "
                "sequence IDs. Generic GEMMs cannot always be assigned to a model "
                "substage from their names alone.",
                "",
                "| Mode | Phase | Kernel category | GPU ms | Calls/GPU |",
                "|---|---|---|---:|---:|",
            ]
        )
        for mode in ("cp", "bp", "fused"):
            summary = mode_summaries.get(mode)
            if summary is None:
                continue
            for phase_name in phase_names:
                for category in kernel_categories:
                    aggregates = [
                        report["kernel_categories"][category]["phases"][phase_name][
                            "aggregate"
                        ]
                        for report in summary["reports"]
                        if category in report.get("kernel_categories", {})
                        and phase_name
                        in report["kernel_categories"][category].get("phases", {})
                    ]
                    if not aggregates:
                        continue
                    gpu_count = sum(int(item["gpu_count"]) for item in aggregates)
                    mean_busy_ms = (
                        sum(
                            float(item["mean_busy_ms"]) * int(item["gpu_count"])
                            for item in aggregates
                        )
                        / gpu_count
                    )
                    calls_per_gpu = (
                        sum(int(item["total_kernel_count"]) for item in aggregates)
                        / gpu_count
                    )
                    lines.append(
                        f"| {mode} | {phase_name} | {category} | "
                        f"{mean_busy_ms:.2f} | {calls_per_gpu:.1f} |"
                    )
    operator_families = sorted(
        {
            family
            for summary in mode_summaries.values()
            for report in summary["reports"]
            for family in report.get("operator_families", {})
        }
    )
    if operator_families:
        lines.extend(
            [
                "",
                "## Operator Breakdown",
                "",
                "Operator values use Kineto launch correlation and autograd sequence "
                "IDs. This attribution is heuristic under activation-checkpoint "
                "replay; when it conflicts with an unambiguous native kernel name, "
                "use the native-kernel table above. Compute is the union of "
                "attributed GPU compute intervals; attention communication is "
                "reported separately.",
                "",
                "| Mode | Phase | Operator | Compute ms | NCCL ms | Exposed NCCL ms |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for mode in ("cp", "bp", "fused"):
            summary = mode_summaries.get(mode)
            if summary is None:
                continue
            for phase_name in ("forward", "backward"):
                for family in operator_families:
                    aggregates = [
                        report["operator_families"][family]["phases"][phase_name][
                            "aggregate"
                        ]
                        for report in summary["reports"]
                        if family in report.get("operator_families", {})
                        and phase_name
                        in report["operator_families"][family].get("phases", {})
                    ]
                    if not aggregates:
                        continue
                    gpu_count = sum(int(item["gpu_count"]) for item in aggregates)

                    def operator_weighted(name: str) -> float:
                        return (
                            sum(
                                float(item[name]) * int(item["gpu_count"])
                                for item in aggregates
                            )
                            / gpu_count
                        )

                    lines.append(
                        f"| {mode} | {phase_name} | {family} | "
                        f"{operator_weighted('mean_compute_ms'):.2f} | "
                        f"{operator_weighted('mean_comm_ms'):.2f} | "
                        f"{operator_weighted('mean_exposed_comm_ms'):.2f} |"
                    )

        lines.extend(
            [
                "",
                "### Operator Detail",
                "",
                "| Mode | Phase | Operator subtype | Compute ms | NCCL ms |",
                "|---|---|---|---:|---:|",
            ]
        )
        operator_names = sorted(
            {
                name
                for summary in mode_summaries.values()
                for report in summary["reports"]
                for name in report.get("operators", {})
            }
        )
        for mode in ("cp", "bp", "fused"):
            summary = mode_summaries.get(mode)
            if summary is None:
                continue
            for phase_name in ("forward", "backward"):
                for operator_name in operator_names:
                    aggregates = [
                        report["operators"][operator_name]["phases"][phase_name][
                            "aggregate"
                        ]
                        for report in summary["reports"]
                        if operator_name in report.get("operators", {})
                        and phase_name
                        in report["operators"][operator_name].get("phases", {})
                    ]
                    if not aggregates:
                        continue
                    gpu_count = sum(int(item["gpu_count"]) for item in aggregates)

                    def detail_weighted(name: str) -> float:
                        return (
                            sum(
                                float(item[name]) * int(item["gpu_count"])
                                for item in aggregates
                            )
                            / gpu_count
                        )

                    lines.append(
                        f"| {mode} | {phase_name} | {operator_name} | "
                        f"{detail_weighted('mean_compute_ms'):.2f} | "
                        f"{detail_weighted('mean_comm_ms'):.2f} |"
                    )
    (args.run_root / "system_trace_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (args.run_root / "system_trace_summary.json").write_text(
        json.dumps(mode_summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
