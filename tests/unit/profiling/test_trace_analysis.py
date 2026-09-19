from __future__ import annotations

import sqlite3
import gzip
import json
import sys

import pytest

from dllm_parallel.core.profiling.trace_analysis import (
    analyze_kineto_trace,
    analyze_nsys_sqlite,
)
from benchmarks.tools.analyze_system_traces import main as analyze_system_traces_main


def test_analyze_nsys_sqlite_is_overlap_aware(tmp_path) -> None:
    database = tmp_path / "trace.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
            "start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER, "
            "shortName INTEGER)"
        )
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            ((1, "gemm_kernel"), (2, "ncclKernel_AllReduce")),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
            (
                (0, 10_000_000, 0, 7, 1),
                (4_000_000, 6_000_000, 0, 7, 2),
                (10_000_000, 12_000_000, 0, 7, 2),
            ),
        )

    result = analyze_nsys_sqlite(database)
    device = result["devices"][0]
    assert device["window_ms"] == pytest.approx(12.0)
    assert device["compute_ms"] == pytest.approx(10.0)
    assert device["comm_ms"] == pytest.approx(4.0)
    assert device["overlap_ms"] == pytest.approx(2.0)
    assert device["exposed_comm_ms"] == pytest.approx(2.0)
    assert device["busy_ms"] == pytest.approx(12.0)
    assert device["idle_ms"] == pytest.approx(0.0)
    assert device["comm_hidden_fraction"] == pytest.approx(0.5)


def test_analyze_nsys_sqlite_requires_cuda_kernels(tmp_path) -> None:
    database = tmp_path / "empty.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)"
        )

    with pytest.raises(ValueError, match="no CUDA kernel table"):
        analyze_nsys_sqlite(database)


def test_analyze_kineto_trace_attributes_correlated_kernels_to_phases(
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace.json.gz"
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.phase.backward",
                "ts": 100.0,
                "dur": 100.0,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::matmul",
                "ts": 120.0,
                "dur": 10.0,
                "args": {"External id": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm_kernel",
                "pid": 0,
                "ts": 300.0,
                "dur": 20.0,
                "args": {"External id": 7, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ncclKernel_AllReduce",
                "pid": 0,
                "ts": 310.0,
                "dur": 20.0,
                "args": {"External id": 7, "device": 0},
            },
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)

    result = analyze_kineto_trace(trace_path)
    backward = result["phases"]["backward"]["aggregate"]
    assert backward["mean_compute_ms"] == pytest.approx(0.02)
    assert backward["mean_comm_ms"] == pytest.approx(0.02)
    assert backward["mean_overlap_ms"] == pytest.approx(0.01)


def test_analyze_kineto_trace_separates_attention_collectives_and_bytes(
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace.json.gz"
    communication_label = (
        "dllm.communication.domain=attention;phase=forward;"
        "collective=all_gather_into_tensor;input_bytes=1024;"
        "logical_bytes=3072"
    )
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.phase.backward",
                "ts": 0.0,
                "dur": 90.0,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": communication_label,
                "ts": 10.0,
                "dur": 20.0,
                "args": {"External id": 2},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "c10d::allgather_",
                "ts": 12.0,
                "dur": 5.0,
                "args": {"External id": 20},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "c10d::allreduce_",
                "ts": 50.0,
                "dur": 5.0,
                "args": {"External id": 30},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ncclKernel_AllGather",
                "pid": 0,
                "ts": 100.0,
                "dur": 20.0,
                "args": {"External id": 20, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm_kernel",
                "pid": 0,
                "ts": 110.0,
                "dur": 10.0,
                "args": {"External id": 40, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ncclKernel_AllReduce",
                "pid": 0,
                "ts": 130.0,
                "dur": 30.0,
                "args": {"External id": 30, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm_kernel",
                "pid": 0,
                "ts": 140.0,
                "dur": 10.0,
                "args": {"External id": 41, "device": 0},
            },
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)

    result = analyze_kineto_trace(trace_path)
    collective = result["communication_collectives"][
        "attention.backward.all_gather_into_tensor"
    ]
    assert collective["calls"] == 1
    assert collective["input_bytes"] == 1024
    assert collective["logical_bytes"] == 3072
    assert collective["aggregate"]["mean_comm_ms"] == pytest.approx(0.02)
    assert collective["aggregate"]["mean_exposed_comm_ms"] == pytest.approx(0.01)

    attention = result["communication_domains"]["attention"]
    other = result["communication_domains"]["other"]
    assert attention["logical_bytes"] == 3072
    assert attention["aggregate"]["mean_comm_ms"] == pytest.approx(0.02)
    assert attention["aggregate"]["mean_exposed_comm_ms"] == pytest.approx(0.01)
    assert other["logical_bytes"] is None
    assert other["aggregate"]["mean_comm_ms"] == pytest.approx(0.03)
    assert other["aggregate"]["mean_exposed_comm_ms"] == pytest.approx(0.02)


def test_system_trace_report_renders_attention_bytes_separately(
    tmp_path,
    monkeypatch,
) -> None:
    trace_dir = tmp_path / "cp" / "system_trace"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "rank_0.trace.json.gz"
    label = (
        "dllm.communication.domain=attention;phase=forward;"
        "collective=all_gather_into_tensor;input_bytes=1073741824;"
        "logical_bytes=3221225472"
    )
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.phase.forward",
                "ts": 0.0,
                "dur": 100.0,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": label,
                "ts": 10.0,
                "dur": 20.0,
                "args": {"External id": 2},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "c10d::allgather_",
                "ts": 12.0,
                "dur": 5.0,
                "args": {"External id": 20},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ncclKernel_AllGather",
                "pid": 0,
                "ts": 100.0,
                "dur": 20.0,
                "args": {"External id": 20, "device": 0},
            },
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)

    monkeypatch.setattr(sys, "argv", ["analyze_system_traces.py", str(tmp_path)])
    assert analyze_system_traces_main() == 0
    summary = (tmp_path / "system_trace_summary.md").read_text(encoding="utf-8")
    assert "## Communication Domain Breakdown" in summary
    assert "## Attention Communication by Collective" in summary
    assert "| cp | attention | 0.02 | 0.02 | 1.0 | 1.000 | 3.000 |" in summary
    assert (
        "| cp | forward | all_gather_into_tensor | 1.0 | 1.000 | 3.000 | 0.02 | 0.02 |"
    ) in summary


def test_analyze_kineto_trace_attributes_forward_and_backward_operators(
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace.json.gz"
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.phase.forward",
                "ts": 0.0,
                "dur": 100.0,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.operator.attention.full",
                "ts": 10.0,
                "dur": 30.0,
                "args": {"External id": 2},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "attention_forward",
                "ts": 20.0,
                "dur": 5.0,
                "args": {"External id": 10, "Sequence number": 7},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.operator.feed_forward.dense",
                "ts": 50.0,
                "dur": 30.0,
                "args": {"External id": 3},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "mlp_forward",
                "ts": 60.0,
                "dur": 5.0,
                "args": {"External id": 20, "Sequence number": 8},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.phase.backward",
                "ts": 100.0,
                "dur": 100.0,
                "args": {"External id": 4},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "AttentionBackward",
                "ts": 120.0,
                "dur": 5.0,
                "args": {"External id": 30, "Sequence number": 7},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "MlpBackward",
                "ts": 150.0,
                "dur": 5.0,
                "args": {"External id": 40, "Sequence number": 8},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "attention_fwd_kernel",
                "pid": 0,
                "ts": 300.0,
                "dur": 20.0,
                "args": {"External id": 10, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "mlp_fwd_kernel",
                "pid": 0,
                "ts": 320.0,
                "dur": 30.0,
                "args": {"External id": 20, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "attention_bwd_kernel",
                "pid": 0,
                "ts": 350.0,
                "dur": 40.0,
                "args": {"External id": 30, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "mlp_bwd_kernel",
                "pid": 0,
                "ts": 390.0,
                "dur": 50.0,
                "args": {"External id": 40, "device": 0},
            },
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)

    result = analyze_kineto_trace(trace_path)
    attention = result["operator_families"]["attention"]
    feed_forward = result["operator_families"]["feed_forward"]
    assert attention["phases"]["forward"]["aggregate"][
        "mean_compute_ms"
    ] == pytest.approx(0.02)
    assert attention["phases"]["backward"]["aggregate"][
        "mean_compute_ms"
    ] == pytest.approx(0.04)
    assert feed_forward["phases"]["forward"]["aggregate"][
        "mean_compute_ms"
    ] == pytest.approx(0.03)
    assert feed_forward["phases"]["backward"]["aggregate"][
        "mean_compute_ms"
    ] == pytest.approx(0.05)
    assert result["operators"]["attention.full"]["aggregate"][
        "mean_compute_ms"
    ] == pytest.approx(0.06)
    assert result["operator_attribution"]["aggregate"][
        "compute_coverage"
    ] == pytest.approx(1.0)


def test_native_attention_kernel_overrides_checkpoint_sequence_collision(
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace.json.gz"
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.operator.feed_forward.moe",
                "ts": 0.0,
                "dur": 50.0,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "moe_forward",
                "ts": 10.0,
                "dur": 5.0,
                "args": {"External id": 10, "Sequence number": 7},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.phase.backward",
                "ts": 100.0,
                "dur": 100.0,
                "args": {"External id": 2},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "CheckpointBackward",
                "ts": 120.0,
                "dur": 5.0,
                "args": {"External id": 30, "Sequence number": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "bdlm_splitd_dkdv_kernel",
                "pid": 0,
                "ts": 300.0,
                "dur": 40.0,
                "args": {"External id": 30, "device": 0},
            },
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)

    result = analyze_kineto_trace(trace_path)
    attention = result["operator_families"]["attention"]
    assert attention["phases"]["backward"]["aggregate"][
        "mean_compute_ms"
    ] == pytest.approx(0.04)
    assert "feed_forward" not in result["operator_families"]
    assert result["kernel_categories"]["attention.backward"]["phases"]["backward"][
        "aggregate"
    ]["mean_compute_ms"] == pytest.approx(0.04)


def test_native_moe_kernel_categories_are_reported(tmp_path) -> None:
    trace_path = tmp_path / "trace.json.gz"
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "dllm.phase.forward",
                "ts": 0.0,
                "dur": 100.0,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "moe_forward",
                "ts": 10.0,
                "dur": 5.0,
                "args": {"External id": 10},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "cutlass_GroupProblemShape_grouped_gemm",
                "pid": 0,
                "ts": 200.0,
                "dur": 30.0,
                "args": {"External id": 10, "device": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "deep_ep_dispatch_kernel",
                "pid": 0,
                "ts": 230.0,
                "dur": 20.0,
                "args": {"External id": 10, "device": 0},
            },
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)

    result = analyze_kineto_trace(trace_path)
    assert result["kernel_categories"]["moe.expert_gemm"]["aggregate"][
        "mean_compute_ms"
    ] == pytest.approx(0.03)
    assert result["kernel_categories"]["moe.transport"]["aggregate"][
        "mean_compute_ms"
    ] == pytest.approx(0.02)
