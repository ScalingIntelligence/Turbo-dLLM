from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.tools.verify_family_profile import validate_profile_pair


def _write_profile(root: Path, mode: str, *, loss: float) -> None:
    case = root / mode
    case.mkdir(parents=True)
    bp_size = 1 if mode == "cp" else 2
    events = [
        {
            "event": "resolved_run_spec",
            "spec": {
                "model": {
                    "id": "test/model",
                    "revision": "test-revision",
                    "seq_len": 1024,
                },
                "kernel": {"runtime_jit": False},
                "topology": {
                    "block_parallel_size": bp_size,
                    "context_parallel_size": 2,
                    "expert_parallel_size": 1,
                    "sequence_parallel": False,
                    "tensor_parallel_size": 1,
                },
            },
        },
        {
            "event": "summary",
            "model_id": "test/model",
            "parallel_context_parallel_size": 2,
            "parallel_block_parallel_size": bp_size,
            "cp_bp_policy": {"attention_policy": "production"},
            "rank_metrics": [{"avg_loss": loss}, {"avg_loss": loss}],
        },
    ]
    (case / "train.log").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def _write_status(root: Path) -> None:
    (root / "status.csv").write_text(
        "mode,status,exit_code\ncp,ok,0\nfused,ok,0\n",
        encoding="utf-8",
    )


def test_validate_profile_pair_writes_machine_readable_result(tmp_path: Path) -> None:
    _write_status(tmp_path)
    _write_profile(tmp_path, "cp", loss=2.0)
    _write_profile(tmp_path, "fused", loss=2.01)

    result = validate_profile_pair(
        tmp_path,
        expected_model="test/model",
        expected_revision="test-revision",
        context_parallel_size=2,
        block_parallel_size=2,
    )

    assert result["status"] == "passed"
    assert result["loss_delta"] == pytest.approx(0.01)
    assert len(result["matched_run_spec_sha256"]) == 64
    assert json.loads((tmp_path / "family_validation.json").read_text())["status"] == "passed"


def test_validate_profile_pair_rejects_semantic_loss_mismatch(tmp_path: Path) -> None:
    _write_status(tmp_path)
    _write_profile(tmp_path, "cp", loss=2.0)
    _write_profile(tmp_path, "fused", loss=2.5)

    with pytest.raises(RuntimeError, match="losses differ"):
        validate_profile_pair(
            tmp_path,
            expected_model="test/model",
            expected_revision="test-revision",
            context_parallel_size=2,
            block_parallel_size=2,
        )


def test_validate_profile_pair_rejects_runtime_jit(tmp_path: Path) -> None:
    _write_status(tmp_path)
    _write_profile(tmp_path, "cp", loss=2.0)
    _write_profile(tmp_path, "fused", loss=2.0)
    log_path = tmp_path / "fused" / "train.log"
    log_path.write_text(
        log_path.read_text().replace(
            '"runtime_jit": false',
            '"runtime_jit": true',
        )
    )

    with pytest.raises(RuntimeError, match="runtime kernel JIT"):
        validate_profile_pair(
            tmp_path,
            expected_model="test/model",
            expected_revision="test-revision",
            context_parallel_size=2,
            block_parallel_size=2,
        )


def test_validate_profile_pair_rejects_unmatched_topology(tmp_path: Path) -> None:
    _write_status(tmp_path)
    _write_profile(tmp_path, "cp", loss=2.0)
    _write_profile(tmp_path, "fused", loss=2.0)
    log_path = tmp_path / "fused" / "train.log"
    log_path.write_text(
        log_path.read_text().replace(
            '"tensor_parallel_size": 1',
            '"tensor_parallel_size": 2',
        )
    )

    with pytest.raises(RuntimeError, match="spec.topology.tensor_parallel_size"):
        validate_profile_pair(
            tmp_path,
            expected_model="test/model",
            expected_revision="test-revision",
            context_parallel_size=2,
            block_parallel_size=2,
        )
