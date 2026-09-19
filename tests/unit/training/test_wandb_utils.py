# Copyright 2026 The bdlm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dllm_parallel.training.metrics import (
    WandBRunLogger,
    make_wandb_logger_kwargs,
    resolve_run_id,
    sanitize_run_id,
)


def test_sanitize_run_id_is_stable_and_wandb_safe() -> None:
    assert sanitize_run_id("run name / lr 1e-3") == "run-name-lr-1e-3"


def test_resolve_run_id_prefers_explicit_env() -> None:
    assert resolve_run_id("name", env={"WANDB_RUN_ID": "explicit"}) == "explicit"


def test_make_wandb_logger_kwargs_adds_resume_and_tags() -> None:
    kwargs = make_wandb_logger_kwargs(
        {
            "project": "proj",
            "name": "abc run",
            "id": "None_1",
            "tags": ["base", "paper", "h100"],
        },
        config_payload={
            "algo": {"name": "standard_block_diffusion", "backbone": "causal_lm"},
            "optim": {"lr": 0.001},
            "parallel": {
                "active_block_mode": "dual_end",
                "context_parallel_size": 1,
                "block_parallel_size": 4,
                "kv_backend": "replicated",
            },
        },
    )

    assert kwargs["id"] == "abc-run"
    assert kwargs["resume"] == "allow"
    assert "base" in kwargs["tags"]
    assert "paper" in kwargs["tags"]
    assert "parallel:topology" in kwargs["tags"]
    assert "bp:4" in kwargs["tags"]
    assert "tp:1" in kwargs["tags"]
    assert "cp:1" in kwargs["tags"]
    assert "cp_attention:1" in kwargs["tags"]
    assert "kv_backend:replicated" in kwargs["tags"]
    assert "lr:0.001" in kwargs["tags"]


def test_make_wandb_logger_kwargs_tags_ring_as_context_parallel_attention() -> None:
    kwargs = make_wandb_logger_kwargs(
        {"project": "proj", "name": "ring run"},
        config_payload={
            "parallel": {
                "active_block_mode": "dual_end",
                "context_parallel_size": 4,
                "block_parallel_size": 4,
                "kv_backend": "ring",
            },
        },
    )

    assert "cp:4" in kwargs["tags"]
    assert "cp_attention:4" in kwargs["tags"]
    assert "kv_backend:ring" in kwargs["tags"]


def test_make_wandb_logger_kwargs_tags_tensor_parallel_only() -> None:
    kwargs = make_wandb_logger_kwargs(
        {"project": "proj", "name": "tp run"},
        config_payload={
            "parallel": {
                "active_block_mode": "all_blocks",
                "context_parallel_size": 1,
                "block_parallel_size": 1,
                "tensor_parallel_size": 2,
                "kv_backend": "replicated",
            },
        },
    )

    assert "parallel:topology" in kwargs["tags"]
    assert "tp:2" in kwargs["tags"]
    assert "bp:1" in kwargs["tags"]
    assert "cp_attention:1" in kwargs["tags"]


def test_wandb_run_logger_initializes_and_logs_rank_zero(monkeypatch) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.logged = []
            self.summary = {}
            self.exit_code = None

        def log(self, metrics, step=None):
            self.logged.append((dict(metrics), step))

        def finish(self, exit_code=0):
            self.exit_code = exit_code

    fake_run = FakeRun()

    class FakeWandb:
        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        @staticmethod
        def init(**kwargs):
            fake_run.kwargs = kwargs
            return fake_run

    monkeypatch.setitem(__import__("sys").modules, "wandb", FakeWandb)
    spec = {
        "model": {"id": "Qwen/Qwen3-8B", "seq_len": 1024},
        "objective": {"block_size": 32},
        "topology": {
            "context_parallel_size": 2,
            "block_parallel_size": 2,
            "tensor_parallel_size": 1,
            "kv_backend": "ring",
        },
        "optimizer": {"lr": 1e-6},
        "logging": {
            "wandb": True,
            "wandb_project": "dllm",
            "wandb_mode": "offline",
            "wandb_tags": ("smoke",),
        },
    }
    logger = WandBRunLogger.from_run_spec(
        spec=spec,
        run_context={"run_id": "abc"},
        rank=0,
    )
    assert logger.enabled()
    assert fake_run.kwargs["project"] == "dllm"
    assert fake_run.kwargs["mode"] == "offline"
    assert fake_run.kwargs["id"] == "abc"
    assert "smoke" in fake_run.kwargs["tags"]
    logger.log_step(
        {
            "step": 1,
            "loss": 2.0,
            "lr": 1e-6,
            "ms": 3.0,
            "input_tokens": 1024,
            "valid_tokens": 17,
            "active_tokens": 15,
            "optimizer_overflow": False,
            "perf": {"unique_tokens_per_s_global": 100.0},
        }
    )
    logger.log_summary(
        {
            "completed_steps": 1,
            "max_avg_ms": 3.0,
            "max_peak_mib": 4.0,
            "mfu_pct": 25.0,
            "throughput": {"unique_tokens_per_s_global": 100.0},
        }
    )
    logger.finish(exit_code=0)
    assert fake_run.logged
    assert fake_run.logged[0][0]["trainer/input_tokens"] == 1024
    assert fake_run.logged[0][0]["trainer/valid_tokens"] == 17
    assert fake_run.logged[0][0]["trainer/active_tokens"] == 15
    assert fake_run.logged[0][0]["optimizer/overflow"] is False
    assert fake_run.logged[0][0]["perf/unique_tokens_per_s_global"] == 100.0
    assert fake_run.logged[1][0]["perf/mfu"] == 0.25
    assert fake_run.logged[1][0]["perf/mfu_pct"] == 25.0
    assert fake_run.exit_code == 0
