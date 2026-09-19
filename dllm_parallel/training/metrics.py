# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""W&B helpers for the canonical production trainer."""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


FAILED_STATES = {"failed", "crashed", "killed"}
RUN_ID_MAX_LEN = 120
RUN_ID_DIGEST_LEN = 12
DEFAULT_POLL_TIMEOUT_S = 3600
DEFAULT_POLL_INTERVAL_S = 60


def sanitize_run_id(name: str, *, max_len: int = RUN_ID_MAX_LEN) -> str:
    """Convert a human run name into a W&B-safe deterministic ID."""

    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip(".-")
    if not run_id:
        run_id = "run"
    if len(run_id) > max_len:
        digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:RUN_ID_DIGEST_LEN]
        run_id = f"{run_id[: max_len - (RUN_ID_DIGEST_LEN + 1)].rstrip('.-')}-{digest}"
    return run_id


def resolve_run_id(
    run_name: str | None,
    *,
    configured_id: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the W&B run ID from explicit config, W&B env, or run name."""

    env = os.environ if env is None else env
    explicit = env.get("WANDB_RUN_ID")
    if explicit:
        return explicit
    if configured_id and not configured_id.startswith("None_"):
        return configured_id
    if run_name:
        return sanitize_run_id(run_name)
    return None


def resolve_resume(env: Mapping[str, str] | None = None) -> str:
    """Resolve W&B resume mode; default to preemption-safe ``allow``."""

    env = os.environ if env is None else env
    return env.get("WANDB_RESUME") or "allow"


def format_tag_value(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "unset"
    return re.sub(r"\s+", "_", text)


def wandb_tags(config_payload: Mapping[str, Any] | None = None) -> list[str]:
    """Build stable W&B tags from common RunSpec fields."""

    tags: list[str] = []
    config_payload = config_payload or {}
    algo = config_payload.get("algo")
    model = config_payload.get("model")
    optim = config_payload.get("optim")
    parallel = config_payload.get("parallel")
    objective = config_payload.get("objective")
    topology = config_payload.get("topology")
    optimizer = config_payload.get("optimizer")

    if isinstance(algo, Mapping):
        for key, prefix in (("name", "algo"), ("backbone", "backbone")):
            value = algo.get(key)
            if value is not None:
                tags.append(f"{prefix}:{format_tag_value(value)}")
    if isinstance(model, Mapping):
        for key in ("length", "seq_len", "hidden_size", "n_blocks", "num_layers"):
            value = model.get(key)
            if value is not None:
                tags.append(f"{key}:{format_tag_value(value)}")
    block_size = config_payload.get("block_size")
    if block_size is None and isinstance(objective, Mapping):
        block_size = objective.get("block_size")
    if block_size is not None:
        tags.append(f"block_size:{format_tag_value(block_size)}")
    if isinstance(optim, Mapping) and optim.get("lr") is not None:
        lr_tag = format_tag_value(optim["lr"])
        tags.extend([f"lr:{lr_tag}", f"learning_rate:{lr_tag}"])
    if isinstance(optimizer, Mapping) and optimizer.get("lr") is not None:
        lr_tag = format_tag_value(optimizer["lr"])
        tags.extend([f"lr:{lr_tag}", f"learning_rate:{lr_tag}"])
    if parallel is None and isinstance(topology, Mapping):
        parallel = topology
    if isinstance(parallel, Mapping):
        mode = parallel.get("active_block_mode", "unknown")
        cp = int(parallel.get("context_parallel_size", 1) or 1)
        bp = int(parallel.get("block_parallel_size", 1) or 1)
        tp = int(parallel.get("tensor_parallel_size", 1) or 1)
        ep = int(parallel.get("expert_parallel_size", 1) or 1)
        if bp > 1 or cp > 1 or tp > 1 or ep > 1:
            kv_backend = parallel.get("kv_backend", "replicated")
            tags.extend([
                "parallel:topology",
                f"active_blocks:{format_tag_value(mode)}",
                f"kv_backend:{format_tag_value(kv_backend)}",
                f"cp:{cp}",
                f"bp:{bp}",
                f"tp:{tp}",
                f"ep:{ep}",
                f"cp_attention:{cp}",
            ])

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def make_wandb_logger_kwargs(
    wandb_config: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return W&B logger kwargs with deterministic resume and merged tags."""

    kwargs = dict(wandb_config)
    name = kwargs.get("name")
    run_id = resolve_run_id(
        str(name) if name is not None else None,
        configured_id=kwargs.get("id"),
    )
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = resolve_resume()
    elif str(kwargs.get("id") or "").startswith("None_"):
        kwargs.pop("id", None)

    tags = list(kwargs.get("tags") or [])
    tags.extend(wandb_tags(config_payload))
    kwargs["tags"] = list(dict.fromkeys(str(tag) for tag in tags if tag is not None))

    return kwargs


@dataclass
class WandBRunLogger:
    """Rank-zero W&B logger for the canonical trainer."""

    run: Any | None = None

    @classmethod
    def from_run_spec(
        cls,
        *,
        spec: Mapping[str, Any],
        run_context: Mapping[str, Any],
        rank: int,
    ) -> "WandBRunLogger":
        logging = spec.get("logging") if isinstance(spec, Mapping) else None
        if not isinstance(logging, Mapping) or not bool(logging.get("wandb", False)):
            return cls()
        if int(rank) != 0:
            return cls()
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "logging.wandb=true requires the wandb package to be installed"
            ) from exc
        if str(logging.get("wandb_mode", "online")) == "disabled":
            return cls()
        kwargs = make_wandb_logger_kwargs(
            {
                "project": logging.get("wandb_project"),
                "entity": logging.get("wandb_entity"),
                "name": logging.get("wandb_name") or _default_wandb_name(spec),
                "group": logging.get("wandb_group"),
                # A display name is not a run identity. Reusing a deterministic
                # name for repeated CP/BP campaigns otherwise silently resumes
                # and merges unrelated histories. Explicit IDs and
                # WANDB_RUN_ID still take precedence for preemption-safe resume.
                "id": logging.get("wandb_id")
                or os.environ.get("WANDB_RUN_ID")
                or run_context.get("run_id"),
                "mode": logging.get("wandb_mode", "online"),
                "dir": logging.get("wandb_dir"),
                "tags": list(logging.get("wandb_tags") or ()),
                "config": {
                    "run_spec": spec,
                    "run_context": run_context,
                },
            },
            config_payload=spec,
        )
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return cls(run=wandb.init(**kwargs))

    def enabled(self) -> bool:
        return self.run is not None

    def log_step(self, payload: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        step = int(payload.get("step", 0) or 0)
        metrics = _flatten_metrics(
            {
                "trainer/loss": payload.get("loss"),
                "trainer/lr": payload.get("lr"),
                "trainer/grad_norm": payload.get("grad_norm"),
                "trainer/gradient_clip_norm": payload.get("gradient_clip_norm"),
                "trainer/step": payload.get("step"),
                "trainer/timed": payload.get("timed"),
                "trainer/input_tokens": payload.get("input_tokens"),
                "trainer/valid_tokens": payload.get("valid_tokens"),
                "trainer/active_tokens": payload.get("active_tokens"),
                "perf/step_time_ms": payload.get("ms"),
                "perf": payload.get("perf"),
                "memory/peak_mib": payload.get("peak_mib"),
                "optimizer/backend": payload.get("optimizer_backend"),
                "optimizer/zero_impl": payload.get("zero_optimizer_impl"),
                "optimizer/step": payload.get("optimizer_step"),
                "optimizer/zero_sample_parallel": payload.get("zero_sample_parallel"),
                "optimizer/overflow": payload.get("optimizer_overflow"),
                "phase_ms": payload.get("phase_ms"),
            }
        )
        self.run.log(metrics, step=step)

    def log_summary(self, payload: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        metrics = _flatten_metrics(
            {
                "summary/max_avg_ms": payload.get("max_avg_ms"),
                "summary/max_peak_mib": payload.get("max_peak_mib"),
                "summary/seq_len": payload.get("seq_len"),
                "summary/batch_size": payload.get("batch_size"),
                "summary/steps": payload.get("steps"),
                "summary/completed_steps": payload.get("completed_steps"),
                "summary/training_elapsed_seconds": payload.get(
                    "training_elapsed_seconds"
                ),
                "summary/duration_limit_reached": payload.get(
                    "duration_limit_reached"
                ),
                "summary/optimizer_backend": payload.get("optimizer_backend"),
                "parallel/data_parallel_size": payload.get("parallel_data_parallel_size"),
                "parallel/model_parallel_size": payload.get("parallel_model_parallel_size"),
                "parallel/tensor_parallel_size": payload.get("parallel_tensor_parallel_size"),
                "parallel/context_parallel_size": payload.get("parallel_context_parallel_size"),
                "parallel/block_parallel_size": payload.get("parallel_block_parallel_size"),
                "perf": {
                    **dict(payload.get("throughput") or {}),
                    "mfu_pct": payload.get("mfu_pct"),
                    "mfu": (
                        float(payload["mfu_pct"]) / 100.0
                        if payload.get("mfu_pct") is not None
                        else None
                    ),
                },
                "evaluation": payload.get("evaluation"),
            }
        )
        self.run.log(metrics, step=int(payload.get("completed_steps", 0) or 0))
        for key, value in metrics.items():
            self.run.summary[key] = value

    def finish(self, exit_code: int = 0) -> None:
        if self.run is None:
            return
        self.run.finish(exit_code=int(exit_code))
        self.run = None


def _default_wandb_name(spec: Mapping[str, Any]) -> str:
    model = spec.get("model") if isinstance(spec, Mapping) else {}
    topology = spec.get("topology") if isinstance(spec, Mapping) else {}
    model_id = str(model.get("id", "dllm")) if isinstance(model, Mapping) else "dllm"
    seq_len = model.get("seq_len", "seq") if isinstance(model, Mapping) else "seq"
    if isinstance(topology, Mapping):
        cp = int(topology.get("context_parallel_size", 1) or 1)
        bp = int(topology.get("block_parallel_size", 1) or 1)
        tp = int(topology.get("tensor_parallel_size", 1) or 1)
        ep = int(topology.get("expert_parallel_size", 1) or 1)
    else:
        cp = bp = tp = ep = 1
    return sanitize_run_id(
        f"{model_id}-{seq_len}-cp{cp}-bp{bp}-tp{tp}-ep{ep}"
    )


def _flatten_metrics(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            out.update(_flatten_metrics(value, prefix=name))
            continue
        if isinstance(value, (str, bool, int, float)):
            out[name] = value
    return out


def poll_wandb_runs(
    *,
    entity: str,
    project: str,
    run_ids: list[str],
    min_step: int = 0,
    timeout_s: int = DEFAULT_POLL_TIMEOUT_S,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
) -> dict[str, dict[str, object]]:
    """Poll W&B until all run IDs are healthy or the timeout expires."""

    try:
        import wandb
    except ModuleNotFoundError:
        return {
            run_id: {"state": "unmonitored", "error": "local wandb is not installed"}
            for run_id in run_ids
        }

    api = wandb.Api()
    deadline = time.time() + timeout_s
    states: dict[str, dict[str, object]] = {
        run_id: {"state": "pending"} for run_id in run_ids
    }
    while time.time() < deadline:
        healthy = 0
        for run_id in run_ids:
            try:
                run = api.run(f"{entity}/{project}/{run_id}")
                summary = dict(run.summary)
                step = int(
                    summary.get(
                        "trainer/step",
                        summary.get(
                            "summary/completed_steps",
                            summary.get("step", summary.get("_step", -1)),
                        ),
                    )
                )
                state = getattr(run, "state", None)
                states[run_id] = {
                    "state": state,
                    "step": step,
                    "url": run.url,
                    "loss": summary.get("trainer/loss", summary.get("loss")),
                    "perf/input_tokens_per_s_global": summary.get(
                        "perf/input_tokens_per_s_global"
                    ),
                    "perf/tokens_per_s_global": summary.get("perf/tokens_per_s_global"),
                    "perf/unique_tokens_per_s_global": summary.get(
                        "perf/unique_tokens_per_s_global"
                    ),
                    "perf/mfu": summary.get("perf/mfu"),
                }
                if step >= min_step and state not in FAILED_STATES:
                    healthy += 1
            except Exception as exc:
                states[run_id] = {"state": "pending", "error": type(exc).__name__}
        if healthy == len(run_ids):
            return states
        time.sleep(poll_interval_s)
    return states
