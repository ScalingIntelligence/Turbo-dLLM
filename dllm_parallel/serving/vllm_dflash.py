"""Validate and launch native vLLM DFlash2 speculative serving."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from dllm_parallel.core.models.backbones.dflash.executor import (
    validate_sglang_dflash2_export,
)


QUALIFIED_VLLM_VERSION = "0.29.0"


def _draft_tokens(block_size: int) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError("DFlash2 block_size must be an integer")
    if block_size < 2:
        raise ValueError("vLLM DFlash2 block_size must be at least 2")
    return block_size - 1


def require_vllm_version(installed_version: str | None = None) -> str:
    """Require the exact vLLM release qualified by Turbo-dLLM."""

    if installed_version is None:
        try:
            installed_version = metadata.version("vllm")
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "vLLM is not installed; use a separate Linux environment with "
                'python -m pip install "turbo-dllm[vllm]"'
            ) from error
    if installed_version != QUALIFIED_VLLM_VERSION:
        raise RuntimeError(
            f"Turbo-dLLM requires vLLM {QUALIFIED_VLLM_VERSION}; "
            f"found {installed_version}"
        )
    return installed_version


def validate_vllm_dflash2_export(
    export_dir: str | Path,
    *,
    expected_block_size: int | None = None,
) -> dict[str, Any]:
    """Validate the shared DFlash2 artifact against vLLM's native contract."""

    root = Path(export_dir).expanduser().resolve()
    shared = validate_sglang_dflash2_export(
        root, expected_block_size=expected_block_size
    )
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError(
            "the native vLLM DFlash2 loader supports model_type='qwen3' only; "
            f"got {config.get('model_type')!r}"
        )
    method = config.get("dflash_config")
    if not isinstance(method, Mapping):
        raise ValueError("vLLM DFlash2 requires a dflash_config mapping")
    top_level = config.get("block_size")
    nested = method.get("block_size")
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError(
            "DFlash2 export has conflicting block_size values: "
            f"top-level={top_level!r}, dflash_config={nested!r}"
        )
    block_size = top_level if top_level is not None else nested
    num_speculative_tokens = _draft_tokens(block_size)
    if expected_block_size is not None and block_size != expected_block_size:
        raise ValueError(
            f"exported block_size={block_size!r}, expected {expected_block_size}"
        )
    result = dict(shared)
    result.update(
        {
            "backend": "vllm",
            "qualified_vllm_version": QUALIFIED_VLLM_VERSION,
            "block_size": block_size,
            "num_speculative_tokens": num_speculative_tokens,
        }
    )
    return result


def build_vllm_dflash_command(
    *,
    target_model: str,
    draft_model: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    served_model_name: str | None = None,
    gpu_memory_utilization: float | None = None,
    trust_remote_code: bool = False,
    python_executable: str | None = None,
) -> list[str]:
    """Build the pinned vLLM command for one validated DFlash2 export."""

    if not target_model.strip():
        raise ValueError("target_model must be nonempty")
    if not host.strip():
        raise ValueError("host must be nonempty")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in [1, 65535]")
    if (
        isinstance(tensor_parallel_size, bool)
        or not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size <= 0
    ):
        raise ValueError("tensor_parallel_size must be a positive integer")
    if max_model_len is not None and (
        isinstance(max_model_len, bool)
        or not isinstance(max_model_len, int)
        or max_model_len <= 0
    ):
        raise ValueError("max_model_len must be a positive integer")
    if gpu_memory_utilization is not None and not (
        0.0 < float(gpu_memory_utilization) <= 1.0
    ):
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if served_model_name is not None and not served_model_name.strip():
        raise ValueError("served_model_name must be nonempty")

    validated = validate_vllm_dflash2_export(draft_model)
    speculative_config = json.dumps(
        {
            "method": "dflash",
            "model": str(Path(validated["path"]).resolve()),
            "num_speculative_tokens": validated["num_speculative_tokens"],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    command = [
        python_executable or sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        target_model,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--speculative-config",
        speculative_config,
    ]
    if max_model_len is not None:
        command.extend(("--max-model-len", str(max_model_len)))
    if served_model_name is not None:
        command.extend(("--served-model-name", served_model_name))
    if gpu_memory_utilization is not None:
        command.extend(("--gpu-memory-utilization", str(gpu_memory_utilization)))
    if trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(("--per-request-spec-decode-metrics", "summary"))
    return command


def _prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return index
    return min(len(left), len(right))


def _metric_int(metrics: Mapping[str, Any], name: str) -> int | None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def evaluate_vllm_dflash_response(
    *,
    models: dict[str, Any],
    response: dict[str, Any],
    model: str,
    expected_text: str,
    encode: Callable[[str], Sequence[int]],
    block_size: int,
) -> dict[str, Any]:
    """Evaluate one deterministic native-vLLM DFlash2 response."""

    num_speculative_tokens = _draft_tokens(block_size)
    model_entries = models.get("data")
    served_ids = (
        {
            item.get("id")
            for item in model_entries
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        if isinstance(model_entries, list)
        else set()
    )
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    generated = choice.get("text") if isinstance(choice, Mapping) else None
    prefix = _prefix_length(encode(str(generated or "")), encode(expected_text))

    response_metrics = response.get("metrics")
    speculative = (
        response_metrics.get("speculative_decoding")
        if isinstance(response_metrics, Mapping)
        else None
    )
    metrics = speculative if isinstance(speculative, Mapping) else {}
    reported_tokens = _metric_int(metrics, "num_spec_tokens")
    num_spec_steps = _metric_int(metrics, "num_spec_steps")
    num_accepted = _metric_int(metrics, "num_accepted_draft_tokens")
    num_draft = _metric_int(metrics, "num_draft_tokens")
    histogram = metrics.get("acceptance_histogram")
    full_blocks = (
        histogram[num_speculative_tokens]
        if isinstance(histogram, list)
        and len(histogram) > num_speculative_tokens
        and isinstance(histogram[num_speculative_tokens], int)
        and not isinstance(histogram[num_speculative_tokens], bool)
        else 0
    )

    errors: list[str] = []
    if model not in served_ids:
        errors.append(f"model {model!r} is not served by vLLM")
    response_model = response.get("model")
    if response_model != model:
        errors.append(f"response model {response_model!r} != requested model {model!r}")
    if not isinstance(speculative, Mapping):
        errors.append("missing response.metrics.speculative_decoding")
    if reported_tokens != num_speculative_tokens:
        errors.append(
            f"num_spec_tokens {reported_tokens!r} != {num_speculative_tokens}"
        )
    if num_spec_steps is None or num_spec_steps <= 0:
        errors.append("num_spec_steps must be positive")
    if num_accepted is None or num_accepted < num_speculative_tokens:
        errors.append("num_accepted_draft_tokens does not cover one draft block")
    if num_draft is None or num_draft < num_speculative_tokens:
        errors.append("num_draft_tokens does not cover one draft block")
    if full_blocks < 1:
        errors.append("no complete draft block was accepted")
    if prefix < num_speculative_tokens:
        errors.append(
            f"target prefix match {prefix} < draft length {num_speculative_tokens}"
        )
    return {
        "passed": not errors,
        "served_model": response_model,
        "num_speculative_tokens": reported_tokens,
        "num_spec_steps": num_spec_steps,
        "num_accepted_draft_tokens": num_accepted,
        "num_draft_tokens": num_draft,
        "full_blocks_accepted": full_blocks,
        "target_prefix_match_tokens": prefix,
        "errors": errors,
    }


def _json_request(
    url: str, *, payload: dict[str, Any] | None, timeout: float
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def verify_vllm_dflash(
    *,
    server_url: str,
    model: str,
    prompt: str,
    expected_text: str,
    encode: Callable[[str], Sequence[int]],
    block_size: int,
    max_tokens: int | None = None,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Gate a running vLLM server using native DFlash2 metrics."""

    num_speculative_tokens = _draft_tokens(block_size)
    generation_tokens = (
        num_speculative_tokens if max_tokens is None else int(max_tokens)
    )
    if generation_tokens < num_speculative_tokens:
        raise ValueError("max_tokens must cover all vLLM DFlash2 speculative tokens")
    root = server_url.rstrip("/")
    models = _json_request(f"{root}/v1/models", payload=None, timeout=30.0)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": generation_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
    }
    response = _json_request(
        f"{root}/v1/completions", payload=payload, timeout=float(timeout)
    )
    return evaluate_vllm_dflash_response(
        models=models,
        response=response,
        model=model,
        expected_text=expected_text,
        encode=encode,
        block_size=block_size,
    )


__all__ = (
    "QUALIFIED_VLLM_VERSION",
    "build_vllm_dflash_command",
    "evaluate_vllm_dflash_response",
    "require_vllm_version",
    "validate_vllm_dflash2_export",
    "verify_vllm_dflash",
)
