"""Validate deterministic DFlash speculative decoding through SGLang."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from dllm_parallel.core.models.backbones.dflash.executor import (
    validate_sglang_dflash2_export,
)


def build_sglang_dflash_command(
    *,
    target_model: str,
    draft_model: str | Path,
    host: str = "127.0.0.1",
    port: int = 30000,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    trust_remote_code: bool = False,
    python_executable: str | None = None,
) -> list[str]:
    """Build the pinned SGLang command for one validated DFlash2 export."""

    if not target_model.strip() or not host.strip():
        raise ValueError("target_model and host must be nonempty")
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
        0.0 < float(gpu_memory_utilization) < 1.0
    ):
        raise ValueError("gpu_memory_utilization must be in (0, 1)")

    validated = validate_sglang_dflash2_export(draft_model)
    root = Path(validated["path"]).resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    method = config.get("dflash_config") or {}
    block_size = config.get("block_size", method.get("block_size"))
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("DFlash2 export has an invalid block_size")
    command = [
        python_executable or sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        target_model,
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(tensor_parallel_size),
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        str(root),
        "--speculative-dflash-block-size",
        str(block_size),
    ]
    if max_model_len is not None:
        command.extend(("--context-length", str(max_model_len)))
    if gpu_memory_utilization is not None:
        command.extend(("--mem-fraction-static", str(gpu_memory_utilization)))
    if trust_remote_code:
        command.append("--trust-remote-code")
    return command


def _prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return index
    return min(len(left), len(right))


def evaluate_sglang_dflash_response(
    *,
    server_info: dict[str, Any],
    response: dict[str, Any],
    expected_text: str,
    encode: Callable[[str], Sequence[int]],
    block_size: int,
) -> dict[str, Any]:
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    generated = choice.get("text")
    if generated is None:
        message = choice.get("message") or {}
        generated = (message.get("reasoning_content") or "") + (
            message.get("content") or ""
        )
    extension = response.get("sglext")
    details = (
        extension.get("spec_tokens_details") if isinstance(extension, dict) else None
    )
    if isinstance(details, list):
        details = details[0] if details and isinstance(details[0], dict) else None
    acceptance = (
        details.get("spec_accept_length") if isinstance(details, dict) else None
    )
    prefix = _prefix_length(encode(str(generated or "")), encode(expected_text))
    errors: list[str] = []
    if server_info.get("speculative_algorithm") != "DFLASH":
        errors.append("server speculative_algorithm must be DFLASH")
    if acceptance is None:
        errors.append("missing response.sglext.spec_tokens_details.spec_accept_length")
    elif float(acceptance) < int(block_size):
        errors.append(f"spec_accept_length {acceptance} < block_size {block_size}")
    if prefix < int(block_size):
        errors.append(f"target prefix match {prefix} < block_size {block_size}")
    return {
        "passed": not errors,
        "speculative_algorithm": server_info.get("speculative_algorithm"),
        "spec_accept_length": acceptance,
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


def verify_sglang_dflash(
    *,
    server_url: str,
    model: str,
    prompt: str,
    expected_text: str,
    encode: Callable[[str], Sequence[int]],
    block_size: int,
    max_tokens: int,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    if block_size <= 0 or max_tokens < block_size:
        raise ValueError(
            "block_size must be positive and max_tokens must cover one block"
        )
    root = server_url.rstrip("/")
    info = _json_request(f"{root}/server_info", payload=None, timeout=30.0)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "return_spec_tokens_details": True,
    }
    response = _json_request(
        f"{root}/v1/completions", payload=payload, timeout=float(timeout)
    )
    return evaluate_sglang_dflash_response(
        server_info=info,
        response=response,
        expected_text=expected_text,
        encode=encode,
        block_size=block_size,
    )
