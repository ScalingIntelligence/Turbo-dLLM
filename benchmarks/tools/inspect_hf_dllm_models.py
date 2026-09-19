#!/usr/bin/env python3
"""Inspect Hugging Face diffusion-LLM configs without downloading weights."""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import asdict
from typing import Any

try:
    from dllm_parallel.core.models import ModelFamilySpec, summarize_hf_config
except ImportError as exc:  # pragma: no cover - environment error path.
    raise SystemExit(
        "dllm_parallel is not importable. Install the package or set PYTHONPATH "
        "explicitly before running this utility."
    ) from exc


DEFAULT_MODELS = (
    "google/diffusiongemma-26B-A4B-it",
    "nvidia/Nemotron-Labs-Diffusion-3B",
    "nvidia/Nemotron-Labs-Diffusion-8B",
    "nvidia/Nemotron-Labs-Diffusion-14B",
)


def fetch_config(model_id: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def summarize(model_id: str, config: dict[str, Any]) -> ModelFamilySpec:
    return summarize_hf_config(model_id, config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Hugging Face diffusion-LLM config metadata."
    )
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--json", action="store_true", help="Emit JSON lines.")
    args = parser.parse_args()

    summaries = [summarize(model_id, fetch_config(model_id)) for model_id in args.models]
    if args.json:
        for summary in summaries:
            print(json.dumps(asdict(summary), sort_keys=True))
        return

    for summary in summaries:
        print(f"{summary.model_id}")
        for key, value in summary.__dict__.items():
            if key == "model_id" or value is None:
                continue
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
