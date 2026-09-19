"""Commands for exporting and validating DFlash2 serving artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Sequence
from pathlib import Path


def run_dflash(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="dllm dflash")
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser(
        "prepare-features",
        help="Capture verifier features for a prepared DFlash2 dataset.",
    )
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--draft-model", required=True)
    prepare.add_argument("--draft-revision", required=True)
    prepare.add_argument("--verifier-model", required=True)
    prepare.add_argument("--verifier-revision", required=True)
    prepare.add_argument("--sequence-length", required=True, type=int)
    prepare.add_argument("--tensor-parallel-size", type=int, default=1)
    prepare.add_argument("--attention-backend", default="flashinfer")
    prepare.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    prepare.add_argument("--distributed-timeout-minutes", type=int, default=120)
    prepare.add_argument("--trust-remote-code", action="store_true")
    export = subparsers.add_parser(
        "export", help="Export a DFlash2 checkpoint for supported serving runtimes."
    )
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument(
        "--base-model",
        type=Path,
        help="Local pinned draft checkpoint; omitted downloads model-id/revision.",
    )
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--checkpoint-tag", default="latest")
    export.add_argument("--model-id", required=True)
    export.add_argument("--model-revision", required=True)
    export.add_argument("--block-size", required=True, type=int)
    validate = subparsers.add_parser(
        "validate-export", help="Validate an exported DFlash2 config."
    )
    validate.add_argument("path", type=Path)
    validate.add_argument("--block-size", required=True, type=int)
    validate_vllm = subparsers.add_parser(
        "validate-vllm", help="Validate a DFlash2 export for native vLLM serving."
    )
    validate_vllm.add_argument("path", type=Path)
    validate_vllm.add_argument("--block-size", type=int)
    serve_sglang = subparsers.add_parser(
        "serve-sglang", help="Launch SGLang DFlash2 speculative serving."
    )
    serve_sglang.add_argument("--target", required=True)
    serve_sglang.add_argument("--draft", required=True, type=Path)
    serve_sglang.add_argument("--host", default="127.0.0.1")
    serve_sglang.add_argument("--port", default=30000, type=int)
    serve_sglang.add_argument("--tensor-parallel-size", default=1, type=int)
    serve_sglang.add_argument("--max-model-len", type=int)
    serve_sglang.add_argument("--gpu-memory-utilization", type=float)
    serve_sglang.add_argument("--trust-remote-code", action="store_true")
    serve_sglang.add_argument("--dry-run", action="store_true")
    serve_vllm = subparsers.add_parser(
        "serve-vllm", help="Launch native vLLM DFlash2 speculative serving."
    )
    serve_vllm.add_argument("--target", required=True)
    serve_vllm.add_argument("--draft", required=True, type=Path)
    serve_vllm.add_argument("--host", default="127.0.0.1")
    serve_vllm.add_argument("--port", default=8000, type=int)
    serve_vllm.add_argument("--tensor-parallel-size", default=1, type=int)
    serve_vllm.add_argument("--max-model-len", type=int)
    serve_vllm.add_argument("--served-model-name")
    serve_vllm.add_argument("--gpu-memory-utilization", type=float)
    serve_vllm.add_argument("--trust-remote-code", action="store_true")
    serve_vllm.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the export and print the exact command without launching.",
    )
    verify = subparsers.add_parser(
        "verify-sglang", help="Gate a running SGLang DFLASH server."
    )
    verify.add_argument("--server-url", required=True)
    verify.add_argument("--model", required=True)
    verify.add_argument("--tokenizer", required=True)
    verify.add_argument("--prompt", required=True)
    verify.add_argument("--expected-text", required=True, type=Path)
    verify.add_argument("--block-size", required=True, type=int)
    verify.add_argument("--max-tokens", required=True, type=int)
    verify.add_argument("--timeout", type=float, default=1800.0)
    verify.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow the tokenizer repository to execute custom Python code.",
    )
    verify_vllm = subparsers.add_parser(
        "verify-vllm", help="Gate a running native vLLM DFlash2 server."
    )
    verify_vllm.add_argument("--server-url", required=True)
    verify_vllm.add_argument("--model", required=True)
    verify_vllm.add_argument("--tokenizer", required=True)
    verify_vllm.add_argument("--draft", required=True, type=Path)
    verify_vllm.add_argument("--prompt", required=True)
    verify_vllm.add_argument("--expected-text", required=True, type=Path)
    verify_vllm.add_argument("--max-tokens", type=int)
    verify_vllm.add_argument("--timeout", type=float, default=1800.0)
    verify_vllm.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow the tokenizer repository to execute custom Python code.",
    )
    args = parser.parse_args(list(argv))

    from dllm_parallel.core.models.backbones.dflash.executor import (
        export_speculators_training_checkpoint,
        validate_sglang_dflash2_export,
    )

    if args.action == "prepare-features":
        from dllm_parallel.core.models.backbones.dflash.feature_capture import (
            capture_dflash_features_with_specforge,
        )

        result = capture_dflash_features_with_specforge(
            source=args.source,
            output=args.output,
            draft_model=args.draft_model,
            draft_revision=args.draft_revision,
            verifier_model=args.verifier_model,
            verifier_revision=args.verifier_revision,
            sequence_length=args.sequence_length,
            tensor_parallel_size=args.tensor_parallel_size,
            attention_backend=args.attention_backend,
            gpu_memory_utilization=args.gpu_memory_utilization,
            distributed_timeout_minutes=args.distributed_timeout_minutes,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if result is not None:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.action == "validate-export":
        result = validate_sglang_dflash2_export(
            args.path, expected_block_size=args.block_size
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.action == "validate-vllm":
        from dllm_parallel.serving.vllm_dflash import validate_vllm_dflash2_export

        result = validate_vllm_dflash2_export(
            args.path, expected_block_size=args.block_size
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.action == "serve-sglang":
        from dllm_parallel.serving.sglang_dflash import build_sglang_dflash_command

        command = build_sglang_dflash_command(
            target_model=args.target,
            draft_model=args.draft,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if args.dry_run:
            print(shlex.join(command))
            return 0
        os.execv(command[0], command)
        return 0
    if args.action == "serve-vllm":
        from dllm_parallel.serving.vllm_dflash import (
            build_vllm_dflash_command,
            require_vllm_version,
        )

        command = build_vllm_dflash_command(
            target_model=args.target,
            draft_model=args.draft,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            served_model_name=args.served_model_name,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if args.dry_run:
            print(shlex.join(command))
            return 0
        require_vllm_version()
        os.execv(command[0], command)
        return 0
    if args.action == "verify-sglang":
        from transformers import AutoTokenizer
        from dllm_parallel.serving.sglang_dflash import verify_sglang_dflash

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=bool(args.trust_remote_code)
        )
        result = verify_sglang_dflash(
            server_url=args.server_url,
            model=args.model,
            prompt=args.prompt,
            expected_text=args.expected_text.read_text(encoding="utf-8"),
            encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
            block_size=args.block_size,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise RuntimeError(
                "SGLang DFLASH serving gate failed: " + "; ".join(result["errors"])
            )
        return 0
    if args.action == "verify-vllm":
        from transformers import AutoTokenizer
        from dllm_parallel.serving.vllm_dflash import (
            validate_vllm_dflash2_export,
            verify_vllm_dflash,
        )

        validated = validate_vllm_dflash2_export(args.draft)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=bool(args.trust_remote_code)
        )
        result = verify_vllm_dflash(
            server_url=args.server_url,
            model=args.model,
            prompt=args.prompt,
            expected_text=args.expected_text.read_text(encoding="utf-8"),
            encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
            block_size=validated["block_size"],
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise RuntimeError(
                "vLLM DFlash2 serving gate failed: " + "; ".join(result["errors"])
            )
        return 0
    base_model = args.base_model
    if base_model is None:
        from huggingface_hub import snapshot_download

        base_model = Path(
            snapshot_download(repo_id=args.model_id, revision=args.model_revision)
        )
    output = export_speculators_training_checkpoint(
        args.checkpoint,
        base_model_dir=base_model,
        output_dir=args.output,
        checkpoint_tag=args.checkpoint_tag,
        expected_model_id=args.model_id,
        expected_model_revision=args.model_revision,
        expected_block_size=args.block_size,
    )
    print(output)
    return 0


__all__ = ("run_dflash",)
