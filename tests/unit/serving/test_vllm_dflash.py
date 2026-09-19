from __future__ import annotations

import json
from pathlib import Path

import pytest

from dllm_parallel.serving.vllm_dflash import (
    QUALIFIED_VLLM_VERSION,
    build_vllm_dflash_command,
    evaluate_vllm_dflash_response,
    require_vllm_version,
    validate_vllm_dflash2_export,
    verify_vllm_dflash,
)


def _draft_config(path: Path, *, block_size: int = 16) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "model_type": "qwen3",
                "dflash_config": {
                    "block_size": block_size,
                    "conv_group_size": 4,
                    "conv_kernel_size": 2,
                    "selector_rank": 8,
                    "selector_top_k": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_vllm_export_derives_draft_length(monkeypatch, tmp_path):
    draft = _draft_config(tmp_path / "draft")
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.validate_sglang_dflash2_export",
        lambda *args, **kwargs: {"path": str(draft), "tensor_count": 17},
    )

    result = validate_vllm_dflash2_export(draft, expected_block_size=16)

    assert result == {
        "path": str(draft),
        "tensor_count": 17,
        "backend": "vllm",
        "qualified_vllm_version": "0.29.0",
        "block_size": 16,
        "num_speculative_tokens": 15,
    }


def test_vllm_export_rejects_invalid_or_ambiguous_block_size(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.validate_sglang_dflash2_export",
        lambda *args, **kwargs: {"path": str(args[0]), "tensor_count": 1},
    )
    too_small = _draft_config(tmp_path / "small", block_size=1)
    with pytest.raises(ValueError, match="at least 2"):
        validate_vllm_dflash2_export(too_small)

    inconsistent = _draft_config(tmp_path / "inconsistent")
    config = json.loads((inconsistent / "config.json").read_text())
    config["block_size"] = 8
    (inconsistent / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="conflicting block_size"):
        validate_vllm_dflash2_export(inconsistent)


def test_vllm_export_rejects_backbones_without_a_native_dflash2_loader(
    monkeypatch, tmp_path
):
    draft = _draft_config(tmp_path / "draft")
    config = json.loads((draft / "config.json").read_text())
    config["model_type"] = "gemma3_text"
    (draft / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.validate_sglang_dflash2_export",
        lambda *args, **kwargs: {"path": str(draft), "tensor_count": 1},
    )

    with pytest.raises(ValueError, match="native vLLM DFlash2 loader.*qwen3"):
        validate_vllm_dflash2_export(draft)


def test_vllm_version_is_exact_and_actionable():
    assert require_vllm_version(QUALIFIED_VLLM_VERSION) == "0.29.0"
    with pytest.raises(RuntimeError, match="requires vLLM 0.29.0.*found 0.29.1"):
        require_vllm_version("0.29.1")


def test_vllm_launch_uses_native_dflash2(monkeypatch, tmp_path):
    draft = tmp_path / "draft"
    draft.mkdir()
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.validate_vllm_dflash2_export",
        lambda *args, **kwargs: {
            "path": str(draft.resolve()),
            "block_size": 16,
            "num_speculative_tokens": 15,
        },
    )

    command = build_vllm_dflash_command(
        target_model="org/target",
        draft_model=draft,
        host="0.0.0.0",
        port=8100,
        tensor_parallel_size=4,
        max_model_len=32768,
        served_model_name="target",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        python_executable="/python",
    )

    config = json.loads(command[command.index("--speculative-config") + 1])
    assert config == {
        "method": "dflash",
        "model": str(draft.resolve()),
        "num_speculative_tokens": 15,
    }
    assert command[:5] == [
        "/python",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "org/target",
    ]
    assert command[command.index("--tensor-parallel-size") + 1] == "4"
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--served-model-name") + 1] == "target"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.9"
    assert "--trust-remote-code" in command
    assert command[-2:] == ["--per-request-spec-decode-metrics", "summary"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_model": ""}, "target_model"),
        ({"port": 0}, "port"),
        ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
        ({"max_model_len": 0}, "max_model_len"),
        ({"gpu_memory_utilization": 1.1}, "gpu_memory_utilization"),
    ],
)
def test_vllm_launch_rejects_invalid_values(monkeypatch, tmp_path, kwargs, message):
    draft = tmp_path / "draft"
    draft.mkdir()
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.validate_vllm_dflash2_export",
        lambda *args, **kwargs: {
            "path": str(draft.resolve()),
            "block_size": 16,
            "num_speculative_tokens": 15,
        },
    )
    arguments = {"target_model": "target", "draft_model": draft, **kwargs}
    with pytest.raises(ValueError, match=message):
        build_vllm_dflash_command(**arguments)


def _valid_models():
    return {"data": [{"id": "target"}]}


def _valid_response():
    return {
        "model": "target",
        "choices": [{"text": "answer"}],
        "metrics": {
            "speculative_decoding": {
                "num_spec_tokens": 5,
                "acceptance_histogram": [0, 0, 0, 0, 0, 1],
                "num_spec_steps": 1,
                "num_accepted_draft_tokens": 5,
                "num_draft_tokens": 5,
            }
        },
    }


def test_vllm_gate_requires_a_complete_draft_block():
    result = evaluate_vllm_dflash_response(
        models=_valid_models(),
        response=_valid_response(),
        model="target",
        expected_text="answer plus suffix",
        encode=lambda text: list(text.encode()),
        block_size=6,
    )

    assert result == {
        "passed": True,
        "served_model": "target",
        "num_speculative_tokens": 5,
        "num_spec_steps": 1,
        "num_accepted_draft_tokens": 5,
        "num_draft_tokens": 5,
        "full_blocks_accepted": 1,
        "target_prefix_match_tokens": 6,
        "errors": [],
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda models, response: models.update(data=[]), "not served"),
        (lambda models, response: response.pop("metrics"), "missing response.metrics"),
        (
            lambda models, response: response["metrics"]["speculative_decoding"].update(
                num_spec_tokens=4
            ),
            "num_spec_tokens",
        ),
        (
            lambda models, response: response["metrics"]["speculative_decoding"].update(
                acceptance_histogram=[0, 1, 0, 0, 0, 0]
            ),
            "complete draft block",
        ),
        (
            lambda models, response: response["choices"][0].update(text="no"),
            "target prefix match",
        ),
    ],
)
def test_vllm_gate_fails_closed(mutate, message):
    models = _valid_models()
    response = _valid_response()
    mutate(models, response)

    result = evaluate_vllm_dflash_response(
        models=models,
        response=response,
        model="target",
        expected_text="answer plus suffix",
        encode=lambda text: list(text.encode()),
        block_size=6,
    )

    assert result["passed"] is False
    assert any(message in error for error in result["errors"])


def test_verify_vllm_uses_models_and_deterministic_completion(monkeypatch):
    from dllm_parallel.serving import vllm_dflash

    requests = []
    responses = iter([_valid_models(), _valid_response()])

    def request(url, *, payload, timeout):
        requests.append((url, payload, timeout))
        return next(responses)

    monkeypatch.setattr(vllm_dflash, "_json_request", request)

    result = verify_vllm_dflash(
        server_url="http://server/",
        model="target",
        prompt="question",
        expected_text="answer plus suffix",
        encode=lambda text: list(text.encode()),
        block_size=6,
    )

    assert result["passed"] is True
    assert requests == [
        ("http://server/v1/models", None, 30.0),
        (
            "http://server/v1/completions",
            {
                "model": "target",
                "prompt": "question",
                "max_tokens": 5,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
            },
            1800.0,
        ),
    ]


def test_verify_vllm_rejects_too_few_generation_tokens():
    with pytest.raises(ValueError, match="max_tokens"):
        verify_vllm_dflash(
            server_url="http://server",
            model="target",
            prompt="question",
            expected_text="answer",
            encode=lambda text: list(text.encode()),
            block_size=6,
            max_tokens=4,
        )
