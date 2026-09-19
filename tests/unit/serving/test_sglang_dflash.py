from __future__ import annotations

from dllm_parallel.serving.sglang_dflash import (
    build_sglang_dflash_command,
    evaluate_sglang_dflash_response,
)


def test_sglang_launch_uses_validated_dflash2_export(monkeypatch, tmp_path) -> None:
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(
        '{"dflash_config":{"block_size":16}}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "dllm_parallel.serving.sglang_dflash.validate_sglang_dflash2_export",
        lambda path: {"path": str(draft.resolve())},
    )

    command = build_sglang_dflash_command(
        target_model="org/target",
        draft_model=draft,
        host="0.0.0.0",
        port=31000,
        tensor_parallel_size=4,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        python_executable="/python",
    )

    assert command[:4] == ["/python", "-m", "sglang.launch_server", "--model-path"]
    assert command[command.index("--speculative-algorithm") + 1] == "DFLASH"
    assert command[command.index("--speculative-draft-model-path") + 1] == str(
        draft.resolve()
    )
    assert command[command.index("--speculative-dflash-block-size") + 1] == "16"
    assert command[command.index("--tp-size") + 1] == "4"
    assert command[command.index("--context-length") + 1] == "32768"
    assert "--trust-remote-code" in command


def test_serving_gate_accepts_deterministic_full_block():
    result = evaluate_sglang_dflash_response(
        server_info={"speculative_algorithm": "DFLASH"},
        response={
            "sglext": {"spec_tokens_details": {"spec_accept_length": 16}},
            "choices": [
                {
                    "text": "answer",
                }
            ],
        },
        expected_text="answer plus suffix",
        encode=lambda text: list(text.encode()),
        block_size=6,
    )
    assert result["passed"] is True
    assert result["spec_accept_length"] == 16
    assert result["target_prefix_match_tokens"] == 6
    assert result["speculative_algorithm"] == "DFLASH"
    assert "server_info" not in result
    assert "response" not in result


def test_serving_gate_rejects_wrong_algorithm_and_missing_acceptance():
    result = evaluate_sglang_dflash_response(
        server_info={"speculative_algorithm": "EAGLE3"},
        response={"choices": [{"text": "answer"}]},
        expected_text="answer",
        encode=lambda text: list(text.encode()),
        block_size=2,
    )
    assert result["passed"] is False
    assert any("DFLASH" in error for error in result["errors"])
    assert any("spec_accept_length" in error for error in result["errors"])


def test_verify_requests_current_sglang_spec_token_details(monkeypatch):
    from dllm_parallel.serving import sglang_dflash

    requests = []
    responses = iter(
        [
            {"speculative_algorithm": "DFLASH"},
            {
                "choices": [{"text": "ok"}],
                "sglext": {"spec_tokens_details": {"spec_accept_length": 2}},
            },
        ]
    )

    def request(url, *, payload, timeout):
        requests.append((url, payload, timeout))
        return next(responses)

    monkeypatch.setattr(sglang_dflash, "_json_request", request)
    result = sglang_dflash.verify_sglang_dflash(
        server_url="http://server",
        model="target",
        prompt="p",
        expected_text="ok",
        encode=lambda text: list(text.encode()),
        block_size=2,
        max_tokens=2,
    )
    assert result["passed"] is True
    assert requests[1][1]["return_spec_tokens_details"] is True
    assert "return_meta_info" not in requests[1][1]


def test_serving_gate_does_not_accept_choice_level_extension():
    result = evaluate_sglang_dflash_response(
        server_info={"speculative_algorithm": "DFLASH"},
        response={
            "choices": [
                {
                    "text": "answer",
                    "sglext": {"spec_tokens_details": {"spec_accept_length": 16}},
                }
            ]
        },
        expected_text="answer",
        encode=lambda text: list(text.encode()),
        block_size=2,
    )

    assert result["passed"] is False
    assert result["spec_accept_length"] is None
    assert any("response.sglext" in error for error in result["errors"])
