from __future__ import annotations

import json

from dllm_parallel.cli.main import run


def test_dflash_export_cli_delegates_all_identity_fields(monkeypatch, tmp_path, capsys):
    captured = {}

    def export(checkpoint, **kwargs):
        captured.update({"checkpoint": checkpoint, **kwargs})
        return tmp_path / "export"

    monkeypatch.setattr(
        "dllm_parallel.core.models.backbones.dflash.executor.export_speculators_training_checkpoint",
        export,
    )
    result = run(
        [
            "dflash",
            "export",
            "--checkpoint",
            "checkpoints",
            "--base-model",
            "draft",
            "--output",
            str(tmp_path / "export"),
            "--model-id",
            "org/draft",
            "--model-revision",
            "abc123",
            "--block-size",
            "16",
        ]
    )
    assert result == 0
    assert captured["expected_model_id"] == "org/draft"
    assert captured["expected_model_revision"] == "abc123"
    assert captured["expected_block_size"] == 16
    assert "export" in capsys.readouterr().out


def test_dflash_export_downloads_exact_base_when_path_is_omitted(monkeypatch, tmp_path):
    captured = {}
    downloaded = tmp_path / "draft"
    downloaded.mkdir()
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: captured.update(kwargs) or str(downloaded),
    )
    monkeypatch.setattr(
        "dllm_parallel.core.models.backbones.dflash.executor.export_speculators_training_checkpoint",
        lambda checkpoint, **kwargs: captured.update(kwargs) or tmp_path / "export",
    )

    assert (
        run(
            [
                "dflash",
                "export",
                "--checkpoint",
                "checkpoints",
                "--output",
                str(tmp_path / "export"),
                "--model-id",
                "org/draft",
                "--model-revision",
                "abc123",
                "--block-size",
                "16",
            ]
        )
        == 0
    )
    assert captured["repo_id"] == "org/draft"
    assert captured["revision"] == "abc123"
    assert captured["base_model_dir"] == downloaded


def test_dflash_prepare_features_delegates_generic_capture_contract(
    monkeypatch, tmp_path, capsys
):
    captured = {}
    source = tmp_path / "prepared"
    output = tmp_path / "features"
    source.mkdir()
    monkeypatch.setattr(
        "dllm_parallel.core.models.backbones.dflash.feature_capture.capture_dflash_features_with_specforge",
        lambda **kwargs: (
            captured.update(kwargs)
            or {"format": "dllm_parallel.dflash_features.compact_sharded"}
        ),
    )

    assert (
        run(
            [
                "dflash",
                "prepare-features",
                "--source",
                str(source),
                "--output",
                str(output),
                "--draft-model",
                "org/draft",
                "--draft-revision",
                "draft-commit",
                "--verifier-model",
                "org/verifier",
                "--verifier-revision",
                "verifier-commit",
                "--sequence-length",
                "1048576",
                "--tensor-parallel-size",
                "4",
            ]
        )
        == 0
    )
    assert captured["source"] == source
    assert captured["output"] == output
    assert captured["sequence_length"] == 1_048_576
    assert captured["tensor_parallel_size"] == 4
    assert captured["draft_model"] == "org/draft"
    assert captured["verifier_model"] == "org/verifier"
    assert '"format"' in capsys.readouterr().out


def test_dflash_verify_sglang_cli_runs_real_contract(monkeypatch, tmp_path, capsys):
    captured = {}

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text.encode())

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        "dllm_parallel.serving.sglang_dflash.verify_sglang_dflash",
        lambda **kwargs: captured.update(kwargs) or {"passed": True, "errors": []},
    )
    expected = tmp_path / "expected.txt"
    expected.write_text("answer", encoding="utf-8")
    assert (
        run(
            [
                "dflash",
                "verify-sglang",
                "--server-url",
                "http://localhost:30000",
                "--model",
                "target",
                "--tokenizer",
                "target",
                "--prompt",
                "question",
                "--expected-text",
                str(expected),
                "--block-size",
                "4",
                "--max-tokens",
                "8",
            ]
        )
        == 0
    )
    assert captured["block_size"] == 4
    assert '"passed": true' in capsys.readouterr().out


def test_dflash_serve_sglang_dry_run_prints_validated_command(
    monkeypatch, tmp_path, capsys
):
    captured = {}
    draft = tmp_path / "draft"
    draft.mkdir()

    def command(**kwargs):
        captured.update(kwargs)
        return ["/python", "-m", "sglang.launch_server", "--model-path", "target"]

    monkeypatch.setattr(
        "dllm_parallel.serving.sglang_dflash.build_sglang_dflash_command", command
    )

    assert (
        run(
            [
                "dflash",
                "serve-sglang",
                "--target",
                "org/target",
                "--draft",
                str(draft),
                "--tensor-parallel-size",
                "4",
                "--dry-run",
            ]
        )
        == 0
    )
    assert captured["target_model"] == "org/target"
    assert captured["draft_model"] == draft
    assert captured["tensor_parallel_size"] == 4
    assert (
        capsys.readouterr()
        .out.strip()
        .startswith("/python -m sglang.launch_server --model-path target")
    )


def test_dflash_verify_remote_code_requires_explicit_opt_in(monkeypatch, tmp_path):
    trust = []

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: (
            trust.append(kwargs["trust_remote_code"]) or Tokenizer()
        ),
    )
    monkeypatch.setattr(
        "dllm_parallel.serving.sglang_dflash.verify_sglang_dflash",
        lambda **kwargs: {"passed": True, "errors": []},
    )
    expected = tmp_path / "expected.txt"
    expected.write_text("answer")
    common = [
        "dflash",
        "verify-sglang",
        "--server-url",
        "http://localhost",
        "--model",
        "target",
        "--tokenizer",
        "target",
        "--prompt",
        "q",
        "--expected-text",
        str(expected),
        "--block-size",
        "1",
        "--max-tokens",
        "1",
    ]
    assert run(common) == 0
    assert run([*common, "--trust-remote-code"]) == 0
    assert trust == [False, True]


def test_dflash_validate_vllm_cli_uses_native_contract(monkeypatch, tmp_path, capsys):
    captured = {}
    draft = tmp_path / "draft"
    draft.mkdir()
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.validate_vllm_dflash2_export",
        lambda path, **kwargs: (
            captured.update({"path": path, **kwargs})
            or {"backend": "vllm", "num_speculative_tokens": 15}
        ),
    )

    assert run(["dflash", "validate-vllm", str(draft), "--block-size", "16"]) == 0
    assert captured == {"path": draft, "expected_block_size": 16}
    assert json.loads(capsys.readouterr().out)["backend"] == "vllm"


def test_dflash_serve_vllm_dry_run_prints_validated_command(
    monkeypatch, tmp_path, capsys
):
    captured = {}
    draft = tmp_path / "draft"
    draft.mkdir()

    def command(**kwargs):
        captured.update(kwargs)
        return ["/python", "-m", "vllm.entrypoints.cli.main", "serve", "target"]

    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.build_vllm_dflash_command", command
    )
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.require_vllm_version",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run imported vLLM")),
    )

    assert (
        run(
            [
                "dflash",
                "serve-vllm",
                "--target",
                "org/target",
                "--draft",
                str(draft),
                "--tensor-parallel-size",
                "4",
                "--max-model-len",
                "32768",
                "--dry-run",
            ]
        )
        == 0
    )
    assert captured["target_model"] == "org/target"
    assert captured["draft_model"] == draft
    assert captured["tensor_parallel_size"] == 4
    assert captured["max_model_len"] == 32768
    assert (
        capsys.readouterr()
        .out.strip()
        .startswith("/python -m vllm.entrypoints.cli.main serve target")
    )


def test_dflash_serve_vllm_executes_same_python(monkeypatch, tmp_path):
    events = []
    draft = tmp_path / "draft"
    draft.mkdir()
    command = ["/qualified/python", "-m", "vllm.entrypoints.cli.main", "serve"]
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.build_vllm_dflash_command",
        lambda **kwargs: command,
    )
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.require_vllm_version",
        lambda: events.append("qualified") or "0.29.0",
    )
    monkeypatch.setattr(
        "dllm_parallel.cli.dflash.os.execv",
        lambda executable, argv: events.append((executable, argv)),
    )

    assert (
        run(["dflash", "serve-vllm", "--target", "target", "--draft", str(draft)]) == 0
    )
    assert events == ["qualified", ("/qualified/python", command)]


def test_dflash_verify_vllm_derives_block_size_from_export(
    monkeypatch, tmp_path, capsys
):
    captured = {}
    draft = tmp_path / "draft"
    draft.mkdir()
    expected = tmp_path / "expected.txt"
    expected.write_text("answer", encoding="utf-8")

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text.encode())

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.validate_vllm_dflash2_export",
        lambda path: {"block_size": 16},
    )
    monkeypatch.setattr(
        "dllm_parallel.serving.vllm_dflash.verify_vllm_dflash",
        lambda **kwargs: captured.update(kwargs) or {"passed": True, "errors": []},
    )

    assert (
        run(
            [
                "dflash",
                "verify-vllm",
                "--server-url",
                "http://localhost:8000",
                "--model",
                "target",
                "--tokenizer",
                "target",
                "--draft",
                str(draft),
                "--prompt",
                "question",
                "--expected-text",
                str(expected),
            ]
        )
        == 0
    )
    assert captured["block_size"] == 16
    assert captured["max_tokens"] is None
    assert '"passed": true' in capsys.readouterr().out
