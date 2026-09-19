from __future__ import annotations

import json
import importlib
from pathlib import Path

import yaml

from dllm_parallel.cli.main import run
from dllm_parallel.recipes import recipe_text


def test_doctor_reports_portable_environment(capsys) -> None:
    assert run(["doctor", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["package"] == "turbo-dllm"
    assert report["python"]
    assert isinstance(report["cuda_available"], bool)
    assert set(report["native_artifacts"]) == {
        "bdlm-flash-attn-3",
        "dllm-native-kernels",
        "flash-attn-4",
    }
    assert set(report["gpu_runtime"]) == {
        "apache-tvm-ffi",
        "cuda-bindings",
        "cuda-python",
        "deepspeed",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "quack-kernels",
        "torch-c-dlpack-ext",
        "transformer-engine",
    }
    assert set(report["model_runtimes"]) == {
        "deep-ep",
        "flash-linear-attention",
        "tilelang",
    }


def test_recipe_commands_list_show_and_copy(tmp_path: Path, capsys) -> None:
    assert run(["recipe", "list"]) == 0
    assert "smoke/cpu-config" in capsys.readouterr().out

    assert run(["recipe", "show", "smoke/cpu-config"]) == 0
    assert "family: causal_lm" in capsys.readouterr().out

    destination = tmp_path / "my-recipe.yaml"
    assert run(["recipe", "copy", "smoke/cpu-config", str(destination)]) == 0
    assert destination.is_file()


def test_config_validate_prints_resolved_run_spec(capsys) -> None:
    assert run(["config", "validate", "--recipe", "smoke/cpu-config"]) == 0
    assert '"event": "validated_run_spec"' in capsys.readouterr().out


def test_train_delegates_to_existing_entrypoint(monkeypatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        "dllm_parallel.training.entrypoint.main",
        lambda argv: captured.append(list(argv)),
    )

    assert run(["train", "--recipe", "smoke/cpu-config", "--steps", "2"]) == 0
    assert captured and captured[0][0] == "--config"
    assert captured[0][-2:] == ["--steps", "2"]


def test_user_errors_are_concise(capsys) -> None:
    assert run(["recipe", "show", "smoke/does-not-exist"]) == 2

    error = capsys.readouterr().err
    assert "unknown packaged recipe" in error
    assert "Traceback" not in error


def test_bundle_install_delegates_to_verified_installer(monkeypatch, capsys) -> None:
    captured: list[str] = []
    doctor_args: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "dllm_parallel.core.kernels.bundle_manifest.install_bundle",
        lambda manifest: captured.append(str(manifest)),
    )
    main_module = importlib.import_module("dllm_parallel.cli.main")
    monkeypatch.setattr(
        main_module, "_doctor", lambda argv: doctor_args.append(tuple(argv)) or 0
    )

    assert run(["bundle", "install", "--manifest", "bundle.json"]) == 0
    assert captured == ["bundle.json"]
    assert doctor_args == [("--training",)]
    assert "installed" in capsys.readouterr().out


def test_bundle_auto_install_delegates_to_release_resolver(monkeypatch, capsys) -> None:
    captured: list[str] = []
    doctor_args: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        "dllm_parallel.core.kernels.bundle_manifest.install_bundle_for_release",
        lambda release: captured.append(release),
    )
    main_module = importlib.import_module("dllm_parallel.cli.main")
    monkeypatch.setattr(
        main_module, "_doctor", lambda argv: doctor_args.append(tuple(argv)) or 0
    )

    assert run(["bundle", "install", "--release", "v0.1.0", "--auto"]) == 0
    assert captured == ["v0.1.0"]
    assert doctor_args == [("--training",)]
    assert "installed" in capsys.readouterr().out


def test_root_launch_command_propagates_child_status(monkeypatch) -> None:
    monkeypatch.setattr("dllm_parallel.cli.launch.run_launch", lambda argv: 23)

    assert run(["launch", "--config", "run.yaml"]) == 23


def test_data_commands_prepare_inspect_validate_and_report_stats(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        '{"input_ids":[1,2,3]}\n{"input_ids":[4,5]}\n',
        encoding="utf-8",
    )
    config = tmp_path / "prepare.yaml"
    config.write_text(
        f"""
source:
  type: jsonl
  path: {source}
records:
  type: pretokenized
tokenizer:
  add_eos: false
supervision:
  policy: full
packing:
  maximum_length: 4
  separator_token_id: 9
output:
  path: {tmp_path / "configured-output"}
""".lstrip(),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"

    assert (
        run(
            [
                "data",
                "prepare",
                "--config",
                str(config),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["format"] == "dllm_parallel.packed_tokens"
    assert prepared["artifact_path"] == str(output.resolve())

    assert run(["data", "inspect", str(output), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["status"] == "present"
    assert inspected["token_count"] == 6

    assert run(["data", "validate", str(output), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True

    assert run(["data", "stats", str(output), "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["records"] == 2
    assert stats["stored_tokens"] == 6


def test_data_validate_returns_concise_checksum_error(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "tokens.i32"
    source.write_bytes(b"\x01\x00\x00\x00\x02\x00\x00\x00")
    config = tmp_path / "prepare.json"
    config.write_text(
        json.dumps(
            {
                "source": {"type": "pretokenized", "path": str(source)},
                "records": {"type": "pretokenized"},
                "tokenizer": {"add_eos": False},
                "packing": {"maximum_length": 8},
                "output": {"path": str(tmp_path / "prepared")},
            }
        ),
        encoding="utf-8",
    )
    assert run(["data", "prepare", "--config", str(config), "--json"]) == 0
    capsys.readouterr()
    payload = tmp_path / "prepared" / "tokens.i32"
    payload.write_bytes(b"\x03\x00\x00\x00\x02\x00\x00\x00")

    assert run(["data", "validate", str(tmp_path / "prepared")]) == 2
    error = capsys.readouterr().err
    assert "checksum" in error
    assert "Traceback" not in error


def test_data_validate_with_config_honors_no_checksums(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    source = tmp_path / "tokens.i32"
    source.write_bytes(b"\x01\x00\x00\x00\x02\x00\x00\x00")
    prepare_config = tmp_path / "prepare.json"
    prepare_config.write_text(
        json.dumps(
            {
                "source": {"type": "pretokenized", "path": str(source)},
                "records": {"type": "pretokenized"},
                "packing": {"maximum_length": 8},
                "output": {"path": str(tmp_path / "prepared")},
            }
        ),
        encoding="utf-8",
    )
    assert run(["data", "prepare", "--config", str(prepare_config)]) == 0
    capsys.readouterr()

    run_config = yaml.safe_load(recipe_text("smoke/cpu-config"))
    run_config["data"] = {
        "input_mode": "dataset",
        "dataset_path": str(tmp_path / "prepared"),
    }
    run_config_path = tmp_path / "run.yaml"
    run_config_path.write_text(yaml.safe_dump(run_config), encoding="utf-8")

    checksum_calls = 0

    def unexpected_checksum(path):
        nonlocal checksum_calls
        checksum_calls += 1
        return "not-used"

    monkeypatch.setattr("dllm_parallel.data.indexed.sha256_file", unexpected_checksum)

    assert (
        run(
            [
                "data",
                "validate",
                str(tmp_path / "prepared"),
                "--config",
                str(run_config_path),
                "--no-checksums",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["run"]["compatible"] is True
    assert checksum_calls == 0
