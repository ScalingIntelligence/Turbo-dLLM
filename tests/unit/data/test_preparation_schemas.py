from __future__ import annotations

from pathlib import Path

import pytest

from dllm_parallel.data.registry import (
    FORMATTER_ENTRY_POINT_GROUP,
    SOURCE_ENTRY_POINT_GROUP,
    get_formatter,
    get_source,
    register_formatter,
    register_source,
)
from dllm_parallel.data import registry
from dllm_parallel.data.schemas import PreparationSpec


def _minimal_mapping(tmp_path: Path) -> dict[str, object]:
    return {
        "source": {"type": "jsonl", "path": str(tmp_path / "records.jsonl")},
        "records": {"type": "pretokenized"},
        "supervision": {"policy": "full"},
        "packing": {"maximum_length": 128, "alignment": 16},
        "output": {"path": str(tmp_path / "prepared")},
    }


def test_preparation_spec_loads_strict_yaml_and_normalizes_defaults(
    tmp_path: Path,
) -> None:
    config = tmp_path / "prepare.yaml"
    config.write_text(
        """
source:
  type: jsonl
  path: records.jsonl
records:
  type: pretokenized
supervision:
  policy: full
packing:
  maximum_length: 128
  alignment: 16
output:
  path: prepared
""".lstrip(),
        encoding="utf-8",
    )

    spec = PreparationSpec.from_path(config)

    assert spec.source.type == "jsonl"
    assert spec.source.path == str((tmp_path / "records.jsonl").resolve())
    assert spec.output.path == str((tmp_path / "prepared").resolve())
    assert spec.records.input_ids_field == "input_ids"
    assert spec.packing.overflow == "reject"
    assert spec.output.format == "auto"
    assert spec.to_mapping()["format"] == "dllm.data.prepare.v1"


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("source", "type", "database", "source.type"),
        ("records", "type", "chat-ish", "records.type"),
        ("supervision", "policy", "sometimes", "supervision.policy"),
        ("packing", "overflow", "guess", "packing.overflow"),
        ("packing", "alignment_policy", "automatic", "alignment_policy"),
        ("output", "format", "pickle", "output.format"),
    ],
)
def test_preparation_spec_rejects_invalid_choices(
    tmp_path: Path,
    section: str,
    field: str,
    value: str,
    match: str,
) -> None:
    mapping = _minimal_mapping(tmp_path)
    mapping[section] = {**mapping[section], field: value}  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=match):
        PreparationSpec.from_mapping(mapping)


def test_preparation_spec_rejects_unknown_keys(tmp_path: Path) -> None:
    mapping = _minimal_mapping(tmp_path)
    mapping["packing"] = {**mapping["packing"], "silently_pad": True}  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"packing.*silently_pad"):
        PreparationSpec.from_mapping(mapping)


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("source", "streaming", "false", "source.streaming"),
        ("source", "loader_kwargs", [], "source.loader_kwargs"),
        ("tokenizer", "add_eos", 1, "tokenizer.add_eos"),
        ("tokenizer", "kwargs", [], "tokenizer.kwargs"),
        ("packing", "maximum_length", True, "packing.maximum_length"),
        ("packing", "pad_token_id", False, "packing.pad_token_id"),
        ("output", "overwrite", "yes", "output.overwrite"),
    ],
)
def test_preparation_spec_rejects_weakly_typed_values(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    match: str,
) -> None:
    mapping = _minimal_mapping(tmp_path)
    mapping[section] = {**mapping.get(section, {}), field: value}  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=match):
        PreparationSpec.from_mapping(mapping)


def test_preparation_spec_enforces_source_and_cross_section_contracts(
    tmp_path: Path,
) -> None:
    mapping = _minimal_mapping(tmp_path)
    mapping["source"] = {"type": "huggingface"}
    with pytest.raises(ValueError, match=r"source\.id"):
        PreparationSpec.from_mapping(mapping)

    mapping = _minimal_mapping(tmp_path)
    mapping["records"] = {"type": "messages"}
    mapping["supervision"] = {"policy": "completion_only"}
    with pytest.raises(ValueError, match=r"completion_only.*prompt_completion"):
        PreparationSpec.from_mapping(mapping)

    mapping = _minimal_mapping(tmp_path)
    mapping["packing"] = {"maximum_length": 127, "alignment": 16}
    with pytest.raises(ValueError, match=r"maximum_length.*alignment"):
        PreparationSpec.from_mapping(mapping)


def test_preparation_spec_limits_lossless_split_and_rejects_unsafe_padding(
    tmp_path: Path,
) -> None:
    mapping = _minimal_mapping(tmp_path)
    mapping["packing"] = {"maximum_length": 128, "overflow": "split"}
    assert PreparationSpec.from_mapping(mapping).packing.overflow == "split"

    mapping["supervision"] = {"policy": "provided"}
    with pytest.raises(ValueError, match=r"split.*full supervision"):
        PreparationSpec.from_mapping(mapping)

    mapping = _minimal_mapping(tmp_path)
    mapping["packing"] = {
        "maximum_length": 128,
        "alignment_policy": "pad_right",
        "pad_token_id": 0,
    }
    with pytest.raises(ValueError, match=r"padding.*valid-token mask"):
        PreparationSpec.from_mapping(mapping)


@pytest.mark.parametrize(
    ("record_type", "policy"),
    [
        ("messages", "assistant_only"),
        ("prompt_completion", "completion_only"),
        ("pretokenized", "provided"),
    ],
)
def test_preparation_spec_rejects_supervised_packed_output_immediately(
    tmp_path: Path,
    record_type: str,
    policy: str,
) -> None:
    mapping = _minimal_mapping(tmp_path)
    mapping["records"] = {"type": record_type}
    mapping["supervision"] = {"policy": policy}
    mapping["output"] = {
        "path": str(tmp_path / "prepared"),
        "format": "packed",
    }

    with pytest.raises(ValueError, match=r"supervised.*output\.format=indexed"):
        PreparationSpec.from_mapping(mapping)


def test_source_and_formatter_registries_reject_accidental_replacement() -> None:
    source_name = "unit_test_source"
    formatter_name = "unit_test_formatter"

    def source(spec):
        return iter(({"value": 1},))

    def formatter(record, spec):
        return record

    register_source(source_name, source)
    register_formatter(formatter_name, formatter)

    assert get_source(source_name) is source
    assert get_formatter(formatter_name) is formatter
    with pytest.raises(ValueError, match="already registered"):
        register_source(source_name, source)
    with pytest.raises(ValueError, match="already registered"):
        register_formatter(formatter_name, formatter)

    def replacement(spec):
        return iter(())

    register_source(source_name, replacement, replace=True)
    assert get_source(source_name) is replacement


def test_registry_names_are_normalized_and_unknown_names_are_clear() -> None:
    def handler(spec):
        return iter(())

    register_source(" Unit-Test-Normalized ", handler, replace=True)
    assert get_source("unit-test-normalized") is handler
    with pytest.raises(KeyError, match="unknown data source"):
        get_source("not-registered")


def test_registry_lazily_loads_versioned_entry_points(monkeypatch) -> None:
    source = lambda spec: iter(({"value": 1},))
    formatter = lambda record, spec: record

    class EntryPoint:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def load(self):
            return self.value

    class EntryPoints(list):
        def select(self, *, group):
            values = {
                SOURCE_ENTRY_POINT_GROUP: [EntryPoint("unit-plugin-source", source)],
                FORMATTER_ENTRY_POINT_GROUP: [
                    EntryPoint("unit-plugin-formatter", formatter)
                ],
            }
            return values[group]

    monkeypatch.setattr(registry.metadata, "entry_points", lambda: EntryPoints())
    registry._DISCOVERED_GROUPS.discard(SOURCE_ENTRY_POINT_GROUP)
    registry._DISCOVERED_GROUPS.discard(FORMATTER_ENTRY_POINT_GROUP)

    assert get_source("unit-plugin-source") is source
    assert get_formatter("unit-plugin-formatter") is formatter


def test_registered_handler_does_not_trigger_entry_point_discovery(monkeypatch) -> None:
    handler = lambda spec: iter(())
    register_source("unit-already-registered", handler, replace=True)
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected discovery")),
    )

    assert get_source("unit-already-registered") is handler


@pytest.mark.parametrize(
    ("entry_name", "entry_value", "expected"),
    [
        ("unit-invalid-plugin", object(), "must be callable"),
        ("unit-duplicate-plugin", lambda spec: iter(()), "already registered"),
    ],
)
def test_registry_rejects_invalid_or_duplicate_entry_points(
    monkeypatch,
    entry_name,
    entry_value,
    expected,
) -> None:
    if "duplicate" in entry_name:
        register_source(entry_name, lambda spec: iter(()), replace=True)

    class EntryPoint:
        name = entry_name

        def load(self):
            return entry_value

    class EntryPoints(list):
        def select(self, *, group):
            assert group == SOURCE_ENTRY_POINT_GROUP
            return [EntryPoint()]

    monkeypatch.setattr(registry.metadata, "entry_points", lambda: EntryPoints())
    registry._DISCOVERED_GROUPS.discard(SOURCE_ENTRY_POINT_GROUP)

    with pytest.raises((TypeError, ValueError), match=expected):
        get_source("unit-trigger-discovery")
