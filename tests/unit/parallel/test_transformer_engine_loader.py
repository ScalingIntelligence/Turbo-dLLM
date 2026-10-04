"""Packaged cuDNN must be visible to Transformer Engine's native loader."""

import ctypes
import importlib
from importlib import metadata
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from dllm_parallel.core.parallel import transformer_engine as te


@pytest.fixture
def packaged_cudnn(monkeypatch, tmp_path):
    files = [
        PurePosixPath("nvidia/cudnn/lib/libcudnn_graph.so.9"),
        PurePosixPath("nvidia/cudnn/lib/libcudnn_ops.so.9"),
    ]
    for file in files:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    distribution = SimpleNamespace(
        files=files, locate_file=lambda file: tmp_path / file
    )
    monkeypatch.setattr(te.torch.version, "cuda", "12.8")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(te, "_CUDNN_HANDLES", {}, raising=False)
    return tmp_path


def test_packaged_dependencies_are_loaded_before_transformer_engine(
    monkeypatch, packaged_cudnn
):
    loaded = set()

    def load(path, mode):
        name = Path(path).name
        if "graph" in name and "libcudnn_ops.so.9" not in loaded:
            raise OSError("libcudnn_ops.so.9: cannot open shared object file")
        loaded.add(name)
        return object()

    def import_module(name):
        assert loaded == {"libcudnn_graph.so.9", "libcudnn_ops.so.9"}
        return name

    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setattr(importlib, "import_module", import_module)
    assert te.load_transformer_engine() == "transformer_engine.pytorch"
    monkeypatch.setattr(
        ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("loaded twice")
    )
    assert (
        te.load_transformer_engine("transformer_engine.pytorch.ops")
        == "transformer_engine.pytorch.ops"
    )


def test_unloadable_library_reports_error_and_can_be_retried(
    monkeypatch, packaged_cudnn
):
    def broken(path, mode):
        raise OSError("missing dependent library")

    monkeypatch.setattr(ctypes, "CDLL", broken)
    with pytest.raises(RuntimeError, match="libcudnn_graph.so.9") as error:
        te.load_transformer_engine()
    assert isinstance(error.value.__cause__, OSError)
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode: object())
    monkeypatch.setattr(importlib, "import_module", lambda name: name)
    assert te.load_transformer_engine() == "transformer_engine.pytorch"


def test_cpu_runtime_preserves_regular_optional_import(monkeypatch):
    monkeypatch.setattr(te.torch.version, "cuda", None)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("CPU load"))
    monkeypatch.setattr(importlib, "import_module", lambda name: name)
    assert te.load_transformer_engine() == "transformer_engine.pytorch"


def test_missing_cudnn_distribution_preserves_normal_import(monkeypatch):
    monkeypatch.setattr(te.torch.version, "cuda", "12.8")
    monkeypatch.setattr(sys, "platform", "linux")

    def missing(name):
        assert name == "nvidia-cudnn-cu12"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    monkeypatch.setattr(importlib, "import_module", lambda name: name)
    assert te.load_transformer_engine() == "transformer_engine.pytorch"
