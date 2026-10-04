from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sysconfig
import zipfile
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "gpu_release_runner", ROOT / ".github/scripts/gpu_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("changed", ["vendor", "profile", "cute"])
def test_changed_inputs_clear_stale_attention_before_rebuild(
    tmp_path, monkeypatch, changed
):
    runner = _runner()
    source, dist, attention = (
        tmp_path / name for name in ("source", "dist", "attention")
    )
    for path in (source, dist, attention):
        path.mkdir()
    (attention / "stale.whl").write_bytes(b"stale")
    paths = (
        "third_party/flash-attention/hopper",
        "third_party/flash-attention/flash_attn/cute",
        "third_party/flash-attention/csrc/cutlass/include",
    )
    for path in paths:
        target = source / path / "input.txt"
        target.parent.mkdir(parents=True)
        target.write_text("old")
    script = source / "scripts/build/build_cuda_wheels.sh"
    script.parent.mkdir(parents=True)
    script.write_text("FLASH_ATTENTION_DISABLE_SPLIT=TRUE")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    git("init", "-q")
    git("config", "user.name", "Release test")
    git("config", "user.email", "release@example.invalid")
    git("add", ".")
    git("commit", "-qm", "old")
    old = git("rev-parse", "HEAD")
    changed_path = (
        source / paths[0 if changed == "vendor" else 1] / "input.txt"
        if changed != "profile"
        else script
    )
    changed_path.write_text("FLASH_ATTENTION_DISABLE_SPLIT=FALSE")
    git("add", ".")
    git("commit", "-qm", "new")
    new = git("rev-parse", "HEAD")
    git("tag", "v0.1.4")
    git("remote", "add", "origin", str(source))
    git("checkout", "-q", "--detach", old)
    monkeypatch.setattr(runner, "SOURCE", source)
    monkeypatch.setattr(runner, "DIST", dist)
    monkeypatch.setattr(runner, "ATTENTION", attention)
    if changed == "cute":
        (dist / "bdlm_flash_attn_3-fixture.whl").write_bytes(b"verified cache fixture")
        monkeypatch.setattr(
            runner, "validate_cached_attention", lambda wheels, revision: None
        )
    else:
        fa4_name = f"flash_attn_4-4.0.0b19+bdlm.{old[:12]}-py3-none-any.whl"
        (dist / fa4_name).write_bytes(b"verified FA4 cache fixture")
    monkeypatch.setenv("DLLM_ATTENTION_IMAGE", "im-cache")
    original_output = subprocess.check_output
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda command, **kw: (
            "fixture CUDA compiler"
            if command[0] == "nvcc"
            else original_output(command, **kw)
        ),
    )
    runner.prepare_source("v0.1.4", new, "owner/repo")
    if changed == "cute":
        assert (
            attention / "bdlm_flash_attn_3-fixture.whl"
        ).read_bytes() == b"verified cache fixture"
        assert not list(attention.glob("flash_attn_4-*.whl"))
    else:
        assert (attention / fa4_name).read_bytes() == b"verified FA4 cache fixture"
        assert not list(attention.glob("bdlm_flash_attn_3-*.whl"))
    assert not (attention / "stale.whl").exists()
    provenance = json.loads((attention / "attention-provenance.json").read_text())
    assert provenance["components"][0]["source_revision"] == old
    assert not dist.exists()
    assert git("rev-parse", "HEAD") == new


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "fa3_revision",
        "fa4_revision",
        "checksum",
        "tvm_abi",
        "cutlass_abi",
        "incomplete",
        "missing_manifest",
    ],
)
def test_cached_wheels_preserve_their_actual_revision(tmp_path, monkeypatch, fault):
    runner = _runner()
    monkeypatch.setattr(runner, "SOURCE", ROOT)
    revision = "a" * 40
    binary = b"native fixture"
    metadata = {
        "source": {"revision": "b" * 40 if fault == "fa3_revision" else revision},
        "build": {
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "python_extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "cuda_architectures": ["sm_90a"],
        },
        "binary": {
            "path": "flash_attn_3/_C.so",
            "sha256": "0" * 64
            if fault == "checksum"
            else hashlib.sha256(binary).hexdigest(),
        },
    }
    fa3 = tmp_path / "bdlm_flash_attn_3-0.1.0-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(fa3, "w") as archive:
        archive.writestr("flash_attn_3/build_metadata.json", json.dumps(metadata))
        archive.writestr("flash_attn_3/_C.so", binary)
        if fault != "missing_manifest":
            archive.writestr(
                "flash_attn_3/bdlm_splitd/_artifacts/fingerprint/manifest.json",
                json.dumps(
                    {
                        "tvm_ffi": "0.1.14.post1" if fault == "tvm_abi" else "0.1.12",
                        "cutlass": "4.6.0" if fault == "cutlass_abi" else "4.5.2",
                        "torch": str(torch.__version__),
                        "torch_cuda": str(torch.version.cuda),
                        "complete": fault != "incomplete",
                    }
                ),
            )
    suffix = "b" * 12 if fault == "fa4_revision" else revision[:12]
    fa4 = tmp_path / f"flash_attn_4-4.0.0b19+bdlm.{suffix}-py3-none-any.whl"
    if fault:
        with pytest.raises(RuntimeError, match="mismatch"):
            runner.validate_cached_attention([fa3, fa4], revision)
    else:
        runner.validate_cached_attention([fa3, fa4], revision)
