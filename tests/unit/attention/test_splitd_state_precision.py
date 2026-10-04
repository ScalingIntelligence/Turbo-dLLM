"""Exercise the packaged runtime's precision contract without CUDA or CuTe."""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def _runtime(monkeypatch):
    package = "_splitd_precision_test"
    module = types.ModuleType(package)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, package, module)
    artifacts = types.ModuleType(package + "._artifacts")
    monkeypatch.setitem(sys.modules, artifacts.__name__, artifacts)
    artifacts.load_packaged_variant = lambda *_: {}
    ffi = types.ModuleType("tvm_ffi")
    ffi.use_torch_stream = lambda *_: nullcontext()
    monkeypatch.setitem(sys.modules, "tvm_ffi", ffi)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda **_: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: object())
    root = ROOT / "third_party/flash-attention/hopper/flash_attn_3/bdlm_splitd"
    for name in ("_capabilities", "_aot_runtime"):
        spec = importlib.util.spec_from_file_location(
            package + "." + name, root / (name + ".py")
        )
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, loaded)
        spec.loader.exec_module(loaded)
    return loaded


def test_packaged_forward_preserves_unrounded_output_state(monkeypatch):
    runtime = _runtime(monkeypatch)
    query = torch.zeros(1, 3, 2, 512, dtype=torch.bfloat16)
    key = torch.zeros(1, 4, 1, 512, dtype=query.dtype)
    precise = 0.123456

    def forward(q, k, v, output, lse, *_):
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        output.fill_(precise)
        lse.fill_(2.0)

    monkeypatch.setattr(
        runtime, "load_packaged_variant", lambda *_: {"forward": forward}
    )
    output, _ = runtime.forward(query, key, key, 1.0, None)
    assert output.dtype == torch.float32
    assert output[0, 0, 0, 0].item() == pytest.approx(precise, abs=1e-8)
    assert output[0, 0, 0, 0].item() != output.to(query.dtype)[0, 0, 0, 0].item()


def test_prepared_backward_keeps_fp32_output_and_input_dtype_gradient(monkeypatch):
    runtime = _runtime(monkeypatch)
    query = torch.zeros(1, 3, 2, 512, dtype=torch.bfloat16)
    key = torch.zeros(1, 4, 1, 512, dtype=query.dtype)
    output = torch.full(query.shape, 0.123456, dtype=torch.float32)
    grad_output = torch.ones_like(query)
    lse = torch.zeros(1, 2, 3)
    seen = []

    def preprocess(saved_output, dout, delta, *_):
        seen.append((saved_output.dtype, dout.dtype))
        delta[:, :, :3].copy_((saved_output * dout.float()).sum(-1).transpose(1, 2))

    monkeypatch.setattr(
        runtime, "load_packaged_variant", lambda *_: {"preprocess": preprocess}
    )
    state = runtime.prepare_backward(
        query, key, key, output, lse, grad_output, None, 1.0, None
    )
    assert seen == [(torch.float32, torch.bfloat16)]
    torch.testing.assert_close(
        state.dpsum[:, :, :3], (output * grad_output.float()).sum(-1).transpose(1, 2)
    )
    with pytest.raises(TypeError, match="output has dtype"):
        runtime.prepare_backward(
            query, key, key, output.to(query.dtype), lse, grad_output, None, 1.0, None
        )


def test_model_output_cast_preserves_fp32_backward_state(monkeypatch):
    from dllm_parallel.core.attention import wide_head_attention as wide

    query = torch.zeros(1, 3, 2, 512, dtype=torch.bfloat16, requires_grad=True)
    key = torch.zeros(1, 4, 1, 512, dtype=query.dtype, requires_grad=True)
    precise = torch.full(query.shape, 0.123456, dtype=torch.float32)
    lse = torch.zeros(1, 2, 3)
    seen = []
    monkeypatch.setattr(wide, "_raw_forward", lambda *_: (precise, lse))

    def backward(q, k, v, state, *args):
        seen.append(state)
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    monkeypatch.setattr(wide, "_raw_backward", backward)
    output, _ = wide._WideFullAttention.apply(query, key, key, 1.0)
    assert output.dtype == query.dtype
    output.float().sum().backward()
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], precise, atol=0, rtol=0)
    assert seen[0].dtype == torch.float32
