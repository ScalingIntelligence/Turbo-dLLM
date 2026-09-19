from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dllm_parallel.core.attention import fa3


def _bdlm_flash_attn_func(
    q,
    k,
    v,
    query_blocks,
    query_is_clean,
    block_size,
    key_start=0,
    softmax_scale=None,
    deterministic=False,
    sm_margin=0,
    return_softmax=False,
    clean_offset=0,
):
    del (
        q,
        k,
        v,
        query_blocks,
        query_is_clean,
        block_size,
        key_start,
        softmax_scale,
        deterministic,
        sm_margin,
        return_softmax,
        clean_offset,
    )


def _op(argument_names: tuple[str, ...]) -> SimpleNamespace:
    schema = SimpleNamespace(
        arguments=[SimpleNamespace(name=name) for name in argument_names]
    )
    return SimpleNamespace(default=SimpleNamespace(_schema=schema))


def _abi_namespace(**overrides: tuple[str, ...]) -> SimpleNamespace:
    suffixes = {
        **fa3._BDLM_OP_ABI,
        **fa3._DFLASH_OP_ABI,
        **overrides,
    }
    return SimpleNamespace(
        **{name: _op(("prefix", *suffix)) for name, suffix in suffixes.items()}
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(binary: Path, interface: Path) -> dict[str, Any]:
    splitd_root = binary.parent / "bdlm_splitd"
    splitd_root.mkdir(exist_ok=True)
    (splitd_root / "__init__.py").write_text("# splitd package\n", encoding="utf-8")
    package_files = {
        "flash_attn_3/bdlm_splitd/__init__.py": _sha256(splitd_root / "__init__.py")
    }
    payload: dict[str, Any] = {
        "format": "bdlm.flash_attn_3.build.v3",
        "variant": "bdlm-cp-splitd-v3",
        "distribution": {
            "name": "bdlm-flash-attn-3",
            "version": "3.0.0+bdlm2",
        },
        "module": "flash_attn_3._C",
        "required_ops": [
            "bdlm_fwd_accum",
            "bdlm_ragged_prefix_fwd",
            "bdlm_ragged_prefix_bwd",
            "dflash_interval_fwd",
            "dflash_interval_bwd",
        ],
        "binary": {
            "path": f"flash_attn_3/{binary.name}",
            "sha256": _sha256(binary),
            "extension_suffix": ".abi3.so",
        },
        "python_interface": {
            "path": interface.name,
            "sha256": _sha256(interface),
        },
        "python_package": {
            "root": "flash_attn_3/bdlm_splitd",
            "sha256": hashlib.sha256(
                json.dumps(
                    package_files, separators=(",", ":"), sort_keys=True
                ).encode()
            ).hexdigest(),
            "files": package_files,
        },
        "source": {
            "sha256": "a" * 64,
            "file_count": 12,
            "revision": "source-revision",
        },
        "build": {
            "python": "3.12.4",
            "python_extension_suffix": ".cpython-312-x86_64-linux-gnu.so",
            "torch": "2.8.0+cu128",
            "torch_cuda": "12.8",
            "cuda_toolkit": "12.8",
            "cuda_architectures": ["sm_80", "sm_89", "sm_90a"],
            "cxx11_abi": False,
            "flash_api_source": "flash_api.cpp",
            "feature_flags": {
                "FLASHATTENTION_DISABLE_BACKWARD": False,
                "FLASHATTENTION_DISABLE_SM8x": False,
            },
        },
    }
    _refresh_build_id(payload)
    return payload


def _refresh_build_id(payload: dict[str, Any]) -> None:
    payload.pop("build_id", None)
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    payload["build_id"] = hashlib.sha256(canonical).hexdigest()


def _validate(
    payload: dict[str, Any],
    *,
    binary: Path,
    interface: Path,
    device_capability: tuple[int, int] | None = (9, 0),
) -> fa3.FlashAttentionKernelMetadata:
    return fa3._validate_build_metadata(
        payload,
        distribution_version="3.0.0+bdlm2",
        binary_path=binary,
        interface_path=interface,
        package_path=binary.parent / "__init__.py",
        metadata_path=binary.parent / "build_metadata.json",
        runtime_torch_version="2.8.1+cu128",
        runtime_torch_cuda="12.8",
        runtime_cxx11_abi=False,
        device_capability=device_capability,
    )


def test_validate_build_metadata_binds_binary_source_and_architectures(tmp_path) -> None:
    binary = tmp_path / "_C.abi3.so"
    interface = tmp_path / "flash_attn_interface.py"
    binary.write_bytes(b"native-binary")
    interface.write_text("# interface\n", encoding="utf-8")
    payload = _metadata(binary, interface)

    result = _validate(
        payload,
        binary=binary,
        interface=interface,
        device_capability=(8, 9),
    )

    assert result.package == "bdlm-flash-attn-3"
    assert result.binary_hash == _sha256(binary)
    assert result.source_hash == "a" * 64
    assert result.cuda_architectures == ("sm_80", "sm_89", "sm_90a")


def test_validate_build_metadata_rejects_modified_native_binary(tmp_path) -> None:
    binary = tmp_path / "_C.abi3.so"
    interface = tmp_path / "flash_attn_interface.py"
    binary.write_bytes(b"native-binary")
    interface.write_text("# interface\n", encoding="utf-8")
    payload = _metadata(binary, interface)
    binary.write_bytes(b"replaced-binary")

    with pytest.raises(RuntimeError, match="native binary hash mismatch"):
        _validate(payload, binary=binary, interface=interface)


def test_validate_build_metadata_rejects_modified_splitd_source(tmp_path) -> None:
    binary = tmp_path / "_C.abi3.so"
    interface = tmp_path / "flash_attn_interface.py"
    binary.write_bytes(b"native-binary")
    interface.write_text("# interface\n", encoding="utf-8")
    payload = _metadata(binary, interface)
    (tmp_path / "bdlm_splitd" / "__init__.py").write_text(
        "# replaced splitd package\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="packaged source hash mismatch"):
        _validate(payload, binary=binary, interface=interface)


def test_validate_build_metadata_rejects_unsupported_device_architecture(tmp_path) -> None:
    binary = tmp_path / "_C.abi3.so"
    interface = tmp_path / "flash_attn_interface.py"
    binary.write_bytes(b"native-binary")
    interface.write_text("# interface\n", encoding="utf-8")
    payload = _metadata(binary, interface)

    with pytest.raises(RuntimeError, match="no code for CUDA device capability"):
        _validate(
            payload,
            binary=binary,
            interface=interface,
            device_capability=(12, 0),
        )


def test_validate_build_metadata_rejects_blackwell_device(tmp_path) -> None:
    binary = tmp_path / "_C.abi3.so"
    interface = tmp_path / "flash_attn_interface.py"
    binary.write_bytes(b"native-binary")
    interface.write_text("# interface\n", encoding="utf-8")
    payload = _metadata(binary, interface)

    with pytest.raises(RuntimeError, match="no code for CUDA device capability"):
        _validate(
            payload,
            binary=binary,
            interface=interface,
            device_capability=(10, 0),
        )


def test_validate_build_metadata_requires_exact_sm89_architecture_list(tmp_path) -> None:
    binary = tmp_path / "_C.abi3.so"
    interface = tmp_path / "flash_attn_interface.py"
    binary.write_bytes(b"native-binary")
    interface.write_text("# interface\n", encoding="utf-8")
    payload = _metadata(binary, interface)
    payload["build"]["cuda_architectures"] = ["sm_80", "sm_90a"]
    _refresh_build_id(payload)

    with pytest.raises(RuntimeError, match="does not match the build contract"):
        _validate(payload, binary=binary, interface=interface)


def test_validate_build_metadata_rejects_unrelated_distribution(tmp_path) -> None:
    binary = tmp_path / "_C.abi3.so"
    interface = tmp_path / "flash_attn_interface.py"
    binary.write_bytes(b"native-binary")
    interface.write_text("# interface\n", encoding="utf-8")
    payload = _metadata(binary, interface)
    payload["distribution"]["name"] = "flash-attn-3"

    with pytest.raises(RuntimeError, match="unexpected distribution"):
        _validate(payload, binary=binary, interface=interface)


def test_verify_bdlm_attention_abi_accepts_exact_schema_suffixes() -> None:
    fa3._verify_bdlm_attention_abi(
        bdlm_flash_attn_func=_bdlm_flash_attn_func,
        flash_attn_3_gpu=_abi_namespace(),
        op_abi=fa3._BDLM_OP_ABI,
    )


@pytest.mark.parametrize(
    ("op_name", "reordered_suffix"),
    [
        (
            "fwd",
            (
                "bdlm_query_blocks",
                "bdlm_query_is_clean",
                "bdlm_block_size",
                "bdlm_clean_offset",
                "bdlm_key_start",
            ),
        ),
        (
            "bwd",
            (
                "bdlm_query_blocks",
                "bdlm_query_is_clean",
                "bdlm_block_size",
                "bdlm_key_start",
                "bdlm_clean_offset",
                "softmax_lse_grad",
            ),
        ),
        (
            "bdlm_fwd_accum",
            (
                "query_indices",
                "bdlm_query_blocks",
                "bdlm_query_is_clean",
                "bdlm_block_size",
                "bdlm_key_start",
                "softmax_scale",
                "bdlm_clean_offset",
                "initial",
            ),
        ),
    ],
)
def test_verify_bdlm_attention_abi_rejects_reordered_clean_offset(
    op_name: str,
    reordered_suffix: tuple[str, ...],
) -> None:
    with pytest.raises(RuntimeError, match=rf"{op_name} ABI is stale or reordered"):
        fa3._verify_bdlm_attention_abi(
            bdlm_flash_attn_func=_bdlm_flash_attn_func,
            flash_attn_3_gpu=_abi_namespace(**{op_name: reordered_suffix}),
            op_abi=fa3._BDLM_OP_ABI,
        )


def test_verify_bdlm_attention_abi_rejects_reordered_python_suffix() -> None:
    def stale_wrapper(
        key_start=0,
        clean_offset=0,
        softmax_scale=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        del key_start, clean_offset, softmax_scale, deterministic, sm_margin, return_softmax

    with pytest.raises(RuntimeError, match="Python ABI suffix is stale or reordered"):
        fa3._verify_bdlm_attention_abi(
            bdlm_flash_attn_func=stale_wrapper,
            flash_attn_3_gpu=_abi_namespace(),
            op_abi=fa3._BDLM_OP_ABI,
        )
