from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import pytest

from dllm_parallel.core.profiling import release_gates
from dllm_parallel.core.profiling.release_gates import (
    assert_release_policy_gates,
    run_release_policy_gates,
    verify_production_wheel_directory,
)


def test_preflight_cli_can_skip_unused_fa3_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    required_fa3: list[bool] = []
    monkeypatch.setattr(
        release_gates,
        "run_runtime_preflight",
        lambda *, require_fa3: (
            required_fa3.append(require_fa3)
            or {"flash_attention_4": {"backend": "test"}}
        ),
    )

    release_gates.main(["preflight", "--skip-fa3"])

    assert required_fa3 == [False]
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert "flash_attention" not in payload


def _write_native_wheels(
    root: Path,
    *,
    corrupt_hash: bool = False,
    aligned_pack_gqa: bool = True,
) -> None:
    binary = b"native-kernel"
    binary_hash = hashlib.sha256(binary).hexdigest()
    if corrupt_hash:
        binary_hash = hashlib.sha256(b"different").hexdigest()
    dllm_wheel = root / "turbo_dllm-0.1.0-cp312-cp312-manylinux_2_35_x86_64.whl"
    with zipfile.ZipFile(dllm_wheel, "w") as archive:
        archive.writestr(
            "turbo_dllm-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp312-cp312-manylinux_2_35_x86_64\n",
        )
        archive.writestr("dllm_parallel/core/_C/kernel.so", binary)
        archive.writestr(
            "dllm_parallel/core/_C/native_kernels.json",
            json.dumps(
                {
                    "format": "dllm_parallel.native_kernels.v1",
                    "kernels": {
                        "kernel": {
                            "binary": "kernel.so",
                            "binary_hash": binary_hash,
                        }
                    },
                }
            ),
        )

    fa3_wheel = root / "bdlm_flash_attn_3-3.0.0-cp312-cp312-manylinux_2_35_x86_64.whl"
    with zipfile.ZipFile(fa3_wheel, "w") as archive:
        archive.writestr(
            "bdlm_flash_attn_3-3.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp312-cp312-manylinux_2_35_x86_64\n",
        )
        archive.writestr("flash_attn_3/build_metadata.json", "{}")
        archive.writestr("flash_attn_3/_C.cpython-312-x86_64-linux-gnu.so", binary)
        archive.writestr("flash_attn_interface.py", "")

    fa4_wheel = root / "flash_attn_4-4.0.0b19+bdlm.test-py3-none-any.whl"
    with zipfile.ZipFile(fa4_wheel, "w") as archive:
        archive.writestr(
            "flash_attn/cute/interface.py",
            "def _validate_pack_gqa_backward_capability(): pass\n",
        )
        archive.writestr(
            "flash_attn/cute/flash_bwd_postprocess.py",
            f"PACK_GQA_DQACCUM_ALIGNMENT_PROPAGATED = {aligned_pack_gqa!r}\n",
        )


def _write_valid_tree(root: Path) -> None:
    package = root / "dllm_parallel"
    core = package / "core"
    native = core / "_C"
    training = package / "training"
    scripts = root / "scripts" / "build"
    hopper = root / "third_party" / "flash-attention" / "hopper"
    package.mkdir()
    native.mkdir(parents=True)
    training.mkdir(parents=True)
    scripts.mkdir(parents=True)
    (hopper / "flash_attn_3").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "turbo-dllm"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )
    (hopper / "pyproject.toml").write_text(
        '[build-system]\nbuild-backend = "setuptools.build_meta"\n',
        encoding="utf-8",
    )
    (hopper / "setup.py").write_text(
        "# bdlm-flash-attn-3\n"
        "# bdlm.flash_attn_3.build.v3\n"
        "# bdlm-cp-splitd-v3\n"
        "# arch=compute_89,code=sm_89\n"
        "# arch=compute_90a,code=sm_90a\n"
        "class BDLMBuildExtension: pass\n"
        "class BDLMWheelCommand: pass\n",
        encoding="utf-8",
    )
    (hopper / "flash_attn_3" / "__init__.py").write_text(
        '__bdlm_variant__ = "bdlm-cp-splitd-v3"\n'
        '__build_metadata_format__ = "bdlm.flash_attn_3.build.v3"\n',
        encoding="utf-8",
    )
    (scripts / "build_cuda_wheels.sh").write_text(
        "FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \\\n"
        "FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \\\n"
        "FLASH_ATTENTION_FORCE_LEGACY_API=1 \\\n"
        "FLASH_ATTENTION_USE_STABLE_API=0 \\\n"
        'BDLM_FA3_SOURCE_REVISION="$SOURCE_REVISION"\n',
        encoding="utf-8",
    )
    (training / "block_diffusion_trainer.py").write_text("", encoding="utf-8")
    (core / "kernels").mkdir()
    (core / "kernels" / "cp_fusion.py").write_text("# cp\n", encoding="utf-8")
    (core / "kernels" / "chunked_linear_ce_native.py").write_text(
        "# ce\n", encoding="utf-8"
    )
    (core / "kernels" / "dflash_cp_fusion.py").write_text(
        "# dflash\n",
        encoding="utf-8",
    )
    (native / "bdlm_cp_fusion.so").write_bytes(b"cp")
    (native / "dllm_fused_linear_ce_v3.so").write_bytes(b"ce")
    (native / "dflash_cp_fusion.so").write_bytes(b"dflash")
    (native / "native_kernels.json").write_text(
        """{
  "format": "dllm_parallel.native_kernels.v1",
  "package_version": "0.1.0",
  "kernels": {
    "bdlm_cp_fusion": {
      "binary": "bdlm_cp_fusion.so",
      "binary_hash": "e44b4bd707c540bca8615b21f850fb41e02029701df241d179c1a5f3acbf5bf1",
      "cuda_arch_list": "8.0;9.0",
      "minimum_device_capability": [8, 0],
      "supported_device_capabilities": [[8, 0], [9, 0]],
      "python_extension_suffix": ".so",
      "required_symbols": ["merge_full_", "merge_compact_", "merge_backward_", "finalize_bshd_"],
      "build_flags": {"extra_cflags": ["-O3"], "extra_cuda_cflags": ["-O3", "--use_fast_math"], "extra_ldflags": []},
      "source_hash": "59d8927b64a1a00a6791b8643e2da06fd11fac422b3a42a0c1a87dbe090e5eaf",
      "source_files": ["dllm_parallel/core/kernels/cp_fusion.py"]
    },
    "dllm_fused_linear_ce_v3": {
      "binary": "dllm_fused_linear_ce_v3.so",
      "binary_hash": "e64c826b6b33f307cecf54ff843c5f9797c1056eb33f9dd60bc632f519712cb9",
      "cuda_arch_list": "8.0;9.0",
      "minimum_device_capability": [8, 0],
      "supported_device_capabilities": [[8, 0], [9, 0]],
      "python_extension_suffix": ".so",
      "required_symbols": ["chunked_linear_ce_forward", "chunked_linear_ce_backward"],
      "build_flags": {"extra_cflags": ["-O3", "-std=c++17"], "extra_cuda_cflags": ["-O3", "--use_fast_math"], "extra_ldflags": ["-lcublas"]},
      "source_hash": "37520c25de6383e98976d9e24695ff7a0cad71876ae4ed33108bfb69b509dd8d",
      "source_files": ["dllm_parallel/core/kernels/chunked_linear_ce_native.py"]
    },
    "dflash_cp_fusion": {
      "binary": "dflash_cp_fusion.so",
      "binary_hash": "004c2f8423543431568f6e585434b18d0d895acddbefd3363fcc57748ff48b33",
      "cuda_arch_list": "8.0;9.0",
      "minimum_device_capability": [8, 0],
      "supported_device_capabilities": [[8, 0], [9, 0]],
      "python_extension_suffix": ".so",
      "required_symbols": ["merge_bshd_", "merge_state_"],
      "build_flags": {"extra_cflags": ["-O3"], "extra_cuda_cflags": ["-O3", "--use_fast_math"], "extra_ldflags": []},
      "source_hash": "09bdf97704ccfe7796d8eaba5ebfd715f5e6cf10953244488c3cd8a5ee7085cb",
      "source_files": ["dllm_parallel/core/kernels/dflash_cp_fusion.py"]
    }
  }
}
""",
        encoding="utf-8",
    )


def test_release_policy_gates_pass_clean_tree(tmp_path) -> None:
    _write_valid_tree(tmp_path)

    violations = run_release_policy_gates(tmp_path)

    assert violations == []


def test_release_policy_gates_reject_import_path_mutation(tmp_path) -> None:
    _write_valid_tree(tmp_path)
    package = tmp_path / "dllm_parallel"
    (package / "bad.py").write_text(
        "import sys\nsys.path.insert(0, 'x')\n", encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="import_path_mutation"):
        assert_release_policy_gates(tmp_path)


def test_release_policy_gates_reject_runtime_fa3_source_path(tmp_path) -> None:
    _write_valid_tree(tmp_path)
    package = tmp_path / "dllm_parallel"
    (package / "bad.py").write_text(
        'FA3_SOURCE = "third_party/flash-attention/hopper"\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="hidden_flash_attention_path"):
        assert_release_policy_gates(tmp_path)


def test_release_policy_gates_reject_production_recipe_directory(tmp_path) -> None:
    _write_valid_tree(tmp_path)
    production = tmp_path / "dllm_parallel" / "recipes" / "production"
    production.mkdir(parents=True)

    with pytest.raises(AssertionError, match="production_recipe_directory"):
        assert_release_policy_gates(tmp_path)


def test_release_policy_gates_reject_public_fa3_dependency(tmp_path) -> None:
    _write_valid_tree(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project.optional-dependencies]\nproduction = ["flash-attn-3"]\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="fa3_public_dependency"):
        assert_release_policy_gates(tmp_path)


def test_release_policy_gates_reject_fa3_wheel_fallback(tmp_path) -> None:
    _write_valid_tree(tmp_path)
    setup = tmp_path / "third_party" / "flash-attention" / "hopper" / "setup.py"
    setup.write_text(
        setup.read_text(encoding="utf-8") + "\nurlretrieve('wheel')\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="fa3_prebuilt_wheel_fallback"):
        assert_release_policy_gates(tmp_path)


def test_release_policy_gates_reject_incomplete_fa3_build_policy(tmp_path) -> None:
    _write_valid_tree(tmp_path)
    builder = tmp_path / "scripts" / "build" / "build_cuda_wheels.sh"
    builder.write_text(
        builder.read_text(encoding="utf-8").replace(
            "FLASH_ATTENTION_FORCE_LEGACY_API=1",
            "",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="fa3_build_policy_incomplete"):
        assert_release_policy_gates(tmp_path)


def test_release_policy_gates_reject_native_package_version_mismatch(
    tmp_path,
) -> None:
    _write_valid_tree(tmp_path)
    manifest = tmp_path / "dllm_parallel" / "core" / "_C" / "native_kernels.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["package_version"] = "0.0.9"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssertionError, match="native_kernel_package_version"):
        assert_release_policy_gates(tmp_path)


def test_release_policy_gates_reject_native_build_flag_mismatch(tmp_path) -> None:
    _write_valid_tree(tmp_path)
    manifest = tmp_path / "dllm_parallel" / "core" / "_C" / "native_kernels.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["kernels"]["bdlm_cp_fusion"]["build_flags"] = {}
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssertionError, match="native_kernel_build_flags"):
        assert_release_policy_gates(tmp_path)


def test_production_wheel_gate_verifies_platform_binaries(tmp_path) -> None:
    _write_native_wheels(tmp_path)

    wheels = verify_production_wheel_directory(tmp_path)

    assert wheels["turbo_dllm"].endswith("manylinux_2_35_x86_64.whl")
    assert wheels["flash_attention"].endswith("manylinux_2_35_x86_64.whl")
    assert wheels["flash_attention_4"].endswith("py3-none-any.whl")


def test_production_wheel_gate_rejects_binary_hash_mismatch(tmp_path) -> None:
    _write_native_wheels(tmp_path, corrupt_hash=True)

    with pytest.raises(RuntimeError, match="binary hash mismatch"):
        verify_production_wheel_directory(tmp_path)


def test_production_wheel_gate_rejects_stock_fa4(tmp_path) -> None:
    _write_native_wheels(tmp_path)
    fa4_wheel = next(tmp_path.glob("flash_attn_4-*.whl"))
    with zipfile.ZipFile(fa4_wheel, "w") as archive:
        archive.writestr(
            "flash_attn/cute/interface.py", "def _flash_attn_bwd(): pass\n"
        )

    with pytest.raises(RuntimeError, match="native Pack-GQA backward"):
        verify_production_wheel_directory(tmp_path)


def test_production_wheel_gate_rejects_unaligned_pack_gqa_backward(
    tmp_path,
) -> None:
    _write_native_wheels(tmp_path, aligned_pack_gqa=False)

    with pytest.raises(RuntimeError, match="aligned packed-GQA"):
        verify_production_wheel_directory(tmp_path)


def test_production_wheel_gate_rejects_universal_native_wheel(tmp_path) -> None:
    _write_native_wheels(tmp_path)
    dllm_wheel = next(tmp_path.glob("turbo_dllm-*.whl"))
    universal = tmp_path / "turbo_dllm-0.1.0-py3-none-any.whl"
    dllm_wheel.rename(universal)

    with pytest.raises(RuntimeError, match="universal tag"):
        verify_production_wheel_directory(tmp_path)
