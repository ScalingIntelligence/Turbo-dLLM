from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_flash_attention_is_isolated_and_licensed() -> None:
    vendor = ROOT / "third_party/flash-attention"
    assert vendor.is_dir()
    assert not (ROOT / "flash-attention").exists()
    assert (vendor / "LICENSE").is_file()
    assert (vendor / "PROVENANCE.md").is_file()
    assert (vendor / "NOTICE").is_file()


def test_flash_attention_vendor_tree_contains_only_release_build_inputs() -> None:
    vendor = ROOT / "third_party/flash-attention"
    required = (
        vendor / "hopper",
        vendor / "flash_attn/cute",
        vendor / "csrc/cutlass/include",
        vendor / "csrc/cutlass/LICENSE.txt",
    )
    excluded = (
        vendor / "assets",
        vendor / "benchmarks",
        vendor / "tests",
        vendor / "training",
        vendor / "tools",
        vendor / "csrc/cutlass/docs",
        vendor / "csrc/cutlass/examples",
        vendor / "csrc/cutlass/media",
        vendor / "csrc/cutlass/python",
        vendor / "csrc/cutlass/test",
        vendor / "csrc/cutlass/tools",
    )

    assert all(path.exists() for path in required)
    assert not any(path.exists() for path in excluded)


def test_flash_attention_provenance_pins_source_and_artifact_identity() -> None:
    provenance = (ROOT / "third_party/flash-attention/PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    required = (
        "2409214a03797b168f648ea30df1adbc09ce658a",
        "390a26b5448aad77a8faba0f2fdd1a08f98cf4a4",
        "3c43ce5aa9a960e8ea26cdf485a121048d09d3ee",
        "bdlm-flash-attn-3",
        "flash-attn-4",
        "Dao-AILab/flash-attention",
    )
    assert all(value in provenance for value in required)


def test_shipped_flash_attention_text_uses_public_product_name() -> None:
    paths = (
        ROOT / "third_party/flash-attention/PROVENANCE.md",
        ROOT / "third_party/flash-attention/NOTICE",
        ROOT / "third_party/flash-attention/flash_attn/cute/flash_bwd_postprocess.py",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "dllm-parallel" not in text, path
        assert "Turbo-dLLM" in text, path


def test_first_party_build_paths_use_third_party_boundary() -> None:
    paths = (
        ROOT / "scripts/build/build_cuda_wheels.sh",
        ROOT / "dllm_parallel/core/profiling/release_gates.py",
        ROOT / "MANIFEST.in",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "third_party/flash-attention" in text, path
        assert 'Path("flash-attention")' not in text, path
