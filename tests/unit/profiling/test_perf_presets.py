from __future__ import annotations

import pytest

from dllm_parallel.core.profiling.perf import (
    DEFAULT_LOG_EVERY_N_STEPS,
    DEFAULT_PROFILER_SCHEDULE,
    HARDWARE_PEAK_FLOPS,
    PerfConfig,
    detect_hardware_preset,
    peak_flops_for_hardware,
)


def test_default_perf_config_matches_h100_release_defaults() -> None:
    cfg = PerfConfig()
    assert cfg.hardware_peak_flops == 989.0e12
    assert cfg.log_every_n_steps == DEFAULT_LOG_EVERY_N_STEPS
    assert cfg.schedule == DEFAULT_PROFILER_SCHEDULE


def test_explicit_hardware_resolves_preset_peak() -> None:
    assert PerfConfig(hardware="a100_sxm_bf16").hardware_peak_flops == 312.0e12


def test_explicit_peak_overrides_preset() -> None:
    cfg = PerfConfig(hardware="a100_sxm_bf16", hardware_peak_flops=500.0e12)
    assert cfg.hardware_peak_flops == 500.0e12


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown hardware preset"):
        peak_flops_for_hardware("not-a-real-gpu")


def test_from_node_resolves_preset_and_explicit() -> None:
    assert PerfConfig.from_node({"hardware": "a100_sxm_bf16"}).hardware_peak_flops == 312.0e12
    assert PerfConfig.from_node({"hardware_peak_flops": 700.0e12}).hardware_peak_flops == 700.0e12


def test_detect_hardware_preset_returns_known_preset() -> None:
    assert detect_hardware_preset() in HARDWARE_PEAK_FLOPS
