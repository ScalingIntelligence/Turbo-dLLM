#!/usr/bin/env python3
# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Benchmark SM90 FA4 backward launch configurations on the fused BDLM shape.

This is an offline kernel-development tool. Production selects one validated
configuration from the packaged FA4 capability table; it does not autotune at
training time.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from typing import Any, Iterator

import torch

from tools.benchmarks.bench_fa4_pack_gqa_backward import (
    Shape,
    _make_state,
    _time_variants,
)


@contextmanager
def _backward_config(config: dict[str, Any]) -> Iterator[None]:
    from flash_attn.cute import interface

    original = interface._tile_size_bwd_sm90

    def selected_config(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return interface.BwdConfig(**config)

    interface._tile_size_bwd_sm90 = selected_config
    try:
        yield
    finally:
        interface._tile_size_bwd_sm90 = original


def _config(
    *,
    tile_m: int,
    tile_n: int,
    stages_pds: int = 2,
    atom_n_dkv: int = 2,
    atom_m_dq: int = 1,
    dq_single_wg: bool = False,
) -> dict[str, Any]:
    return {
        "m_block_size": tile_m,
        "n_block_size": tile_n,
        "num_stages_Q": 2,
        "num_stages_dO": 2,
        "num_stages_PdS": stages_pds,
        "SdP_swapAB": True,
        "dKV_swapAB": False,
        "dQ_swapAB": False,
        "AtomLayoutMSdP": 1,
        "AtomLayoutNdKV": atom_n_dkv,
        "AtomLayoutMdQ": atom_m_dq,
        "dQ_single_wg": dq_single_wg,
    }


def _configs() -> dict[str, dict[str, Any]]:
    return {
        "m64_n128": _config(tile_m=64, tile_n=128),
        "m128_n128": _config(
            tile_m=128,
            tile_n=128,
            atom_m_dq=2,
        ),
        "m64_n128_dq_single": _config(
            tile_m=64,
            tile_n=128,
            dq_single_wg=True,
        ),
        "m64_n128_atom_n1": _config(
            tile_m=64,
            tile_n=128,
            atom_n_dkv=1,
        ),
        "m64_n128_pds1": _config(
            tile_m=64,
            tile_n=128,
            stages_pds=1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=tuple(_configs()), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FA4 tuning requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise RuntimeError("FA4 SM90 tuning requires compute capability 9.x")

    shape = Shape(
        q_length=16_384,
        k_length=40_960,
        active_length=8_192,
        parallel_size=4,
        rank=0,
        block_size=32,
    )
    name = str(args.config)
    config = _configs()[name]
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "shape": shape.to_dict(),
        "config": name,
        "launch": config,
    }
    with _backward_config(config):
        state = _make_state(shape, device, 1234)
        timing = _time_variants(
            device,
            2,
            5,
            variants=[("pack_gqa", state.backward_args(pack_gqa=True))],
        )[0]
    report["milliseconds"] = timing["milliseconds"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
