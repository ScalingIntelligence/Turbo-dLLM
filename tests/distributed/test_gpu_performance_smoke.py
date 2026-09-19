from __future__ import annotations

import math
import time

import pytest
import torch


pytestmark = pytest.mark.performance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bfloat16_matmul_reports_finite_throughput(record_property) -> None:
    """Exercise the installed CUDA stack without encoding a hardware baseline."""

    device = torch.device("cuda", 0)
    left = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    right = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    for _ in range(2):
        torch.mm(left, right)
    torch.cuda.synchronize(device)

    started = time.perf_counter()
    iterations = 4
    for _ in range(iterations):
        torch.mm(left, right)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    tflops = (iterations * 2 * 2048**3) / elapsed / 1.0e12

    record_property("cuda_bfloat16_matmul_tflops", tflops)
    assert elapsed > 0
    assert math.isfinite(tflops) and tflops > 0
