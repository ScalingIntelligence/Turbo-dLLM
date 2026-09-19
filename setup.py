# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from pathlib import Path

from setuptools import Distribution, setup


class NativeArtifactDistribution(Distribution):
    """Mark production wheels as platform-specific when CUDA artifacts exist."""

    def has_ext_modules(self) -> bool:
        native_dir = Path(__file__).parent / "dllm_parallel" / "core" / "_C"
        return any(
            path
            for pattern in ("*.so", "*.pyd", "*.dylib")
            for path in native_dir.glob(pattern)
        )


setup(distclass=NativeArtifactDistribution)
