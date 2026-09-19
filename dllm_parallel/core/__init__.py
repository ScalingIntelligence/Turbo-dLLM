# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Reusable block-diffusion training library.

`dllm_parallel.core` is the application-independent layer: parallelism, the
shared transformer building blocks, attention backends, model backbones,
objectives, schedules, optimizers, kernels, checkpoint format, and profiling.
It must never import from `dllm_parallel.training` (the one-way layering that
keeps the core installable on its own). Imports here stay torch-free; heavy
submodules load lazily on first use.
"""
