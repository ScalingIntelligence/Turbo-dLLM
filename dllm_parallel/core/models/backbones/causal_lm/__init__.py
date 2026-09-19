# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Causal-LM to block-diffusion conversion backbone."""

from dllm_parallel.core.models.backbones.causal_lm.executor import (
    CausalLMBackboneExecutor,
    build_executor,
    summarize_config,
)

__all__ = ["CausalLMBackboneExecutor", "build_executor", "summarize_config"]
