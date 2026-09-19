# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Canonical public training API."""

from dllm_parallel.training.entrypoint import main as train
from dllm_parallel.training.run_spec import RunSpec, load_run_spec

__all__ = ["RunSpec", "load_run_spec", "train"]
