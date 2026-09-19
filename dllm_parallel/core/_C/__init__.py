# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Packaged native extension namespace.

Production builds place compiled CP/BP and fused CE extension modules in this
package. Developer recipes may still JIT-build missing modules explicitly, but
production startup requires the packaged modules to be importable.
"""
