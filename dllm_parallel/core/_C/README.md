# `dllm_parallel.core._C`

Import namespace for the project's **compiled** native CUDA extensions. A `_C`
package is the conventional home for a Python project's compiled C/C++/CUDA
modules (cf. `torch._C`); the binaries themselves are build artifacts, not
source.

## What lives here

- `__init__.py` — marks the package (committed).
- `<kernel>.so` — compiled extensions, e.g. `bdlm_cp_fusion.so`,
  `dllm_fused_linear_ce_v2.so` (**build artifacts, git-ignored**).
- `native_kernels.json` — manifest with the package version, per-kernel
  source/binary hashes, compiler and linker flags, ABI suffix, required
  symbols, CUDA arch list, and device capabilities
  (**build artifact, git-ignored**).

## Source of truth

The actual kernel source is **not** here — it lives as ordinary `.cpp`/`.cu`
files under [`dllm_parallel/core/csrc/`](../csrc), and the build/compile metadata
(sources, symbols, compile flags) is declared once in
[`dllm_parallel/core/kernels/_native_specs.py`](../kernels/_native_specs.py).

## Building

Compile the extensions from `csrc/` and (re)generate the manifest into this
directory before building a wheel/container:

```bash
python -m dllm_parallel.core.kernels.build      # or: dllm-build-kernels
```

When a release image builds kernels before installing the wheel, its build
system must pass the version from the package metadata explicitly:

```bash
dllm-build-kernels --package-version "${PACKAGE_VERSION}"
```

Installed-package builds infer the same value from distribution metadata. The
runtime rejects artifacts whose recorded version differs from the installed
package.

This must run on a machine with the CUDA toolchain and a target GPU (the build
records the device/ABI it was produced for). At runtime, production startup
loads the packaged `.so` and verifies it against `native_kernels.json`; only
developer builds JIT-compile from `csrc/` on demand
(`kernel.runtime_jit=true`).
