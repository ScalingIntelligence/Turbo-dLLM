# BDLM Split-D FlashAttention

`bdlm_splitd` is the Hopper training kernel for attention heads with dimension
512. It implements dense attention and the exact block-diffusion mask used by
`dllm_parallel`, including forward output, FP32 log-sum-exp statistics, and
backward gradients for Q, K, V, and an optional LSE gradient.

## Runtime contract

- GPU architecture: SM90a (Hopper)
- Tensor layout: BSHD
- Head dimension: 512
- Dtypes: BF16 and FP16
- Q heads per KV head: 1, 2, 4, or 8
- Mask modes: dense, clean-prefix/full block diffusion, and exact packed
  clean-context intervals
- Metadata: contiguous CUDA query/key coordinates, intervals, and clean roles

Masked calls use a nonnegative logical `key_start`. Full block-diffusion masks
are selected explicitly with `full_mask=True` and require a positive
`clean_offset`; prefix masks require `clean_offset=0`.

Interval calls carry one half-open clean-context interval and one local active
block coordinate per query, plus one logical coordinate and clean/active role
per key. This represents sliding and full shared-encoder/decoder attention
without materializing a dense mask.

Sparse interval schedules use CSR offsets and opaque `int32` tile work items.
The high-level planner constructs these work items and marks tiles whose mask
predicate is identically true, allowing the native kernel to omit redundant
elementwise mask evaluation without changing the visible query-key pairs.

Production wheels contain the complete AOT specialization matrix. Startup
verifies the capability manifest and SHA-256 digest of every shared object.
Missing or incompatible artifacts are fatal. Production execution never JIT
compiles and never falls back to another attention implementation.

## Build and development

Building a wheel compiles and packages all AOT variants. Compiler dependencies
are build dependencies and are not imported by the production runtime.
Developer JIT is available only through the explicit runtime-JIT policy and the
`developer-jit` package extra.
