# Native development

The modified FlashAttention source is pinned under
`third_party/flash-attention`; its license, upstream revision, component tree
hashes, and artifact identities are in `PROVENANCE.md`.

Maintainers build explicit architectures with:

```bash
scripts/build/build_cuda_wheels.sh --cuda-arch-list '9.0'
scripts/verify/artifacts.sh --directory dist
```

Builds require a clean committed source snapshot, CUDA 12.8, compatible PyTorch,
and the `kernel-build` extra. Runtime JIT is never an implicit release fallback.
Do not edit or reformat vendored sources during repository-only refactors; make
native changes isolated, reviewed, and hardware-qualified.
