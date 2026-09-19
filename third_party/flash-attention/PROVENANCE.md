# FlashAttention fork provenance

- Upstream project: `Dao-AILab/flash-attention`
- Upstream repository: `https://github.com/Dao-AILab/flash-attention`
- Upstream source revision: `2409214a03797b168f648ea30df1adbc09ce658a`
- Upstream tag at that revision: `fa4-v4.0.0.beta23`
- Monorepo import commit: `89467c2d5c5314741f2fb43c469162d62bcaa76b`
- Pre-relocation source tree: `390a26b5448aad77a8faba0f2fdd1a08f98cf4a4`
- Pre-relocation Hopper tree: `362d9ebb9d86532ad3c1a6679a8bcc7e135346f6`
- Pre-relocation CuTe tree: `1777795f43146d45aff1b0ede1a20c347e46db91`
- Pre-curation CUTLASS tree: `d44f3350d470728eb780d0a25de38187b1e2b39c`
- Retained CUTLASS include tree: `3c43ce5aa9a960e8ea26cdf485a121048d09d3ee`

The upstream import and local patch series are anchored by monorepo commit
`89467c2`; the source and tree identifiers above provide the immutable audit
trail for this curated copy. The fork produces two native artifact identities:
`bdlm-flash-attn-3` for the modified Hopper extension and `flash-attn-4` for the
coordinated CuTe package. They are built and published independently from the
portable Turbo-dLLM (`turbo-dllm`) wheel.

Relocating this source under `third_party/` did not change the Hopper or CuTe
tree objects listed above. This repository retains only the upstream source
required by the two native wheel builds: the Hopper package, the CuTe package,
and CUTLASS headers. Upstream papers, generated documentation, media,
benchmarks, tests, training utilities, and unrelated kernels remain recoverable
from the pinned upstream revision and monorepo import commit. Release builds
record the Turbo-dLLM revision in artifact metadata and verify the coordinated
wheel set before publishing.

See `LICENSE` for the upstream BSD 3-Clause license and `NOTICE` for the
redistribution notice.
