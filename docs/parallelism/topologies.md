# Parallel topologies

`RunSpec.topology` can combine data (DP), tensor (TP), context (CP), block (BP),
sequence (SP), and supported expert parallelism (EP).

- CP-only uses context-sharded block-masked attention.
- BP-only requires explicit clean-prefix replication.
- Fused CP/BP requires BP to be at least CP and divisible by CP.
- Pipeline parallelism is unsupported and rejected.
- EP is supported for compatible DiffusionGemma MoE configurations; dense
  backbones reject it.

The product of enabled axes must match world size. Validate a config before
allocating GPUs, and use the installed launcher to inspect the exact command
without starting workers:

```bash
dllm launch --config ./run.yaml --nproc-per-node 8 --dry-run
```

For a multi-node static launch, invoke the same command on each node with its
own rank:

```bash
dllm launch --config ./run.yaml \
  --nnodes 2 --node-rank 0 --nproc-per-node 8 \
  --master-addr trainer-0 --master-port 29500
```

For elastic jobs, provide `--rdzv-backend`, `--rdzv-endpoint`, and `--rdzv-id`.
Use torchrun-style node ranges such as `--nnodes 1:4`. Run `dllm launch --help`
for preflight and configuration override options.
