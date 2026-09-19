# Checkpointing and resume

Checkpoints contain the model, optimizer, scheduler, data position,
configuration, and random state needed to resume training. A run with
`checkpointing.auto_resume: true` resumes from its newest complete checkpoint.

Inspect a checkpoint without loading model tensors:

```bash
dllm-checkpoint inspect ./checkpoints/run
dllm-checkpoint validate ./checkpoints/run
```

Only load trusted PyTorch checkpoints. Turbo-dLLM rejects incompatible formats
and topologies before resuming.

## DFlash2 export

```bash
dllm dflash export \
  --checkpoint ./checkpoints/dflash2 \
  --output ./exports/dflash2 \
  --model-id org/draft \
  --model-revision COMMIT \
  --block-size 16
```

The exact base draft is downloaded automatically. Pass `--base-model` to use a
local copy. For training, export, and both serving backends, follow the
[DFlash2 guide](../getting-started/dflash2-training-and-serving.md).
