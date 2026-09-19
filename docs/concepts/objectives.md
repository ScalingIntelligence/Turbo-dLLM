# Objectives

The runtime supports standard block diffusion, fast-dLLM-v2, DFlash
distillation, and DiffusionGemma-native objectives. Objective
selection, block size, schedule, weighting, and model family are resolved by
the typed `RunSpec` before device initialization.

Objective selection does not alter corruption, target construction, loss
scaling, or schedule math. Unsupported objective/model combinations fail
validation.
The compatibility matrix has one implementation in the model registry and is
included in `dllm-train --print-supported-configs`. See
[supported models](../models/supported.md) for the human-readable form. Use a
packaged example as the starting point for each model family.
