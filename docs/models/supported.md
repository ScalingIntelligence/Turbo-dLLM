# Supported model families

| Family key | Supported objectives | Use |
|---|---|---|
| `causal_lm` | `fast_dllm_v2` | Generic dense causal backbones |
| `dflash` | `dflash_distillation` | DFlash speculator/distillation training |
| `diffusion_gemma` | `standard_block_diffusion`, `diffusiongemma_native_sft` | DiffusionGemma dense/MoE execution |
| `nemotron_labs_diffusion` | `standard_block_diffusion` | Nemotron Labs Diffusion execution |
| `qwen3_8` | `fast_dllm_v2` | Qwen3.8-specific optimized execution |

Model downloads remain the user's responsibility.
Run `dllm-train --print-supported-configs` for the machine-readable registry.
Unsupported model and objective combinations fail during validation.

## Muse-Glimmer DFlash2

`incoai/Muse-Glimmer-30B-DFlash2` is supported through the generic `dflash`
family; Turbo-dLLM does not include a separate Muse-Glimmer backbone. Trained
DFlash2 drafts can be exported with `dllm dflash export`. SGLang supports the
Qwen3 and Muse-Glimmer exports. Native vLLM serving currently supports the
Qwen3 export contract. See
[train and serve DFlash2](../getting-started/dflash2-training-and-serving.md).
