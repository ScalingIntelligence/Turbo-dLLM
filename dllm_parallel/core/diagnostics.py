"""Provider-neutral diagnostics for portable and GPU training environments."""

from __future__ import annotations

import os
import site
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeCheck:
    """One actionable environment check."""

    name: str
    status: str
    detail: str
    remediation: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeReadinessReport:
    """Result of a portable or training-runtime inspection."""

    checks: tuple[RuntimeCheck, ...]

    @property
    def failures(self) -> tuple[RuntimeCheck, ...]:
        return tuple(check for check in self.checks if check.status == "fail")

    @property
    def ready(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
        }

    def require_ready(self) -> None:
        if self.ready:
            return
        lines = ["training runtime is not ready:"]
        for check in self.failures:
            lines.append(f"- {check.name}: {check.detail}")
            if check.remediation:
                lines.append(f"  fix: {check.remediation}")
        raise RuntimeError("\n".join(lines))


def _run_check(
    name: str,
    operation: Callable[[], str],
    *,
    remediation: str,
) -> RuntimeCheck:
    try:
        detail = operation()
    except Exception as exc:
        return RuntimeCheck(
            name=name,
            status="fail",
            detail=str(exc) or type(exc).__name__,
            remediation=remediation,
        )
    return RuntimeCheck(name=name, status="pass", detail=detail)


def inspect_portable_runtime() -> RuntimeReadinessReport:
    """Inspect the CPU-safe package environment without requiring a GPU."""

    checks = [
        _run_check(
            "torch",
            _verify_torch_import,
            remediation="reinstall turbo-dllm in a clean virtual environment",
        )
    ]
    in_virtualenv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    user_site = str(site.getusersitepackages())
    user_site_active = bool(site.ENABLE_USER_SITE and user_site in sys.path)
    if in_virtualenv or not user_site_active:
        checks.append(
            RuntimeCheck(
                name="python_isolation",
                status="pass",
                detail=(
                    "virtual environment active"
                    if in_virtualenv
                    else "user site-packages disabled"
                ),
            )
        )
    else:
        checks.append(
            RuntimeCheck(
                name="python_isolation",
                status="warning",
                detail=f"user site-packages is active: {user_site}",
                remediation=(
                    "use python -m venv .venv to prevent stale editable installs "
                    "from shadowing the release environment"
                ),
            )
        )
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        checks.append(
            RuntimeCheck(
                name="pythonpath",
                status="warning",
                detail=f"PYTHONPATH is set: {pythonpath}",
                remediation=(
                    "unset PYTHONPATH unless repository-local imports are intentional"
                ),
            )
        )
    return RuntimeReadinessReport(tuple(checks))


def inspect_training_runtime(
    spec: Any | None = None,
    *,
    distributed: bool = False,
    full_bundle: bool = False,
    local_processes: int = 1,
) -> RuntimeReadinessReport:
    """Inspect prerequisites before model allocation or worker launch."""

    if int(local_processes) <= 0:
        raise ValueError("local_processes must be positive")
    required_devices = int(local_processes)
    checks = list(inspect_portable_runtime().checks)
    checks.append(
        _run_check(
            "cuda",
            lambda: _verify_cuda_runtime(required_devices),
            remediation=(
                "run on a host with a visible supported NVIDIA GPU and a "
                "CUDA-enabled PyTorch build"
            ),
        )
    )
    runtime_jit = bool(
        getattr(getattr(spec, "kernel", None), "runtime_jit", False)
    )
    checks.append(
        _run_check(
            "native_kernels",
            (
                _verify_jit_toolchain
                if runtime_jit
                else lambda: _verify_packaged_kernels(required_devices)
            ),
            remediation=(
                "install a CUDA development image containing nvcc"
                if runtime_jit
                else "run dllm bundle install --release <release-tag> --auto in "
                "this Python environment"
            ),
        )
    )
    checks.append(
        _run_check(
            "flash_attention_4",
            _verify_fa4,
            remediation=(
                "install the coordinated GPU bundle with "
                "dllm bundle install --release <release-tag> --auto"
            ),
        )
    )
    checks.append(
        _run_check(
            "flash_attention_3",
            _verify_fa3,
            remediation=(
                "reinstall the complete coordinated GPU bundle; do not mix "
                "FlashAttention wheels from another release"
            ),
        )
    )

    family = ""
    tensor_parallel_size = 1
    expert_parallel_size = 1
    optimizer = "auto"
    if spec is not None:
        family = str(getattr(getattr(spec, "model", None), "family", ""))
        tensor_parallel_size = int(
            getattr(getattr(spec, "topology", None), "tensor_parallel_size", 1)
        )
        expert_parallel_size = int(
            getattr(getattr(spec, "topology", None), "expert_parallel_size", 1)
        )
        optimizer = str(
            getattr(getattr(spec, "optimizer", None), "backend", "auto")
        )
        if family == "qwen3_8":
            checks.append(
                _run_check(
                    "qwen3_8_runtime",
                    _verify_qwen38,
                    remediation="install the matching turbo-dllm[qwen3_8] extra",
                )
            )
        if family == "dflash":
            checks.append(
                _run_check(
                    "dflash_runtime",
                    _verify_dflash,
                    remediation="reinstall the coordinated GPU bundle",
                )
            )
        if expert_parallel_size > 1:
            checks.append(
                _run_check(
                    "deep_ep",
                    _verify_deepep,
                    remediation=(
                        "install the qualified DeepEP binary bundle or the pinned "
                        "DeepEP source revision documented by Turbo-dLLM"
                    ),
                )
            )
        checks.extend(_data_checks(spec))
    if full_bundle or (tensor_parallel_size > 1 and family != "qwen3_8"):
        checks.append(
            _run_check(
                "transformer_engine",
                _verify_transformer_engine,
                remediation="install the matching turbo-dllm[gpu] extra",
            )
        )
    if full_bundle or optimizer == "deepspeed_zero2" or (
        optimizer == "auto" and distributed
    ):
        checks.append(
            _run_check(
                "deepspeed",
                _verify_deepspeed,
                remediation="install the matching turbo-dllm[gpu] extra",
            )
        )
    return RuntimeReadinessReport(tuple(checks))


def _verify_torch_import() -> str:
    import torch

    return f"torch {torch.__version__}"


def _verify_cuda_runtime(required_devices: int) -> str:
    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA is not available to this Python environment")
    count = int(torch.cuda.device_count())
    if count < int(required_devices):
        raise RuntimeError(
            f"launch requires {required_devices} local GPU(s), but PyTorch reports "
            f"{count} visible"
        )
    capabilities = [
        tuple(int(value) for value in torch.cuda.get_device_capability(index))
        for index in range(int(required_devices))
    ]
    formatted = ", ".join(f"sm{major}{minor}" for major, minor in capabilities)
    return (
        f"CUDA {torch.version.cuda}; {count} visible GPU(s); assigned {formatted}"
    )


def _verify_packaged_kernels(required_devices: int) -> str:
    import torch

    from dllm_parallel.core.kernels.runtime import verify_packaged_native_kernels

    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    capabilities = tuple(
        tuple(int(value) for value in torch.cuda.get_device_capability(index))
        for index in range(min(int(required_devices), visible))
    )
    names = verify_packaged_native_kernels(device_capabilities=capabilities)
    return f"{len(names)} packaged kernels verified: {', '.join(names)}"


def _verify_jit_toolchain() -> str:
    from dllm_parallel.core.kernels.runtime import verify_runtime_jit_toolchain

    return f"nvcc ready: {verify_runtime_jit_toolchain()}"


def _verify_fa4() -> str:
    from dllm_parallel.core.attention.flex import verify_flex_attention_runtime

    result = verify_flex_attention_runtime()
    return f"flash-attn-4 {result.fa4_version}"


def _verify_fa3() -> str:
    from dllm_parallel.core.attention.fa3 import verify_flash_attention_kernels

    result = verify_flash_attention_kernels()
    return f"{result.package} {result.version}; build {result.build_id}"


def _verify_qwen38() -> str:
    from dllm_parallel.core.models.backbones.qwen3_8.model import (
        verify_qwen38_runtime,
    )

    result = verify_qwen38_runtime()
    return (
        "flash-linear-attention "
        f"{result['flash_linear_attention_version']}; tilelang "
        f"{result['tilelang_version']}"
    )


def _verify_dflash() -> str:
    from dllm_parallel.core.attention.dflash_fa4 import verify_dflash_fa4_runtime

    result = verify_dflash_fa4_runtime()
    return str(result)


def _verify_transformer_engine() -> str:
    import transformer_engine.pytorch  # noqa: F401

    try:
        version = metadata.version("transformer-engine")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return f"transformer-engine {version}"


def _verify_deepspeed() -> str:
    from dllm_parallel.core.optim.zero import verify_deepspeed_runtime_available

    verify_deepspeed_runtime_available()
    return f"deepspeed {metadata.version('deepspeed')}"


def _verify_deepep() -> str:
    from dllm_parallel.core.parallel.expert.deepep import validate_deepep_install

    result = validate_deepep_install()
    return f"deep-ep {result.deep_ep_version or 'unknown'}"


def _data_checks(spec: Any) -> tuple[RuntimeCheck, ...]:
    data = getattr(spec, "data", None)
    dataset_fix = (
        "prepare the dataset with dllm data prepare and update the RunSpec path"
    )
    paths: list[tuple[str, str | None, str]] = []
    if str(getattr(data, "input_mode", "")) == "dataset":
        paths.append(("dataset", getattr(data, "dataset_path", None), dataset_fix))
    if str(getattr(data, "target_features", "")) == "offline":
        paths.append(
            (
                "target_features",
                getattr(data, "target_feature_path", None),
                "generate the offline target features and update data.target_feature_path",
            )
        )
    evaluation = getattr(spec, "evaluation", None)
    evaluation_path = getattr(evaluation, "dataset_path", None)
    if evaluation_path:
        paths.append(("evaluation_dataset", evaluation_path, dataset_fix))
    model = getattr(spec, "model", None)
    draft_vocab_path = getattr(model, "draft_vocab_path", None)
    if draft_vocab_path:
        paths.append(
            (
                "draft_vocab",
                draft_vocab_path,
                "generate the draft vocabulary artifact and update model.draft_vocab_path",
            )
        )
    checkpointing = getattr(spec, "checkpointing", None)
    load_checkpoint_dir = getattr(checkpointing, "load_checkpoint_dir", None)
    if load_checkpoint_dir:
        paths.append(
            (
                "load_checkpoint",
                load_checkpoint_dir,
                "set checkpointing.load_checkpoint_dir to an existing checkpoint",
            )
        )
    results: list[RuntimeCheck] = []
    for name, value, remediation in paths:
        path = Path(str(value)).expanduser() if value else None
        if path is not None and path.exists():
            results.append(
                RuntimeCheck(name=name, status="pass", detail=str(path.resolve()))
            )
        else:
            results.append(
                RuntimeCheck(
                    name=name,
                    status="fail",
                    detail=f"configured path does not exist: {value!r}",
                    remediation=remediation,
                )
            )
    return tuple(results)


__all__ = (
    "RuntimeCheck",
    "RuntimeReadinessReport",
    "inspect_portable_runtime",
    "inspect_training_runtime",
)
