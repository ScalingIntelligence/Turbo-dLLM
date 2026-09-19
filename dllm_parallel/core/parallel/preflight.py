# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Startup validation for production DLLM parallel training."""

from __future__ import annotations

import argparse
import csv
import os
import pwd
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class ParallelMeshSizes:
    data_parallel_size: int
    model_parallel_size: int
    local_parallel_size: int


@dataclass(frozen=True)
class GPUProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int | None
    owner: str | None


CommandRunner = Callable[[Sequence[str]], str]


def prepare_parallel_environment(
    *,
    sequence_parallel: bool,
    context_parallel_size: int = 1,
    block_parallel_size: int = 1,
    tensor_parallel_size: int,
    expert_parallel_size: int = 1,
    tensor_parallel_overlap: bool,
) -> None:
    """Apply typed launch policy before CUDA context creation."""

    if sequence_parallel:
        _require_cuda_allocator_option("expandable_segments:True")
    requires_ordered_overlap = (
        (
            int(tensor_parallel_size) > 1
            and (bool(sequence_parallel) or bool(tensor_parallel_overlap))
        )
        or int(context_parallel_size) > 1
        or int(block_parallel_size) > 1
        or int(expert_parallel_size) > 1
    )
    if requires_ordered_overlap:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    else:
        os.environ.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
    if int(expert_parallel_size) > 1:
        _validate_expert_parallel_environment()


def launch_environment_metadata() -> dict[str, str | None]:
    """Return launch environment values that affect distributed performance."""

    names = (
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUDA_HOME",
        "PYTORCH_CUDA_ALLOC_CONF",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_ID",
        "CUDA_VISIBLE_DEVICES",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "DLLM_BLOCK_SCHEDULE_POLICY",
    )
    return {name: os.environ.get(name) for name in names}


def launch_environment_policy(
    *,
    sequence_parallel: bool,
    context_parallel_size: int,
    block_parallel_size: int,
    tensor_parallel_size: int,
    expert_parallel_size: int = 1,
    tensor_parallel_overlap: bool,
) -> dict[str, object]:
    """Return the typed launch environment policy implied by topology."""

    requires_expandable_segments = bool(sequence_parallel)
    requires_cuda_device_max_connections_one = (
        (
            int(tensor_parallel_size) > 1
            and (bool(sequence_parallel) or bool(tensor_parallel_overlap))
        )
        or int(context_parallel_size) > 1
        or int(block_parallel_size) > 1
        or int(expert_parallel_size) > 1
    )
    return {
        "requires_pytorch_cuda_alloc_expandable_segments": (
            requires_expandable_segments
        ),
        "requires_cuda_device_max_connections_one": (
            requires_cuda_device_max_connections_one
        ),
        "source": "RunSpec.topology",
    }


def validate_supported_training_axes(
    *,
    pipeline_parallel_size: int = 1,
) -> None:
    """Reject axes that are represented in planning but not in the trainer."""

    if int(pipeline_parallel_size) != 1:
        raise ValueError(
            "pipeline_parallel_size > 1 is not implemented in the DLLM trainer"
        )


def infer_parallel_mesh_sizes(
    *,
    world_size: int,
    context_parallel_size: int,
    block_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    data_parallel_size: int | None = None,
) -> ParallelMeshSizes:
    """Validate and infer DP size from the fused CP/BP/TP mesh."""

    world_size = _positive("world_size", world_size)
    context_parallel_size = _positive("context_parallel_size", context_parallel_size)
    block_parallel_size = _positive("block_parallel_size", block_parallel_size)
    tensor_parallel_size = _positive("tensor_parallel_size", tensor_parallel_size)
    pipeline_parallel_size = _positive("pipeline_parallel_size", pipeline_parallel_size)
    expert_parallel_size = _positive("expert_parallel_size", expert_parallel_size)
    if not _supported_local_layout(context_parallel_size, block_parallel_size):
        raise ValueError(
            "v1 supports BP-only (context_parallel_size=1), CP-only "
            "(block_parallel_size=1), or fused CP/BP with "
            "block_parallel_size >= context_parallel_size and "
            "block_parallel_size divisible by context_parallel_size"
        )
    local_parallel_size = max(context_parallel_size, block_parallel_size)
    model_parallel_size = (
        local_parallel_size
        * tensor_parallel_size
        * pipeline_parallel_size
        * expert_parallel_size
    )
    if data_parallel_size is None:
        if world_size % model_parallel_size != 0:
            raise ValueError(
                "world_size must divide evenly by local_parallel_size * "
                "tensor_parallel_size * pipeline_parallel_size * "
                "expert_parallel_size"
            )
        data_parallel_size = world_size // model_parallel_size
    data_parallel_size = _positive("data_parallel_size", data_parallel_size)
    if world_size != data_parallel_size * model_parallel_size:
        raise ValueError(
            "world_size must equal data_parallel_size * model_parallel_size"
        )
    return ParallelMeshSizes(
        data_parallel_size=data_parallel_size,
        model_parallel_size=model_parallel_size,
        local_parallel_size=local_parallel_size,
    )


def _require_cuda_allocator_option(option: str) -> None:
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if current is None or current.strip() == "":
        raise ValueError(
            "sequence-parallel training requires "
            f"PYTORCH_CUDA_ALLOC_CONF to include {option!r} before launch"
        )
    key = option.split(":", maxsplit=1)[0]
    parts = [part.strip() for part in current.split(",") if part.strip()]
    if any(part.split(":", maxsplit=1)[0] == key for part in parts):
        return
    raise ValueError(
        "sequence-parallel training requires PYTORCH_CUDA_ALLOC_CONF to include "
        f"{option!r} before launch"
    )


def _validate_expert_parallel_environment() -> None:
    from dllm_parallel.core.parallel.expert import validate_deepep_install

    validate_deepep_install()


def _supported_local_layout(
    context_parallel_size: int,
    block_parallel_size: int,
) -> bool:
    if context_parallel_size == 1 or block_parallel_size == 1:
        return True
    return (
        block_parallel_size >= context_parallel_size
        and block_parallel_size % context_parallel_size == 0
    )


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_idle_assigned_gpus(
    *,
    local_processes: int,
    environ: Mapping[str, str] | None = None,
    command_runner: CommandRunner | None = None,
) -> None:
    """Fail before launch when an assigned GPU already has a compute process."""

    local_processes = _positive("local_processes", local_processes)
    environment = os.environ if environ is None else environ
    runner = _run_command if command_runner is None else command_runner
    inventory = _parse_gpu_inventory(
        runner(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            )
        )
    )
    assigned = _assigned_gpu_uuids(
        inventory,
        environment=environment,
        local_processes=local_processes,
    )
    processes = _parse_gpu_processes(
        runner(
            (
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            )
        )
    )
    conflicts = [process for process in processes if process.gpu_uuid in assigned]
    if not conflicts:
        return
    def memory_text(process: GPUProcess) -> str:
        return (
            str(process.used_memory_mib)
            if process.used_memory_mib is not None
            else "unknown"
        )

    details = "; ".join(
        f"gpu={process.gpu_uuid} pid={process.pid} "
        f"owner={process.owner or 'unknown'} process={process.process_name} "
        f"memory_mib={memory_text(process)}"
        for process in conflicts
    )
    raise RuntimeError(
        "assigned GPUs are not idle before worker launch: "
        f"{details}. Request an exclusive allocation or clean the assigned GPUs; "
        "the launcher will not terminate another process."
    )


def _run_command(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"GPU launch preflight failed to execute {' '.join(command)!r}: {exc}"
        ) from exc
    return result.stdout


def _parse_gpu_inventory(output: str) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for row in csv.reader(output.splitlines()):
        if len(row) < 2:
            continue
        index, gpu_uuid = (field.strip() for field in row[:2])
        if index and gpu_uuid:
            inventory[index] = gpu_uuid
            inventory[gpu_uuid] = gpu_uuid
    if not inventory:
        raise RuntimeError("GPU launch preflight found no GPUs in nvidia-smi output")
    return inventory


def _assigned_gpu_uuids(
    inventory: Mapping[str, str],
    *,
    environment: Mapping[str, str],
    local_processes: int,
) -> frozenset[str]:
    selector = next(
        (
            environment[name]
            for name in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS")
            if environment.get(name, "").strip()
        ),
        "",
    )
    if selector:
        identifiers = [part.strip() for part in selector.split(",") if part.strip()]
    else:
        numeric_indices = sorted(
            (int(key), key) for key in inventory if key.isdigit()
        )
        identifiers = [key for _, key in numeric_indices[:local_processes]]
    if len(identifiers) < local_processes:
        raise RuntimeError(
            f"GPU launch preflight resolved {len(identifiers)} assigned GPUs for "
            f"{local_processes} local processes"
        )
    unknown = [identifier for identifier in identifiers if identifier not in inventory]
    if unknown:
        raise RuntimeError(
            "GPU launch preflight could not map assigned GPU identifiers from "
            f"CUDA/Slurm environment: {unknown}"
        )
    return frozenset(inventory[identifier] for identifier in identifiers)


def _parse_gpu_processes(output: str) -> list[GPUProcess]:
    processes: list[GPUProcess] = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 3:
            continue
        gpu_uuid, pid_text, process_name = (field.strip() for field in row[:3])
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        memory_mib: int | None = None
        if len(row) >= 4:
            try:
                memory_mib = int(row[3].strip())
            except ValueError:
                pass
        processes.append(
            GPUProcess(
                gpu_uuid=gpu_uuid,
                pid=pid,
                process_name=process_name,
                used_memory_mib=memory_mib,
                owner=_process_owner(pid),
            )
        )
    return processes


def _process_owner(pid: int) -> str | None:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (FileNotFoundError, KeyError, PermissionError):
        return None


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-idle-gpus", action="store_true")
    parser.add_argument("--local-processes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.require_idle_gpus:
        require_idle_assigned_gpus(local_processes=args.local_processes)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except (RuntimeError, ValueError) as exc:
        print(f"dllm launch preflight error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
