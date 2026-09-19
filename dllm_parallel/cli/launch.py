"""Package-native local and distributed training launcher."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dllm_parallel.core.diagnostics import inspect_training_runtime
from dllm_parallel.core.parallel.preflight import require_idle_assigned_gpus
from dllm_parallel.recipes import recipe_text


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _node_range(value: int | str) -> tuple[int, int]:
    parts = str(value).split(":")
    if len(parts) not in {1, 2} or any(not part.isdigit() for part in parts):
        raise ValueError("nnodes must be a positive integer or range such as 1:4")
    minimum = int(parts[0])
    maximum = int(parts[-1])
    if minimum <= 0 or maximum <= 0 or minimum > maximum:
        raise ValueError(
            "nnodes must be positive and ranges must satisfy minimum <= maximum"
        )
    return minimum, maximum


def build_launch_command(
    *,
    config: str | Path,
    trainer_args: Sequence[str],
    nproc_per_node: int = 1,
    nnodes: int | str = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    rdzv_backend: str | None = None,
    rdzv_endpoint: str | None = None,
    rdzv_id: str | None = None,
    max_restarts: int = 0,
    monitor_interval: float = 5.0,
) -> list[str]:
    """Build an argv-safe command using only the installed package."""

    _positive("nproc-per-node", nproc_per_node)
    minimum_nodes, maximum_nodes = _node_range(nnodes)
    _positive("monitor-interval", monitor_interval)
    if node_rank < 0 or node_rank >= maximum_nodes:
        raise ValueError("node-rank must be non-negative and smaller than max nnodes")
    if not 1 <= master_port <= 65535:
        raise ValueError("master-port must be between 1 and 65535")
    if max_restarts < 0:
        raise ValueError("max-restarts must be non-negative")
    rendezvous = (rdzv_backend, rdzv_endpoint, rdzv_id)
    if any(value is not None for value in rendezvous) and not all(
        value is not None for value in rendezvous
    ):
        raise ValueError(
            "rdzv-backend, rdzv-endpoint, and rdzv-id must be supplied together"
        )
    if minimum_nodes != maximum_nodes and rdzv_backend is None:
        raise ValueError(
            "elastic nnodes range requires rdzv-backend, rdzv-endpoint, and rdzv-id"
        )
    if any(value == "" for value in rendezvous if value is not None):
        raise ValueError("rendezvous values must not be empty")
    if not master_addr:
        raise ValueError("master-addr must not be empty")

    training = [
        "--module",
        "dllm_parallel.training",
        "--config",
        str(config),
        *map(str, trainer_args),
    ]
    use_torchrun = (
        nproc_per_node > 1
        or maximum_nodes > 1
        or minimum_nodes != maximum_nodes
        or rdzv_backend is not None
        or max_restarts > 0
    )
    if not use_torchrun:
        return [sys.executable, "-m", "dllm_parallel.training", *training[2:]]

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={nproc_per_node}",
        f"--node-rank={node_rank}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        f"--max-restarts={max_restarts}",
        f"--monitor-interval={monitor_interval:g}",
    ]
    if rdzv_backend is not None:
        command.extend(
            (
                f"--rdzv-backend={rdzv_backend}",
                f"--rdzv-endpoint={rdzv_endpoint}",
                f"--rdzv-id={rdzv_id}",
            )
        )
    command.extend(training)
    return command


def build_launch_environment(
    *,
    run_dir: str | Path,
    cache_dir: str | Path,
    environ: Mapping[str, str] | None = None,
    force_cache_locations: bool = False,
) -> dict[str, str]:
    """Return a child environment without repository path injection."""

    environment = dict(os.environ if environ is None else environ)
    run_root = str(Path(run_dir).expanduser().resolve())
    cache_root = str(Path(cache_dir).expanduser().resolve())
    environment["DLLM_RUN_DIR"] = run_root
    environment["DLLM_CACHE_DIR"] = cache_root
    environment.setdefault("USER", getpass.getuser() or "dllm")
    environment.setdefault("LOGNAME", environment["USER"])
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    python_bin = str(Path(sys.executable).parent)
    path_entries = environment.get("PATH", "").split(os.pathsep)
    if python_bin not in path_entries:
        environment["PATH"] = os.pathsep.join(
            [python_bin, *(entry for entry in path_entries if entry)]
        )
    cache_values = {
        "XDG_CACHE_HOME": cache_root,
        "HF_HOME": str(Path(cache_root) / "hf"),
        "HF_HUB_CACHE": str(Path(cache_root) / "hf" / "hub"),
        "HF_DATASETS_CACHE": str(Path(cache_root) / "hf" / "datasets"),
        "HF_MODULES_CACHE": str(Path(cache_root) / "hf" / "modules"),
        "TORCHINDUCTOR_CACHE_DIR": str(Path(cache_root) / "torchinductor"),
        "TORCH_EXTENSIONS_DIR": str(Path(cache_root) / "torch_extensions"),
        "TRITON_CACHE_DIR": str(Path(cache_root) / "triton"),
    }
    for name, value in cache_values.items():
        if force_cache_locations:
            environment[name] = value
        else:
            environment.setdefault(name, value)
    return environment


def _validate_config(config: Path, trainer_args: Sequence[str]):
    if not config.is_file():
        raise ValueError(f"config file does not exist: {config}")
    from dllm_parallel.training.run_spec import preflight_run_spec_from_argv

    try:
        return preflight_run_spec_from_argv(
            ["--config", str(config), *map(str, trainer_args)]
        )
    except SystemExit as exc:
        raise ValueError("invalid training arguments") from exc
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        raise ValueError(f"invalid RunSpec YAML in {config}: {problem}") from exc


def _validate_launch_world_sizes(
    spec,
    *,
    nnodes: int | str,
    nproc_per_node: int,
) -> None:
    from dllm_parallel.core.parallel.preflight import infer_parallel_mesh_sizes

    minimum_nodes, maximum_nodes = _node_range(nnodes)
    node_counts = {minimum_nodes, maximum_nodes}
    if minimum_nodes < maximum_nodes:
        node_counts.add(minimum_nodes + 1)
    for nodes in sorted(node_counts):
        world_size = nodes * nproc_per_node
        try:
            infer_parallel_mesh_sizes(
                world_size=world_size,
                context_parallel_size=spec.topology.context_parallel_size,
                block_parallel_size=spec.topology.block_parallel_size,
                tensor_parallel_size=spec.topology.tensor_parallel_size,
                expert_parallel_size=spec.topology.expert_parallel_size,
            )
        except ValueError as exc:
            raise ValueError(
                f"launch world_size={world_size} ({nodes} node(s) x "
                f"{nproc_per_node} process(es)) is incompatible with RunSpec "
                f"topology: {exc}"
            ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dllm launch",
        description="Launch training from an installed Turbo-dLLM package.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", help="Packaged recipe logical name.")
    source.add_argument("--config", type=Path, help="RunSpec YAML path.")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--nnodes", default="1")
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--rdzv-backend")
    parser.add_argument("--rdzv-endpoint")
    parser.add_argument("--rdzv-id")
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument(
        "--gpu-preflight",
        choices=("auto", "idle", "off"),
        default="auto",
        help="Check assigned GPUs: auto under Slurm, always, or never.",
    )
    parser.add_argument(
        "--runtime-preflight",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Verify CUDA, native artifacts, and config-specific runtimes before "
            "starting workers (default: auto)."
        ),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _execute(
    args: argparse.Namespace, trainer_args: Sequence[str], config: Path
) -> int:
    # Validate launcher topology before touching user configuration or the filesystem.
    command = build_launch_command(
        config=config,
        trainer_args=trainer_args,
        nproc_per_node=args.nproc_per_node,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        rdzv_backend=args.rdzv_backend,
        rdzv_endpoint=args.rdzv_endpoint,
        rdzv_id=args.rdzv_id,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
    )
    spec = _validate_config(config, trainer_args)
    _validate_launch_world_sizes(
        spec,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.run_dir
        or (
            Path(os.environ["DLLM_RUN_DIR"]) if os.environ.get("DLLM_RUN_DIR") else None
        )
        or Path.cwd() / "runs" / timestamp
    )
    if args.cache_dir is not None:
        cache_dir = args.cache_dir
    elif os.environ.get("DLLM_CACHE_DIR"):
        cache_dir = Path(os.environ["DLLM_CACHE_DIR"])
    elif os.environ.get("XDG_CACHE_HOME"):
        cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "dllm_parallel"
    else:
        cache_dir = Path.home() / ".cache" / "dllm_parallel"
    environment = build_launch_environment(
        run_dir=run_dir,
        cache_dir=cache_dir,
        force_cache_locations=args.cache_dir is not None,
    )

    print(f"DLLM_RUN_DIR={environment['DLLM_RUN_DIR']}")
    print(f"DLLM_CACHE_DIR={environment['DLLM_CACHE_DIR']}")
    print(f"DLLM_COMMAND={shlex.join(command)}")
    if args.dry_run:
        return 0

    if args.runtime_preflight == "auto":
        _, maximum_nodes = _node_range(args.nnodes)
        inspect_training_runtime(
            spec,
            distributed=(maximum_nodes * args.nproc_per_node > 1),
            local_processes=args.nproc_per_node,
        ).require_ready()
        environment["DLLM_RUNTIME_PREFLIGHT_DONE"] = "1"
    else:
        environment["DLLM_RUNTIME_PREFLIGHT"] = "off"

    Path(environment["DLLM_RUN_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(environment["DLLM_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    if args.gpu_preflight == "idle" or (
        args.gpu_preflight == "auto" and environment.get("SLURM_JOB_ID")
    ):
        require_idle_assigned_gpus(local_processes=args.nproc_per_node)
    try:
        completed = subprocess.run(command, check=False, env=environment)
    except KeyboardInterrupt:
        return 130
    return (
        int(completed.returncode)
        if completed.returncode >= 0
        else 128 + abs(int(completed.returncode))
    )


def run_launch(argv: Sequence[str]) -> int:
    """Parse launcher arguments, run the child, and propagate its exit status."""

    args, trainer_args = _parser().parse_known_args(list(argv))
    if trainer_args[:1] == ["--"]:
        trainer_args = trainer_args[1:]
    if args.recipe:
        with tempfile.TemporaryDirectory(prefix="dllm-recipe-") as directory:
            config = Path(directory) / "recipe.yaml"
            config.write_text(recipe_text(args.recipe), encoding="utf-8")
            return _execute(args, trainer_args, config)
    return _execute(args, trainer_args, args.config)


__all__ = (
    "build_launch_command",
    "build_launch_environment",
    "run_launch",
)
