"""Distributed-map executors (DESIGN §14.4).

The map backend is abstracted behind a tiny `Executor` interface; user code never
names a backend. The launch **topology** is auto-detected from the environment
(torchrun / SLURM / MPI / Ray / bare) and selects:

  - `LocalExecutor`            — single node, zero overhead (multiprocessing/Rust
                                 threadpool); the common case.
  - `StaticPartitionExecutor`  — torchrun/SLURM: each rank computes a deterministic,
                                 disjoint `sample_id` partition, writes its own
                                 layer **fragment**, then a single commit merges +
                                 registers (the one sync point).
  - `RayExecutor`              — multi-node via Ray (optional dependency).

`FERROLOAD_EXECUTOR=local|static|ray` overrides the auto-selection.

A map is **embarrassingly parallel (map) + a trivial reduce (manifest merge)** —
no shuffle (DESIGN §14.3). The unit of work is a sample partition; each worker
streams its inputs (projection), computes, and writes only the new modality.
"""
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from ._core import LayerWriter


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #
@dataclass
class Topology:
    """Launch topology. `num_nodes` is THE deciding variable (local vs distributed);
    `local_size` sets intra-node parallelism."""
    num_nodes: int
    node_rank: int
    local_size: int
    world_size: int
    rank: int
    source: str = "bare"


def _int(name, default=None):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _local_size():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x != ""]) or 1
    return os.cpu_count() or 1


def detect_topology():
    """Derive `Topology` from launcher env vars, in priority order (DESIGN §14.4)."""
    # torchrun / PyTorch elastic
    if _int("WORLD_SIZE") is not None and (
        "TORCHELASTIC_RUN_ID" in os.environ
        or "LOCAL_WORLD_SIZE" in os.environ
        or _int("RANK") is not None
    ):
        world = _int("WORLD_SIZE", 1)
        local = _int("LOCAL_WORLD_SIZE", world) or world
        rank = _int("RANK", 0)
        nnodes = _int("NNODES") or max(1, world // max(local, 1))
        node_rank = _int("GROUP_RANK", _int("NODE_RANK", rank // max(local, 1)))
        return Topology(nnodes, node_rank, local, world, rank, "torchrun")
    # SLURM
    if _int("SLURM_NTASKS") is not None or _int("SLURM_PROCID") is not None:
        nnodes = _int("SLURM_NNODES", 1)
        world = _int("SLURM_NTASKS") or nnodes
        local = _int("SLURM_NTASKS_PER_NODE") or max(1, world // max(nnodes, 1))
        rank = _int("SLURM_PROCID", 0)
        node_rank = _int("SLURM_NODEID", 0)
        return Topology(nnodes, node_rank, local, world, rank, "slurm")
    # MPI
    world = _int("OMPI_COMM_WORLD_SIZE") or _int("PMI_SIZE")
    if world is not None:
        rank = _int("OMPI_COMM_WORLD_RANK") or _int("PMI_RANK") or 0
        local = _int("OMPI_COMM_WORLD_LOCAL_SIZE") or world
        nnodes = max(1, world // max(local, 1))
        return Topology(nnodes, rank // max(local, 1), local, world, rank, "mpi")
    # Ray (sizes resolved by Ray at runtime; treat as distributed)
    if os.environ.get("RAY_ADDRESS"):
        ls = _local_size()
        return Topology(2, 0, ls, ls, 0, "ray")
    # bare / single process
    ls = _local_size()
    return Topology(1, 0, ls, ls, 0, "bare")


def _ray_available():
    if os.environ.get("RAY_ADDRESS"):
        return True
    try:
        import ray  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Plan + executors
# --------------------------------------------------------------------------- #
@dataclass
class MapPlan:
    """A unit of map work handed to an `Executor`: where to write the layer, how
    to declare it, and a `process(writer, positions) -> n_written` closure that
    reads inputs, runs the user fn, and writes rows."""
    root: str
    layer_name: str
    modalities: dict                       # {name: (ext, kind, codec)} for LayerWriter
    total: int
    process: Callable                       # (writer, positions) -> int
    sample_id: Callable                     # position -> global sample_id
    progress: bool = False
    _meta: dict = field(default_factory=dict)


class Executor:
    """Map backend. `run(plan)` writes (and, for distributed, commits) the layer."""
    def run(self, plan: MapPlan) -> int:        # pragma: no cover - interface
        raise NotImplementedError


class LocalExecutor(Executor):
    """Single-process / single-node. Writes the whole layer and registers it.
    Decode of inputs is already parallel across cores inside Rust (GIL released)."""
    def __init__(self, topo: Optional[Topology] = None):
        self.topo = topo or detect_topology()

    def run(self, plan: MapPlan) -> int:
        writer = LayerWriter(plan.root, plan.layer_name, plan.modalities)
        written = plan.process(writer, list(range(plan.total)))
        if written > 0:
            writer.close()                      # register/append (skip no-op resume)
        return written


class StaticPartitionExecutor(Executor):
    """torchrun/SLURM: deterministic, disjoint partition per rank, no queue. Each
    rank writes its own fragment; a single commit merges + registers. If
    `torch.distributed` is initialized it barriers and rank 0 commits; with a
    single rank it commits directly; otherwise it writes the fragment and asks you
    to run `ferroload.commit_layer(...)` once every rank has finished."""
    def __init__(self, topo: Optional[Topology] = None):
        self.topo = topo or detect_topology()

    def run(self, plan: MapPlan) -> int:
        t = self.topo
        world, rank = max(t.world_size, 1), t.rank
        # disjoint, reproducible partition of the dense sample_id space
        positions = [p for p in range(plan.total) if plan.sample_id(p) % world == rank]
        writer = LayerWriter(plan.root, plan.layer_name, plan.modalities, partition=rank)
        written = plan.process(writer, positions)
        writer.close()                           # always write fragment + .done marker
        self._commit(plan)
        return written

    def _commit(self, plan: MapPlan):
        t = self.topo
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                if t.rank == 0:
                    LayerWriter.commit(plan.root, plan.layer_name, plan.modalities)
                dist.barrier()
                return
        except Exception:
            pass
        if max(t.world_size, 1) <= 1:
            LayerWriter.commit(plan.root, plan.layer_name, plan.modalities)
        else:
            print(
                f"[ferroload] rank {t.rank}/{t.world_size}: wrote layer fragment "
                f"'{plan.layer_name}'. Run "
                f"ferroload.commit_layer({plan.root!r}, {plan.layer_name!r}, modalities) "
                f"once all ranks finish (no torch.distributed barrier available).",
                flush=True,
            )


class RayExecutor(Executor):
    """Multi-node via Ray (Ray Data blocks == our shards). Optional dependency."""
    def __init__(self, topo: Optional[Topology] = None):
        self.topo = topo or detect_topology()

    def run(self, plan: MapPlan) -> int:        # pragma: no cover - needs a cluster
        raise NotImplementedError(
            "RayExecutor is not implemented in this build. Run the map under "
            "torchrun/SLURM (StaticPartitionExecutor) or on a single node "
            "(LocalExecutor), or set FERROLOAD_EXECUTOR=local."
        )


_BY_NAME = {
    "local": LocalExecutor,
    "static": StaticPartitionExecutor,
    "static_partition": StaticPartitionExecutor,
    "ray": RayExecutor,
}


def select_executor(topo: Optional[Topology] = None, override: Optional[str] = None) -> Executor:
    """Pick an executor (DESIGN §14.4 selection rule). `override` or
    `FERROLOAD_EXECUTOR` forces the choice; otherwise `num_nodes` decides."""
    topo = topo or detect_topology()
    choice = override or os.environ.get("FERROLOAD_EXECUTOR")
    if choice:
        cls = _BY_NAME.get(choice.lower())
        if cls is None:
            raise ValueError(f"unknown FERROLOAD_EXECUTOR={choice!r}; use local|static|ray")
        return cls(topo)
    if topo.num_nodes <= 1:
        return LocalExecutor(topo)
    if _ray_available():
        return RayExecutor(topo)
    if topo.source in ("torchrun", "slurm", "mpi"):
        return StaticPartitionExecutor(topo)
    raise RuntimeError(
        "multi-node topology detected but no Ray/torchrun/SLURM backend found; "
        "set FERROLOAD_EXECUTOR=local|static|ray"
    )


def commit_layer(root, name, modalities=None):
    """Merge partition fragments into a layer and register it (the manual commit
    step for a distributed map without a torch.distributed barrier)."""
    return LayerWriter.commit(root, name, modalities)
