"""Ferroload — pure-Rust multimodal dataset format + loader.

The compiled extension lives at `ferroload._core`; its classes are re-exported
here, and the PyTorch glue is in `ferroload.loader`.
"""
from ._core import Writer, Sampler, LayerWriter, __version__
from . import loader, executor
from .loader import (
    FerroTorchDataset, FerroIterableDataset, subset_dataset,
    FerroSampler, PrefetchLoader, batched, numpy_collate,
    FerroLoader, make_loader,
)
from .dataset import Dataset, Modality, Annotation   # Python handle + map output types
from .executor import (
    Topology, detect_topology, select_executor, commit_layer,
    Executor, LocalExecutor, StaticPartitionExecutor, RayExecutor,
)

# `Dataset`/`Writer` are the canonical names. The `Ferro`-prefixed names are kept
# as deprecated back-compat aliases (the prefix stutters inside the package).
FerroDataset = Dataset
FerroWriter = Writer

__all__ = [
    "Dataset", "Writer", "FerroDataset", "FerroWriter", "Sampler", "LayerWriter",
    "Modality", "Annotation",
    "FerroTorchDataset", "FerroIterableDataset", "subset_dataset",
    "FerroSampler", "PrefetchLoader", "batched", "numpy_collate",
    "FerroLoader", "make_loader",
    "Topology", "detect_topology", "select_executor", "commit_layer",
    "Executor", "LocalExecutor", "StaticPartitionExecutor", "RayExecutor",
    "loader", "executor", "__version__",
]
