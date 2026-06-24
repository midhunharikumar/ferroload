"""Deprecated shim. The canonical module is `ferroload.loader` (in the package).

Kept so older scripts importing `ferroload_loader` keep working once the
`ferroload` package is installed (`maturin develop`/wheel). Prefer:

    from ferroload import loader
"""
from ferroload.loader import (  # noqa: F401
    FerroTorchDataset,
    subset_dataset,
)
