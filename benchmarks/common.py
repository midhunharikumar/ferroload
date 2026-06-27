"""Shared benchmark utilities: timing, peak-RAM, dir sizes, result records.

Fairness principle: every format is built from the SAME encoded image bytes, and
every loader decodes to the SAME target (decode -> optional resize -> uint8 HWC).
So the per-sample CPU work is identical across loaders; only the format + loader
machinery differ.
"""
import io
import json
import os
import resource
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
from PIL import Image


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def peak_rss_bytes() -> int:
    """Peak resident set of THIS process. On macOS ru_maxrss is bytes; on Linux KB.
    (Worker subprocesses aren't included — noted as a caveat in the report.)"""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return rss if sys.platform == "darwin" else rss * 1024


def decode_resize(img_bytes: bytes, target: Optional[int]) -> np.ndarray:
    """Decode encoded image -> RGB uint8 HWC, optionally resized to target x target.
    The canonical, identical-across-loaders transform."""
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if target is not None and im.size != (target, target):
        im = im.resize((target, target), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


@dataclass
class RunResult:
    dataset: str
    fmt: str            # hf_arrow | webdataset | ferroload
    loader: str         # e.g. "torch_dl" | "webloader" | "ferro_native"
    num_workers: int
    batch_size: int
    target_px: Optional[int]
    n_samples: int
    seconds: float
    samples_per_s: float
    first_batch_s: float
    peak_rss_mb: float
    notes: str = ""

    def row(self):
        return asdict(self)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t


def run_loader(iterable_batches, *, warmup_batches=2, max_samples=20000,
               sample_count_fn=len):
    """Drive a batch iterator: skip `warmup_batches` (warms page cache / spins up
    workers), then time until `max_samples` consumed. Returns (n, seconds,
    first_batch_s). `sample_count_fn(batch)` returns the batch's sample count."""
    it = iter(iterable_batches)

    # first-batch latency (cold): time to produce the very first batch
    t0 = time.perf_counter()
    try:
        first = next(it)
    except StopIteration:
        return 0, 0.0, 0.0
    first_batch_s = time.perf_counter() - t0
    consumed = sample_count_fn(first)

    # additional warmup
    for _ in range(max(0, warmup_batches - 1)):
        try:
            consumed += sample_count_fn(next(it))
        except StopIteration:
            break

    # timed steady-state pass
    n = 0
    t = time.perf_counter()
    for batch in it:
        n += sample_count_fn(batch)
        if n >= max_samples:
            break
    seconds = time.perf_counter() - t
    return n, seconds, first_batch_s


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [r.row() if isinstance(r, RunResult) else r for r in results]
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return path
