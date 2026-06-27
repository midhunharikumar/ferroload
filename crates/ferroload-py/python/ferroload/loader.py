"""PyTorch glue for Ferroload — idiomatic: the Dataset yields **per-sample items**
and a normal `DataLoader` batches them with its `collate_fn` (default or custom).
There is no special "collate" method; batching is PyTorch's job.

The performance trick (parallel Rust decode + GIL-released I/O) lives *under*
PyTorch's batched-fetch hook `__getitems__`, so it's transparent: a standard
DataLoader gets the fast path automatically, and any custom `collate_fn` still
works because `__getitems__` returns a plain list of sample dicts.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

# Subclass torch's IterableDataset when torch is present (so a DataLoader treats
# the streaming view as iterable via its isinstance check); fall back to `object`
# so the module still imports without torch.
try:
    from torch.utils.data import IterableDataset as _IterableBase
except Exception:  # pragma: no cover - torch optional
    _IterableBase = object


def subset_dataset(tds, indices):
    """Restrict a `FerroTorchDataset` to `indices` — e.g. the result of
    `ds.subset("label < 3 AND split='train'")`. Uses `torch.utils.data.Subset`
    when torch is available (which forwards the batched `__getitems__` fast path),
    else a lightweight index-remapping wrapper.

        ids = ds.subset("split='train'")
        train = subset_dataset(FerroTorchDataset(ds, ...), ids)
        DataLoader(train, batch_size=64)
    """
    indices = list(indices)
    try:
        from torch.utils.data import Subset
        return Subset(tds, indices)
    except ImportError:
        return _IndexRemap(tds, indices)


class _IndexRemap:
    """torch-free fallback for `subset_dataset` (keeps the batched fast path)."""

    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]

    def __getitems__(self, idx):
        return self.base.__getitems__([self.indices[k] for k in idx])


def _to_torch(sample: dict) -> dict:
    import torch
    out = {}
    for k, v in sample.items():
        if hasattr(v, "dtype") and hasattr(v, "shape"):   # numpy array
            out[k] = torch.from_numpy(v)
        else:
            out[k] = v
    return out


def _to_jax(sample: dict) -> dict:
    import jax.numpy as jnp
    out = {}
    for k, v in sample.items():
        if hasattr(v, "dtype") and hasattr(v, "shape"):   # numpy array
            out[k] = jnp.asarray(v)
        else:
            out[k] = v
    return out


def _resolve_columns(fds, columns):
    """Bucket a flat list of column names into (images, videos, arrays, raw, meta)
    by consulting the dataset's declared modalities (`<name>: {ext,kind,codec}`):
    image/video codecs decode, `.npy` tensor columns load as arrays, other
    modalities pass as raw bytes, and names that aren't modalities are metadata keys."""
    mods = fds.modalities()
    images, videos, arrays, raw, meta = [], [], [], [], []
    for c in columns:
        m = mods.get(c)
        codec = m.get("codec") if m else None
        if m is None:
            meta.append(c)
        elif codec == "image":
            images.append(c)
        elif codec == "video":
            videos.append(c)
        elif codec == "npy":
            arrays.append(c)
        else:
            raw.append(c)
    return images, videos, arrays, raw, meta


def _load_npy_list(raws):
    import io
    import numpy as np
    return [None if r is None else np.load(io.BytesIO(r), allow_pickle=False) for r in raws]


class FerroTorchDataset:
    """Map-style torch Dataset over a Ferroload dataset, flexible in modalities.

    Args:
        fds:     a `ferroload.Dataset`
        columns: a flat list of column names — each is resolved to its kind from the
                 dataset's modalities (image/video decode, `.npy` -> arrays, other
                 modalities -> raw bytes, non-modalities -> metadata). Merged with any
                 explicit `images`/`videos`/`arrays`/`raw`/`meta` below.
        images:  modalities to decode to arrays (each resized to `resize`)
        videos:  modalities to decode to [T,H,W,3] (needs the `video` feature)
        arrays:  `.npy` tensor modalities (e.g. a `map` embedding output) -> ndarray
        raw:     modalities to return as raw bytes (e.g. audio streams)
        meta:    metadata keys to attach (read from the index, no I/O)
        resize:  (H, W) applied to every decoded image column, OR a per-column
                 dict {col: (H, W) | None} for different sizes per column (covers
                 video columns too; None means no resize for that column)
        out:     "numpy" or "torch"

    Returns per-sample dicts; feed to a DataLoader and let its collate_fn batch.
    Absent modalities yield None plus a `<name>_present` flag.
    """

    def __init__(self, fds, images: Optional[Sequence[str]] = None,
                 videos: Optional[Sequence[str]] = None,
                 raw: Optional[Sequence[str]] = None,
                 meta: Optional[Sequence[str]] = None,
                 arrays: Optional[Sequence[str]] = None,
                 columns: Optional[Sequence[str]] = None,
                 resize=(224, 224), video_resize=None, num_frames: int = 16,
                 out: str = "numpy"):
        self.fds = fds
        images = list(images or [])
        videos = list(videos or [])
        arrays = list(arrays or [])
        raw = list(raw or [])
        meta = list(meta or [])
        if columns:
            ci, cv, ca, cr, cm = _resolve_columns(fds, columns)
            images += ci; videos += cv; arrays += ca; raw += cr; meta += cm
        self.images = images
        self.videos = videos
        self.arrays = arrays
        self.raw = raw
        self.meta = meta
        # `resize` is either a global (H, W) tuple (or None) applied to every
        # decoded column, or a per-column dict {col: (H, W) | None}.
        self.resize = resize
        if isinstance(resize, dict):
            # per-column dict covers videos too; video_resize only used as an
            # explicit (H, W) fallback for video columns not in the dict.
            self.video_resize = video_resize if isinstance(video_resize, (tuple, list)) else None
        elif video_resize is None:
            self.video_resize = resize
        elif video_resize is False:
            self.video_resize = None
        else:
            self.video_resize = video_resize
        self.num_frames = num_frames
        # check capability on the raw reader (the Dataset handle always defines a
        # decode_video method that only works when the core was built with video)
        _cap = getattr(fds, "reader", fds)
        if self.videos and not hasattr(_cap, "decode_video"):
            raise RuntimeError(
                "video decode not available — rebuild with the video feature: "
                "maturin develop --release --features video (needs system ffmpeg)"
            )
        if out not in ("numpy", "torch", "jax"):
            raise ValueError("out must be 'numpy', 'torch', or 'jax'")
        self.out = out

    def _img_resize(self, col):
        r = self.resize
        return r.get(col) if isinstance(r, dict) else r

    def _vid_resize(self, col):
        r = self.resize
        return r.get(col, self.video_resize) if isinstance(r, dict) else self.video_resize

    def __len__(self):
        return len(self.fds)

    def __getitem__(self, i):
        return self.__getitems__([i])[0]

    def __getitems__(self, indices):
        idx = list(indices)
        img_cols = {m: self.fds.decode_many(idx, m, self._img_resize(m)) for m in self.images}
        vid_cols = {v: self.fds.decode_video(idx, v, self.num_frames, self._vid_resize(v))
                    for v in self.videos}
        arr_cols = {a: _load_npy_list(self.fds.read_many(idx, a)) for a in self.arrays}
        raw_cols = {r: self.fds.read_many(idx, r) for r in self.raw}
        meta_cols = self.fds.meta_batch(idx, self.meta) if self.meta else {}

        samples = []
        for k in range(len(idx)):
            s = {}
            for m in self.images:
                s[m] = img_cols[m][k]
                s[f"{m}_present"] = img_cols[m][k] is not None
            for v in self.videos:
                s[v] = vid_cols[v][k]
                s[f"{v}_present"] = vid_cols[v][k] is not None
            for a in self.arrays:
                s[a] = arr_cols[a][k]
                s[f"{a}_present"] = arr_cols[a][k] is not None
            for r in self.raw:
                b = raw_cols[r][k]
                s[r] = b
                s[f"{r}_present"] = b is not None
            for key in self.meta:
                s[key] = meta_cols[key][k]
            if self.out == "torch":
                s = _to_torch(s)
            elif self.out == "jax":
                s = _to_jax(s)
            samples.append(s)
        return samples


# --------------------------------------------------------------------------
# Streaming / iterable view (WebDataset-style: shuffle shard blocks + buffer)
# --------------------------------------------------------------------------

class FerroIterableDataset(_IterableBase):
    """Iterable (streaming) view over the *same* `ferroload.Dataset` — the
    counterpart to the map-style `FerroTorchDataset`. Instead of random per-sample
    access, it reads the dataset in **contiguous blocks** (sequential, shard-local,
    object-store-friendly), decodes each block in parallel in Rust, and yields
    through a **shuffle buffer** of `shuffle_buffer` samples for approximate
    shuffling. Hand it to a `DataLoader` (no sampler) for cloud-scale streaming.

    Sharding for DDP + DataLoader workers is automatic: blocks are split across
    `world_size` ranks (fixed, for `__len__`), then across the loader's workers at
    iteration time via `torch.utils.data.get_worker_info()`. Call `set_epoch(e)`
    each epoch to reshuffle block order. Decode/columns config is identical to
    `FerroTorchDataset` (it wraps one internally), so nothing is duplicated.
    """

    def __init__(self, fds, *, columns=None, images=None, videos=None, raw=None,
                 meta=None, arrays=None, resize=(224, 224), video_resize=None,
                 num_frames=16, out="numpy", shuffle=True, shuffle_buffer=8192,
                 block_size=1024, world_size=1, rank=0, seed=0):
        super().__init__()
        # the heavy lifting (decode, columns, out conversion) is the map view's job
        self.tds = FerroTorchDataset(
            fds, columns=columns, images=images, videos=videos, raw=raw, meta=meta,
            arrays=arrays, resize=resize, video_resize=video_resize,
            num_frames=num_frames, out=out,
        )
        self.fds = fds
        self.block_size = max(1, int(block_size))
        self.shuffle = bool(shuffle)
        self.shuffle_buffer = max(1, int(shuffle_buffer))
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        n = len(fds)
        blocks = [(i, min(i + self.block_size, n)) for i in range(0, n, self.block_size)]
        # fixed per-rank split (disjoint across ranks); worker split happens in __iter__
        self._rank_blocks = blocks[self.rank::self.world_size]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        # samples this rank yields per epoch (across all its workers)
        return sum(e - s for s, e in self._rank_blocks)

    def __iter__(self):
        # split this rank's blocks across the DataLoader's workers
        worker_id, num_workers = 0, 1
        try:
            import torch.utils.data as _tud
            info = _tud.get_worker_info()
            if info is not None:
                worker_id, num_workers = info.id, info.num_workers
        except Exception:
            pass
        my_blocks = self._rank_blocks[worker_id::num_workers]
        rng = random.Random(
            (self.seed * 1_000_003) ^ (self.epoch * 9176) ^ (self.rank << 8) ^ worker_id
        )
        if self.shuffle:
            my_blocks = list(my_blocks)
            rng.shuffle(my_blocks)

        if not self.shuffle:
            for start, end in my_blocks:
                yield from self.tds.__getitems__(list(range(start, end)))
            return

        # WebDataset-style streaming shuffle buffer
        buf = []
        K = self.shuffle_buffer
        for start, end in my_blocks:
            for s in self.tds.__getitems__(list(range(start, end))):
                if len(buf) < K:
                    buf.append(s)
                else:
                    j = rng.randrange(K)
                    out, buf[j] = buf[j], s
                    yield out
        rng.shuffle(buf)
        yield from buf


# --------------------------------------------------------------------------
# Deterministic distributed sampling + async prefetch
# --------------------------------------------------------------------------

class FerroSampler:
    """Deterministic, DDP-aware, resumable index sampler (torch `Sampler`-compatible).

    Partitions `range(total)` by `(world_size, rank)` and **block-shuffles** per
    epoch, backed by the Rust `ferroload._core.Sampler`. Pass to
    `DataLoader(sampler=...)`; call `set_epoch(e)` each epoch to reshuffle (same
    contract as torch's `DistributedSampler`). Worker-level splitting is handled
    by the DataLoader's `num_workers`, so the sampler itself uses `num_workers=1`
    (it yields the whole rank slice).
    """

    def __init__(self, total, world_size=1, rank=0, seed=0, shuffle=True, shuffle_block=1024):
        from ._core import Sampler as _S
        self._S = _S
        self.total = int(total)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.shuffle_block = int(shuffle_block)
        self.epoch = 0
        self._len = len(self._plan(0))

    def _plan(self, epoch, resume_from=0):
        return self._S(self.total, self.world_size, self.rank, 1, 0,
                       self.seed, self.shuffle, self.shuffle_block).indices(epoch, resume_from)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(self._plan(self.epoch))

    def __len__(self):
        return self._len


def batched(indices, batch_size, drop_last=False):
    """Yield lists of `batch_size` indices from an iterable/sampler."""
    batch = []
    for i in indices:
        batch.append(int(i))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch


def numpy_collate(samples):
    """Minimal NumPy collate: stack arrays, array scalars, list the rest."""
    import numpy as np
    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        v0 = vals[0]
        if isinstance(v0, np.ndarray):
            out[k] = np.stack(vals)
        elif isinstance(v0, (int, float, bool, np.integer, np.floating, np.bool_)):
            out[k] = np.array(vals)
        else:
            out[k] = vals
    return out


class PrefetchLoader:
    """Background-thread prefetch over `(dataset, batches)`.

    A worker thread pulls index-batches from `batches`, calls the dataset's
    batched `__getitems__` (which decodes/reads in Rust with the **GIL released**),
    optionally collates, and pushes onto a bounded queue — so the next batch is
    prepared while the current one is consumed/trained on. `depth` batches are
    buffered ahead (a count-based budget; byte-budgeted prefetch is a Rust-core
    roadmap item). Single-pass: recreate it per epoch.

        sampler = FerroSampler(len(ds), world_size=W, rank=R)
        tds = FerroTorchDataset(ds, images=["image"], meta=["label"], resize=(224,224))
        for epoch in range(E):
            sampler.set_epoch(epoch)
            for batch in PrefetchLoader(tds, batched(sampler, 64),
                                        collate_fn=numpy_collate, depth=3):
                train_step(batch)
    """

    def __init__(self, dataset, batches, collate_fn=None, depth=2):
        self.dataset = dataset
        self.batches = batches
        self.collate_fn = collate_fn or (lambda x: x)
        self.depth = max(1, int(depth))

    def __iter__(self):
        import queue
        import threading
        q = queue.Queue(maxsize=self.depth)

        def worker():
            try:
                for idxs in self.batches:
                    samples = self.dataset.__getitems__(list(idxs))
                    q.put(("ok", self.collate_fn(samples)))
            except Exception as e:  # surface to the consumer
                q.put(("err", e))
            finally:
                q.put(("done", None))

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            tag, payload = q.get()
            if tag == "done":
                break
            if tag == "err":
                raise payload
            yield payload


def _prefetch(gen, depth=2):
    """Run a generator on a daemon thread, buffering up to `depth` items ahead on a
    bounded queue. The producer's heavy work (Rust block decode / remote fetch,
    which release the GIL) then overlaps with the consumer (training). Single-pass."""
    import queue
    import threading

    q = queue.Queue(maxsize=max(1, int(depth)))

    def worker():
        try:
            for item in gen:
                q.put(("ok", item))
        except Exception as e:  # surface to the consumer
            q.put(("err", e))
        finally:
            q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        tag, payload = q.get()
        if tag == "done":
            break
        if tag == "err":
            raise payload
        yield payload


# --------------------------------------------------------------------------
# One-call initializer
# --------------------------------------------------------------------------

class FerroLoader:
    """Open a dataset and iterate collated minibatches in one object. The access
    pattern is a single knob, `streaming`:

      - `streaming=False` (default) — **map-style**: a deterministic `FerroSampler`
        + random per-sample reads + background prefetch. Exact shuffle, `ds[i]`.
      - `streaming=True` — **iterable/streaming**: a `FerroIterableDataset` that
        shuffles shard-local blocks and yields through a `shuffle_buffer`
        (WebDataset-style). Fewer, larger sequential reads — the object-store
        (s3://, gs://) friendly path. Approximate shuffle, no random `ds[i]`.

    Either way it's the same `Dataset`, columns config, decode, and cache.
    Call `set_epoch(e)` each epoch to reshuffle.

        dl = make_loader("/data/ds", batch_size=64,
                         columns=["image", "video", "label"], resize=(224, 224))
        # cloud-scale streaming (no sampler; block-shuffle + buffer):
        dl = make_loader("s3://bucket/ds", batch_size=256, columns=["image", "caption"],
                         streaming=True, shuffle_buffer=16384)
        for epoch in range(E):
            dl.set_epoch(epoch)
            for batch in dl:
                ...
    """

    def __init__(self, root, batch_size=32, *, columns=None, images=None, videos=None,
                 raw=None, meta=None, arrays=None, resize=(224, 224), video_resize=None,
                 num_frames=16, out="numpy", shuffle=True, world_size=1, rank=0, seed=0,
                 prefetch=2, drop_last=False, collate_fn=None, cache_dir=None,
                 streaming=False, shuffle_buffer=8192, block_size=1024):
        from ._core import Dataset
        # `root` may be a local path or a remote URL (s3://, gs://, az://); remote
        # shard bytes stream through a local cache at `cache_dir`.
        self.ds = Dataset.open(root, cache_dir)
        self.batch_size = int(batch_size)
        self.prefetch = prefetch
        self.drop_last = drop_last
        self.streaming = bool(streaming)
        cfg = dict(columns=columns, images=images, videos=videos, raw=raw, meta=meta,
                   arrays=arrays, resize=resize, video_resize=video_resize,
                   num_frames=num_frames, out=out)
        if self.streaming:
            # iterable / streaming access pattern: shard-block shuffle + buffer
            self.iter_ds = FerroIterableDataset(
                self.ds, shuffle=shuffle, shuffle_buffer=shuffle_buffer,
                block_size=block_size, world_size=world_size, rank=rank, seed=seed, **cfg)
            self.tds = None
            self.sampler = None
        else:
            # map-style access pattern: deterministic sampler + random reads
            self.tds = FerroTorchDataset(self.ds, **cfg)
            self.sampler = FerroSampler(len(self.ds), world_size=world_size, rank=rank,
                                        seed=seed, shuffle=shuffle)
            self.iter_ds = None
        if collate_fn is None:
            if out == "torch":
                from torch.utils.data._utils.collate import default_collate
                collate_fn = default_collate
            else:
                collate_fn = numpy_collate
        self.collate_fn = collate_fn

    def set_epoch(self, epoch):
        (self.iter_ds or self.sampler).set_epoch(epoch)

    def __len__(self):
        n = len(self.iter_ds) if self.streaming else len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.streaming:
            # overlap block decode / remote fetch with training (prefetch batches ahead)
            return _prefetch(self._iter_streaming(), self.prefetch)
        return iter(PrefetchLoader(
            self.tds,
            batched(self.sampler, self.batch_size, self.drop_last),
            collate_fn=self.collate_fn,
            depth=self.prefetch,
        ))

    def _iter_streaming(self):
        # the parallel Rust decode happens per block inside the iterable view; here
        # we just group its sample stream into collated minibatches.
        batch = []
        for sample in self.iter_ds:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch and not self.drop_last:
            yield self.collate_fn(batch)


def make_loader(root, batch_size=32, **kwargs):
    """Convenience factory — see `FerroLoader`.

        from ferroload import make_loader
        dl = make_loader("/data/ds", 64, images=["image"], meta=["label"])
        for batch in dl: ...
    """
    return FerroLoader(root, batch_size, **kwargs)
