# Ferroload — Python API Reference

The single source of truth for the **current** Python surface (the extension
`ferroload._core` + the `ferroload` package). For the Rust core API see
`USAGE.md`; for the design target see `DESIGN.md` §10 (aspirational).

```python
import ferroload
from ferroload import loader            # PyTorch glue
```

## Quick start (one-call loader)

```python
from ferroload import make_loader
dl = make_loader("/data/ds", batch_size=64,
                 columns=["image", "video", "label"],  # kinds resolved from the manifest
                 resize=(224, 224), out="torch")        # or out="numpy"
for epoch in range(epochs):
    dl.set_epoch(epoch)                              # reshuffle
    for batch in dl:                                 # batch["image"], batch["video"], batch["label"], ...
        train_step(batch)
```

`make_loader` (alias of `FerroLoader`) bundles `Dataset.open` + `FerroTorchDataset`
+ `FerroSampler` + background prefetch. Everything below is the lower-level API it
is built from.

A flat `columns=[...]` list is resolved against the dataset's modalities: image /
video codecs decode, `.npy` tensor columns (e.g. a `map` embedding output) load as
arrays, other modalities pass as raw bytes, and names that aren't modalities are
metadata keys. You can still bucket types explicitly with
`images=`/`videos=`/`arrays=`/`raw=`/`meta=` (and mix both — the explicit lists are
merged with the resolved ones).

`resize` is a global `(H, W)` for every decoded column, or a **per-column dict** for
different sizes per column (covers videos; `None` = no resize for that column):

```python
make_loader(ROOT, batch_size=8, columns=["image", "thumb", "video"],
            resize={"image": (224, 224), "thumb": (32, 32), "video": (64, 64)})
```

### Framework views

`Dataset.open(root)` returns a handle with fluent, framework-typed views (all take
the same `images=`/`videos=`/`raw=`/`meta=`/`resize=` config as `FerroTorchDataset`):

```python
ds = ferroload.Dataset.open(root)
ds.torch(images=["image"], meta=["label"], resize=(224, 224))   # torch-ready Dataset
ds.numpy(images=["image"])                                      # NumPy per-sample dicts
ds.jax(images=["image"])                                        # JAX arrays
ds.reader                                                       # the raw _core.Dataset
```

Each returns a per-sample-dict dataset (a `FerroTorchDataset` with the matching
`out=`); pass it to a `DataLoader` or your own batching. All core reader methods
(`get`, `read`, `decode_*`, `meta_batch`, `subset`, …) are delegated through the
handle, so `ds.get(0)` / `ds.subset(...)` still work directly.

## Conventions

- **Canonical names:** `ferroload.Dataset` / `ferroload.Writer`. `FerroDataset` /
  `FerroWriter` are kept as aliases.
- **Projection vs single column:** `get(i, modalities=[...])` takes a **list**
  (it projects a whole sample); `read* / decode_*` take a single `modality: str`.
- **`resize` is `(height, width)`** everywhere, both must be `> 0` (else
  `ValueError`).
- **Defaults are image-centric:** `read*`/`decode_many` default `modality="image"`;
  pass `modality=` explicitly for audio/video/other datasets.
- **Exceptions:** `IndexError` (bad sample index), `FileNotFoundError` (missing
  root/shard), `RuntimeError` (`reader too old`), `ValueError` (bad predicate /
  resize / decode).

---

## `ferroload.Writer(root: str, name: str)`

Streaming writer (alias `FerroWriter`). **Stateful** — `declare` must precede
`add`, and `close` is required to commit (writes the index, then the manifest
atomically, and a `versions/vN.json` snapshot).

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `declare` | `(name, ext, kind="tensor", codec="raw")` | `None` | modality declaration; `kind` ∈ `tensor`/`scalar`/`annotation` |
| `add` | `(key, blobs: dict[str, bytes], meta: dict[str, scalar] = None)` | `int` (sample_id) | `blobs` are **pre-encoded bytes**; `meta` values are scalar/JSON |
| `close` | `()` | `None` | commit; required (idempotent guard) |

```python
w = ferroload.Writer("/data/ds", "demo")
w.declare("image", "jpg", "tensor", "image")
w.declare("text",  "json", "scalar", "passthrough")
w.add("s0000", {"image": jpg_bytes}, {"label": 3, "split": "train"})
w.close()
```

---

## `ferroload.Dataset`

Open with `Dataset.open(root, cache_dir=None)` (alias `FerroDataset`). `root` is a
local path **or an object-store URL** — `s3://bucket/prefix`, `gs://…`, `az://…`,
also `file://` / `memory://`. For a URL, shard bytes stream via ranged GETs through
a content-addressed local cache at `cache_dir` (default `$FERROLOAD_CACHE` or a temp
dir); batched reads coalesce into one `get_ranges` per shard, all in Rust with the
GIL released. Credentials come from the environment (`AWS_*` / `GOOGLE_*` /
`AZURE_*`). Remote needs a build with `--features aws` (or `gcp`/`azure`), and remote
datasets are read-only.

```python
ds = ferroload.Dataset.open("s3://my-bucket/datasets/laion-pop")        # streams from S3
dl = make_loader("s3://my-bucket/datasets/laion-pop", batch_size=256,
                 columns=["image", "caption"], resize=(256, 256), cache_dir="/mnt/nvme/ferro-cache")
```

### Access & introspection

| Member | Signature | Returns |
|---|---|---|
| `open` | `Dataset.open(root: str)` | `Dataset` |
| `__len__` | `len(ds)` | `int` |
| `__getitem__` | `ds[i]` | sample `dict` (== `get(i)`) |
| `num_shards` | `()` | `int` |
| `name` | property | `str` |
| `version` | property | `int` |
| `modalities` | `()` | `dict[str, dict]` — `{name: {ext, kind, codec, …}}` |
| `schema` | `()` | `list[dict]` — index columns |
| `manifest` | `()` | `dict` — full manifest incl. `extensions` |
| `verify` | `()` | `int` — samples verified |

### Reads (one sample)

```python
get(i, modalities: list[str] | None = None) -> dict
# {sample_id, basename, <modality>: bytes, <modality>_present: bool, meta: dict}
# modalities=None reads all declared; a list projects (others never fetched).
```

### Reads (one modality, batched)

| Method | Signature | Returns | Use |
|---|---|---|---|
| `read` | `(i, modality="image")` | `bytes \| None` | single sample, minimal overhead |
| `read_many` | `(indices, modality="image")` | `list[bytes \| None]` | batch; I/O in Rust, **GIL released** |
| `read_batch` | `(indices, modality="image")` | `(bytes, list[(off, len)])` | batch into **one contiguous buffer**; slice with `memoryview(buf)[o:o+l]` (zero-copy). Best for large blobs; an alternative to `read_many` |

### Decode (parallel, GIL released, zero-copy NumPy)

| Method | Signature | Returns |
|---|---|---|
| `decode_many` | `(indices, modality="image", resize=(H,W) \| None)` | `list[ndarray[H,W,3] uint8 \| None]` |
| `decode_audio` | `(indices, modality="audio")` | `list[ndarray[channels, samples] float32 \| None]` |
| `decode_video` | `(indices, modality="video", num_frames=16, resize=(H,W) \| None)` | `list[ndarray[T,H,W,3] uint8 \| None]` — **needs `--features video`** |

`resize` makes mixed-resolution inputs uniform so they stack into a batch.
Absent modalities yield `None` (sparse-tolerant).

### Metadata & subsetting

| Method | Signature | Returns |
|---|---|---|
| `meta_batch` | `(indices, keys: list[str])` | `dict[str, ndarray \| list]` — typed array when uniform, else list (strings/nested); **no shard I/O** |
| `subset` | `(where_sql: str, return_indices=False)` | a new (subset) **`Dataset`** by default, or `list[int]` of ids when `return_indices=True` |

```python
train = ds.subset("duration_s < 16 AND lang = 'en' AND has_audio")  # -> Dataset
train.torch(images=["image"], meta=["label"])                       # use it directly
ids = ds.subset("split = 'val'", return_indices=True)               # -> list[int]
```
The subset is an index-remapped view over the same reader (supports `get`,
`decode_many`, the framework views, and further `.subset()`). `subset` is a
lightweight `WHERE` evaluator (AND/OR/NOT, comparisons, `<col>_present`).
DataFusion SQL over the Parquet index is the production swap-in (DESIGN §6).

### Enrichment — `map`

Run a function over the dataset and store its results as a new, **additive
layer** joined on `sample_id`. The base data is never rewritten; the pass is
**idempotent and resumable**; outputs read back as ordinary modalities/metadata.

```python
ds.map(fn, inputs, outputs, name=None, batch_size=32, batched=False,
       resume=True, decode=True, resize=None, num_workers=0,
       progress=False, executor=None) -> Dataset
```

`fn` is bound to `inputs` **positionally** — it receives one argument per name in
`inputs`, in order, and never references column names itself, so the same
function is reusable across datasets. By default (`batched=False`) `fn` is
**unitary**: it takes one sample's value(s) and returns that sample's output (or a
tuple in `outputs` order). With `batched=True` it takes one list per input and
returns one list (or a tuple of lists). Image inputs are decoded in parallel in
Rust either way, so per-sample mode keeps the fast decode — only `fn` runs per row.

| Param | Meaning |
|---|---|
| `inputs` | modality and/or metadata names, bound **positionally** to `fn`'s arguments (a bare str for a single input). Image-codec modalities arrive as `[H,W,C]` uint8 arrays (unless `decode=False`); `.npy` tensor-layer outputs (from a previous `map`) arrive as arrays; other modalities arrive as raw `bytes`; metadata keys arrive as scalars. Absent → `None`. |
| `outputs` | a list of names (all arrays) **or** `{name: kind}`. Kinds: `'array'`/'tensor' → a new `.npy`-backed modality; `'bytes'`/'raw' or media shorthands `'video'`/'audio'/'image' → a new modality storing the **raw bytes** the fn returns (e.g. a downloaded `.mp4`); `'scalar'`/'text'/'annotation' → metadata. Also accepts the typed objects `ferroload.Modality(ext, codec=...)` (tensor/blob) and `ferroload.Annotation()`, or a dict `{'type':'bytes','ext':'mp4','codec':'video'}` for full control. The output order defines how `fn`'s returned tuple is unpacked; with a single output, return the value (or column) directly. A `None` per-sample value (or `None` element when batched) is skipped (sparse layer). |
| `name` | layer name (default `"map_" + "_".join(outputs)`). |
| `batched` | if `False` (default) `fn` runs once per sample on scalar args; if `True` it runs once per batch on the input column lists — use for vectorized or I/O-fan-out work. |
| `batch_size` | rows decoded/read per call (the unit of the resume loop, and the list length in `batched=True`). |
| `resume` | skip `sample_id`s already in the layer (default `True`); re-running only computes what's missing. |
| `decode` / `resize` | decode image inputs to arrays (default `True`); optional `(h, w)`. |
| `num_workers` | reserved — intra-process decode is already parallel across cores in Rust (GIL released). |
| `executor` | map backend (see *Distributed map* below). Default: auto-selected from the launch topology. |

Returns a fresh `Dataset` with the layer visible (preserving the subset view if
`self` is a subset). Read tensor outputs back with `read_array(i, modality)` /
`read_arrays(indices, modality)` (NumPy); scalar/text outputs appear in `meta`.

```python
# compute a depth map (tensor) + a caption (text) for every sample — per-sample,
# positional: `img` is bound from inputs=["image"]; returns (depth, caption)
def enrich(img):                                # one [H,W,C] image
    return estimate_depth(img), caption(img)    # tuple in `outputs` order

ds = ds.map(enrich, inputs=["image"],
            outputs={"depth": "array", "caption": "text"}, name="features")

ds.read_array(0, "depth")          # -> ndarray
ds.get(0)["meta"]["caption"]       # -> str
ds.subset("label = 0").map(...)    # map only a subset (sparse layer)
```

**Download a media column from a URL into a real modality.** A `'video'` (or
`'bytes'`) output stores exactly what `fn` returns — no NumPy wrapping — so it
becomes a genuine `.mp4` member in the layer's shards (decodable later with
`decode_video`, served by the loader, etc.). The unitary form is just
`def download(url): return requests.get(url).content`. But downloads are
I/O-bound, so this is the case for `batched=True` — fan the whole batch out
across threads in one call:

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def download(urls):                                # batched: one call per batch of URLs
    with ThreadPoolExecutor(max_workers=16) as ex:
        def get(u):
            try:    return requests.get(u, timeout=30).content
            except Exception: return None          # None -> that sample is skipped (sparse)
        return list(ex.map(get, urls))             # single output -> return the list

ds = ds.map(download, inputs=["video_url"],
            outputs={"video": "video"}, batched=True,   # -> mp4/video modality
            name="video", batch_size=64)            # idempotent: re-run resumes failed/missing

ds.read(0, "video")                                 # -> bytes (the downloaded mp4)
# ds.decode_video([0,1,2], "video", num_frames=8)   # if built with --features video
```

Because the pass is resumable, re-running `map` only fetches the URLs that aren't
already stored (those that errored to `None` or were added since). Use
`{'type':'bytes','ext':'jpg','codec':'image'}` to download images instead, etc.

A layer's shards live at `shards/<name>/` and its index at `index/<name>.json`;
the layer is registered in the manifest with a version bump, and a base-only
reader simply ignores it.

#### Distributed map (executors)

The map backend is abstracted behind an `Executor` (DESIGN §14.4); user code
doesn't name one. The launch **topology** is auto-detected from the environment
and selects the backend:

| Topology | Executor | Behavior |
|---|---|---|
| single node (bare / `torchrun` 1 node) | `LocalExecutor` | one process, writes + registers the layer (decode already parallel in Rust) |
| `torchrun` / SLURM multi-node | `StaticPartitionExecutor` | each rank computes a disjoint `sample_id` partition, writes its **own** layer fragment, then a single commit merges + registers (under `torch.distributed`, rank 0 commits after a barrier) |
| Ray cluster | `RayExecutor` | reserved (raises with guidance; run under torchrun/SLURM or single-node) |

`FERROLOAD_EXECUTOR=local|static|ray` overrides the choice; pass `executor=` to
`map` to force one explicitly. Helpers: `ferroload.detect_topology()`,
`ferroload.select_executor()`, and `ferroload.commit_layer(root, name, modalities)`
(the manual commit step when there's no `torch.distributed` barrier).

```python
# torchrun --nnodes=2 --nproc-per-node=8 enrich.py   (each rank runs this)
ds = ferroload.Dataset.open(root)
ds.map(compute_depth, inputs=["image"], outputs={"depth": "array"}, name="depth")
# StaticPartitionExecutor: each rank writes shards/depth/shard-<rank>-*.tar +
# index/depth.part-<rank>.json; rank 0 commits the merged layer after a barrier.
```

### Return-type summary

`get`→dict · `read`→bytes|None · `read_many`→list[bytes|None] ·
`read_batch`→(bytes, spans) · `decode_*`→list[ndarray|None] · `meta_batch`→dict ·
`subset`→Dataset (or `list[int]` with `return_indices=True`) ·
`map`→Dataset · `read_array`→ndarray|None · `read_arrays`→list[ndarray|None] ·
`verify`→int.

---

## `ferroload.loader` (PyTorch)

The dataset yields **per-sample dicts**; a normal `DataLoader` batches them with
its `collate_fn` (default or custom). The parallel Rust decode runs under
PyTorch's batched-fetch hook `__getitems__`, transparently.

### `FerroTorchDataset(fds, columns=None, images=None, videos=None, raw=None, meta=None, arrays=None, resize=(224,224), video_resize=None, num_frames=16, out="numpy")`

- `images` — decoded to `[H,W,3]` (resized to `resize`)
- `videos` — decoded to `[T,H,W,3]` (resized to `video_resize`, default = `resize`;
  `video_resize=False` keeps native sizes). Needs `--features video`.
- `raw` — returned as raw `bytes` (decode in your own transform/collate)
- `meta` — attached from the index (no I/O)
- `out` — `"numpy"` or `"torch"` (zero-copy `torch.from_numpy`)

Absent modalities → `None` + `<name>_present=False`.

```python
from torch.utils.data import DataLoader
ds  = ferroload.Dataset.open("/data/ds")
tds = loader.FerroTorchDataset(ds, images=["image"], meta=["label"],
                               resize=(224,224), out="torch")
dl  = DataLoader(tds, batch_size=64, num_workers=4)        # default collate stacks
for batch in dl:
    batch["image"]   # [B,224,224,3] uint8 ; batch["label"] [B] int64
```

### Distributed sampling + async prefetch

```python
# Deterministic, DDP-aware, resumable sampler (torch Sampler-compatible)
sampler = loader.FerroSampler(len(ds), world_size=W, rank=R, seed=0)
sampler.set_epoch(epoch)                       # reshuffle per epoch (like DistributedSampler)
DataLoader(tds, batch_size=64, sampler=sampler)

# Or background-thread prefetch (overlaps the GIL-released Rust decode):
from ferroload.loader import FerroSampler, PrefetchLoader, batched, numpy_collate
for epoch in range(E):
    sampler.set_epoch(epoch)
    for batch in PrefetchLoader(tds, batched(sampler, 64),
                                collate_fn=numpy_collate, depth=3):
        train_step(batch)                      # next batch decodes while this one trains
```

| Symbol | Purpose |
|---|---|
| `ferroload.Sampler(total, world_size=1, rank=0, num_workers=1, worker_id=0, seed=0, shuffle=True, shuffle_block=1024)` | raw Rust planner; `.indices(epoch=0, resume_from=0) -> list[int]` |
| `loader.FerroSampler(total, world_size=1, rank=0, seed=0, shuffle=True, shuffle_block=1024)` | torch-compatible; `set_epoch`, `__iter__`, `__len__` |
| `loader.batched(indices, batch_size, drop_last=False)` | chunk indices into batches |
| `loader.numpy_collate(samples)` | stack arrays / array scalars / list strings |
| `loader.PrefetchLoader(dataset, batches, collate_fn=None, depth=2)` | background-thread prefetch (single-pass; recreate per epoch) |

The sampler partitions by `(world_size, rank)` and block-shuffles deterministically
from `(seed, epoch)`; worker-level splitting is left to the DataLoader. Prefetch
`depth` is a count-based budget (byte-budgeted prefetch lives in the Rust-core
roadmap).

### `loader.subset_dataset(tds, indices)`

Restrict a `FerroTorchDataset` to `indices` (e.g. from `ds.subset(...)`). Uses
`torch.utils.data.Subset` (which forwards the batched fast path) when torch is
present, else a lightweight remapper.

```python
ids   = ds.subset("split = 'train'", return_indices=True)
train = loader.subset_dataset(tds, ids)
# (or simply: train = ds.subset("split = 'train'").torch(...))
DataLoader(train, batch_size=64)
```

---

## Notes & current limits

- **Sampling / distributed.** The Rust deterministic rank×worker sampler is now
  exposed as `ferroload.Sampler` and wrapped by `loader.FerroSampler` (a
  torch-compatible, resumable, DDP-aware sampler). `loader.PrefetchLoader` adds
  background-thread prefetch that overlaps the GIL-released Rust decode. You can
  still use PyTorch's own `DataLoader`/`DistributedSampler` if you prefer.
  *Remaining roadmap:* **byte-budgeted** prefetch in the Rust core, and pinned
  memory / DLPack hand-off.
- **`read_batch` vs `read_many`.** Both batch one modality with the GIL released.
  `read_many` returns a list of `bytes` (simplest); `read_batch` returns one
  contiguous buffer + spans (fewer Python objects, better for large blobs). The
  loader currently uses `read_many`/`decode_*`; `read_batch` is the lower-overhead
  alternative when you manage slicing yourself.
- **Writer** takes pre-encoded `bytes` (no tensor/array input) and is stateful
  (`declare` → `add*` → `close`).
- **CLI:** `ferroload {inspect,verify,import-hf,import-files,subset,list,add}` —
  see `python/README_HF.md`.
