# Ferroload — A Pure-Rust Multimodal Dataloader for PyTorch

**Status:** Design draft v1
**Scope:** Distributed, sharded, indexable, iterable dataloader for large-scale
video + image + audio + text training, with a self-contained dataset format,
in-Rust decoding, and SQL-based subsetting.

---

## 1. Goals & non-goals

### Goals

- **Pure-Rust core.** All I/O, indexing, sharding, sampling, prefetch, and
  orchestration live in Rust. Python is a thin binding.
- **Cloud-native storage.** Read/write S3, GCS, Azure Blob, and local/NFS through
  one abstraction.
- **Self-contained dataset format.** A dataset is a single root URI. Manifest,
  index, shards, and side-indexes live together so a dataset can be copied,
  moved, or versioned as one unit.
- **Four modalities.** Video, image, audio, text — decoded in Rust into tensors.
- **Random + range + iterable access.** `ds[i]`, `ds[a:b]`, and `for x in ds`.
- **Distributed + sharded.** Deterministic, disjoint partitioning across the
  `world_size x num_workers` grid, with clean resume.
- **SQL subsetting.** Create a subset of a dataset from a SQL predicate over
  rich per-sample metadata, either virtually or materialized.
- **PyTorch-native.** Map-style `Dataset` and `IterableDataset`, integrating with
  `DataLoader(num_workers=N)`, pinned memory, and DLPack zero-copy tensors.

### Non-goals (v1)

- Tokenization (returned as raw text/bytes; tokenize in Python).
- Training-loop / checkpoint orchestration (we expose resume state, not a trainer).
- On-the-fly transcoding of source media beyond decode + standard augment.

---

## 2. High-level architecture

```
+-------------------------------------------------------------+
|                        PyTorch (Python)                     |
|   Dataset / IterableDataset  •  DataLoader(num_workers=N)    |
+----------------------------+--------------------------------+
                             | PyO3 / maturin (thin binding)
+----------------------------v--------------------------------+
|                      ferroload-core (Rust)                  |
|                                                             |
|  +-----------+  +-----------+  +-----------+  +-----------+  |
|  |  Format   |  |   Index   |  |  Sampler  |  | Prefetch  |  |
|  | manifest  |  | (Parquet  |  | rank x    |  | tokio +   |  |
|  | + writer  |  | +DataFusion| | worker    |  | ready q   |  |
|  +-----------+  +-----------+  +-----------+  +-----------+  |
|        |              |             |              |         |
|  +-----v--------------v-------------v--------------v------+  |
|  |                  Decoders (feature-gated)             |  |
|  |  image (zune/image) • audio (symphonia/rubato)       |  |
|  |  video (ffmpeg CPU | NVDEC) • text (passthrough)     |  |
|  +------------------------------------------------------+  |
|        |                                                    |
|  +-----v------------------------------------------------+   |
|  |        object_store: S3 / GCS / Azure / local        |   |
|  +------------------------------------------------------+   |
+-------------------------------------------------------------+
```

### Crate layout (Cargo workspace)

```
ferroload/
  Cargo.toml                # workspace
  crates/
    ferroload-core/         # pure Rust: format, index, sampler, prefetch
    ferroload-codec/        # decoders, feature-gated per modality
    ferroload-format/       # manifest + writer/packer (dataset creation)
    ferroload-cli/          # `ferroload` binary: pack, index, subset, inspect
    ferroload-py/           # PyO3 bindings -> Python wheel (maturin)
```

Keeping the core Python-free makes it independently testable and reusable (a Rust
trainer or a Go service could embed it). The binding stays thin.

---

## 3. Self-contained dataset format

The biggest change from the first sketch: **a dataset is one root URI**, not a
separate index path plus a shards path. Everything is nested under the root and
described by a manifest, so the dataset is portable and versionable.

### 3.1 On-disk / on-bucket layout

```
s3://bucket/datasets/howto-av/          <-- the ONE URI you pass
  manifest.json                          # source of truth (see 3.2)
  index/
    index.parquet                        # global sample index (may be partitioned)
    _stats.json                          # column stats, counts, byte totals
  shards/
    shard-00000.tar                      # WebDataset-compatible tar
    shard-00000.tar.idx                  # side-index: member -> (offset, len)
    shard-00001.tar
    shard-00001.tar.idx
    ...
  subsets/
    train_en.parquet                     # materialized subset views (optional)
    val.parquet
  indexes/                               # capability artifacts (e.g. vector search)
    clip_emb_hnsw/                       # registered in manifest "extensions"
  versions/
    v1.json                              # manifest snapshots for time-travel
    v2.json
```

A reader is constructed with just:

```python
ds = ferroload.Dataset("s3://bucket/datasets/howto-av")
```

It reads `manifest.json`, which points at the index and shards. No second path.

### 3.2 Manifest (`manifest.json`)

```json
{
  "format_version": 1,
  "min_reader_version": 1,
  "name": "howto-av",
  "created_utc": "2026-06-18T00:00:00Z",
  "version": 5,
  "modalities": {
    "video": { "ext": "mp4", "decode": "video" },
    "audio": { "ext": "flac", "decode": "audio" },
    "image": { "ext": "jpg",  "decode": "image" },
    "text":  { "ext": "json", "decode": "passthrough" }
  },
  "index": { "path": "index/index.parquet", "rows": 12000000 },
  "shards": { "dir": "shards/", "count": 4096, "shard_bytes_target": 1073741824 },
  "decode_defaults": { "video": { "num_frames": 16 }, "audio": { "sample_rate": 16000 } },
  "schema": [
    { "name": "image_embedding", "dtype": "list<float32>[768]",
      "semantic": "embedding",
      "attrs": { "dim": 768, "metric": "cosine", "source_model": "clip-vit-l14" } }
  ],
  "stats_path": "index/_stats.json",

  "extensions": {
    "vector_index": [
      {
        "name": "clip_emb_hnsw", "ext_version": 1,
        "column": "image_embedding", "kind": "hnsw", "metric": "cosine", "dim": 768,
        "path": "indexes/clip_emb_hnsw/", "backend": "object_store",
        "params": { "M": 32, "ef_construction": 200 },
        "built_over_dataset_version": 4, "covers_rows": 12000000, "stale": false
      }
    ]
  }
}
```

### 3.3 Atomic commit & versioning

- The **manifest is the source of truth**; orphan shards not referenced by it are
  ignored. This makes writes safe.
- A write/append: stage new shards + new index fragment, then write the new
  manifest **last**. A crash mid-write leaves the old manifest valid.
- Each commit snapshots the manifest into `versions/vN.json`, enabling time-travel
  reads (`Dataset(root, version=1)`). Shards are append-only and content-addressed
  by name, so old versions stay readable.

### 3.4 Why WebDataset-compatible tar + side-index

Tar is a great container (streamable, ubiquitous, append-friendly), but vanilla
WebDataset is **sequential-only** — you can't seek to sample N. We add a per-shard
`.tar.idx` side-index produced at pack time:

```
member_name            offset      length
clip0001.mp4           512         842133
clip0001.flac          843264      19200
clip0001.json          862976      318
clip0002.mp4           ...
```

With it, random access becomes a byte-range GET; existing WebDataset tars can be
adopted by generating side-indexes without repacking. A WebDataset "sample" is the
set of members sharing a basename, so one index row references all its modalities.

### 3.5 Manifest extensibility & capability extensions

The manifest is designed to grow new capabilities **without changing the core
format or breaking older readers**. Three mechanisms:

1. **Versioning with a reader floor.** `format_version` identifies the core
   structure; `min_reader_version` is the *minimum* reader version required to read
   the dataset *safely*. Additive features bump `format_version` but leave
   `min_reader_version` alone, so old readers keep working. Only a genuinely
   breaking change raises `min_reader_version`.

2. **Reserved `extensions` namespace.** Optional capabilities live under
   `extensions.<name>` as self-describing, versioned blocks (`ext_version` +
   payload). The core reader never needs to understand them.
   - **Forward compatibility:** a reader that doesn't know an extension simply
     ignores it — the dataset stays fully usable (you just don't get that feature).
   - **Preserve-unknown (critical):** any tool that rewrites the manifest must
     **round-trip extension blocks it doesn't recognize**, so an older writer can't
     silently drop a newer capability.
   - **Graceful degradation:** capabilities are opt-in accelerators, never required
     to read base data.

3. **Column-level semantics + free-form attrs.** Schema entries carry an optional
   `semantic` tag and an open `attrs` map. An embedding column added by an
   enrichment map is just `semantic: "embedding"` with `attrs: {dim, metric,
   source_model}`. The core treats it as an ordinary column; an extension gives it
   meaning later.

**Worked example — a vector search index over an embedding column (the requested
case).** It needs *no* format change today:

```
1. enrichment map computes embeddings:
     ds.map(clip_encode, inputs=["image"],
            outputs={"image_embedding": Embedding(dim=768, metric="cosine")})
   -> adds an embedding column (a layer); manifest schema records semantic+attrs.

2. later, build the index when the capability exists:
     ferroload build-index --column image_embedding --kind hnsw --metric cosine
   -> writes artifacts to indexes/clip_emb_hnsw/ (object-store backed),
      registers an "extensions.vector_index" block, bumps version atomically.

3. query (hybrid: SQL filter + ANN):
     ds.search(query_vec, k=50, where="language='en' AND duration_s<16")
   -> DataFusion predicate narrows candidates, the registered index does ANN.
```

The index blobs live under the dataset root (`indexes/<name>/`) on the same object
store, so the dataset stays self-contained and portable. The extension records
**`built_over_dataset_version` + `covers_rows` + `stale`**, so when new data is
appended the index is flagged stale/partial until rebuilt — no silent wrong results.
The same pattern serves future capabilities (full-text indexes, dedup/near-dup maps,
precomputed statistics, tokenized caches): add an `extensions.<name>` block and an
optional `indexes/<name>/` artifact directory; old readers ignore both.

---

## 4. Storage layer

Use the **`object_store`** crate (Arrow/DataFusion ecosystem). One `ObjectStore`
trait covers S3, GCS, Azure, and local FS, with:

- `get(path)` — full object,
- `get_range(path, Range<usize>)` — single byte range (this is the core primitive),
- `get_ranges(path, &[Range])` — coalesced multi-range (used for index ranges and
  multi-modal samples in the same shard),
- `put` / multipart upload — used by the writer.

Both requirements — "load a specific index" and "load a range of indices" — reduce
to byte-range GETs, so they're native at the I/O layer. Credentials, retries,
timeouts, and request coalescing are handled by the crate.

---

## 5. Global index & schema

The index is Parquet (columnar, compresses well, predicate/projection pushdown,
partitionable for billions of rows). Each row = one logical training sample.

**Locator columns** (how to fetch):

```
sample_id        u64        stable id, used for ordering & sharding
shard_id         u32        which shard
basename         string     member basename within the shard
video_off,len    u64,u64    byte span of the video member  (null if absent)
audio_off,len    u64,u64
image_off,len    u64,u64
text_off,len     u64,u64
```

**Metadata columns** (what to filter on — keep flat for pushdown):

```
duration_s   f32     fps        f32     width   u32   height u32
has_audio    bool    sample_rate u32    channels u8
language     string  caption_len u32    label   string/int
split        string  source     string  nsfw_score f32   ...
```

Schema is open: the writer accepts arbitrary scalar metadata per sample and adds
columns. Nested/locator data stays structured; queryable metadata stays flat so
DataFusion can push predicates down to row-groups.

---

## 6. SQL subsetting (DataFusion)

A subset is **a filtered index** — nothing in the shards moves.

Embed **DataFusion** (Arrow-native Rust SQL engine; plugs directly into
`object_store`). It reads the Parquet index with predicate + projection pushdown,
so a `WHERE` over a billion rows only scans relevant columns/row-groups.

```python
train_en = ds.subset("""
    SELECT sample_id FROM dataset
    WHERE has_audio AND duration_s BETWEEN 2 AND 16
      AND width >= 224 AND language = 'en'
      AND nsfw_score < 0.1
    ORDER BY sample_id
""")
```

- **Deterministic ordering.** Every DDP rank runs the identical query and gets the
  identical `ORDER BY sample_id` result *before* sharding — so partitions stay
  disjoint with zero cross-rank communication.
- **Virtual (default).** The subset is an in-memory `Vec<u64>` of sample_ids layered
  over the same shards. Zero copy, instant, ideal for experiments.
- **Materialized.** `train_en.save("subsets/train_en.parquet")` persists the filtered
  index for reproducibility/versioning. Shards are untouched; only the index is
  rewritten. Yields a frozen, citable subset definition.
- **Composable.** Subsets of subsets are AND-ed predicates. Cross-modal constraints
  (`has_audio AND caption_len > 0`) work because all modalities share a row. External
  label tables can be `JOIN`-ed if ever needed.

**Caveat:** SQL operates on an **index snapshot**. For a live-growing dataset,
re-run subsetting against the latest manifest version — the engine filters a
point-in-time index, not a moving stream.

---

## 7. Distributed sampling & sharding

A deterministic sampler producing disjoint slices across both DDP ranks and
DataLoader workers.

**Inputs:** `world_size`, `rank`, `num_workers`, `worker_id`, `epoch`, `seed`,
and the (ordered) id list (full dataset or a subset).

**Global worker id:** `g = rank * num_workers + worker_id`, total `G = world_size * num_workers`.

**Algorithm:**

1. Seeded permutation of the id list from `(seed, epoch)` — **block shuffle**
   (shuffle blocks of contiguous ids, then within blocks) to preserve shard
   locality and keep I/O coalesced.
2. Partition the permuted list across `G` buckets; bucket `g` is this worker's slice.
3. Track a per-worker consumed counter so resume = "skip first K of my slice."

This is communication-free and bit-for-bit reproducible from `(seed, epoch)`.

**Fork safety (critical):** with `num_workers > 0`, PyTorch forks workers. Tokio +
fork is unsafe, so the async runtime is created **lazily inside each worker** via
`worker_init_fn`, never in the parent. Each worker gets its own runtime scoped to
its slice.

---

## 8. Decoders (feature-gated)

Per-modality, returning tensors. Cargo features keep the build lean and let video
backends be optional.

| Modality | Crate / backend | Feature | Notes |
|---|---|---|---|
| Image | `image` + `zune-jpeg` | `image` (default) | Pure Rust, fast JPEG/PNG/WebP. |
| Audio | `symphonia` + `rubato` | `audio` (default) | Pure Rust decode (mp3/aac/flac/wav) + resample. |
| Video (CPU) | `ffmpeg-next` (libav FFI) | `video-cpu` (default) | Portable; links system ffmpeg. |
| Video (GPU) | NVDEC bindings | `video-nvdec` (opt-in) | CUDA toolchain; falls back to CPU if absent. |
| Text | passthrough | always | Returns raw string/bytes; tokenize in Python. |

**Honest caveat on "pure Rust":** there is no production-grade pure-Rust *video*
decoder. The core (I/O, index, sampler, prefetch, image, audio) is pure Rust; the
video codec links ffmpeg (CPU) or NVDEC (GPU) under the hood — same approach as
Decord/PyTorchVideo. Both are feature-gated: `video-cpu` is the portable default,
`video-nvdec` is opt-in and falls back to CPU when CUDA is unavailable.

**Video specifics:** temporal sampling (uniform / random / dense) selects T frames;
optional resize/crop; returns `[T, C, H, W]`. Audio aligned to the clip window when
both are present.

---

## 9. Prefetch, GIL, and tensor handoff

- A bounded Tokio task pool issues range GETs concurrently; decoded samples land in
  a ready queue (bounded, applies backpressure).
- The binding wraps fetch/decode in `py.allow_threads`, so the GIL is released
  while Rust does I/O and CPU decode — Python isn't blocked.
- Tensors are handed to PyTorch zero-copy via **DLPack** into **pinned memory**, so
  the H2D copy can overlap compute.
- A Rust `collate` pads variable-length audio / frame counts and emits attention
  masks; exposed as `collate_fn`.

---

## 10. Python API

> **Aspirational / target API.** This section is the design *target* and does not
> match the current implementation one-to-one — it shows `ds[a:b]` slicing,
> `for x in ds`, `ds.subset("SELECT …")` returning a view, `worker_init_fn`,
> `collate_fn=train.collate`, `state_dict()` resume, and remote/`world_size` at
> `open`, several of which are not yet built. For the **actual, current** Python
> surface see `PYTHON_API.md`; unbuilt items are tracked in the README roadmap.

### Reading

```python
import ferroload, torch
from torch.utils.data import DataLoader

ds = ferroload.Dataset(
    "s3://bucket/datasets/howto-av",      # single self-describing root
    modalities=["video", "audio", "text"],# subset of available modalities to load
    video={"num_frames": 16, "sampling": "uniform", "resize": (224, 224)},
    audio={"sample_rate": 16000, "mono": True},
    world_size=W, rank=R,                  # DDP dims; worker dims auto-filled
)

ds[1234]                 # specific index -> dict of tensors + raw text
ds[1000:1064]            # range of indices -> batch
len(ds)
for sample in ds:        # iterable, internal async prefetch
    ...

# SQL subset
train = ds.subset("SELECT sample_id FROM dataset WHERE split='train' ORDER BY sample_id")

loader = DataLoader(
    train, batch_size=64, num_workers=8,
    worker_init_fn=train.worker_init,    # creates per-worker tokio runtime + sharding
    collate_fn=train.collate,
    pin_memory=True,
)
```

### Resume

```python
state = loader.dataset.state_dict()      # {epoch, seed, per-worker consumed}
loader.dataset.load_state_dict(state)    # exact-resume, no replay
```

### Writing / dataset creation

```python
with ferroload.DatasetWriter(
    "s3://bucket/datasets/howto-av",
    modalities={"video": "mp4", "audio": "flac", "text": "json"},
    shard_bytes_target=1 << 30,           # ~1 GiB shards
) as w:
    for rec in source_records():
        w.add(
            key=rec.id,
            video=rec.mp4_bytes,
            audio=rec.flac_bytes,
            text={"caption": rec.caption, "lang": rec.lang},
            meta={"duration_s": rec.dur, "width": rec.w, "height": rec.h,
                  "has_audio": True, "language": rec.lang, "split": "train"},
        )
# on close: rolls/uploads tars, writes .tar.idx, appends index.parquet,
# computes stats, writes manifest atomically, snapshots versions/vN.json
```

### CLI

```
ferroload pack    <src> <dst-root>     # build a dataset from raw media
ferroload index   <root>               # (re)build .tar.idx + index.parquet
ferroload subset  <root> --sql "..."   # materialize a subset view
ferroload inspect <root>               # manifest, counts, schema, stats
ferroload verify  <root>               # checksum shards vs manifest
ferroload build-index <root> --column <c> --kind hnsw   # capability ext (e.g. vector search)
ferroload search  <root> --query ... -k 50 [--where SQL] # hybrid SQL filter + ANN
```

---

## 11. Sample record shape (returned to Python)

```python
{
  "video": Tensor[T, C, H, W] (uint8 or float),   # if requested & present
  "audio": Tensor[channels, samples],             # if requested & present
  "image": Tensor[C, H, W],                        # single-frame modality
  "text":  str | dict,                             # raw, tokenize downstream
  "sample_id": int,
  "meta": { ... },                                 # passthrough metadata
}
```

Collated batch pads ragged dims and adds `*_mask` tensors.

---

## 12. Extensibility: open schema & heterogeneous mixing

Datasets vary. Some videos carry **depth maps**, some images carry **bounding
boxes** for detection, some carry segmentation masks, optical flow, keypoints, or
fields we haven't thought of. And we must be able to **load such datasets together**
even when their schemas differ. So modalities are an *open registry*, not a fixed
enum, and per-sample data is split into three kinds, each handled differently.

### 12.1 Three kinds of per-sample data

| Kind | Examples | Where it lives | Returned as |
|---|---|---|---|
| **Tensor modality** | video, image, audio, **depth**, optical flow, seg masks | shard member + byte offset, decoded by a codec | Tensor |
| **Structured annotation** | **bounding boxes**, keypoints, polygons, timestamped captions | **inline in the Parquet index** as nested Arrow columns | ragged Tensor / list + mask |
| **Scalar metadata** | duration, language, label, width | flat index column | scalar (also SQL-filterable) |

**Rule of thumb:** *small + structured + queryable → inline in the index;
large + dense → a shard blob with a codec.* Depth maps go the blob route;
bounding boxes go the inline route. Both are first-class without format changes.

### 12.2 Open modality / codec registry

The manifest's `modalities` map is open. Each entry declares an extension, a kind,
and a codec:

```json
"modalities": {
  "video": { "ext": "mp4",       "kind": "tensor",     "codec": "video" },
  "depth": { "ext": "depth.png", "kind": "tensor",     "codec": "depth16" },
  "flow":  { "ext": "flow.npy",  "kind": "tensor",     "codec": "npy" },
  "boxes": { "ext": "det.json",  "kind": "annotation", "codec": "coco_bbox" },
  "text":  { "ext": "json",      "kind": "scalar",     "codec": "passthrough" }
}
```

A **`Codec` trait** (`fn decode(&self, bytes, params) -> Tensor`) backs each tensor
modality. Built-ins cover video/image/audio/depth(PNG16/EXR/npy)/raw. **Custom
codecs** are registrable two ways:

```python
# Python callback (runs in the worker, GIL held only for the call)
ds.register_codec("hyperspectral", lambda b, p: my_decode(b))
```

or as a Rust plugin compiled into a build. Unknown modalities with **no registered
codec are still loadable as raw bytes**, so a dataset never becomes unreadable.

### 12.3 Nested annotation columns

Variable-length annotations live inline in the index as Arrow nested types, e.g.
bounding boxes as `List<Struct{class:int, x:f32, y:f32, w:f32, h:f32, score:f32}>`.
Benefits: cheap (no extra GET), batch-collated into ragged tensors + masks, and
**SQL-queryable** — e.g. `WHERE array_length(boxes) BETWEEN 1 AND 20`.

### 12.4 Schema evolution within a dataset

Adding a field in `v2` uses Parquet schema evolution: old rows read back the new
column as null. Readers treat any absent field as "not present" (see masks below),
so old and new versions coexist.

### 12.5 Loading datasets together — `MixDataset`

```python
mix = ferroload.MixDataset(
    [ferroload.Dataset(a), ferroload.Dataset(b), ferroload.Dataset(c)],
    weights=[0.5, 0.3, 0.2],     # per-source interleave probability
    schema="union",               # "union" (default) | "intersection"
)
```

- **Union schema.** The result schema is the union of all sources' modalities and
  columns. A field missing from a source is returned as `None` plus a `*_present`
  mask, so a dataset without depth yields `depth=None, depth_present=False` and the
  model masks that loss term. (`intersection` keeps only shared fields if you prefer
  strictness.)
- **Deterministic mixing.** Sample ids are namespaced `(source_idx, local_id)`, so
  the rank x worker sampler stays disjoint and reproducible across the union.
- **Weighted interleave.** `weights` balances over- vs under-sized sources without
  physically rebalancing data.
- **SQL across the union.** DataFusion `UNION`s the sources with schema coercion;
  you can subset the whole mixture in one query.

### 12.6 Conflict detection

```
ferroload schema <root...>     # prints union schema + per-source diff
                               # flags type conflicts (e.g. label int vs string),
                               # codec mismatches, and unit mismatches before runtime
```

Catching `label: int64` vs `label: string` (or `duration_s` vs `duration_ms`) at
schema-check time avoids silent corruption when mixing.

---

## 13. Layered storage, enrichment & projection

A dataset is rarely written once and frozen. You compute depth from images,
detect boxes, precompute features, or add captions later — and at read time you
often want only *some* modalities. The format supports this with a **layered,
columnar storage model** plus **modality projection**.

### 13.1 Layers (additive, non-destructive)

Existing shards are never mutated (they may be remote, immutable, or shared).
Instead, a dataset is a **base layer plus zero or more enrichment layers**, each
contributing its own modalities/columns, joined on `sample_id`:

```
howto-av/
  manifest.json                 # lists ALL layers + their storage groups
  index/base.parquet            # sample_id, image_off/len, text, meta...
  index/depth.parquet           # sample_id, depth_off/len        <- added later
  index/boxes.parquet           # sample_id, boxes (inline annotation)  <- added later
  shards/base/shard-*.tar       # image + text blobs
  shards/depth/shard-*.tar      # depth blobs only                <- added later
  versions/v3.json
```

Because `sample_id` is **dense and contiguous**, joining layers is positional
alignment — O(1), no hash join. Partial enrichment (depth for only some samples)
is fine: missing entries surface through the `*_present` mask from section 12.
Each enrichment **bumps the manifest version and commits the manifest last**
(atomic); the prior version stays readable without the new column, and a layer can
be dropped by de-listing it.

### 13.2 Storage groups (hybrid default)

Each modality is assigned to a **storage group** = its own set of shards.

- **Co-located group:** several modalities packed together (good locality when you
  read them together).
- **Columnar group:** a modality alone in its own shards (skippable on read,
  independently enriched).

The **default is hybrid**: the importer may co-locate small modalities with their
primary (e.g., `text`/`audio` next to `video`), while enrichment layers are
*always* their own group. Any modality can be pinned to its own group. This is the
knob that trades all-modality locality against projection/enrichment efficiency.

### 13.3 Enrichment: a distributed map that writes a layer

```python
ds.map(
    fn=compute_depth,                       # `img` <- inputs; per-sample (batched=False)
    inputs=["image"],                       # reads ONLY image (projection)
    outputs={"depth": Modality("png", codec="depth16")},  # tensor -> new shard layer
    batch_size=32, num_workers=8, devices="cuda", resume=True,
)

ds.map(detect_objects, inputs=["image"],
       outputs={"boxes": Annotation()})     # annotation -> inline index fragment, no shards
```

- **Output kind decides the sink.** Tensor outputs become a new shard group +
  offset columns; annotation/scalar outputs become an inline index fragment. Same
  engine, two sinks.
- **Idempotent + resumable.** Output shards are named by input partition; a
  per-layer progress set tracks completed `sample_id`s, so re-runs skip done work.
- **Reuses the training sampler** for partitioning (see section 14 for distributed
  execution across nodes).

### 13.4 Projection reads (read only what you need)

The reader/loader takes a modality selection and skips everything else at two
levels — the Parquet index reads only the requested columns (projection pushdown),
and the fetch layer issues byte-range GETs only for the selected modalities:

```python
ferroload.Dataset(root, modalities=["text", "boxes"])   # never opens image/depth shards
ferroload.Dataset(root, modalities=["image", "depth"])  # never opens text
```

With columnar/layered grouping, projection becomes *"don't open those objects at
all"* — the largest possible I/O saving, because you read fewer bytes rather than
ranged-reading inside a shared object.

### 13.5 Optional compaction

Many layers ⇒ many objects per sample when you *do* read everything. A `ferroload
compact <root> --group train` step re-packs selected layers into co-located shards
to produce an optimal read-time layout, without losing the original layers.

---

## 14. Performance: fastest reads & distributed map

Two throughput goals: (a) **reads as fast as possible**, and (b) **map/enrichment
that scales across single and multiple nodes**. Both come from the same principles
— minimize bytes moved, hide latency with concurrency, keep stages overlapped.

### 14.1 The pipeline & the one rule

```
index lookup (in-mem, ns-us) -> fetch (ranged GET / cache, latency-hidden)
   -> decode (CPU/GPU, parallel) -> augment -> collate -> pinned host -> async H2D
```

Effective throughput = **the slowest stage**, not the sum — *if* stages overlap.
So the whole design is about (1) shrinking each stage and (2) running them
concurrently with bounded queues.

### 14.2 Making reads fastest

1. **Read fewer bytes — projection (section 13.4).** Biggest single win: never
   fetch modalities you won't use.
2. **Hide object-store latency with concurrency.** S3/GCS are latency-bound per
   request (~tens of ms) but high-bandwidth. The Tokio fetcher keeps many GETs in
   flight so you saturate bandwidth, not latency. One tiny serial GET per sample is
   the anti-pattern.
3. **Coalesce + go sequential.** `get_ranges` merges nearby ranges into fewer large
   GETs. **Block-shuffle** (shuffle blocks, then within blocks) keeps a microbatch's
   samples in the same shard so coalescing works; it's a tunable dial between full
   randomness and sequential megabyte reads. Streaming/iterable mode reads whole
   shards sequentially — far faster than random KB reads.
4. **Local NVMe cache tier (cache-aside, content-addressed).** Epoch 1 fills the
   cache; subsequent epochs read from local NVMe at GB/s. Critical for multi-epoch
   training. Locally, `mmap` shards and lean on the page cache.
5. **Don't double-compress.** Media (jpg/mp4/flac) is already compressed — store
   tars uncompressed (or lz4 for raw tensors only) so decode isn't bottlenecked by
   inflate.
6. **Decode is often the real limit (esp. video).** Mitigate with GIL-free parallel
   decode across `num_workers`, **NVDEC** GPU decode, or — via the enrichment map —
   **offline pre-extraction** (frames→JPEG, or precompute features) stored as a
   cheap-to-read layer. This is the deepest lever: pay heavy decode once, read
   tensors forever.
7. **Zero-copy handoff.** DLPack into **pinned memory**, async H2D overlapping
   compute, double-buffered prefetch queue. No copies between fetch→decode→tensor.
8. **In-memory / mmap index, O(1) positional lookup.** Dense `sample_id` means
   index access is array indexing, never a scan.

**Knobs exposed:** `prefetch_bytes` / `max_inflight_bytes` (byte budgets, see 14.5),
`io_concurrency`, `shuffle_block`, `bucket_by` (duration/bytes),
`cache_dir`/`cache_bytes`, `num_workers`, `decode_backend` (cpu/nvdec),
`max_member_bytes`, and per-modality `storage_group`.

### 14.3 Distributed map across single & multiple nodes

Enrichment must scale like training does. The map is **embarrassingly parallel
(map) + a trivial reduce (manifest merge)** — no shuffle.

**Work partitioning — shard-aligned.** The unit of work is a **shard (or shard
range)**, not a single sample. Each worker streams whole shards sequentially
(fast reads), computes, and writes its *own* output shards. The `sample_id` space
is partitioned over the global grid `nodes × procs/node × threads` using the same
deterministic sampler as training, so partitions are disjoint and reproducible.

**Partition-local writes, atomic commit.** During compute there is **no
coordination**: worker *w* writes `shards/depth/shard-w-*.tar` + a partial index
fragment. A final **commit step** concatenates fragment manifests into one layer
and writes the manifest atomically (the only synchronization point). Output shards
named by input partition make the whole job **idempotent and resumable** — a
completed-shard marker lets re-runs skip finished work and recover from stragglers
or crashes.

**Locality & overlap.** Prefer assigning a node the shards it already has cached;
co-locate compute with the bucket region. Reads (projection: only `image`) are
prefetched ahead, compute runs on batches, writes buffer into rolled output
shards — all overlapped.

**Scaling.** Throughput grows near-linearly with workers until it hits bucket
egress bandwidth, compute/decode, or write bandwidth. Because enrichment reads via
projection (only inputs) and writes only the new small modality, both ends move
minimal bytes.

### 14.4 Executor abstraction & automatic topology

The map backend is **abstracted behind an `Executor` interface**; caller code never
names Ray (or any backend). Single-vs-multi-node is **auto-detected** from the
launch environment — no manual flag.

```python
class Executor(Protocol):
    def map(self, plan: MapPlan) -> LayerCommit: ...   # plan = shard partitions + fn + outputs

# interchangeable implementations:
#   LocalExecutor            — multiprocessing / Rust threadpool; multi-GPU on one box
#   RayExecutor              — multi-node, Ray Data blocks == our shards
#   StaticPartitionExecutor  — torchrun/SLURM, deterministic slice, no queue
```

User code is backend-agnostic:

```python
ds.map(compute_depth, inputs=["image"], outputs={"depth": ...})   # backend auto-selected
```

**Topology is derived from launcher env vars** (which already encode node count and
per-node size), read in priority order:

| Source | Nodes | Per-node size | Rank |
|---|---|---|---|
| torchrun / PyTorch | `NNODES` (or `WORLD_SIZE/LOCAL_WORLD_SIZE`) | `LOCAL_WORLD_SIZE` | `NODE_RANK`, `RANK` |
| SLURM | `SLURM_NNODES` | `SLURM_NTASKS_PER_NODE` / `SLURM_GPUS_ON_NODE` | `SLURM_NODEID`, `SLURM_PROCID` |
| MPI | `OMPI_COMM_WORLD_SIZE` / `PMI_SIZE` | local comm size | world rank |
| Ray | `ray.is_initialized()` / `RAY_ADDRESS` | cluster resources | — |
| none (bare) | 1 | `CUDA_VISIBLE_DEVICES` count or `os.cpu_count()` | 0 |

```python
@dataclass
class Topology:
    num_nodes: int     # THE deciding variable: local vs distributed
    node_rank: int
    local_size: int    # gpus/procs per node -> intra-node parallelism
    world_size: int
```

**Selection rule:**

```
executor =
  env override (FERROLOAD_EXECUTOR=...) if set
  elif topo.num_nodes == 1:         LocalExecutor(local_size)   # zero Ray overhead
  elif ray available / RAY_ADDRESS: RayExecutor
  elif under SLURM/torchrun:        StaticPartitionExecutor
  else: raise (with guidance)
```

So `num_nodes` flips local <-> distributed and `local_size` sets intra-node
parallelism, both pulled from the launcher. Single node stays fully native (no Ray
process/object-store overhead); multi-node transparently uses Ray (or static
partition) with no code change. Swapping backends later is a one-class change.

**Single node detail:** the `LocalExecutor` runs an internal threadpool /
multiprocessing over local shards; model map fns (depth/detection) get GPUs
assigned round-robin (one or more workers per GPU) with batched inference.

> **Implementation status (v1).** `LocalExecutor` and `StaticPartitionExecutor`
> ship and are tested (`python/test_map.py`): topology auto-detection
> (`ferroload.detect_topology`), the selection rule, per-rank partition fragments
> (`shards/<name>/shard-<rank>-*.tar` + `index/<name>.part-<rank>.json`), and the
> merge/commit (`LayerWriter::commit`) are all implemented. Under
> `torch.distributed`, rank 0 commits after a barrier; otherwise call
> `ferroload.commit_layer(root, name, modalities)` once all ranks finish.
> `RayExecutor` is reserved (raises with guidance) — multi-node currently runs via
> the static-partition path under torchrun/SLURM.

### 14.5 Scaling hardening (large video & sparse columns)

Derived from a review of issues reported by users of comparable systems
(WebDataset, MosaicML Streaming, PyTorch `DataLoader`, S3, Arrow/Parquet, Ray
Data, NVIDIA DALI). See `PERF_REVIEW.md` for the full findings + sources. Four
changes are folded in here because they are load-bearing once samples vary
~1000x in size or columns become sparse.

**1. Byte-budgeted prefetch & in-flight I/O (highest impact).** A fixed
item-count `prefetch_depth`/`prefetch_factor` is unsafe when one video is
1000x another — it causes the memory spikes and OOMs reported across
WebDataset/MDS/PyTorch. So the prefetch queue and concurrent-fetch bound are
governed by **byte budgets** (`prefetch_bytes`, `max_inflight_bytes`), with item
count only a secondary cap. Peak host memory is then bounded and predictable:

```
peak_host ~= num_workers * (prefetch_bytes + decode_working_set)
             + shuffle_index_bytes + cache_pinned_bytes
```

Every term is a byte budget the user can size. (`shuffle_index_bytes` is small
because we shuffle ids/offsets, not media — see point 3.)

**2. Per-member size cap + clip-level sampling for long video.** The writer must
not produce a multi-GB shard from one giant video (it wrecks shuffle granularity
and memory; ~1 GB shards are the established sweet spot). A `max_member_bytes`
cap applies; above it the importer either gives the video its own single-member
shard or, for long-form video, splits it into **clip-level samples** (2-16 s) with
clip rows in the index. **Decision:** clip-level sampling is the default for
long-form video (this resolves the prior "video sharding granularity" open
question). Decode does temporal subsampling *during demux* (decode only kept
frames), never decode-all-then-drop.

**3. Duration/byte bucketing within shuffle blocks.** A batch mixing one huge
clip with many tiny samples causes head-of-line blocking and padding waste, and a
slow sample stalls the whole batch off the GPU. The sampler supports **bucketing**
(`bucket_by="duration"|"bytes"`): within a block-shuffle window, samples of similar
size are grouped so batches are balanced. Shuffling operates on **sample_ids /
offsets, not materialized bytes**, so shuffle-buffer memory is independent of media
size (avoids the WebDataset `shuffle(n)` blow-up). Presence-aware grouping handles
the sparse-modality version of the same imbalance.

**4. Per-source / per-layer index fragments; union is logical only.** Mixing many
datasets must **not** materialize one wide physical table of mostly-null modality
columns (wide-schema metadata overhead, wasted scans). Each layer/source keeps its
own index fragment; the `MixDataset`/union schema is a **logical view** resolved by
projection pushdown + positional join on dense `sample_id`. Null offsets are a
"skip, emit `*_present` mask" with **zero I/O** — sparse mixes never pay to fetch
absent modalities. (Parquet stores the nulls themselves cheaply via definition
levels, so the cost to avoid is wide-table scanning, not null storage.)

Supporting changes also adopted: return tensors + **Arrow-backed metadata** (never
large Python `dict`/`list`) across the worker boundary to avoid copy-on-write
memory growth; recommend `persistent_workers=True`; configure `object_store` with a
large connection pool + multipart parallel range GETs for big members; size Ray
blocks by bytes to avoid object-store spilling; and spill oversized annotations to a
blob modality above a threshold.

---

## 15. Build plan (phased)

1. **Core skeleton.** Workspace, `object_store` storage layer, manifest read/write,
   tar reader + `.tar.idx` reader. Bytes-in/bytes-out, no decode. Unit tests on local FS.
2. **Index + sampler.** Parquet index read, DataFusion subsetting, rank x worker
   deterministic sampler, resume state. Property tests for disjoint/complete partition.
3. **Writer + CLI.** `DatasetWriter`, shard rolling, `.idx` generation, atomic
   manifest commit, `ferroload pack/index/inspect/verify`.
4. **Decoders.** image -> audio -> video-cpu (feature-gated), then video-nvdec.
5. **PyO3 binding.** Map-style + iterable, prefetch, GIL release, DLPack pinned
   tensors, `worker_init_fn`, `collate_fn`. maturin wheels.
6. **Scale + DDP validation.** Multi-rank, multi-worker correctness; throughput
   benchmarks vs WebDataset/MDS; S3 + GCS soak test.
7. **Layers, enrichment & projection.** Layered index/shards, `ds.map` writing a
   new layer, modality projection on read, optional `compact`.
8. **Performance & distributed map.** NVMe cache tier, coalesced/block-shuffle I/O,
   distributed map (static partition + dynamic work queue), idempotent resume.

---

## 16. Key risks & decisions (locked)

| Decision | Choice | Note |
|---|---|---|
| Decode location | In Rust | GIL-free; video links ffmpeg/NVDEC. |
| Shard format | WebDataset tar + side-index | Compatible w/ existing tars. |
| Parallelism | Integrate with `num_workers` | Lazy per-worker tokio runtime (fork-safe). |
| Video backend | CPU + NVDEC, feature-gated | CPU default, GPU opt-in w/ fallback. |
| Modalities | video, image, audio, text + open registry | image = single-frame path. |
| Dataset format | Single self-contained root + manifest | Portable, versioned, atomic commit. |
| Subsetting | SQL via DataFusion | Virtual or materialized; snapshot semantics. |
| Storage layout | Layered + hybrid storage groups | Additive enrichment; projection-friendly. |
| Enrichment | `ds.map` -> new layer | Distributed, idempotent, resumable. |
| Read projection | Per-modality, skips whole objects | Read only requested modalities. |
| Cache tier | Local NVMe, content-addressed | Multi-epoch reads at GB/s. |
| Prefetch/I-O bounds | Byte budgets, not item counts | Safe under 1000x sample-size variance. |
| Long-form video | Clip-level samples + `max_member_bytes` cap | Avoids multi-GB shards / HOL blocking. |
| Batch balancing | Duration/byte bucketing; shuffle ids not bytes | Avoids stragglers + shuffle-buffer OOM. |
| Heterogeneous mix | Per-source/layer index fragments; logical union | No wide sparse mega-table; null = skip+mask. |
| Extensibility | `extensions` namespace + column `semantic`/`attrs` | Additive, version-floored, preserve-unknown; e.g. vector index over embeddings. |
| Distributed map | Pluggable `Executor`; backend auto-selected | Map-only + atomic manifest reduce. |
| Local vs distributed | Auto from env-derived `Topology` (`num_nodes`) | No manual flag; Ray hidden behind interface. |
| Scale-out backend | Local (default) / Ray / static-partition | Ray opt-in for multi-node only; Redis dropped. |

### Open questions for next round

- Default storage-group policy for the importer: which base modalities to co-locate
  vs split out by default?
- Default `max_member_bytes` and clip length (s) for long-form video splitting?
- Additional `Executor` backends worth shipping beyond Local / Ray / static-partition
  (e.g., Kubernetes Jobs)?
- Checksums/content-addressing scheme for shards (xxhash vs sha256).
```