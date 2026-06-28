# Changelog

All notable changes to Ferroload. Versions are unified across the Rust crates
(`ferroload-core`/`io`/`codec`) and the Python package (`ferroload`).

## [Unreleased]

## [0.1.1] - 2026-06-28

### Loader: streaming / iterable access pattern (one knob)
- New `FerroIterableDataset` — a streaming (`IterableDataset`) **view** over the
  same `ferroload.Dataset`, alongside the existing map-style `FerroTorchDataset`.
  It reads contiguous, shard-local blocks sequentially, decodes each block in
  parallel in Rust, and yields through a `shuffle_buffer` (WebDataset-style
  approximate shuffle). That's the object-store-friendly access pattern: few large
  sequential reads instead of many random GETs.
- Get it via `ds.iterable(...)` (counterpart to `ds.torch()`), or flip
  `make_loader(..., streaming=True, shuffle_buffer=…, block_size=…)`. Same
  `Dataset`, columns, decode, and cache — only the iteration strategy differs, so
  there's no second data pipeline. DDP/worker sharding is automatic (blocks split
  across `world_size` ranks, then across DataLoader workers); `set_epoch` reshuffles.
- Map-style (exact shuffle, random `ds[i]`) remains the default; streaming trades
  exact shuffle for sequential-read throughput on large/cloud datasets.

### Remote (object-store) streaming: `Dataset.open("s3://…")`
- `Dataset.open` (Python and Rust `Dataset::open_url`) now accepts object-store
  URLs — `s3://`, `gs://`, `az://`, plus `file://` / `memory://`. Shard bytes are
  streamed via **ranged GETs** through a content-addressed local cache (default
  `$FERROLOAD_CACHE` or a temp dir; override with `cache_dir=`/`make_loader(..., cache_dir=)`).
- The whole read path stays in Rust on a process-wide Tokio runtime, driven from
  the synchronous, **GIL-released** decode path — so cloud reads aren't GIL-bound.
  Batched reads group by shard and issue a single coalesced `get_ranges` (few
  requests, parallelized + cached). The local-filesystem path is unchanged
  (positional `read_exact_at`), so local performance is unaffected.
- Credentials/region come from the environment (`AWS_*` / `GOOGLE_*` / `AZURE_*`).
  The `cloud` feature builds all backends at once (S3 + GCS + Azure; pure-Rust
  rustls, so portable) and is what the published wheel and `scripts/build-package.sh`
  ship; `--features aws`/`gcp`/`azure` build a single backend. Base build is local-only.
- Remote datasets are **read-only** (no `map`/writes) for now. `ferroload-io`
  gains `Storage::from_url`, a shared blocking runtime, and coalesced cached
  `get_ranges`; `ferroload-core` gains the feature-gated `remote` backend.

### Loader: per-column `resize`
- `resize` now also accepts a per-column dict `{col: (H, W) | None}` (covers image
  and video columns; `None` = no resize for that column), alongside the existing
  global `(H, W)` tuple. Resolved per column in `FerroTorchDataset`.

### Loader: `columns=[...]` resolves each column's kind from the manifest
- `make_loader` / `FerroLoader` / `FerroTorchDataset` (and `ds.torch()/.numpy()/.jax()`)
  accept a flat `columns=[...]` list instead of bucketing names into
  `images=`/`videos=`/`raw=`/`meta=`. Each name is resolved against the dataset's
  modalities: image/video codecs decode, `.npy` tensor columns load as arrays
  (new `arrays=` bucket), other modalities pass as raw bytes, and non-modality
  names are metadata keys. Explicit buckets still work and are merged with the
  resolved ones.

### `map` inputs/outputs are positional + per-sample (breaking)
- **Positional input binding.** `map`'s `fn` is now bound to `inputs` by position —
  it receives one argument per name in `inputs`, in order, and no longer indexes a
  `batch` dict by column name. This makes map functions generic and reusable across
  datasets (e.g. `def download(url): ...` instead of `batch["thumbnail_loc"]`).
- **Per-sample by default.** `fn` runs once per sample (`batched=False`, the new
  default) and returns that sample's output, or a tuple in `outputs` order. Image
  inputs are still decoded in parallel in Rust, so per-sample mode keeps the fast
  decode — only `fn` runs per row.
- **New `batched=True` mode.** Opt in for vectorized or I/O-fan-out work: `fn`
  receives one list per input and returns a list (or tuple of lists). Recommended
  for threaded downloads.
- **Outputs by position.** A single output returns the value/list directly; multiple
  outputs return a tuple aligned to the `outputs` declaration. The old
  `return {name: [...]}` dict form is removed. `_check_outputs` is replaced by
  `_run_fn`. `Modality(...)`/`Annotation()`/string/dict output specs are unchanged.

## [0.13.0]

### Enrichment aligned to DESIGN §13–14 (distributed map)
- **Positional join (no hash join).** Layers join to the base by a direct
  `sample_id -> row` table (dense/contiguous ids, O(1)) instead of a per-layer
  HashMap (DESIGN §13.1). `IndexRow` gains an optional `shard` filename so a
  partitioned layer's `shard-<rank>-*` files resolve without renames (base
  unaffected; field omitted when absent).
- **Layer layout** now matches the design: `index/<name>.json` + `shards/<name>/`
  (was `layers/<name>/...`).
- **Distributed map.** `LayerWriter` gains a **partitioned** mode
  (`create_partition(part)`) that writes its own `shard-<part>-*` shards + an
  `index/<name>.part-<part>.json` fragment and a `.done` marker, touching no shared
  state, plus `LayerWriter::commit(root, name, modalities)` that merges all
  fragments into one layer and registers it atomically (DESIGN §14.3).
- **Executor abstraction (DESIGN §14.4).** New `ferroload.executor`: `Topology` +
  `detect_topology()` (torchrun/SLURM/MPI/Ray/bare), `LocalExecutor`,
  `StaticPartitionExecutor` (per-rank partition → fragment → commit; rank 0 commits
  after a `torch.distributed` barrier), and a reserved `RayExecutor`.
  `select_executor()` + `FERROLOAD_EXECUTOR` override; `commit_layer()` for the
  manual commit. `Dataset.map(..., executor=None)` auto-selects from topology
  (single-node default is unchanged).
- **Typed outputs (DESIGN §13.3).** `map(outputs=...)` now also accepts
  `ferroload.Modality(ext, codec=...)` (tensor/blob) and `ferroload.Annotation()`,
  alongside the existing string/dict shorthands.
- Bindings: `LayerWriter(partition=...)` + `LayerWriter.commit`; tests in
  `tests/layers.rs` (partitioned commit) and `python/test_map.py` (typed outputs,
  topology detection, simulated 3-rank static-partition + commit).

## [0.12.0]

### Enrichment — `Dataset.map`
- `Dataset.map(fn, inputs, outputs, name=..., batch_size=..., resume=True)` runs a
  function over the dataset and stores results as a new **additive layer** joined
  on `sample_id`. The base data is never rewritten, the pass is **idempotent and
  resumable**, and outputs read back as ordinary modalities/metadata.
  - Tensor outputs (`'array'`) become a new `.npy`-backed modality in
    `layers/<name>/shards/`; **raw-bytes** outputs (`'bytes'`/'raw' or media
    shorthands `'video'`/'audio'/'image', or `{'type':'bytes','ext','codec'}`)
    store the bytes the fn returns verbatim as a new modality (e.g. downloading a
    `video_url` column into a real `.mp4` modality); scalar/text outputs
    (`'scalar'`/'text') are stored inline in the layer index and merged into
    `meta` on read.
  - Image inputs are decoded to arrays in parallel in Rust (GIL released);
    `.npy` tensor-layer inputs are auto-loaded as arrays (chained maps);
    metadata keys are passed as scalars.
  - Read tensor outputs back with `read_array(i, m)` / `read_arrays(indices, m)`.
  - Works on a `subset(...)` view (writes a sparse layer over those ids).
- **Core**: `manifest.layers[]` + `LayerRef`; layer-aware `Dataset` reads
  (`resolve` to base or layer shard, merged `meta`, projection across layers);
  new `LayerWriter` (writes layer shards + index fragment, registers the layer in
  the manifest atomically with a version bump; re-opening appends, for resume).
  Back-compat: a manifest without `layers` reads exactly as before.
- **Bindings**: `_core.LayerWriter` (+`existing_ids` for resume); `Dataset.modalities()`
  now includes layer modalities; new `Dataset.root` getter.

## [0.11.0]

### Subsetting returns a Dataset (breaking)
- `Dataset.subset(where_sql)` now returns a **new (subset) `Dataset`** — an
  index-remapped view supporting `get`/`decode_many`/`meta_batch`/`.torch()` and
  further `.subset()`. Pass `return_indices=True` for the old `list[int]` behavior.
  (CLI `subset` and any callers wanting ids updated to `return_indices=True`.)

## [0.10.0]

### Fluent framework views
- `ferroload.Dataset.open(root)` now returns a handle with
  `.torch(...)` / `.numpy(...)` / `.jax(...)` that return per-sample-dict datasets
  in the requested array type; `out="jax"` added to the loader. All core reader
  methods are delegated; `ds.reader` exposes the raw `_core.Dataset`.

## [0.9.0]

### Distributed sampling + async prefetch
- `ferroload.Sampler` — the Rust deterministic rank×worker sampler, exposed to
  Python (`.indices(epoch, resume_from)`).
- `loader.FerroSampler` — torch-compatible, DDP-aware, resumable sampler with
  `set_epoch` (drop-in for `DistributedSampler`).
- `loader.PrefetchLoader` (+ `batched`, `numpy_collate`) — background-thread
  prefetch that overlaps the GIL-released Rust decode with consumption.
- `ferroload.make_loader(root, batch_size, …)` / `FerroLoader` — one-call
  initializer bundling open + dataset + sampler + prefetch.

## [0.8.0]

### Python API (tightened — see `API_REVIEW.md`)
- **Canonical names** `ferroload.Dataset` / `ferroload.Writer` (aliases of
  `FerroDataset`/`FerroWriter`).
- `Dataset.__getitem__` (`ds[i]`), and introspection: `name`, `version`,
  `modalities()`, `schema()`, `manifest()`.
- `decode_audio(indices, modality)` — WAV/PCM decode to `[channels, samples]`.
- `resize=(height, width)` validated (`> 0`).
- **Precise exceptions:** `IndexError` / `FileNotFoundError` / `RuntimeError` /
  `ValueError` (was: everything `ValueError`).
- `loader.subset_dataset(tds, ids)` to consume `Dataset.subset()` results.

### Packaging
- Maturin **mixed package**: extension at `ferroload._core`, pure-Python
  `ferroload` package (`loader`, `cli`), abi3 wheel, `ferroload` console script.
- Removed `strip` from maturin config (it invalidated the arm64 ad-hoc signature
  and hung `import` under macOS AMFI/Gatekeeper).

### Docs
- Added `PYTHON_API.md` (canonical Python reference) and `API_REVIEW.md` (audit).
- Fixed stale `README.md` build/import + status matrix; corrected `lib.rs`
  module docstring; flagged `DESIGN.md` §10 as aspirational.
- Unified crate versions to 0.8.0.

## Earlier (pre-changelog, milestone summary)
- **Format core:** self-contained dataset (manifest + tar shards + side-index +
  index), atomic versioned commits, extensible manifest (`extensions` namespace).
- **Index:** JSON (default) + Parquet/Arrow backend (`parquet` feature),
  projection, presence masks.
- **Sampler:** deterministic rank×worker (Rust).
- **Subsetting:** lightweight `WHERE` evaluator (DataFusion is the prod swap-in).
- **I/O:** `object_store` (local/memory; S3/GCS/Azure gated) + content-addressed
  cache.
- **Codec:** image (PNG/JPEG), audio (WAV), temporal frame sampling; video decode
  gated behind `video-ffmpeg`/`video-nvdec`.
- **Python:** writer/reader, parallel image/video decode, `meta_batch`, `subset`,
  PyTorch `FerroTorchDataset`, HuggingFace importers, `ferroload` CLI + catalog.
