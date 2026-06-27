# ferroload-rs

Pure-Rust implementation of the **Ferroload** multimodal dataset format and
runtime, plus Python bindings. A fast, cloud-native dataset format for ML
training: sharded tar data with a **columnar, DuckDB-queryable Parquet index**,
streamed from local disk or object storage with parallel in-Rust decode.
Organized as a Cargo workspace of focused crates.

## Quickstart

Build the Python package from source with [maturin](https://www.maturin.rs/) into
your env:

```bash
pip install maturin
cd crates/ferroload-py
maturin develop --release          # add --features cloud for s3:// gs:// az://
                                   #     --features turbojpeg for libjpeg-turbo decode
```

Write a dataset, then open, query, and train on it:

```python
import ferroload

# --- write ---
w = ferroload.Writer("/data/ds", "demo")
w.declare("image", "jpg", "tensor", "image")
w.add("s0000", {"image": open("a.jpg", "rb").read()}, {"label": 3, "caption": "a cat"})
w.close()

# --- read / query ---
ds = ferroload.Dataset.open("/data/ds")          # local path, or "gs://…" / "s3://…" / "az://…"
ds[0]                                            # full sample dict; ds.get(0, ["image"]) to project
ds.subset("label = 3", return_indices=True)      # query the Parquet index -> [sample_id, …]
```

Train with PyTorch — `make_loader(out="torch")` yields collated torch tensors with
parallel in-Rust decode (no `num_workers` needed):

```python
import torch
import ferroload

dl = ferroload.make_loader("/data/ds", batch_size=64, images=["image"],
                           meta=["label"], resize=(224, 224), out="torch")
device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(epochs):
    dl.set_epoch(epoch)                          # deterministic reshuffle (DDP-aware)
    for batch in dl:
        x = batch["image"].permute(0, 3, 1, 2).float().div_(255).to(device)  # NHWC u8 -> NCHW f32
        y = batch["label"].to(device)
        loss = loss_fn(model(x), y)
        loss.backward(); opt.step(); opt.zero_grad()

# Prefer the vanilla API? `ferroload.loader.FerroTorchDataset` plugs straight into
# torch.utils.data.DataLoader; cloud-scale streaming via make_loader(..., streaming=True).
```

Remote datasets stream over the network — `open()` reads only the manifest, and the
index is queryable **directly by DuckDB** without Ferroload:

```sql
SELECT sample_id FROM read_parquet('gs://bucket/ds/index/part-*.parquet')
WHERE label = 3 AND caption LIKE '%cat%';
```

Enrich a dataset with `Dataset.map(fn, …)` to add derived modalities/columns as
additive, idempotent layers; a `ferroload` CLI ships too. See
[Documentation](#documentation).

## Performance

Benchmarked 3-way against **WebDataset** and **HF `datasets` (Arrow)** (the loader
🤗 diffusers training uses) on CIFAR-10, Stanford-Cars, and FFHQ-256, plus GCS
streaming. All formats are built from the *same encoded image bytes* — full report
in **[BENCHMARKS.md](BENCHMARKS.md)**.

![Local throughput](benchmarks/charts/throughput.png)
![GCS streaming](benchmarks/charts/gcs_streaming.png)

- **Tiny images:** Ferroload wins apples-to-apples — even in the *same*
  `DataLoader(nw=8)` it's **1.8× HF, 3.9× WebDataset** (no multiprocessing tax).
- **JPEG decode-bound:** Ferroload **native** (libjpeg-turbo + SIMD `fast_image_resize`)
  now beats WebDataset nw=8 from one process; even the pure-Rust default beats HF
  nw=8. (In a worker `DataLoader` it's IPC-bound, ~on par with HF — run native.)
- **GCS streaming:** **8.9× faster** after coalescing remote reads — now ahead of
  WebDataset, while keeping random access + a DuckDB-queryable index.
- **Storage:** always smaller than WebDataset (≈2× for tiny images).

## Build & test

```bash
export CARGO_TARGET_DIR=/tmp/ferro-target   # if the mounted FS blocks cargo's temp deletes

cargo test                                        # core (Parquet index is default) + io + codec
cargo test -p ferroload-core --features remote    # + remote object-store / ranged reads
cargo run  -p ferroload-core --example synthetic_av
```

The core/io/codec unit tests, the integration/combinations/layers suites, and the
doctest all pass; the Python end-to-end suites live in `python/` (e.g.
`python/test_map.py`) and the benchmark harness in [`benchmarks/`](benchmarks/).

Python wheel (abi3) via maturin:

```bash
cd crates/ferroload-py
maturin build --release            # -> target/wheels/ferroload-*-abi3-*.whl  (pip install it)
#   --features video      in-Rust video decode (needs system ffmpeg + clang)
#   --features turbojpeg  libjpeg-turbo JPEG decode (needs libjpeg-turbo)
python -c "import ferroload; print(ferroload.__version__)"
ferroload --help                   # CLI is installed too
```

> macOS note: don't set `strip = true` in `[tool.maturin]` — stripping invalidates
> the linker's ad-hoc signature on arm64 and makes `import` hang (AMFI/Gatekeeper).

## Workspace layout

```
ferroload-rs/
  Cargo.toml                       # workspace (py crate excluded; built separately)
  crates/
    ferroload-core/                # format core: manifest, shards, index, sampler, subset
      src/manifest.rs              #   extensible manifest (preserve-unknown + extensions)
      src/shard.rs                 #   USTAR tar writer w/ exact offsets + random read
      src/sideindex.rs             #   per-shard member -> (offset,len)
      src/index.rs                 #   sample index + projection + lazy sharded reader
      src/index_parquet.rs         #   columnar Parquet/Arrow index (sharded, lazy, zstd) — default
      src/sampler.rs               #   deterministic rank x worker sampler
      src/subset.rs                #   WHERE-clause subsetting (+ column projection / RG pruning)
      src/dataset.rs               #   DatasetWriter / Dataset + atomic versioned commit
      examples/synthetic_av.rs     #   runnable worked example
      tests/integration.rs
    ferroload-io/                  # object_store storage (local/mem; cloud gated) + cache
    ferroload-codec/               # image (PNG/JPEG, opt-in turbojpeg) + WAV audio; video gated
    ferroload-py/                  # maturin mixed package -> Python `ferroload`
      python/ferroload/            #   __init__.py, loader.py (torch glue), cli.py
      src/lib.rs                   #   PyO3 bindings (the `_core` extension)
  benchmarks/                      # 3-way benchmark harness + report + charts
  notebooks/ferroload_demo.ipynb   # runnable usage walkthrough
  python/                          # dev scripts: importers, benchmarks, tests
```

## Status matrix

| Component | Crate | State | Tested here |
|---|---|---|---|
| Manifest + extensibility | core | implemented | ✅ |
| Tar shards + side-index + random read | core | implemented | ✅ |
| Sample index + projection + presence masks | core | implemented | ✅ |
| Atomic versioned commit | core | implemented | ✅ |
| Deterministic rank×worker sampler | core | implemented | ✅ |
| WHERE-clause subsetting | core | implemented | ✅ (DataFusion is the production swap-in) |
| Enrichment `map` → additive layers (idempotent/resumable) | core + py | implemented | ✅ positional join, typed outputs |
| Distributed `map`: Local + StaticPartition executors, auto-topology | core + py | implemented | ✅ (Ray executor reserved) |
| Sharded lazy Parquet/Arrow index (zstd; column projection + row-group pruning; DuckDB-queryable) | core | implemented | ✅ (default format) |
| object_store storage + ranged reads (incl. remote coalesced/columnar) | io + core | implemented | ✅ (local + in-memory) |
| Content-addressed NVMe cache | io | implemented | ✅ |
| S3 / GCS / Azure backends | io (`aws`/`gcp`/`azure`) | implemented | ⛔ needs cloud creds |
| Image decode (PNG/JPEG; SIMD resize; opt-in libjpeg-turbo via `turbojpeg`) | codec | implemented | ✅ |
| Audio decode (WAV/PCM) | codec | implemented | ✅ |
| Temporal frame sampling | codec | implemented | ✅ |
| Video decode (ffmpeg / NVDEC) | codec (`video-ffmpeg`/`video-nvdec`) | implemented | ⛔ needs system ffmpeg + clang |
| Python bindings: write/read/projection/subset | py | implemented | ✅ |
| Python parallel decode (image; video gated) | py | implemented | ✅ image / ⛔ video |
| PyTorch loader (`FerroTorchDataset`, per-sample dicts) | py (loader) | implemented | ✅ |
| `ferroload` CLI + dataset catalog | py (cli) | implemented | ✅ |
| Installable wheel (maturin, abi3) | py | implemented | ✅ (built + installed) |

Everything marked ⛔ is real code that is **feature-gated** because this sandbox
lacks the native libs (ffmpeg/clang) or credentials (cloud); it compiles where
those are present.

## Documentation

Built with MkDocs (Material) and deployed to GitHub Pages — see [`docs/`](docs/) or
run `pip install -r requirements-docs.txt && mkdocs serve`. It covers the
[Python API](docs/python/api.md), the
[HuggingFace + CLI quickstart](docs/python/quickstart.md),
[Rust core usage](docs/rust/usage.md), and [benchmarks](docs/benchmarks.md). There's
also a runnable `notebooks/ferroload_demo.ipynb` walkthrough, and design/API notes
at the repo root (`DESIGN.md`, `PYTHON_API.md`, `EXAMPLES.md`, `API_REVIEW.md`,
`PERF_REVIEW.md`).

## Roadmap

- DataFusion SQL over the Parquet index (replaces the lightweight WHERE evaluator).
- **Byte-budgeted** prefetch in the Rust core + pinned memory / DLPack hand-off
  (the rank×worker sampler and a background-thread prefetch loader are wired —
  `ferroload.Sampler` / `loader.FerroSampler` / `loader.PrefetchLoader`).
- Distributed/Ray `map` executor (the local single-process executor ships now —
  `Dataset.map`, writing additive layers).
- Faster/GPU decode: decode-at-scale (libjpeg-turbo DCT), nvJPEG/nvImageCodec —
  see [benchmarks/DECODE_OPTIMIZATION.md](benchmarks/DECODE_OPTIMIZATION.md).
