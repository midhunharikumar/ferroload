# ferroload-rs

Pure-Rust implementation of the **Ferroload** multimodal dataset format and
runtime, plus Python bindings. Organized as a Cargo workspace of focused crates.

## Workspace layout

```
ferroload-rs/
  Cargo.toml                       # workspace (py crate excluded; built separately)
  crates/
    ferroload-core/                # format core: manifest, shards, index, sampler, subset
      src/manifest.rs              #   extensible manifest (preserve-unknown + extensions)
      src/shard.rs                 #   USTAR tar writer w/ exact offsets + random read
      src/sideindex.rs             #   per-shard member -> (offset,len)
      src/index.rs                 #   sample index + projection + pluggable backend
      src/index_parquet.rs         #   parquet/arrow backend (feature `parquet`)
      src/sampler.rs               #   deterministic rank x worker sampler
      src/subset.rs                #   WHERE-clause subsetting over metadata
      src/dataset.rs               #   DatasetWriter / Dataset + atomic versioned commit
      examples/synthetic_av.rs     #   runnable worked example
      tests/integration.rs
    ferroload-io/                  # object_store storage (local/mem; cloud gated) + cache
    ferroload-codec/               # image + WAV-audio decoders; video gated; frame sampling
    ferroload-py/                  # maturin mixed package -> Python `ferroload`
      pyproject.toml               #   wheel config (module `ferroload._core`)
      python/ferroload/            #   __init__.py, loader.py (torch glue), cli.py
      src/lib.rs                   #   PyO3 bindings (the `_core` extension)
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
| Parquet/Arrow index backend | core (`parquet`) | implemented | ✅ (feature build) |
| object_store storage + ranged reads | io | implemented | ✅ (local + in-memory) |
| Content-addressed NVMe cache | io | implemented | ✅ |
| S3 / GCS / Azure backends | io (`aws`/`gcp`/`azure`) | implemented | ⛔ needs cloud creds |
| Image decode (PNG/JPEG) | codec | implemented | ✅ |
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

## Build & test

The mounted output filesystem blocks the temp-file deletes cargo does while
linking, so point the build target at local disk:

```bash
export CARGO_TARGET_DIR=/tmp/ferro-target

cargo test                                   # core (default) + io + codec
cargo test -p ferroload-core --features parquet   # + parquet/arrow backend
cargo run -p ferroload-core --example synthetic_av
```

Test counts (this environment): core 28 + parquet 29, io 4, codec 9, integration 1,
combinations 4, layers 2 (enrichment `map`), doctest 1 — all passing. The Python
`map` end-to-end suite is `python/test_map.py`.

### Python package (maturin)

It's a maturin mixed package; build/install with maturin (a fresh venv/conda env):

```bash
cd crates/ferroload-py
pip install maturin
maturin develop --release            # dev install into the active env
#   add --features video for in-Rust video decode (needs system ffmpeg)
maturin build   --release            # -> target/wheels/ferroload-*-abi3-*.whl  (pip install it)
python -c "import ferroload; print(ferroload.__version__)"
ferroload --help                     # CLI is installed too
```

```python
import ferroload
w = ferroload.Writer("/data/ds", "demo")          # canonical (FerroWriter = deprecated alias)
w.declare("image", "jpg", "tensor", "image")
w.add("s0000", {"image": open("a.jpg","rb").read()}, {"label": 3})
w.close()

ds = ferroload.Dataset.open("/data/ds")           # canonical (FerroDataset = deprecated alias)
ds[0]                                              # full sample dict (map-style)
ds.get(0, ["image"])                               # projection: only fetch 'image'
ds.decode_many([0,1,2], "image", resize=(224,224)) # parallel decode -> NumPy [H,W,C]

# Enrich: run a fn over the data, store results as a new additive layer
# (joined on sample_id; idempotent + resumable; base never rewritten).
# fn is bound to `inputs` positionally and runs once per sample (batched=False).
def enrich(img):                                   # `img` <- inputs=["image"]
    return estimate_depth(img), classify(img)      # (array -> modality, text -> metadata)
ds = ds.map(enrich, inputs=["image"], outputs={"depth": "array", "tag": "text"}, name="features")
ds.read_array(0, "depth")                          # consume tensor output -> NumPy
ds.get(0)["meta"]["tag"]                            # scalar/text output -> metadata
```

📖 **Documentation** is built with MkDocs (Material) and deploys to GitHub Pages —
see [`docs/`](docs/) or run `pip install -r requirements-docs.txt && mkdocs serve`.
It covers the [Python API](docs/python/api.md), the
[HuggingFace + CLI quickstart](docs/python/quickstart.md),
[Rust core usage](docs/rust/usage.md), and [benchmarks](docs/benchmarks.md).
There's also a runnable `notebooks/ferroload_demo.ipynb` walkthrough.
(`DESIGN.md` lives at the repo root and is intentionally not part of the docs site.)

> macOS note: don't set `strip = true` in `[tool.maturin]` — stripping invalidates
> the linker's ad-hoc signature on arm64 and makes `import` hang (AMFI/Gatekeeper).

## Roadmap

- DataFusion SQL over the Parquet index (replaces the lightweight WHERE evaluator).
- **Byte-budgeted** prefetch in the Rust core + pinned memory / DLPack hand-off
  (the rank×worker sampler and a background-thread prefetch loader are now wired —
  `ferroload.Sampler` / `loader.FerroSampler` / `loader.PrefetchLoader`).
- Distributed/Ray `map` executor (the local single-process executor ships now —
  `Dataset.map`, writing additive layers); GPU (NVDEC/nvJPEG) decode.

See `USAGE.md`/`EXAMPLES.md` (Rust core API), `PYTHON_API.md` (Python API),
`API_REVIEW.md` (API audit), and `DESIGN.md` / `PERF_REVIEW.md` (design +
bottleneck analysis).
