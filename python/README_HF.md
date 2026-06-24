# Quickstart: test Ferroload with a HuggingFace dataset

End-to-end: build the Python extension, import a small HF dataset, read it back,
and benchmark against HF `datasets`. These exact steps were run successfully on
`ylecun/mnist`.

## 1. Build & install the `ferroload` package

The project is a maturin mixed Rust/Python package: a `ferroload` Python package
(with `ferroload.loader` and the `ferroload` CLI) wrapping the compiled extension
at `ferroload._core`. Use a virtualenv/conda env.

### Dev install (editable-style, optimized)

```bash
cd crates/ferroload-py
pip install maturin
maturin develop --release            # builds + installs into the active env
python -c "import ferroload; from ferroload import loader; print(ferroload.__version__)"
ferroload --help                      # console command is installed too
```

### Build a wheel you can install anywhere

```bash
cd crates/ferroload-py
maturin build --release               # -> target/wheels/ferroload-<ver>-cp39-abi3-<platform>.whl
pip install ../../target/wheels/ferroload-*.whl   # or wherever CARGO_TARGET_DIR points
```

Notes:
- The wheel is **abi3** (`cp39-abi3`), so one wheel works on CPython ≥ 3.9, but it
  is **platform-specific** (macOS arm64 wheel ≠ Linux x86_64 wheel — build on each
  target, or use `cibuildwheel`).
- HuggingFace import needs extras: `pip install 'ferroload[hf]'` (datasets/pillow/
  huggingface_hub).
- **Video:** add `--features video` (and the macOS ffmpeg env vars from §3d). That
  wheel links your system ffmpeg dylibs and is only portable to machines with the
  same ffmpeg unless you bundle them with `delocate`/`auditwheel`.

## 2. Install the importer's Python deps

```bash
pip install datasets pillow         # add --break-system-packages on managed Pythons
```

## 3. Import a small HF dataset

```bash
export PYTHONPATH=/tmp/pyimp
python3 python/import_hf.py ylecun/mnist /tmp/ds_mnist --limit 64
```

Expected:

```
packed 64 samples -> /tmp/ds_mnist
reopened: 64 samples, 1 shard(s)
sample0: image_bytes=549, meta={'label': 5}
verify: 64 samples OK
```

It streams the dataset (only `--limit` rows are pulled), auto-detects the image
column, stores images as PNG members and other columns as metadata. Use
`--name CONFIG`, `--split`, and `--image-col` for other datasets, e.g.:

```bash
python3 python/import_hf.py rotten_tomatoes /tmp/ds_rt --limit 50   # text-only
python3 python/import_hf.py uoft-cs/cifar10 /tmp/ds_cifar --limit 64
```

## 3b. Import a JSONL-manifest + media-files dataset (e.g. video)

Some datasets aren't Arrow — they're a `*.jsonl` manifest plus referenced media
files (e.g. `MiG-NJU/OmniVideo-Test`: 505 rows + `videos/*.mp4`, ~7 GB total).
Use the file-based importer, which pulls **only the first `--limit` media files**
(not the whole repo):

```bash
python3 python/import_hf_files.py MiG-NJU/OmniVideo-Test /tmp/ds_omni \
    --jsonl test_505.jsonl --media-field video_path --modality video --ext mp4 --limit 6
```

Verified run (6 clips, 62.3 MB): each mp4 is stored as a `video` modality blob
with all JSONL fields (question, options, answer, task, duration, resolution, …)
as metadata; reopen + `verify` pass; a metadata-only projection read fetches
**zero** video bytes:

```
packed 6 samples, 62.3 MB of video -> /tmp/ds_omni
reopened: 6 samples, 1 shard(s)
sample0: video_bytes=7729223, present=True
  task=fine_grained_perception answer=B dur=521
projection (no modalities): fetched video? False
verify: 6 samples OK
```

## 3c. Feed a DataLoader (idiomatic — no special collate)

The dataset yields **per-sample dicts** and a normal `DataLoader` batches them
with its `collate_fn` (default or your own). The parallel Rust decode happens
transparently under PyTorch's batched-fetch hook `__getitems__`.

```python
import ferroload, ferroload_loader            # python/ on PYTHONPATH
from torch.utils.data import DataLoader
fd = ferroload.Dataset.open("/tmp/ds_mnist")

ds = ferroload_loader.FerroTorchDataset(
        fd,
        images=["image"],        # decoded (resized) to arrays
        raw=["video"],           # returned as raw bytes (decode in your transform)
        meta=["label","caption"],# attached from the index (no I/O)
        resize=(224,224), out="torch")

loader = DataLoader(ds, batch_size=64, num_workers=4)   # default collate stacks tensors
for batch in loader:
    batch["image"]   # [B,224,224,3] uint8
    batch["label"]   # [B] int64
    batch["caption"] # list[str]

# custom collate works exactly like normal PyTorch:
def my_collate(samples):       # samples = list[dict]
    ...
loader = DataLoader(ds, batch_size=64, collate_fn=my_collate)
```

A sample dict is flexible in modalities — multiple images, multiple video streams,
audio, metadata, in any combination; absent modalities come back as `None` with a
`<name>_present` flag. See `python/test_combinations.py`.

## 3d. Video decode (build with the `video` feature)

In-Rust video decode uses ffmpeg, so it's opt-in and needs system libraries on
the build machine (it is *not* in the default build).

```bash
# macOS prerequisites
brew install ffmpeg pkg-config
xcode-select --install            # clang/Command Line Tools, if not already present

# IMPORTANT (macOS/Homebrew): ffmpeg-sys-next's bindgen runs clang, which defaults
# to /usr/include where Homebrew does NOT put headers. Point it at the brew prefix,
# or you'll get: fatal error: '.../libavcodec/avfft.h' file not found
export PKG_CONFIG_PATH="$(brew --prefix ffmpeg)/lib/pkgconfig"
export BINDGEN_EXTRA_CLANG_ARGS="-I$(brew --prefix ffmpeg)/include"
export FFMPEG_DIR="$(brew --prefix ffmpeg)"
pkg-config --cflags libavcodec    # sanity: must print an -I under your brew prefix

# If you build under conda, make sure Homebrew's pkg-config wins:
#   which pkg-config   # should be under $(brew --prefix)/bin, not miniconda
#   export PATH="$(brew --prefix)/bin:$PATH"

# build the extension WITH video
cd crates/ferroload-py
maturin develop --release --features video
python3 -c "import ferroload; print(hasattr(ferroload.Dataset,'decode_video'))"  # -> True
```

Pack a few clips and decode them:

```bash
# pack ~6 clips from a video dataset (pulls only those files, not the whole repo)
python python/import_hf_files.py MiG-NJU/OmniVideo-Test /tmp/ds_omni \
    --jsonl test_505.jsonl --media-field video_path --modality video --ext mp4 --limit 6

# decode: T frames per clip -> [T,H,W,3] uint8 arrays
python python/test_video.py /tmp/ds_omni --modality video --num-frames 8
```

From Python directly, or via the loader:

```python
clips = fd.decode_video([0,1,2], "video", num_frames=8)   # list of [T,H,W,3] arrays
ds = ferroload_loader.FerroTorchDataset(fd, videos=["video"], num_frames=8, meta=["task"])
# DataLoader(ds, batch_size=...) -> sample["video"] is [T,H,W,3]
```

Note: the ffmpeg decode path is feature-gated and was authored against
`ffmpeg-next` 7 but not compile-tested in CI (no ffmpeg there) — if the first
build surfaces an API mismatch, it'll be a small fix in `crates/ferroload-codec/src/video.rs`.

## 3e. The `ferroload` CLI

After `maturin develop`/`maturin build`, a `ferroload` command is available
(entry point `ferroload_cli:main`). Before install, run it as
`python python/ferroload_cli.py <cmd>`.

```bash
ferroload import-hf ylecun/mnist /tmp/cli_mnist --limit 50   # pack + auto-register
ferroload import-files MiG-NJU/OmniVideo-Test /tmp/ds_omni \
    --jsonl test_505.jsonl --media-field video_path --modality video --ext mp4 --limit 6
ferroload inspect /tmp/cli_mnist        # manifest, schema, counts, stats
ferroload verify  /tmp/cli_mnist        # checks every member reads back
ferroload subset  /tmp/cli_mnist --where "label < 3" --out lt3   # -> subsets/lt3.json
ferroload list                          # datasets in the local catalog
ferroload add s3://bucket/datasets/foo  # register a remote dataset
```

The catalog lives at `~/.ferroload/catalog.json` (override with `$FERROLOAD_HOME`);
imports auto-register, and `list` shows name / location / rows / modalities / URI.

## 4. Read it back in Python

```python
import ferroload
ds = ferroload.Dataset.open("/tmp/ds_mnist")
print(len(ds))
s = ds.get(0)                  # {'image': b'...PNG...', 'image_present': True, 'meta': {'label': 5}, ...}
s = ds.get(0, ["image"])       # projection: fetch only 'image'
ds.verify()
```

## 5. Benchmark vs HuggingFace (local)

```bash
python3 python/bench_read.py --dataset ylecun/mnist --n 3000 --repeats 3
```

See `../BENCHMARKS.md` for results and caveats.

## Notes

- Dataset ids need their namespace now: use `ylecun/mnist`, not `mnist`.
- The importer writes the dataset under the `out` path; on this sandbox write to
  `/tmp/...` (the mounted folder is fine for reading, but local disk avoids any
  rename quirks).
- This uses the reference Python reader. The full async/prefetch/cloud runtime is
  the next milestone (see ../README.md roadmap).
```