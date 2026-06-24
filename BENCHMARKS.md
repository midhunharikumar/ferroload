# Ferroload vs HuggingFace `datasets` — local read benchmark

Compares **sequential local read throughput** of a HuggingFace Arrow dataset
(on disk) against the Ferroload format, using the **same encoded image bytes**
for both so it is apples-to-apples.

Script: `python/bench_read.py`. Run it yourself:

```bash
PYTHONPATH=/dir/with/ferroload.so \
python3 python/bench_read.py --dataset ylecun/mnist --n 3000 --repeats 3
```

## Method

1. `load_dataset(<id>, split="train[:N]")` downloads once, then reads from
   on-disk Arrow — the standard "HF dataset on local" path.
2. The identical encoded bytes (`Image(decode=False)`) are packed into a
   Ferroload dataset, so both stores hold byte-for-byte the same images.
3. Two regimes, each timed best-of-`repeats` (warm page cache):
   - **raw**: return encoded image bytes + label (no pixel decode)
   - **decode**: also PNG-decode to pixels with PIL (the real training cost)

## A perf bug we found and fixed

The first version of the reader called `File::open()` on the shard **on every
`get()`** — one syscall per sample. On tiny images that dominated, and on a
warm-cache Mac it made ferroload *slower* than HF's memory-mapped Arrow. The fix:
cache shard file handles and read positionally with `read_exact_at` (no reopen, no
seek). We also added two lighter read paths: `read(i, modality)` (just the bytes,
no dict/meta) and `read_many(indices, modality)` (one FFI call, all I/O in Rust
with the **GIL released** — the path a DataLoader worker should use).

## Results — `ylecun/mnist`, N=10000, best of 5 (Linux sandbox, **release**, 4 cores)

| Benchmark | time (s) | samples/s | MB/s | vs HF |
|---|---:|---:|---:|---:|
| HF  raw bytes | 0.0933 | 107,208 | 29.5 | 1.0× |
| ferro raw `get()` (full dict) | 0.0112 | 895,529 | 246.7 | **8.4×** |
| ferro raw `read()` (minimal) | 0.0060 | 1,653,496 | 455.6 | **15×** |
| ferro raw `read_batch()` (contiguous batched) | 0.0046 | 2,177,206 | 599.9 | **20×** |
| HF  decode (PIL) | 0.3467 | 28,842 | 7.9 | 1.0× |
| ferro decode (PIL bytes) | 0.1973 | 50,672 | 14.0 | **1.8×** |
| ferro decode (Rust //, zero-copy NumPy) | 0.0237 | 422,201 | 116.3 | **14.6×** |

On these tiny (28×28) images, per-sample Python overhead dominates, so the Rust
paths win big once the per-call `open()` was removed. `decode_many` returns
**zero-copy NumPy** `uint8` arrays decoded in parallel with the GIL released.

> Earlier drafts of this doc cited a one-off ~1.9× from a single run before the
> reopen bug was found; treat the table above as the current, reproducible result.
> Numbers vary by machine — re-run on yours.

## Decode is the bottleneck for real (large) images — and where to speed up

On a dataset with real ~256×256 JPEGs, **decode costs ~100× the raw read**. So
once reads are fast, the only thing that matters is decode. Two findings:

- **Build `--release`.** Image decode is compute-heavy; a *debug* build of the Rust
  decoder is ~7× slower than libjpeg/PIL. In release it wins. (Raw byte reads are
  I/O-bound and barely care about debug vs release — which is why the tables above
  still held in debug.)
- **Decode in Rust, parallel, GIL released** (`decode_many`) beats single-threaded
  PIL. Synthetic 500×(256×256) JPEG, 4 cores, **release**:

  | decode path | img/s |
  |---|---:|
  | PIL, 1 thread | 1,992 |
  | **ferro `decode_many` (Rust //, zero-copy NumPy)** | **5,087  (2.55×)** |

  Smaller multiple than tiny images because here decode *compute* dominates (less
  Python overhead to eliminate). It scales with cores — expect a larger gap on
  8–10 core machines. `decode_many` moves the decoded buffer straight into NumPy
  (`into_pyarray`, no copy), so the previous serial GIL-held copy is gone.

### Where to speed up further (in priority order)

1. **Parallel/Rust decode** — `decode_many` (done); scales with cores. Biggest win
   for large images because decode dominates.
2. **GPU decode** (nvJPEG / NVDEC) — for the heaviest image/video workloads.
3. **Offline pre-decode/resize as a layer** — decode+resize once via the enrichment
   `map`, store decoded/resized tensors, then training reads with *zero* decode.
   Best when you run many epochs.
4. **Resize-on-decode** — decode JPEGs directly to the target size (skip full-res).
5. **Async prefetch overlap** — hide decode behind GPU compute in the training loop
   (the micro-benchmark measures decode in isolation; real loops overlap it).
6. **Zero-copy reads** — return `memoryview` over an mmap'd shard to drop the final
   bytes copy (helps the raw path; minor next to decode).

Usage of the fast decode path:

```python
import numpy as np
for s in range(0, n, 256):
    for buf, (h, w, c) in ds.decode_many(list(range(s, s+256)), "image"):
        arr = np.frombuffer(buf, np.uint8).reshape(h, w, c)   # zero-copy view
```

## Important caveats (read before trusting these numbers)

- **MNIST images are tiny (~280 B).** This is an *overhead-bound* micro-benchmark
  — it mostly measures per-sample access cost, not bandwidth. With large
  images/video the decode/bandwidth terms dominate and the ratio will shift; run
  the script on your real data.
- **Warm page cache, single process, local SSD.** No cold-cache, no S3/GCS, no
  `num_workers` parallelism, no async prefetch. The real Ferroload runtime adds
  byte-budgeted prefetch, a cache tier, and GIL-free decode — none of which are
  exercised here.
- **This is the reference Python binding reader**, which decodes nothing in Rust
  yet and reads JSON index + tar via plain file I/O. The production Parquet index
  + `object_store` + Rust decoders are expected to widen the gap, especially under
  concurrency and on cloud storage.
- The HF path is also doing a tiny bit more work (Arrow row dict construction);
  both are honest "idiomatic usage" of each library.

## Try larger media

```bash
python3 python/bench_read.py --dataset uoft-cs/cifar10 --n 2000     # 32x32
python3 python/bench_read.py --dataset <your image/video set> --n 1000
```
