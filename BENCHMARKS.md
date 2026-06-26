# Benchmarks

A 3-way data-loading comparison — **Ferroload** vs **WebDataset** vs **HF
`datasets` (Arrow)** (the loader 🤗 diffusers training uses) — across three image
datasets, plus a GCS streaming benchmark. Full harness, methodology, and raw
numbers live in [`benchmarks/`](benchmarks/) ([REPORT.md](benchmarks/REPORT.md),
[results/](benchmarks/results)). An earlier 2-way local-read micro-benchmark is
kept as an [appendix](#appendix-earlier-2-way-local-read-micro-benchmark).

> **Fairness:** all three formats are built from the **same encoded image bytes**
> and every loader decodes to the **same target** (decode → optional resize →
> `uint8` HWC), so per-sample CPU work is identical — only the format + loader
> machinery differ. Hardware: Apple M2 Pro (10 cores, 17 GB). Subsets sized to
> fit local disk; throughput (samples/s) is scale-independent. GCS over a
> ~3.6 MB/s link, single client. Numbers are relative — the *patterns* are the
> signal.

## Datasets

| Dataset | Source | N | Stored | Decode target | Regime |
|---|---|--:|---|---|---|
| CIFAR-10 | `cifar10` | 50,000 | PNG 32×32 + label | native 32 | overhead-bound |
| Stanford-Cars | `roskyluo/stanford_cars_blip` | 8,420 | JPEG + BLIP caption | →224 | JPEG-decode-bound |
| FFHQ-256 | `merkol/ffhq-256` | 10,000 | JPEG 256² | →224 | JPEG-decode-bound |

## Throughput (best config per loader)

![Local data-loading throughput](benchmarks/charts/throughput.png)

Apples-to-apples — **all loaders at `num_workers=8`** (Ferroload in the same
`torch DataLoader`, one decode thread per worker), plus Ferroload **native** (its
recommended in-process path: all cores, no worker processes, no IPC):

- **Tiny images (CIFAR-10): Ferroload wins every framing.** Even in the *same*
  DataLoader at nw=8 it does **164k samp/s — 1.8× HF, 3.9× WebDataset** (native
  191k). When the cost is per-sample overhead, in-process decode crushes
  multiprocessing; this is not an artifact of "native vs workers".
- **JPEG decode-bound (Cars / FFHQ): roughly a tie with HF, behind WebDataset.**
  At equal parallelism Ferroload (~4.7–5.1k) is ~on par with HF (slightly behind)
  and below WebDataset (~8–9.5k, sequential tar streaming). Ferroload **native**
  (~6.3–6.5k) beats HF nw=8 — but that edge is **avoiding multiprocessing IPC**
  (its DataLoader run is ~30% slower than native), not faster per-core decode. The
  `turbojpeg` codec brings the per-core JPEG decode up to ≈ PIL/libjpeg-turbo.

  **Takeaway: run Ferroload native (don't wrap it in a worker DataLoader);** its
  real wins are tiny-image throughput, storage, and remote streaming below.

## Storage footprint

![Storage footprint per format](benchmarks/charts/storage.png)

HF Arrow ≈ raw bytes; **Ferroload is always smaller than WebDataset** (≈2× smaller
for tiny images — WebDataset pads every tar member and stores a separate `.json`
meta member per sample, while Ferroload keeps meta in a compact zstd Parquet index).

## GCS streaming

![GCS streaming throughput before/after](benchmarks/charts/gcs_streaming.png)

WebDataset streams whole tar shards sequentially (bandwidth-bound). Ferroload's
decode path originally issued **per-sample ranged GETs** (latency-bound, 113
samp/s). After routing the decode path through the **coalesced** reader (one
`get_ranges` per shard), Ferroload hits **1008 samp/s — 8.9× faster, and 2.3×
faster than WebDataset** — while keeping random access + a queryable index that
WebDataset doesn't have.

## Two gaps the benchmark exposed — and fixed

![JPEG decode with turbojpeg](benchmarks/charts/jpeg_decode.png)

| Fix | Change | Before | After | Δ |
|---|---|--:|--:|--:|
| **Coalesced remote reads** | `decode_many`/`read_many` issue one coalesced `get_ranges` per shard instead of one GET per sample | 113 samp/s | **1008** | **8.9×** |
| **libjpeg-turbo decode** (opt-in `turbojpeg` feature) | JPEG via libjpeg-turbo (SIMD C) instead of pure-Rust zune-jpeg | 5.2–5.4k | **6.1–6.4k** | **+18%** |

Both verified for correctness (`test_map.py`, `test_combinations.py`). The
coalescing fix is unconditional; `turbojpeg` is opt-in (default build stays
pure-Rust).

## Where each format wins

- **Ferroload** — tiny-image throughput (no multiprocessing tax), fastest GCS
  streaming (after fix), smaller-than-WebDataset storage, and unique capabilities:
  **O(manifest) remote `open`**, **ranged columnar `subset`**, and a
  **DuckDB-queryable index** over the same dataset you train on.
- **WebDataset** — simple sequential tar streaming; strong raw JPEG-decode
  throughput at high worker counts; leanest worker RAM.
- **HF `datasets` (Arrow)** — smallest on disk (raw bytes), good queryable
  metadata, scales with `num_workers`; no tensor-shard / remote-range story.

See [benchmarks/REPORT.md](benchmarks/REPORT.md) for per-worker tables,
first-batch latency, peak RAM, the decoder-isolation analysis, limitations, and
reproduction steps.

---

# Appendix: earlier 2-way local-read micro-benchmark

> Kept for history. This earlier benchmark compared **only** Ferroload vs HF
> `datasets` for *local sequential read* on MNIST/CIFAR via `python/bench_read.py`,
> before the 3-way harness above. It's where the per-sample `File::open()` reopen
> bug (and the GIL-released `read_many`/`decode_many` paths) were first found.

Compares **sequential local read throughput** of a HuggingFace Arrow dataset
(on disk) against the Ferroload format, using the **same encoded image bytes**
for both so it is apples-to-apples. Script: `python/bench_read.py`.

**A perf bug found and fixed:** the first reader called `File::open()` on the shard
**on every `get()`** — one syscall per sample. On tiny images that dominated and
made ferroload *slower* than HF's mmap'd Arrow. Fix: cache shard file handles +
positional `read_exact_at`; plus lighter `read(i, modality)` and
`read_many(indices, modality)` (one FFI call, I/O in Rust with the GIL released).

**Results — `ylecun/mnist`, N=10000, best of 5 (release, 4 cores):**

| Benchmark | samples/s | vs HF |
|---|---:|---:|
| HF raw bytes | 107,208 | 1.0× |
| ferro raw `get()` (full dict) | 895,529 | 8.4× |
| ferro raw `read()` (minimal) | 1,653,496 | 15× |
| ferro raw `read_batch()` (contiguous batched) | 2,177,206 | 20× |
| HF decode (PIL) | 28,842 | 1.0× |
| ferro decode (PIL bytes) | 50,672 | 1.8× |
| ferro decode (Rust //, zero-copy NumPy) | 422,201 | 14.6× |

On tiny (28×28) images, per-sample Python overhead dominates, so the Rust paths
win big once the per-call `open()` was removed. On real ~256×256 JPEGs decode
costs ~100× the raw read, so decode throughput is what matters — which is exactly
what the 3-way benchmark above measures (and what the `turbojpeg` + coalescing
fixes improve). Caveats then: warm page cache, single process, local SSD, no
`num_workers`, no cloud — all of which the 3-way harness above now exercises.
