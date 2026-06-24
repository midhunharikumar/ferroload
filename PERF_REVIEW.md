# Ferroload — Performance & Bottleneck Review (scaling: large video, null columns)

**Method:** reviewed the DESIGN.md architecture against issues reported by users of
the building blocks we depend on or resemble — WebDataset, MosaicML
StreamingDataset (MDS), PyTorch `DataLoader`, S3/object stores, Apache
Arrow/Parquet, Ray Data, and NVIDIA DALI. Each finding below is tagged:
**[covered]** already handled by the design, **[gap]** needs a design change, or
**[risk]** inherent trade-off to monitor.

---

## A. Executive summary

1. **For large video, the bottleneck is decode, not I/O.** Multiple large-scale
   reports converge on this: decode is the primary limiter when processing
   tens of millions of videos, and the fix is GPU decode (NVDEC/PyNvVideoCodec) +
   frame subsampling + offline pre-extraction. Our NVDEC path and enrichment-layer
   pre-extraction address this, but only if we make decode genuinely off-critical-path.
2. **Variable sample size (huge videos mixed with small samples) causes
   head-of-line blocking, straggler batches, and memory spikes.** This is the
   single most under-specified area in our current design. Count-based prefetch and
   count-based shards are unsafe when one sample can be 1000x another.
3. **Null/sparse columns are cheap in Parquet storage but cause two real problems:**
   wide union schemas (many mostly-null modality columns) and ragged/imbalanced
   collation. Projection + per-source index fragments mitigate; we should not
   physically materialize one giant wide table.
4. **Multiprocessing + Python objects + large samples = memory blowups.** Confirmed
   repeatedly in MDS and PyTorch. We must keep heavy data in Rust/Arrow buffers and
   budget prefetch in **bytes**, not item counts.

---

## B. Large-video bottlenecks

### B1. Decode dominates throughput  **[covered, with caveat]**
Reports from large-scale video pipelines and NVIDIA DALI confirm decode is
GPU/CPU-bound, not I/O-bound, at scale; NVDEC/NVJPEG + GPUDirect and chunked,
pipelined frame decode are the standard remedies. Our design has NVDEC
(feature-gated) and offline frame/feature pre-extraction via the enrichment map.
*Caveat / action:* make pre-extraction a first-class, recommended profile (store
decoded frames or features as a layer), and ensure the CPU `ffmpeg` default path
does **temporal subsampling during demux** (decode only the frames we keep), not
decode-all-then-drop.

### B2. One giant video creates an oversized shard  **[gap]**
Our writer only rolls shards *between* samples (`shard_bytes_target`), so a single
5 GB video produces a 5 GB shard. WebDataset/MDS practice targets ~1 GB shards;
oversized shards wreck shuffling granularity and memory.
*Action:* add a per-member size cap. Above it, either (a) store the video as its own
single-member shard, or (b) split long videos into **clip-level samples** (this is
exactly the open question "video sharding granularity"). Recommend clip-level as the
default for long-form video.

### B3. Variable sample size → head-of-line blocking & GPU starvation  **[gap]**
A single slow/large sample can delay the whole batch reaching the GPU (documented
HOL blocking; GPUs commonly sit ~57% idle on naive pipelines). Spiky prefetch
queues from variable batch sizes are a known `DataLoader` failure mode.
*Action:* (1) **duration/byte bucketing** — group similar-sized clips within a
shuffle block so batches are balanced and padding waste is low; (2) deeper,
**byte-budgeted** prefetch (see D2); (3) decode workers pull work-stealing style so
one big item doesn't stall a fixed worker.

### B4. Shuffle buffer memory scales with sample size  **[risk]**
WebDataset's `shuffle(n)` buffer holds N samples in RAM; with large videos, even a
modest N blows memory — a widely reported limitation. MDS likewise OOMs at epoch
boundaries when the dataset can't be cached.
*Action:* keep our **block-shuffle on sample_ids / offsets, not on materialized
bytes** (shuffle the index, then fetch) so buffer cost is independent of media size.
Document a memory formula (below).

### B5. Object-store behavior on large objects  **[covered, tune]**
S3 over HTTP/1.1 suffers head-of-line blocking and higher latency vs GCS HTTP/2;
high-throughput clients need large connection pools (1000s) and parallel/multipart
range GETs. A single large video benefits from **split-range parallel GETs**.
*Action:* configure `object_store` with a large connection pool, enable multipart
parallel range reads for big members, prefer regional buckets, and lean on the NVMe
cache for multi-epoch. (All consistent with §14.2; just make the large-object path
explicit.)

### B6. Ray Data object-store spilling on large video blocks  **[gap for map]**
Ray Data spills blocks that exceed object-store memory to disk (big slowdowns), and
`map_batches` has reported memory-retention/OOM issues; heap is bounded by
`num_execution_slots * max_block_size`.
*Action:* for the Ray executor, size blocks by **bytes** (small `max_block_size`
for video), avoid materializing whole decoded videos in the object store (stream
shard→compute→write), and pin output writes to local fragments. Treat large-object
spilling as a tuning parameter we set sane defaults for.

---

## C. Null / sparse-column bottlenecks

### C1. Parquet nulls are cheap to store  **[covered]**
Parquet encodes nulls via definition levels (RLE/bit-packed), so mostly-null columns
cost little on disk. Good news: enrichment layers and partial presence don't bloat
storage.

### C2. Wide union schema = many mostly-null columns  **[gap]**
When mixing many datasets (§12), a single physical index with every modality's
offset columns becomes a **wide sparse table**. Wide schemas add per-column metadata
and footer overhead, and naive readers scan columns they don't need.
*Action:* do **not** materialize one global wide table. Keep **per-layer / per-source
index fragments** and rely on projection pushdown + positional join on dense
`sample_id`. The `MixDataset` union schema is a *logical* view, not a physical
mega-table.

### C3. Nested annotation columns (bboxes) have reconstruction cost  **[risk]**
Lists/structs in Parquet use repetition+definition levels; reconstruction and ragged
collation cost CPU, and pathological high-cardinality annotations (thousands of boxes)
can dominate a batch.
*Action:* keep small annotations inline (current design), but add a size threshold
above which an annotation spills to a blob modality. Collate masks in Rust to keep it
off the Python critical path.

### C4. Null offsets must never trigger a fetch  **[covered]**
A null modality (e.g., no depth for this sample) must skip the GET entirely; the
`*_present` mask drives this. Confirm the fetch planner treats null offset == "skip,
emit mask", with zero I/O — important so sparse mixes don't pay for absent data.

### C5. Imbalanced presence → straggler batches  **[risk]**
If 10% of samples carry video and 90% don't, random batches have wildly uneven
decode cost (same HOL family as B3).
*Action:* optional **presence-aware sampling** (group by presence signature) and the
same byte-budgeted prefetch; surface masks so the model skips absent-modality loss.

---

## D. Cross-cutting (both axes)

### D1. Multiprocessing + Python objects copy-on-write blowup  **[gap]**
PyTorch `DataLoader` with `num_workers>0` leaks memory when iterating Python
`list`/`dict` because refcount touches trigger copy-on-write; the documented fix is
to use numpy/pyarrow-backed data. MDS has multiple OOM reports tied to workers and
stale shared memory.
*Action:* return tensors + **Arrow-backed metadata**, never large Python dicts,
across the worker boundary. Keep media bytes in Rust-owned buffers (we release the
GIL anyway). Document `persistent_workers=True` to avoid per-epoch worker respawn
cost.

### D2. Prefetch budget must be in bytes, not items  **[gap — important]**
Every variable-size finding above (B3, B4, B6, C5) points here. A fixed
`prefetch_depth`/`prefetch_factor` is unsafe when samples vary 1000x. 
*Action:* change the prefetch and in-flight I/O bounds to a **byte budget**
(`max_inflight_bytes`, `prefetch_bytes`) with item count as a secondary cap. This
single change defuses most large-video memory spikes.

### D3. Memory formula to document  **[action]**
Peak host memory ≈
`num_workers × (prefetch_bytes + decode_working_set) + shuffle_index_bytes + cache_pinned_bytes`.
Make every term a byte budget so users can size it; today some are item counts.

### D4. Epoch-boundary stalls & shard re-download  **[covered]**
MDS reports long stalls when shards must be re-downloaded at epoch boundaries.
Our NVMe cache + deterministic resume + block-shuffle-on-index address this; ensure
the cache persists across epochs and the sampler prefetches the next epoch's first
shards before the boundary.

---

## E. Recommended design changes (priority order)

1. **Byte-budgeted prefetch & in-flight I/O** (D2) — defuses large-video memory
   spikes and stragglers. *Highest impact, lowest effort.*
2. **Per-member size cap + clip-level sampling for long video** (B2) — prevents
   oversized shards; resolves the video-granularity open question.
3. **Duration/byte bucketing within shuffle blocks** (B3, C5) — kills HOL blocking
   and padding waste.
4. **Per-source/per-layer index fragments, projection-only union** (C2) — avoids
   wide sparse mega-tables.
5. **Arrow-backed metadata across worker boundary** (D1) — avoids COW memory leaks.
6. **First-class offline pre-extraction profile + temporal-subsample-on-demux** (B1)
   — the deepest decode lever.
7. **Large-object I/O + Ray block-size defaults** (B5, B6) — multipart range GETs,
   byte-sized Ray blocks, large connection pools.
8. **Annotation spill threshold + Rust-side mask collation** (C3) — bounds ragged
   collation cost.

Items 1–4 should be folded into DESIGN.md §14 (performance) and the open questions
before implementation.

---

## Sources

- [webdataset/FAQ.md](https://github.com/webdataset/webdataset/blob/main/FAQ.md) and [WebDataset README](https://github.com/webdataset/webdataset) — shard size trade-offs, sequential I/O, shuffle buffer memory.
- [WebDataset sharding guide](https://rom1504.github.io/webdataset/sharding/) — shard sizing.
- [Why I Chose WebDataset for 50TB (Medium)](https://medium.com/red-buffer/why-did-i-choose-webdataset-for-training-on-50tb-of-data-98a563a916bf) — shuffle buffer vs memory.
- [Training a Large Video Model on a Single Machine in a Day (arXiv)](https://arxiv.org/pdf/2309.16669) — IO + preprocessing can't keep up with GPU for video.
- [FFCV: Removing Data Bottlenecks (arXiv)](https://arxiv.org/pdf/2306.12517) — data-loading bottleneck framing.
- [MosaicML streaming #771 MemoryError](https://github.com/mosaicml/streaming/issues/771), [#652 OOM with workers](https://github.com/mosaicml/streaming/issues/652), [#740 multi-GPU](https://github.com/mosaicml/streaming/issues/740), [#876 NCCL timeouts](https://github.com/mosaicml/streaming/issues/876) — worker memory, shared-memory, epoch-boundary shuffle stalls.
- [MosaicML streaming FAQs & tips](https://docs.mosaicml.com/projects/streaming/en/latest/getting_started/faqs_and_tips.html) — stale shared memory cleanup.
- [PyTorch Data Loading Optimization tutorial](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html) — COW memory with Python objects, prefetch_factor, persistent_workers.
- [AWS: Data loading best practices with S3 clients](https://aws.amazon.com/blogs/machine-learning/applying-data-loading-best-practices-for-ml-training-with-amazon-s3-clients/) — connection pools, range requests.
- [Inflated lakehouse costs — S3 HTTP/1.1 (Onehouse)](https://www.onehouse.ai/blog/inflated-data-lakehouse-costs-and-latencies-blame-s3s-choice-of-http-1-1) — HOL blocking on S3 vs GCS HTTP/2.
- [MinatoLoader (arXiv)](https://arxiv.org/pdf/2509.10712) and [Head-of-line blocking (Wikipedia)](https://en.wikipedia.org/wiki/Head-of-line_blocking) — slow-sample HOL blocking, GPU idle.
- [Ray Data #26441 spilling](https://github.com/ray-project/ray/issues/26441), [#49757 map_batches memory](https://github.com/ray-project/ray/issues/49757), [Ray object spilling docs](https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html), [Ray Data performance tips](https://docs.ray.io/en/latest/data/performance-tips.html) — block sizing, spilling, heap bounds.
- [Arrow/Parquet encoding Part 1: nullability](https://arrow.apache.org/blog/2022/10/05/arrow-parquet-encoding-part-1/) and [Part 3: nested lists/structs](https://arrow.apache.org/blog/2022/10/17/arrow-parquet-encoding-part-3/) — null definition levels, nested repetition/definition cost.
- [NVIDIA DALI FAQ](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/FAQ.html) and [DALI video decoder](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.experimental.decoders.video.html) — NVDEC/NVJPEG, GPU decode, GPUDirect.
- [Breaking the Bottleneck: GPU-Optimised Video Processing (TDS)](https://towardsdatascience.com/breaking-the-bottleneck-gpu-optimised-video-processing-for-deep-learning/) — NVDEC decode-on-GPU, decode as bottleneck.
