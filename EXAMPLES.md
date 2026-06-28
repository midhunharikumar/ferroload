# ferroload-core — Worked Example

A complete, runnable example lives at
`crates/ferroload-core/examples/synthetic_av.rs`. It builds a small synthetic
**video + audio + text (+ sparse depth)** dataset and exercises every format-core
capability. This document walks through it and shows its actual output.

## Run it

```bash
export CARGO_TARGET_DIR=/tmp/ferro-target   # mounted FS blocks cargo's link-time temp deletes
cargo run --example synthetic_av
```

## Verified output

```
wrote 8 samples across 2 shard(s)
sample 5: video="VIDEO-5" present={"audio": true, "depth": false, "text": false, "video": true}
projected sample 5 fetched modalities: []
samples WITH depth: [0, 3, 6]
verified 8 samples
reopened: version=2, extensions=["vector_index"]
```

## What each line demonstrates

**`wrote 8 samples across 2 shard(s)`** — the writer rolled shards by byte budget
(`shard_bytes_target(8192)`), so 8 small samples spilled into 2 tar shards, each
with its own `.tar.idx` side-index.

**`sample 5: ... present={...}`** — a full random-access read of sample 5. The
`present` map reports which declared modalities exist for this sample. `video` and
`audio` are present; `depth` is absent (it's only written on every 3rd sample); and
`text` is a *scalar* modality kept in `meta`, not a shard blob — so it has no blob
and shows `false` in the blob-presence map (its value is in `sample.meta`).

**`projected sample 5 fetched modalities: []`** — projection in action: we asked for
only the `text` modality, which has no shard blob, so **zero shard bytes were
read**. Asking for `["video"]` instead would fetch just the video member and skip
audio/depth entirely. This is the I/O-saving projection path.

**`samples WITH depth: [0, 3, 6]`** — sparse-modality handling. Depth was written
for samples 0, 3, 6 only. For all other samples the read emits `present["depth"] =
false` with **no I/O** for the missing member — exactly how heterogeneous/enriched
datasets avoid paying for absent data.

**`verified 8 samples`** — `Dataset::verify()` re-reads every member of every
sample by its recorded `(offset, length)` and confirms the byte lengths match,
validating the tar offsets and side-index end to end.

**`reopened: version=2, extensions=["vector_index"]`** — extensibility +
versioning. After the initial commit (v1), we loaded the manifest, added a
`caption_embedding` column marked `semantic: "embedding"`, registered a
`vector_index` extension, bumped to v2, and committed atomically. Reopening shows
the extension survived and `versions/v1.json` + `versions/v2.json` both exist for
time-travel — all with **no change to the core format**.

## Resulting on-disk layout

```
$TMPDIR/ferroload_example_av/
  manifest.json              # version 2, includes extensions.vector_index
  index/index.json           # 8 rows (scaffold JSON; Parquet in production)
  shards/
    shard-00000.tar
    shard-00000.tar.idx
    shard-00001.tar
    shard-00001.tar.idx
  versions/
    v1.json                  # pre-extension snapshot
    v2.json                  # post-extension snapshot
```

## Enrichment layers (`map`)

Running an enrichment pass (`LayerWriter` in Rust, `Dataset.map` in Python) adds
an **additive layer** beside the base — the base shards/index are never rewritten:

```
$ROOT/
  manifest.json              # version bumped; gains a `layers: [...]` entry
  index/
    index.json               # base index
    depth.json               # <- layer index fragment (rows keyed by sample_id)
  shards/
    shard-00000.tar          # base shards
    depth/                   # <- one shards group per layer (name)
      shard-00000.tar        #    new tensor modalities (e.g. depth) as .npy members
      shard-00000.tar.idx
```

A **distributed** map (StaticPartitionExecutor) writes partition-local fragments
first — `index/depth.part-<rank>.json` + `shards/depth/shard-<rank>-*.tar` — which
the commit step merges into the single `index/depth.json` and registers.

On read, the base and every layer are joined on `sample_id`: a layer modality
resolves to its layer shard, layer scalar/annotation outputs merge into `meta`,
and projection works across base+layers. A reader that doesn't know about
`layers` simply ignores the field (back-compat). See `tests/layers.rs` (Rust) and
`python/test_map.py` (end-to-end), and `PYTHON_API.md` for the `map` API.

## Mapping to the design

| Example behavior | DESIGN.md section |
|---|---|
| Self-contained root + manifest | §3.1–3.2 |
| Tar shards + side-index (random access) | §3.4 |
| Atomic commit + version snapshots | §3.3 |
| Manifest extensibility / vector index | §3.5 |
| Projection reads | §13.4 |
| Sparse modality presence masks | §12, §14.5 |
| Per-member size cap (see `max_member_bytes` test) | §14.5 |
| Enrichment `map` → additive layers (`LayerWriter`) | §13.3, §14 |

## Where to go next

The same dataset root is what the planned runtime crates consume: `ferroload-io`
(S3/GCS fetch + cache), `ferroload-codec` (decode), and `ferroload-py` (PyTorch
`Dataset`/`IterableDataset`). The format written here is forward-compatible with
all of them.
