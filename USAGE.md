# ferroload-core — Usage

API reference for the format-core crate. Every snippet below corresponds to
behavior covered by the test suite (`cargo test`).

## 1. Add the dependency

```toml
[dependencies]
ferroload-core = { path = "crates/ferroload-core" }
serde_json = "1"
```

Key imports:

```rust
use ferroload_core::dataset::{Dataset, DatasetWriter, Sample, commit_manifest};
use ferroload_core::manifest::{Manifest, Modality, Column};
use std::collections::BTreeMap;
```

## 2. Declare modalities

A modality has an extension, a `kind` (`tensor` | `annotation` | `scalar`), and a
`codec`. Helpers cover the common cases:

```rust
Modality::tensor("mp4", "video")   // blob stored in a shard, decoded by `video` codec
Modality::tensor("png", "depth16") // depth map
Modality::scalar("json")           // inline scalar/metadata, no shard blob
```

`Modality.attrs` is an open map for forward-compatible attributes (e.g. an
embedding's `dim`/`metric`).

## 3. Write a dataset

`DatasetWriter` streams samples into rolling tar shards and accumulates the index.

```rust
let mut w = DatasetWriter::create("/data/my-ds", "my-ds")?
    .shard_bytes_target(512 * 1024 * 1024)  // ~512 MiB shards
    .max_member_bytes(256 * 1024 * 1024)    // one big member -> dedicated shard
    .declare("video", Modality::tensor("mp4", "video"))
    .declare("audio", Modality::tensor("flac", "audio"))
    .declare("depth", Modality::tensor("png", "depth16"))
    .declare("text",  Modality::scalar("json"));

let mut blobs = BTreeMap::new();
blobs.insert("video".to_string(), video_bytes);
blobs.insert("audio".to_string(), audio_bytes);
// `depth` omitted for this sample -> it becomes a sparse/absent modality

let mut meta = BTreeMap::new();
meta.insert("caption".to_string(), serde_json::json!("a cat"));
meta.insert("duration_s".to_string(), serde_json::json!(8));

let sample_id = w.add("clip00001", &blobs, &meta)?;   // returns u64 sample_id
let manifest = w.close()?;                            // writes index + manifest + v1 snapshot
```

`close()` is the only commit point: it writes the index, then the manifest
**last** (temp file + atomic rename), and snapshots `versions/v1.json`.

## 4. Read a dataset

```rust
let ds = Dataset::open("/data/my-ds")?;   // also enforces min_reader_version
ds.len();                                  // number of samples

// full sample (all declared modalities)
let s: Sample = ds.get(7, None)?;
let video: &Vec<u8> = &s.blobs["video"];
let caption = &s.meta["caption"];

// PROJECTION: read only what you need -> other modalities are never fetched
let s = ds.get(7, Some(&["text".to_string()]))?;
assert!(!s.blobs.contains_key("video"));

// PRESENCE MASK: sparse modality absent -> mask=false, zero I/O
let s = ds.get(8, None)?;
if !s.present["depth"] { /* mask the depth loss term */ }

// fast metadata-only access (no shard read)
let row = ds.row(7)?;

// integrity check across all shards/members
let verified = ds.verify()?;
```

`Sample` fields: `sample_id`, `basename`, `blobs` (modality → bytes, only
requested+present), `present` (modality → bool), `meta` (inline metadata).

## 5. Extend the manifest (new capabilities, no format change)

The manifest preserves unknown keys and exposes an `extensions` namespace. To add,
e.g., a vector search index over an embedding column:

```rust
let mut m = Manifest::load(&root.join("manifest.json"))?;
m.version += 1;                                   // bump version
m.put_extension("vector_index", serde_json::json!([{
    "name": "clip_emb_hnsw", "ext_version": 1,
    "column": "caption_embedding", "kind": "hnsw",
    "dim": 768, "metric": "cosine",
    "path": "indexes/clip_emb_hnsw/",
    "built_over_dataset_version": 1, "covers_rows": ds.len(), "stale": false
}]));
commit_manifest(&root, &m)?;                      // atomic; snapshots versions/v2.json
```

Mark a column as carrying an embedding so a future capability can find it:

```rust
m.schema.push(Column {
    name: "caption_embedding".into(),
    dtype: "list<float32>[768]".into(),
    semantic: Some("embedding".into()),
    attrs: BTreeMap::from([
        ("dim".into(), serde_json::json!(768)),
        ("metric".into(), serde_json::json!("cosine")),
    ]),
});
```

Older readers ignore unknown extensions; the dataset stays fully usable.

## 5b. Enrich with an additive layer (`LayerWriter`)

`LayerWriter` adds new modalities/metadata **beside** the base, joined on
`sample_id`, without rewriting the base shards. It's the Rust primitive behind
Python's `Dataset.map`. Re-opening an existing layer **appends** (so an interrupted
pass resumes); `existing_ids()` reports what's already done.

```rust
use ferroload_core::dataset::LayerWriter;
use ferroload_core::manifest::Modality;
use std::collections::BTreeMap;

// declare the tensor modalities this layer contributes (stored in its shards);
// scalar/annotation outputs need no declaration — they go inline in the layer index.
let mods = BTreeMap::from([("depth".to_string(), Modality::tensor("npy", "npy"))]);
let mut w = LayerWriter::create(&root, "features", mods)?;

let done: std::collections::HashSet<u64> = w.existing_ids().into_iter().collect();
for sid in 0..ds.len() as u64 {
    if done.contains(&sid) { continue; }              // resume: skip finished
    let mut blobs = BTreeMap::new();
    blobs.insert("depth".to_string(), depth_npy_bytes(sid));   // tensor -> shard
    let mut meta = BTreeMap::new();
    meta.insert("tag".to_string(), serde_json::json!("indoor")); // scalar -> index
    w.add(sid, &blobs, &meta)?;
}
w.close()?;                                           // registers the layer; bumps version
```

After `close()`, reopening the dataset exposes `depth` as a normal modality and
`tag` merged into each sample's `meta`; `resolve`/`get`/projection/`verify` all
span base + layers. See `tests/layers.rs`.

## 6. Versioning & resume

- Every `commit_manifest` snapshots `versions/v{version}.json` for time-travel.
- `manifest.json` is the source of truth; orphan shards are ignored, so a crash
  before the rename leaves the previous version intact.

## 7. Error handling

All fallible calls return `ferroload_core::Result<T>`. `Error` variants:
`Io`, `Json`, `NotFound`, `Format`, and `ReaderTooOld { required, have }`.

## Notes / current scaffold limits

- The index and side-index serialize as JSON in this milestone; the production
  Parquet backend lands behind the `parquet` feature.
- Member names use the USTAR short-name limit (≤100 bytes) — fine for
  `basename.ext`.
- Storage is local-FS here; `object_store` (S3/GCS/Azure) is a later crate.
