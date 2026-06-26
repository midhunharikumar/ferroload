//! End-to-end glue: a [`DatasetWriter`] that produces a self-contained dataset
//! root, and a [`Dataset`] reader that resolves index rows into materialized
//! samples via random-access shard reads.
//!
//! Layout produced (see DESIGN.md §3):
//! ```text
//! <root>/manifest.json
//! <root>/index/part-00000.parquet    (sharded columnar index, 1:1 with data shards)
//! <root>/shards/shard-00000.tar (+ .tar.idx)
//! <root>/versions/vN.json
//! ```

use crate::error::{Error, Result};
use crate::index::{IndexBackend, IndexRow, LazyIndex, ShardLoader, DEFAULT_INDEX_SHARD_CACHE};
use crate::index_parquet::ParquetIndex;
use crate::manifest::{IndexShardRef, Manifest, Modality};
use std::sync::Arc;
use crate::shard::ShardWriter;
use crate::sideindex::SideIndex;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A fully materialized sample returned by the reader.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub sample_id: u64,
    pub basename: String,
    /// modality -> raw bytes (only for requested + present modalities).
    pub blobs: BTreeMap<String, Vec<u8>>,
    /// modality -> present? (drives loss masking for sparse data).
    pub present: BTreeMap<String, bool>,
    pub meta: BTreeMap<String, Value>,
}

/// Streaming writer for a self-contained dataset root.
pub struct DatasetWriter {
    root: PathBuf,
    manifest: Manifest,
    shard_bytes_target: u64,
    max_member_bytes: u64,
    /// Rows of the **current** (not-yet-flushed) data shard. Flushed to an index
    /// shard 1:1 when its data shard finishes.
    cur_shard_rows: Vec<IndexRow>,
    /// The index-shard directory accumulated so far (mirrors the data shards).
    index_shards: Vec<IndexShardRef>,
    total_rows: u64,
    next_id: u64,
    shard_id: i64,
    shard: Option<ShardWriter>,
}

impl DatasetWriter {
    pub fn create(root: impl AsRef<Path>, name: &str) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(root.join("shards"))?;
        std::fs::create_dir_all(root.join("index"))?;
        std::fs::create_dir_all(root.join("versions"))?;
        Ok(DatasetWriter {
            root,
            manifest: Manifest::new(name),
            shard_bytes_target: 512 * 1024 * 1024,
            max_member_bytes: u64::MAX,
            cur_shard_rows: Vec::new(),
            index_shards: Vec::new(),
            total_rows: 0,
            next_id: 0,
            shard_id: -1,
            shard: None,
        })
    }

    pub fn shard_bytes_target(mut self, n: u64) -> Self {
        self.shard_bytes_target = n;
        self
    }

    /// Cap on a single member's size (DESIGN §14.5). Members larger than this are
    /// given their own dedicated shard so one giant video can't bloat a shard.
    pub fn max_member_bytes(mut self, n: u64) -> Self {
        self.max_member_bytes = n;
        self
    }

    pub fn declare(mut self, name: &str, modality: Modality) -> Self {
        self.manifest.modalities.insert(name.to_string(), modality);
        self
    }

    fn shard_path(&self, id: i64) -> PathBuf {
        self.root.join("shards").join(format!("shard-{:05}.tar", id))
    }

    fn roll_shard(&mut self) -> Result<()> {
        self.finish_shard()?;
        self.shard_id += 1;
        let p = self.shard_path(self.shard_id);
        self.shard = Some(ShardWriter::create(p)?);
        Ok(())
    }

    fn finish_shard(&mut self) -> Result<()> {
        if let Some(w) = self.shard.take() {
            let path = w.path().to_path_buf();
            let locs = w.finish()?;
            let si = SideIndex::from_locs(&locs);
            si.save(&path.with_extension("tar.idx"))?;
            // the index shard mirrors the data shard 1:1 — flush it now.
            self.flush_index_shard()?;
        }
        Ok(())
    }

    /// Write the current data shard's rows to `index/part-<NNNNN>.parquet`
    /// (columnar, queryable) and register an [`IndexShardRef`]. No-op when empty.
    fn flush_index_shard(&mut self) -> Result<()> {
        if self.cur_shard_rows.is_empty() {
            return Ok(());
        }
        // all rows in the buffer belong to the same data shard.
        let part_id = self.cur_shard_rows[0].shard_id;
        let start_id = self.cur_shard_rows[0].sample_id;
        let rows = self.cur_shard_rows.len() as u64;
        let rel = format!("index/part-{part_id:05}.parquet");
        ParquetIndex.write(&self.root.join(&rel), &self.cur_shard_rows)?;
        self.index_shards.push(IndexShardRef { path: rel, start_id, rows });
        self.total_rows += rows;
        self.cur_shard_rows.clear();
        Ok(())
    }

    fn ensure_shard(&mut self, incoming_bytes: u64, dedicated: bool) -> Result<()> {
        let need_new = match &self.shard {
            None => true,
            Some(w) => dedicated || w.bytes_written() + incoming_bytes >= self.shard_bytes_target,
        };
        if need_new {
            self.roll_shard()?;
        }
        Ok(())
    }

    /// Add one logical multimodal sample.
    /// `blobs`: tensor-modality bytes (written into a shard).
    /// `meta`: inline scalar/annotation metadata.
    pub fn add(
        &mut self,
        key: &str,
        blobs: &BTreeMap<String, Vec<u8>>,
        meta: &BTreeMap<String, Value>,
    ) -> Result<u64> {
        let total: u64 = blobs.values().map(|b| b.len() as u64).sum();
        let dedicated = total > self.max_member_bytes;
        self.ensure_shard(total, dedicated)?;

        let mut offsets = BTreeMap::new();
        for (modality, data) in blobs {
            let ext = self
                .manifest
                .modalities
                .get(modality)
                .map(|m| m.ext.clone())
                .unwrap_or_else(|| modality.clone());
            let member = format!("{key}.{ext}");
            let loc = self.shard.as_mut().unwrap().append(&member, data)?;
            offsets.insert(modality.clone(), [loc.offset, loc.length]);
        }

        let row = IndexRow {
            sample_id: self.next_id,
            shard_id: self.shard_id as u32,
            basename: key.to_string(),
            offsets,
            meta: meta.clone(),
            shard: None,
        };
        self.cur_shard_rows.push(row);
        self.next_id += 1;

        if dedicated {
            // close the dedicated shard so the next sample starts fresh
            self.finish_shard()?;
        }
        Ok(self.next_id - 1)
    }

    /// Finalize: flush the final partial index shard, then write the manifest
    /// (atomically, last) and a version snapshot. Emits the **sharded** index
    /// (`index/part-*.parquet` + `manifest.index_shards`). Sets
    /// `min_reader_version = 2` (the sharded format).
    pub fn close(mut self) -> Result<Manifest> {
        self.finish_shard()?; // flushes the last partial index shard

        let n_shards = (self.shard_id + 1).max(0) as u32;
        self.manifest.index_shards = std::mem::take(&mut self.index_shards);
        self.manifest.index.rows = self.total_rows;
        self.manifest.min_reader_version = crate::manifest::READER_VERSION;
        self.manifest.shards.dir = "shards/".to_string();
        self.manifest.shards.count = n_shards;
        self.manifest.shards.shard_bytes_target = self.shard_bytes_target;

        commit_manifest(&self.root, &self.manifest)?;
        Ok(self.manifest)
    }
}

/// Atomic commit: snapshot to versions/vN.json, then write manifest.json LAST via
/// a temp file + rename so a crash never leaves a torn manifest.
pub fn commit_manifest(root: &Path, manifest: &Manifest) -> Result<()> {
    let json = manifest.to_json()?;
    let snap = root
        .join("versions")
        .join(format!("v{}.json", manifest.version));
    std::fs::write(&snap, &json)?;

    let tmp = root.join(".manifest.json.tmp");
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, root.join("manifest.json"))?;
    Ok(())
}

const NO_ROW: u32 = u32::MAX;

/// A loaded enrichment layer. Because `sample_id` is **dense and contiguous**
/// (DESIGN §13.1), the join to the base is **positional, O(1), no hash**: `by_pos`
/// is a direct `sample_id -> row index` table (`NO_ROW` = absent for this layer).
struct LoadedLayer {
    /// Shards dir **relative to the dataset root** (e.g. `shards/<name>`); joined
    /// to root for local reads or to the in-store prefix for remote reads.
    shards_rel: String,
    rows: Vec<IndexRow>,
    by_pos: Vec<u32>,
    modalities: BTreeMap<String, crate::manifest::Modality>,
}

impl LoadedLayer {
    #[inline]
    fn row_for(&self, sample_id: u64) -> Option<&IndexRow> {
        match self.by_pos.get(sample_id as usize).copied() {
            Some(p) if p != NO_ROW => Some(&self.rows[p as usize]),
            _ => None,
        }
    }
}

/// Join an in-store key prefix with a root-relative path.
#[cfg(feature = "remote")]
fn join_key(prefix: &str, rel: &str) -> String {
    if prefix.is_empty() {
        rel.to_string()
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), rel)
    }
}

/// Turn a raw object-store error into a message with an actionable hint. Detects
/// auth/permission failures (so a credential mismatch reads clearly) and otherwise
/// points at the likely non-credential causes.
#[cfg(feature = "remote")]
fn remote_err(url: &str, what: &str, e: impl std::fmt::Display) -> Error {
    let msg = e.to_string();
    let lc = msg.to_lowercase();
    let auth = [
        "unauthenticated", "unauthorized", "401", "403", "forbidden",
        "access denied", "accessdenied", "invalidaccesskeyid",
        "signaturedoesnotmatch", "credential", "permission", "token",
        "no such host", "could not find", "default credentials",
    ]
    .iter()
    .any(|m| lc.contains(m));
    let hint = if auth {
        "this looks like an authentication / permissions problem. Set credentials \
         for your backend and confirm the identity can read this bucket:\n         \
         S3    -> AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (+ AWS_REGION)\n         \
         GCS   -> GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json\n         \
         Azure -> AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_ACCESS_KEY\n         \
         (public buckets need anonymous access, which this build doesn't enable yet)."
    } else {
        "the store authenticated and reached the server, but reading this object \
         failed. Likely causes: wrong bucket/prefix or a missing object; the dataset's \
         manifest.json / index not at this path; or a server-side Content-Encoding \
         mismatch on the object (e.g. it was uploaded gzip-transcoded). Inspect it with \
         your cloud CLI, e.g. `gsutil stat <object>` / `aws s3api head-object`."
    };
    Error::Format(format!("reading {what} from {url}:\n  {msg}\n  hint: {hint}"))
}

/// Where a dataset's shard bytes come from. `Local` is the default, fast path:
/// positional `read_exact_at` against cached file handles, no async. `Remote`
/// streams ranged GETs from an object store through a content-addressed local
/// cache, driven on a Rust-owned Tokio runtime (so it never blocks the GIL).
enum Backend {
    Local {
        handles: std::sync::Mutex<std::collections::HashMap<PathBuf, std::sync::Arc<std::fs::File>>>,
    },
    #[cfg(feature = "remote")]
    Remote {
        store: ferroload_io::CachedStorage,
        prefix: String,
    },
}

/// Reader over a self-contained dataset (base + any enrichment layers), backed by
/// either the local filesystem or a remote object store.
pub struct Dataset {
    /// Local filesystem root, or — for a remote dataset — the source URL (display
    /// only; remote datasets are read-only, so `map`/writes aren't supported).
    root: PathBuf,
    manifest: Manifest,
    index: LazyIndex,
    layers: Vec<LoadedLayer>,
    /// modality name -> index into `layers` (base modalities are absent here).
    modality_layer: std::collections::HashMap<String, usize>,
    backend: Backend,
}

impl Dataset {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest = Manifest::load(&root.join("manifest.json"))?;
        manifest.check_reader_compat()?;
        let base_len = manifest.index.rows as usize;

        // base index: lazy/sharded, only the directory is loaded here.
        let index = LazyIndex::new(
            manifest.index_shards.clone(),
            ShardLoader::Local { root: root.clone() },
            DEFAULT_INDEX_SHARD_CACHE,
        );

        let mut layers = Vec::new();
        let mut modality_layer = std::collections::HashMap::new();
        for (li, lref) in manifest.layers.iter().enumerate() {
            let lrows = ParquetIndex.read(&root.join(&lref.index))?;
            Self::push_layer(&mut layers, &mut modality_layer, li, lref, lrows, base_len);
        }

        Ok(Dataset {
            root,
            manifest,
            index,
            layers,
            modality_layer,
            backend: Backend::Local {
                handles: std::sync::Mutex::new(std::collections::HashMap::new()),
            },
        })
    }

    /// Build one layer table entry + its positional join (shared by local/remote open).
    fn push_layer(
        layers: &mut Vec<LoadedLayer>,
        modality_layer: &mut std::collections::HashMap<String, usize>,
        li: usize,
        lref: &crate::manifest::LayerRef,
        lrows: Vec<IndexRow>,
        base_len: usize,
    ) {
        // positional join table: sample_id -> row index (dense, O(1) lookup)
        let cap = base_len.max(lrows.iter().map(|r| r.sample_id as usize + 1).max().unwrap_or(0));
        let mut by_pos = vec![NO_ROW; cap];
        for (idx, r) in lrows.iter().enumerate() {
            by_pos[r.sample_id as usize] = idx as u32;
        }
        for m in lref.modalities.keys() {
            modality_layer.insert(m.clone(), li);
        }
        layers.push(LoadedLayer {
            shards_rel: lref.shards_dir.clone(),
            rows: lrows,
            by_pos,
            modalities: lref.modalities.clone(),
        });
    }

    /// Open a dataset that lives on a remote object store (`s3://`, `gs://`,
    /// `az://`; also `file://` / `memory://`). Shard bytes are streamed via ranged
    /// GETs through a content-addressed local cache at `cache_dir`; the (small)
    /// manifest + index are fetched fresh. Read-only — `map`/writes aren't
    /// supported on a remote dataset. Requires the `remote` feature plus the
    /// matching backend feature (`aws`/`gcp`/`azure`).
    #[cfg(feature = "remote")]
    pub fn open_url(url: &str, cache_dir: impl AsRef<Path>) -> Result<Self> {
        use ferroload_io::{CachedStorage, Storage};
        let (storage, prefix) = Storage::from_url(url)
            .map_err(|e| Error::Format(format!("remote open {url}: {e}")))?;

        let fetch = |rel: &str| -> Result<Vec<u8>> {
            storage
                .get_blocking(&join_key(&prefix, rel))
                .map(|b| b.to_vec())
                .map_err(|e| remote_err(url, rel, e))
        };

        let manifest = Manifest::from_json(
            std::str::from_utf8(&fetch("manifest.json")?)
                .map_err(|e| Error::Format(format!("manifest utf8: {e}")))?,
        )?;
        manifest.check_reader_compat()?;
        let base_len = manifest.index.rows as usize;

        // Layers are small and still loaded eagerly; the base index is lazy and
        // fetches nothing at open — shards stream on demand through the cache.
        let mut layers = Vec::new();
        let mut modality_layer = std::collections::HashMap::new();
        for (li, lref) in manifest.layers.iter().enumerate() {
            let lrows: Vec<IndexRow> = ParquetIndex::read_bytes(fetch(&lref.index)?)?;
            Self::push_layer(&mut layers, &mut modality_layer, li, lref, lrows, base_len);
        }
        // `fetch`'s borrow of `storage` ends here (NLL), so it can move into the cache.
        let store = CachedStorage::new(storage, cache_dir.as_ref().to_path_buf())?;

        let index = LazyIndex::new(
            manifest.index_shards.clone(),
            ShardLoader::Remote { store: store.clone(), prefix: prefix.clone() },
            DEFAULT_INDEX_SHARD_CACHE,
        );

        Ok(Dataset {
            root: PathBuf::from(url),
            manifest,
            index,
            layers,
            modality_layer,
            backend: Backend::Remote { store, prefix },
        })
    }

    /// Local file handle for a root-relative shard path (cached). Local backend only.
    fn local_file(&self, rel: &str) -> Result<std::sync::Arc<std::fs::File>> {
        match &self.backend {
            Backend::Local { handles } => {
                let path = self.root.join(rel);
                let mut guard = handles.lock().unwrap();
                if let Some(f) = guard.get(&path) {
                    return Ok(f.clone());
                }
                let f = std::sync::Arc::new(std::fs::File::open(&path)?);
                guard.insert(path, f.clone());
                Ok(f)
            }
            #[cfg(feature = "remote")]
            Backend::Remote { .. } => Err(Error::Format("local_file on a remote dataset".into())),
        }
    }

    /// Read one member `[off, off+len)` of a root-relative shard, dispatching to
    /// the local filesystem (positional `read_exact_at`) or the remote object
    /// store (ranged GET via the local cache). The remote read runs on a
    /// Rust-owned runtime, so the caller can keep the GIL released across it.
    fn read_member(&self, rel: &str, off: u64, len: u64) -> Result<Vec<u8>> {
        match &self.backend {
            Backend::Local { .. } => {
                let f = self.local_file(rel)?;
                let mut buf = vec![0u8; len as usize];
                #[cfg(unix)]
                {
                    use std::os::unix::fs::FileExt;
                    f.read_exact_at(&mut buf, off)?;
                }
                #[cfg(not(unix))]
                {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut fc = f.try_clone()?;
                    fc.seek(SeekFrom::Start(off))?;
                    fc.read_exact(&mut buf)?;
                }
                Ok(buf)
            }
            #[cfg(feature = "remote")]
            Backend::Remote { store, prefix } => store
                .get_range_blocking(&join_key(prefix, rel), off..off + len)
                .map(|b| b.to_vec())
                .map_err(|e| remote_err(&self.root.to_string_lossy(), rel, e)),
        }
    }

    /// Root-relative shard path for a base row.
    fn shard_rel_base(row: &IndexRow) -> String {
        match &row.shard {
            Some(name) => format!("shards/{name}"),
            None => format!("shards/shard-{:05}.tar", row.shard_id),
        }
    }
    /// Root-relative shard path for a layer row (explicit filename for partitioned
    /// layers, else the canonical `shard-{id:05}.tar`).
    fn shard_rel_layer(shards_rel: &str, row: &IndexRow) -> String {
        let name = match &row.shard {
            Some(n) => n.clone(),
            None => format!("shard-{:05}.tar", row.shard_id),
        };
        format!("{}/{}", shards_rel.trim_end_matches('/'), name)
    }

    /// Resolve `(shard_rel, off, len)` for sample position `i` + `modality`,
    /// across the base dataset and enrichment layers. `None` if absent.
    fn resolve(&self, i: usize, modality: &str) -> Result<Option<(String, u64, u64)>> {
        let row = self.index.get(i)?;
        if let Some([o, l]) = row.offsets.get(modality).copied() {
            return Ok(Some((Self::shard_rel_base(&row), o, l)));
        }
        let sample_id = row.sample_id;
        if let Some(&li) = self.modality_layer.get(modality) {
            let layer = &self.layers[li];
            if let Some(lrow) = layer.row_for(sample_id) {
                if let Some([o, l]) = lrow.offsets.get(modality) {
                    return Ok(Some((Self::shard_rel_layer(&layer.shards_rel, lrow), *o, *l)));
                }
            }
        }
        Ok(None)
    }

    /// All modality names (base + layers).
    pub fn all_modalities(&self) -> Vec<String> {
        let mut v: Vec<String> = self.manifest.modalities.keys().cloned().collect();
        for layer in &self.layers {
            v.extend(layer.modalities.keys().cloned());
        }
        v
    }

    /// Minimal-overhead read of one modality's bytes (no Sample/meta allocation).
    /// Returns `None` if the modality is absent for this sample (zero I/O).
    pub fn read_blob(&self, i: usize, modality: &str) -> Result<Option<Vec<u8>>> {
        match self.resolve(i, modality)? {
            Some((rel, off, len)) => Ok(Some(self.read_member(&rel, off, len)?)),
            None => Ok(None),
        }
    }

    /// Read one modality for many samples into a **single contiguous buffer**,
    /// returning the buffer plus per-sample `(offset, length)` spans. Missing
    /// modalities get a zero-length span. Local reads go straight into the buffer
    /// slice; remote reads are **grouped per shard and issued as one coalesced
    /// `get_ranges`** (few HTTP requests, parallelized + cached). The batched read
    /// path for a DataLoader worker.
    pub fn read_blobs_contig(
        &self,
        indices: &[usize],
        modality: &str,
    ) -> Result<(Vec<u8>, Vec<(usize, usize)>)> {
        let mut plan: Vec<Option<(String, u64, u64)>> = Vec::with_capacity(indices.len());
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(indices.len());
        let mut total = 0usize;
        for &i in indices {
            match self.resolve(i, modality)? {
                Some((rel, o, l)) => {
                    spans.push((total, l as usize));
                    total += l as usize;
                    plan.push(Some((rel, o, l)));
                }
                None => {
                    spans.push((total, 0));
                    plan.push(None);
                }
            }
        }
        let mut buf = vec![0u8; total];
        match &self.backend {
            Backend::Local { .. } => {
                for (k, item) in plan.iter().enumerate() {
                    if let Some((rel, off, _len)) = item {
                        let (start, l) = spans[k];
                        let f = self.local_file(rel)?;
                        let slice = &mut buf[start..start + l];
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::FileExt;
                            f.read_exact_at(slice, *off)?;
                        }
                        #[cfg(not(unix))]
                        {
                            use std::io::{Read, Seek, SeekFrom};
                            let mut fc = f.try_clone()?;
                            fc.seek(SeekFrom::Start(*off))?;
                            fc.read_exact(slice)?;
                        }
                    }
                }
            }
            #[cfg(feature = "remote")]
            Backend::Remote { store, prefix } => {
                // group by shard key -> one coalesced get_ranges per shard
                let mut by_key: std::collections::HashMap<String, Vec<usize>> =
                    std::collections::HashMap::new();
                for (k, item) in plan.iter().enumerate() {
                    if let Some((rel, _, _)) = item {
                        by_key.entry(join_key(prefix, rel)).or_default().push(k);
                    }
                }
                for (key, ks) in by_key {
                    let ranges: Vec<std::ops::Range<u64>> = ks
                        .iter()
                        .map(|&k| {
                            let (_, off, len) = plan[k].as_ref().unwrap();
                            *off..*off + *len
                        })
                        .collect();
                    let got = store
                        .get_ranges_blocking(&key, &ranges)
                        .map_err(|e| remote_err(&self.root.to_string_lossy(), &key, e))?;
                    for (&k, bytes) in ks.iter().zip(got) {
                        let (start, l) = spans[k];
                        buf[start..start + l].copy_from_slice(bytes.as_ref());
                    }
                }
            }
        }
        Ok((buf, spans))
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn len(&self) -> usize {
        self.index.len()
    }
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Materialize sample `i`, optionally projecting to a subset of modalities.
    /// Modalities not selected — and modalities absent from the row — are never
    /// fetched (zero I/O), but absence is recorded in `present`. Layer modalities
    /// and layer metadata (from enrichment `map`) are merged in.
    pub fn get(&self, i: usize, modalities: Option<&[String]>) -> Result<Sample> {
        let (sample_id, basename, mut meta) = {
            let row = self.index.get(i)?;
            (row.sample_id, row.basename.clone(), row.meta.clone())
        };
        // merge scalar/annotation metadata contributed by enrichment layers
        for layer in &self.layers {
            if let Some(lrow) = layer.row_for(sample_id) {
                for (k, v) in &lrow.meta {
                    meta.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }

        let selected: Vec<String> = match modalities {
            None => self.all_modalities(),
            Some(sel) => sel.to_vec(),
        };

        let mut blobs = BTreeMap::new();
        let mut present = BTreeMap::new();
        for m in &selected {
            match self.resolve(i, m)? {
                Some((rel, off, len)) => {
                    present.insert(m.clone(), true);
                    blobs.insert(m.clone(), self.read_member(&rel, off, len)?);
                }
                None => {
                    present.insert(m.clone(), false);
                }
            }
        }

        Ok(Sample {
            sample_id,
            basename,
            blobs,
            present,
            meta,
        })
    }

    /// Number of shards on disk (for verification).
    pub fn shard_count(&self) -> u32 {
        self.manifest.shards.count
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Dataset {
    /// Read just the index row (no shard data I/O) — fast metadata access. For a
    /// sharded dataset this loads + caches the row's index shard on demand.
    pub fn row(&self, i: usize) -> Result<Arc<IndexRow>> {
        self.index.get(i)
    }

    /// Subset by a `WHERE` predicate over inline metadata; returns matching
    /// `sample_id`s in ascending order (deterministic, see DESIGN §6). Streams the
    /// index shard-by-shard.
    pub fn subset(&self, where_sql: &str) -> Result<Vec<u64>> {
        self.index.subset_ids_streaming(where_sql)
    }

    /// Number of index-shard objects loaded so far (0 right after `open()`, until
    /// a sample is touched). Backs the laziness assertion.
    pub fn index_shard_loads(&self) -> u64 {
        self.index.loaded_count()
    }

    /// Verify each referenced shard exists and members read back to declared
    /// lengths. Returns the number of samples verified.
    pub fn verify(&self) -> Result<usize> {
        let mods = self.all_modalities();
        for i in 0..self.len() {
            for m in &mods {
                if let Some((rel, off, len)) = self.resolve(i, m)? {
                    let bytes = self.read_member(&rel, off, len)?;
                    if bytes.len() as u64 != len {
                        let sid = self.index.get(i)?.sample_id;
                        return Err(Error::Format(format!(
                            "sample {sid} modality {m}: length mismatch"
                        )));
                    }
                }
            }
        }
        Ok(self.len())
    }
}

/// Relative paths for a layer's storage group (DESIGN §13.1 layout):
/// shards under `shards/<name>/`, index fragment under `index/`.
fn layer_shards_rel(name: &str) -> String {
    format!("shards/{name}/")
}
fn layer_index_rel(name: &str) -> String {
    format!("index/{name}.parquet")
}
fn layer_part_index_rel(name: &str, part: u32) -> String {
    format!("index/{name}.part-{part}.parquet")
}

/// Streaming writer for an enrichment **layer** (the sink of `map`).
///
/// Two modes (DESIGN §13.3, §14.3):
/// - **Single-process** (`create`): writes `shards/<name>/shard-NNNNN.tar` + the
///   layer index `index/<name>.parquet`, and registers the layer in the manifest on
///   `close()` (atomic, version bump). Re-opening **appends** (resume).
/// - **Partitioned** (`create_partition(part)`): a distributed worker writes its
///   *own* `shards/<name>/shard-<part>-NNNNN.tar` + an index fragment
///   `index/<name>.part-<part>.parquet` and a `.done` marker on `close()`, touching
///   **no** shared state. A final [`LayerWriter::commit`] merges all fragments
///   into `index/<name>.parquet` and registers the layer (the one sync point).
pub struct LayerWriter {
    root: PathBuf,
    name: String,
    shards_rel: String,
    index_rel: String,
    shards_dir: PathBuf,
    modalities: BTreeMap<String, Modality>,
    shard_bytes_target: u64,
    partition: Option<u32>,
    rows: Vec<IndexRow>,
    shard_id: i64,
    shard: Option<ShardWriter>,
    closed: bool,
}

impl LayerWriter {
    /// Single-process writer (registers the layer on `close`).
    pub fn create(
        root: impl AsRef<Path>,
        name: &str,
        modalities: BTreeMap<String, Modality>,
    ) -> Result<Self> {
        Self::open_internal(root, name, modalities, None)
    }

    /// Partition-local writer for distributed map: writes its own shards + an
    /// index fragment, no manifest changes. Finish the job with `commit`.
    pub fn create_partition(
        root: impl AsRef<Path>,
        name: &str,
        modalities: BTreeMap<String, Modality>,
        part: u32,
    ) -> Result<Self> {
        Self::open_internal(root, name, modalities, Some(part))
    }

    fn open_internal(
        root: impl AsRef<Path>,
        name: &str,
        modalities: BTreeMap<String, Modality>,
        partition: Option<u32>,
    ) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let shards_rel = layer_shards_rel(name);
        let index_rel = match partition {
            None => layer_index_rel(name),
            Some(p) => layer_part_index_rel(name, p),
        };
        let shards_dir = root.join(&shards_rel);
        std::fs::create_dir_all(&shards_dir)?;
        std::fs::create_dir_all(root.join("index"))?;

        // resume/append: load this fragment's existing rows, continue shard numbering
        let index_abs = root.join(&index_rel);
        let (rows, shard_id) = if index_abs.exists() {
            let existing = ParquetIndex.read(&index_abs)?;
            let max_shard = existing.iter().map(|r| r.shard_id as i64).max().unwrap_or(-1);
            (existing, max_shard)
        } else {
            (Vec::new(), -1)
        };

        Ok(LayerWriter {
            root,
            name: name.into(),
            shards_rel,
            index_rel,
            shards_dir,
            modalities,
            shard_bytes_target: 512 * 1024 * 1024,
            partition,
            rows,
            shard_id,
            shard: None,
            closed: false,
        })
    }

    pub fn shard_bytes_target(mut self, n: u64) -> Self {
        self.shard_bytes_target = n;
        self
    }

    /// `sample_id`s already present in this fragment — for resume.
    pub fn existing_ids(&self) -> Vec<u64> {
        self.rows.iter().map(|r| r.sample_id).collect()
    }

    fn shard_name(&self, id: i64) -> String {
        match self.partition {
            Some(p) => format!("shard-{p}-{id:05}.tar"),
            None => format!("shard-{id:05}.tar"),
        }
    }

    fn finish_shard(&mut self) -> Result<()> {
        if let Some(w) = self.shard.take() {
            let path = w.path().to_path_buf();
            let locs = w.finish()?;
            SideIndex::from_locs(&locs).save(&path.with_extension("tar.idx"))?;
        }
        Ok(())
    }

    fn ensure_shard(&mut self, incoming: u64) -> Result<()> {
        let need = match &self.shard {
            None => true,
            Some(w) => w.bytes_written() + incoming >= self.shard_bytes_target,
        };
        if need {
            self.finish_shard()?;
            self.shard_id += 1;
            let path = self.shards_dir.join(self.shard_name(self.shard_id));
            self.shard = Some(ShardWriter::create(path)?);
        }
        Ok(())
    }

    /// Add one enriched sample: `blobs` (tensor/bytes outputs) go to layer shards,
    /// `meta` (scalar/annotation outputs) is recorded inline in the layer index.
    pub fn add(
        &mut self,
        sample_id: u64,
        blobs: &BTreeMap<String, Vec<u8>>,
        meta: &BTreeMap<String, Value>,
    ) -> Result<()> {
        if self.closed {
            return Err(Error::Format("layer writer is closed".into()));
        }
        let total: u64 = blobs.values().map(|b| b.len() as u64).sum();
        let mut offsets = BTreeMap::new();
        let mut shard_name = None;
        if total > 0 {
            self.ensure_shard(total)?;
            for (modality, data) in blobs {
                let ext = self
                    .modalities
                    .get(modality)
                    .map(|m| m.ext.clone())
                    .unwrap_or_else(|| modality.clone());
                let member = format!("{sample_id}.{ext}");
                let loc = self.shard.as_mut().unwrap().append(&member, data)?;
                offsets.insert(modality.clone(), [loc.offset, loc.length]);
            }
            // partitioned shards carry an explicit filename (can't derive from id)
            if self.partition.is_some() {
                shard_name = Some(self.shard_name(self.shard_id));
            }
        }
        self.rows.push(IndexRow {
            sample_id,
            shard_id: self.shard_id.max(0) as u32,
            basename: sample_id.to_string(),
            offsets,
            meta: meta.clone(),
            shard: shard_name,
        });
        Ok(())
    }

    /// Finalize this writer. Single-process: write `index/<name>.parquet` and register
    /// the layer in the manifest. Partitioned: write the fragment + a `.done`
    /// marker only (no manifest changes — call `commit` after all parts finish).
    pub fn close(mut self) -> Result<()> {
        self.finish_shard()?;
        ParquetIndex.write(&self.root.join(&self.index_rel), &self.rows)?;
        match self.partition {
            Some(p) => {
                // completed-shard marker: lets a re-run skip this finished part
                let marker = self.root.join(format!("index/{}.part-{p}.done", self.name));
                std::fs::write(marker, b"")?;
            }
            None => {
                self.register(self.rows.len() as u64)?;
            }
        }
        self.closed = true;
        Ok(())
    }

    /// Register/replace this layer in the manifest (atomic, version bump).
    fn register(&self, rows: u64) -> Result<()> {
        let mut manifest = Manifest::load(&self.root.join("manifest.json"))?;
        let lref = crate::manifest::LayerRef {
            name: self.name.clone(),
            index: layer_index_rel(&self.name),
            shards_dir: self.shards_rel.clone(),
            modalities: self.modalities.clone(),
            rows,
        };
        manifest.layers.retain(|l| l.name != self.name);
        manifest.layers.push(lref);
        manifest.version += 1;
        commit_manifest(&self.root, &manifest)
    }

    /// **Commit step** for a partitioned (distributed) map: merge every
    /// `index/<name>.part-*.parquet` fragment (and any prior `index/<name>.parquet`)
    /// into the single layer index, then register the layer in the manifest
    /// atomically. The fragments + `.done` markers are consumed. Idempotent:
    /// merging is keyed by `sample_id`, so re-running is safe.
    pub fn commit(
        root: impl AsRef<Path>,
        name: &str,
        modalities: BTreeMap<String, Modality>,
    ) -> Result<u64> {
        let root = root.as_ref().to_path_buf();
        let index_dir = root.join("index");
        let merged_rel = layer_index_rel(name);

        // start from any already-committed rows, keyed by sample_id (dedup/idempotent)
        let mut merged: std::collections::BTreeMap<u64, IndexRow> = std::collections::BTreeMap::new();
        let merged_abs = root.join(&merged_rel);
        if merged_abs.exists() {
            for r in ParquetIndex.read(&merged_abs)? {
                merged.insert(r.sample_id, r);
            }
        }

        // collect fragment files: index/<name>.part-*.parquet
        let prefix = format!("{name}.part-");
        let mut part_files: Vec<PathBuf> = Vec::new();
        if index_dir.exists() {
            for entry in std::fs::read_dir(&index_dir)? {
                let entry = entry?;
                let fname = entry.file_name().to_string_lossy().into_owned();
                if fname.starts_with(&prefix) && fname.ends_with(".parquet") {
                    part_files.push(entry.path());
                }
            }
        }
        part_files.sort();
        for pf in &part_files {
            for r in ParquetIndex.read(pf)? {
                merged.insert(r.sample_id, r);
            }
        }

        // write the unified layer index (ascending sample_id) + register
        let rows: Vec<IndexRow> = merged.into_values().collect();
        ParquetIndex.write(&merged_abs, &rows)?;

        let mut manifest = Manifest::load(&root.join("manifest.json"))?;
        let lref = crate::manifest::LayerRef {
            name: name.to_string(),
            index: merged_rel,
            shards_dir: layer_shards_rel(name),
            modalities,
            rows: rows.len() as u64,
        };
        manifest.layers.retain(|l| l.name != name);
        manifest.layers.push(lref);
        manifest.version += 1;
        commit_manifest(&root, &manifest)?;

        // consume fragments + done markers (the merged index is now authoritative)
        for pf in &part_files {
            let _ = std::fs::remove_file(pf);
            let done = pf.with_extension("done"); // <name>.part-<p>.done
            let _ = std::fs::remove_file(done);
        }
        Ok(rows.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_root(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ferroload_ds_{}_{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn blob(bytes: &[u8]) -> Vec<u8> {
        bytes.to_vec()
    }

    #[test]
    fn write_then_read_roundtrip() {
        let root = unique_root("rt");
        let mut w = DatasetWriter::create(&root, "demo")
            .unwrap()
            .declare("image", Modality::tensor("jpg", "image"))
            .declare("text", Modality::scalar("json"));

        for i in 0..3u64 {
            let mut blobs = BTreeMap::new();
            blobs.insert("image".to_string(), blob(format!("IMG{i}").as_bytes()));
            let mut meta = BTreeMap::new();
            meta.insert("label".to_string(), serde_json::json!(i));
            w.add(&format!("s{i:04}"), &blobs, &meta).unwrap();
        }
        let manifest = w.close().unwrap();
        assert_eq!(manifest.index.rows, 3);

        let ds = Dataset::open(&root).unwrap();
        assert_eq!(ds.len(), 3);
        let s = ds.get(1, None).unwrap();
        assert_eq!(s.blobs["image"], b"IMG1".to_vec());
        assert_eq!(s.meta["label"], serde_json::json!(1));
        assert_eq!(ds.verify().unwrap(), 3);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn projection_skips_unrequested_modalities() {
        let root = unique_root("proj");
        let mut w = DatasetWriter::create(&root, "demo")
            .unwrap()
            .declare("image", Modality::tensor("jpg", "image"))
            .declare("depth", Modality::tensor("png", "depth16"));
        let mut blobs = BTreeMap::new();
        blobs.insert("image".to_string(), blob(b"IMG"));
        blobs.insert("depth".to_string(), blob(b"DEPTH"));
        w.add("s0", &blobs, &BTreeMap::new()).unwrap();
        w.close().unwrap();

        let ds = Dataset::open(&root).unwrap();
        // request only image -> depth not fetched
        let s = ds.get(0, Some(&["image".to_string()])).unwrap();
        assert!(s.blobs.contains_key("image"));
        assert!(!s.blobs.contains_key("depth"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn sparse_modality_presence_mask_zero_io() {
        let root = unique_root("sparse");
        let mut w = DatasetWriter::create(&root, "demo")
            .unwrap()
            .declare("image", Modality::tensor("jpg", "image"))
            .declare("depth", Modality::tensor("png", "depth16"));
        // sample 0 has depth, sample 1 does not
        let mut b0 = BTreeMap::new();
        b0.insert("image".to_string(), blob(b"I0"));
        b0.insert("depth".to_string(), blob(b"D0"));
        w.add("s0", &b0, &BTreeMap::new()).unwrap();
        let mut b1 = BTreeMap::new();
        b1.insert("image".to_string(), blob(b"I1"));
        w.add("s1", &b1, &BTreeMap::new()).unwrap();
        w.close().unwrap();

        let ds = Dataset::open(&root).unwrap();
        let s1 = ds.get(1, None).unwrap();
        assert_eq!(s1.present["depth"], false); // absent -> mask false, no fetch
        assert_eq!(s1.present["image"], true);
        assert!(!s1.blobs.contains_key("depth"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn max_member_bytes_forces_dedicated_shard() {
        let root = unique_root("cap");
        let mut w = DatasetWriter::create(&root, "demo")
            .unwrap()
            .max_member_bytes(4) // tiny cap
            .declare("video", Modality::tensor("mp4", "video"));
        let mut big = BTreeMap::new();
        big.insert("video".to_string(), blob(b"BIGVIDEOPAYLOAD")); // > cap
        w.add("big", &big, &BTreeMap::new()).unwrap();
        let mut small = BTreeMap::new();
        small.insert("video".to_string(), blob(b"x"));
        w.add("small", &small, &BTreeMap::new()).unwrap();
        let m = w.close().unwrap();
        // big got its own shard, small a separate one
        assert!(m.shards.count >= 2);

        let ds = Dataset::open(&root).unwrap();
        assert_eq!(ds.get(0, None).unwrap().blobs["video"], b"BIGVIDEOPAYLOAD".to_vec());
        assert_eq!(ds.get(1, None).unwrap().blobs["video"], b"x".to_vec());
        std::fs::remove_dir_all(&root).ok();
    }

    /// Build an `n`-sample dataset with a small `shard_bytes_target` so the index
    /// is split across several shards. Returns the root + closed manifest.
    fn multi_shard_dataset(tag: &str, n: u64) -> (PathBuf, Manifest) {
        let root = unique_root(tag);
        let mut w = DatasetWriter::create(&root, "demo")
            .unwrap()
            .shard_bytes_target(4096) // small (tar pads members) -> many shards
            .declare("image", Modality::tensor("bin", "raw"));
        for i in 0..n {
            let mut blobs = BTreeMap::new();
            blobs.insert("image".to_string(), format!("IMG-{i:05}").into_bytes());
            let mut meta = BTreeMap::new();
            meta.insert("label".to_string(), serde_json::json!(i % 4));
            w.add(&format!("s{i:05}"), &blobs, &meta).unwrap();
        }
        let m = w.close().unwrap();
        (root, m)
    }

    #[test]
    fn writer_emits_contiguous_index_shards() {
        let n = 50u64;
        let (root, m) = multi_shard_dataset("idxshards", n);

        assert!(m.index_shards.len() >= 2, "expected multiple index shards");
        assert_eq!(m.index.rows, n);
        assert!(!root.join("index/index.json").exists(), "no monolithic index written");
        assert_eq!(m.min_reader_version, crate::manifest::READER_VERSION);

        // contiguous + gapless, sum(rows) == total, every part exists on disk
        let mut next = 0u64;
        let mut sum = 0u64;
        for s in &m.index_shards {
            assert_eq!(s.start_id, next, "shards must be contiguous + gapless");
            assert!(s.rows > 0);
            assert!(root.join(&s.path).exists(), "missing index shard {}", s.path);
            next += s.rows;
            sum += s.rows;
        }
        assert_eq!(sum, n);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn lazy_reader_correct_and_lazy() {
        let n = 50u64;
        let (root, m) = multi_shard_dataset("lazyread", n);
        let shards = m.index_shards.clone();

        let ds = Dataset::open(&root).unwrap();
        assert_eq!(ds.len() as u64, n);
        // len()/open touched no index shard
        assert_eq!(ds.index_shard_loads(), 0, "open must be O(manifest)");

        // one get() loads exactly one index shard
        let s7 = ds.get(7, None).unwrap();
        assert_eq!(s7.blobs["image"], b"IMG-00007".to_vec());
        assert_eq!(ds.index_shard_loads(), 1, "one sample touched -> one shard");

        // first + last id of every shard read back correctly (spans boundaries)
        for sref in &shards {
            for &id in &[sref.start_id, sref.start_id + sref.rows - 1] {
                let got = ds.read_blob(id as usize, "image").unwrap().unwrap();
                assert_eq!(got, format!("IMG-{id:05}").into_bytes());
            }
        }

        // a fresh sequential scan loads each shard exactly once
        let ds2 = Dataset::open(&root).unwrap();
        for i in 0..n as usize {
            let _ = ds2.get(i, None).unwrap();
        }
        assert_eq!(ds2.index_shard_loads(), shards.len() as u64);
        assert_eq!(ds2.verify().unwrap(), n as usize);

        // out-of-range is a clean error, not a panic
        assert!(ds.get(n as usize, None).is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn subset_streaming_matches_eager() {
        let n = 50u64;
        let (root, _m) = multi_shard_dataset("subset", n);
        let ds = Dataset::open(&root).unwrap();
        let ids = ds.subset("label = 2").unwrap();
        let expect: Vec<u64> = (0..n).filter(|i| i % 4 == 2).collect();
        assert_eq!(ids, expect);
        // presence flag derived from offsets still works through the streaming scan
        let all = ds.subset("image_present").unwrap();
        assert_eq!(all, (0..n).collect::<Vec<_>>());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn parquet_index_meta_is_queryable() {
        let root = unique_root("pqmeta");
        let mut w = DatasetWriter::create(&root, "demo")
            .unwrap()
            .shard_bytes_target(4096) // multiple parquet index shards
            .declare("image", Modality::tensor("bin", "raw"));
        let n = 30u64;
        for i in 0..n {
            let mut blobs = BTreeMap::new();
            blobs.insert("image".to_string(), format!("IMG-{i:05}").into_bytes());
            let mut meta = BTreeMap::new();
            let animal = if i % 2 == 0 { "cat" } else { "dog" };
            meta.insert("caption".to_string(), serde_json::json!(format!("a photo of a {animal}")));
            meta.insert("width".to_string(), serde_json::json!(64 + i as i64));
            w.add(&format!("s{i:05}"), &blobs, &meta).unwrap();
        }
        let m = w.close().unwrap();

        // every index shard is a real Parquet file carrying the meta as columns
        for s in &m.index_shards {
            assert!(s.path.ends_with(".parquet"));
            let part = ParquetIndex.read(&root.join(&s.path)).unwrap();
            assert!(part[0].meta.contains_key("caption"));
            assert!(part[0].meta.contains_key("width"));
        }

        // DuckDB-style queries over inline meta (string + numeric columns)
        let ds = Dataset::open(&root).unwrap();
        let cats = ds.subset("caption = 'a photo of a cat'").unwrap();
        assert_eq!(cats, (0..n).filter(|&i| i % 2 == 0).collect::<Vec<u64>>());
        let wide = ds.subset("width >= 80").unwrap();
        assert_eq!(wide, (0..n).filter(|&i| 64 + i >= 80).collect::<Vec<u64>>());

        // meta still round-trips through the lazy reader
        assert_eq!(ds.get(3, None).unwrap().meta["caption"], serde_json::json!("a photo of a dog"));
        assert_eq!(ds.get(4, None).unwrap().meta["width"], serde_json::json!(68));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn atomic_commit_writes_version_snapshot() {
        let root = unique_root("commit");
        let w = DatasetWriter::create(&root, "demo")
            .unwrap()
            .declare("text", Modality::scalar("json"));
        w.close().unwrap();
        assert!(root.join("manifest.json").exists());
        assert!(root.join("versions/v1.json").exists());
        assert!(!root.join(".manifest.json.tmp").exists()); // temp cleaned by rename
        std::fs::remove_dir_all(&root).ok();
    }
}
