//! The global sample index: one row per logical multimodal sample.
//!
//! A row carries the locator for each tensor modality (`shard_id` + per-modality
//! `(offset, length)`) and inline scalar/annotation metadata. The backend is
//! abstracted behind [`IndexBackend`]; the default scaffold backend serializes
//! JSON, and a Parquet backend is planned under the `parquet` feature (the
//! production format per DESIGN.md).

use crate::error::{Error, Result};
use crate::manifest::IndexShardRef;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexRow {
    pub sample_id: u64,
    pub shard_id: u32,
    pub basename: String,
    /// modality name -> [offset, length] within this row's shard.
    #[serde(default)]
    pub offsets: BTreeMap<String, [u64; 2]>,
    /// inline scalar/annotation metadata (SQL-filterable in production).
    #[serde(default)]
    pub meta: BTreeMap<String, Value>,
    /// Optional explicit shard filename (relative to the shard dir). Used by
    /// **partitioned** enrichment layers, whose shards are named `shard-<part>-*`
    /// and so can't be reconstructed from `shard_id` alone. `None` (the default,
    /// and always for the base) means the canonical `shard-{shard_id:05}.tar`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
}

impl IndexRow {
    /// True if `modality` is present (has a non-null offset) in this row.
    pub fn has(&self, modality: &str) -> bool {
        self.offsets.contains_key(modality)
    }

    /// Project this row down to a subset of modalities/meta keys. `None` selects
    /// everything. Returns a row carrying only the requested fields — the basis
    /// of cheap projection reads.
    pub fn project(&self, modalities: Option<&[String]>, meta_keys: Option<&[String]>) -> IndexRow {
        let offsets = match modalities {
            None => self.offsets.clone(),
            Some(sel) => self
                .offsets
                .iter()
                .filter(|(k, _)| sel.iter().any(|s| s == *k))
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        };
        let meta = match meta_keys {
            None => self.meta.clone(),
            Some(sel) => self
                .meta
                .iter()
                .filter(|(k, _)| sel.iter().any(|s| s == *k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        IndexRow {
            sample_id: self.sample_id,
            shard_id: self.shard_id,
            basename: self.basename.clone(),
            offsets,
            meta,
            shard: self.shard.clone(),
        }
    }
}

/// Pluggable index storage. The production backend is
/// [`ParquetIndex`](crate::index_parquet::ParquetIndex) — a directory of columnar
/// Parquet shards, queryable directly by DuckDB / Arrow / DataFusion.
pub trait IndexBackend {
    fn write(&self, path: &Path, rows: &[IndexRow]) -> Result<()>;
    fn read(&self, path: &Path) -> Result<Vec<IndexRow>>;
}

/// Default number of decoded index shards kept resident in [`LazyIndex`].
pub const DEFAULT_INDEX_SHARD_CACHE: usize = 16;

/// Knows how to fetch the raw bytes of an index shard (`index/part-*.json`),
/// either from the local filesystem or — under the `remote` feature — through the
/// content-addressed [`ferroload_io::CachedStorage`] (so re-opens hit local disk).
pub enum ShardLoader {
    Local {
        root: PathBuf,
    },
    #[cfg(feature = "remote")]
    Remote {
        store: ferroload_io::CachedStorage,
        prefix: String,
    },
}

impl ShardLoader {
    /// Read one index-shard object (root-relative `rel`) to bytes.
    fn load(&self, rel: &str) -> Result<Vec<u8>> {
        match self {
            ShardLoader::Local { root } => Ok(std::fs::read(root.join(rel))?),
            #[cfg(feature = "remote")]
            ShardLoader::Remote { store, prefix } => store
                .get_blocking(&remote_key(prefix, rel))
                .map(|b| b.to_vec())
                .map_err(|e| Error::Format(format!("loading index shard {rel}: {e}"))),
        }
    }

    /// Subset one shard (`rel`) by a predicate, with column projection + row-group
    /// pruning. Local reads the file and decodes only the needed columns; remote
    /// fetches just the footer + projected column chunks via ranged GETs.
    fn subset_shard(&self, rel: &str, pred: &crate::subset::Predicate) -> Result<Vec<u64>> {
        match self {
            ShardLoader::Local { root } => {
                let bytes = std::fs::read(root.join(rel))?;
                crate::index_parquet::ParquetIndex::subset_shard(bytes, pred)
            }
            #[cfg(feature = "remote")]
            ShardLoader::Remote { store, prefix } => {
                crate::index_parquet::ParquetIndex::subset_shard_remote(
                    store,
                    &remote_key(prefix, rel),
                    pred,
                )
            }
        }
    }
}

/// Join an in-store key prefix with a root-relative path.
#[cfg(feature = "remote")]
fn remote_key(prefix: &str, rel: &str) -> String {
    if prefix.is_empty() {
        rel.to_string()
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), rel)
    }
}

/// Bounded LRU of decoded index shards: `shard_idx -> Arc<Vec<IndexRow>>`.
struct ShardCache {
    cap: usize,
    map: HashMap<usize, Arc<Vec<IndexRow>>>,
    /// recency order, front = least-recently-used, back = most-recent.
    order: VecDeque<usize>,
    /// instrumentation: number of shard objects fetched + parsed from the loader.
    loads: u64,
}

impl ShardCache {
    fn new(cap: usize) -> Self {
        ShardCache { cap: cap.max(1), map: HashMap::new(), order: VecDeque::new(), loads: 0 }
    }

    fn touch(&mut self, k: usize) {
        if let Some(pos) = self.order.iter().position(|&x| x == k) {
            self.order.remove(pos);
        }
        self.order.push_back(k);
    }

    fn get(&mut self, k: usize) -> Option<Arc<Vec<IndexRow>>> {
        let v = self.map.get(&k).cloned()?;
        self.touch(k);
        Some(v)
    }

    fn insert(&mut self, k: usize, v: Arc<Vec<IndexRow>>) {
        if !self.map.contains_key(&k) {
            while self.map.len() >= self.cap {
                match self.order.pop_front() {
                    Some(old) => {
                        self.map.remove(&old);
                    }
                    None => break,
                }
            }
        }
        self.map.insert(k, v);
        self.touch(k);
    }
}

/// Lazy, sharded index reader. `open()` builds only the directory (from
/// `manifest.index_shards`); row data for a shard is fetched, parsed, and LRU-cached
/// the first time a sample in it is touched. `len()` is O(1) and loads nothing.
pub struct LazyIndex {
    dir: Vec<IndexShardRef>,
    /// `prefix_sum[k]` = total rows before shard `k`; length `dir.len() + 1`.
    prefix_sum: Vec<u64>,
    total: usize,
    loader: ShardLoader,
    cache: Mutex<ShardCache>,
}

impl LazyIndex {
    /// Build from the manifest's (ordered, contiguous, gapless) shard directory.
    pub fn new(dir: Vec<IndexShardRef>, loader: ShardLoader, cache_cap: usize) -> Self {
        let mut prefix_sum = Vec::with_capacity(dir.len() + 1);
        let mut acc = 0u64;
        prefix_sum.push(0);
        for s in &dir {
            acc += s.rows;
            prefix_sum.push(acc);
        }
        LazyIndex {
            dir,
            prefix_sum,
            total: acc as usize,
            loader,
            cache: Mutex::new(ShardCache::new(cache_cap)),
        }
    }

    pub fn len(&self) -> usize {
        self.total
    }
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Number of shard objects fetched + parsed so far (for the laziness assertion
    /// in tests / diagnostics). 0 immediately after `open()`.
    pub fn loaded_count(&self) -> u64 {
        self.cache.lock().unwrap().loads
    }

    /// Positional `i` -> `(shard_idx, local_row_idx)` via the prefix-sum table.
    fn locate(&self, i: usize) -> (usize, usize) {
        let target = i as u64;
        let k = match self.prefix_sum.binary_search(&target) {
            // exact hit on a shard boundary => start of that shard (idx in 0..len)
            Ok(idx) => idx.min(self.dir.len().saturating_sub(1)),
            Err(idx) => idx - 1,
        };
        (k, i - self.prefix_sum[k] as usize)
    }

    /// Fetch + parse shard `sidx` (or serve from the LRU), returning its rows.
    fn ensure_loaded(&self, sidx: usize) -> Result<Arc<Vec<IndexRow>>> {
        {
            let mut c = self.cache.lock().unwrap();
            if let Some(rows) = c.get(sidx) {
                return Ok(rows);
            }
        }
        let bytes = self.loader.load(&self.dir[sidx].path)?;
        let rows = crate::index_parquet::ParquetIndex::read_bytes(bytes)?;
        let arc = Arc::new(rows);
        let mut c = self.cache.lock().unwrap();
        c.insert(sidx, arc.clone());
        c.loads += 1;
        Ok(arc)
    }

    /// Row at position `i` (loads + caches its shard on demand). Cloned out of the
    /// shared shard so the caller owns it without holding the cache lock.
    pub fn get(&self, i: usize) -> Result<Arc<IndexRow>> {
        if i >= self.total {
            return Err(Error::NotFound(format!("index row {i}")));
        }
        let (sidx, local) = self.locate(i);
        let shard = self.ensure_loaded(sidx)?;
        Ok(Arc::new(shard[local].clone()))
    }

    /// Streaming subset scan: evaluate a `WHERE` predicate over every shard in
    /// order (loaded directly, **bypassing the LRU** so a full scan doesn't evict
    /// the working set), returning matching `sample_id`s ascending. Same result as
    /// the eager [`subset_ids`](crate::subset::subset_ids) on the same data.
    pub fn subset_ids_streaming(&self, where_sql: &str) -> Result<Vec<u64>> {
        let pred = crate::subset::Predicate::parse(where_sql)?;
        let mut ids = Vec::new();
        for sref in &self.dir {
            // column projection + row-group pruning happen per shard; remote shards
            // fetch only the footer + projected column chunks via ranged GETs.
            ids.extend(self.loader.subset_shard(&sref.path, &pred)?);
        }
        ids.sort_unstable();
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u64) -> IndexRow {
        let mut offsets = BTreeMap::new();
        offsets.insert("image".into(), [512, 100]);
        if id % 2 == 0 {
            offsets.insert("depth".into(), [2048, 200]); // sparse: only even rows
        }
        let mut meta = BTreeMap::new();
        meta.insert("label".into(), serde_json::json!(id));
        meta.insert("lang".into(), serde_json::json!("en"));
        IndexRow { sample_id: id, shard_id: 0, basename: format!("s{id}"), offsets, meta, shard: None }
    }

    #[test]
    fn presence_mask_for_sparse_modality() {
        assert!(row(0).has("depth"));
        assert!(!row(1).has("depth")); // null depth -> absent (skip + mask)
        assert!(row(1).has("image"));
    }

    #[test]
    fn projection_selects_subset() {
        let r = row(0);
        let p = r.project(Some(&["image".into()]), Some(&["lang".into()]));
        assert!(p.offsets.contains_key("image"));
        assert!(!p.offsets.contains_key("depth")); // projected out
        assert!(p.meta.contains_key("lang"));
        assert!(!p.meta.contains_key("label"));
    }

}
