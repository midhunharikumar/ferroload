//! The global sample index: one row per logical multimodal sample.
//!
//! A row carries the locator for each tensor modality (`shard_id` + per-modality
//! `(offset, length)`) and inline scalar/annotation metadata. The backend is
//! abstracted behind [`IndexBackend`]; the default scaffold backend serializes
//! JSON, and a Parquet backend is planned under the `parquet` feature (the
//! production format per DESIGN.md).

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

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

/// Pluggable index storage. Swap in Parquet/Arrow for production.
pub trait IndexBackend {
    fn write(&self, path: &Path, rows: &[IndexRow]) -> Result<()>;
    fn read(&self, path: &Path) -> Result<Vec<IndexRow>>;
}

/// Default JSON-backed index used by the scaffold and tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonIndex;

impl IndexBackend for JsonIndex {
    fn write(&self, path: &Path, rows: &[IndexRow]) -> Result<()> {
        std::fs::write(path, serde_json::to_vec_pretty(rows)?)?;
        Ok(())
    }
    fn read(&self, path: &Path) -> Result<Vec<IndexRow>> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

/// In-memory view over loaded rows with positional access (dense `sample_id`).
pub struct IndexReader {
    rows: Vec<IndexRow>,
}

impl IndexReader {
    pub fn new(rows: Vec<IndexRow>) -> Self {
        IndexReader { rows }
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn get(&self, i: usize) -> Result<&IndexRow> {
        self.rows
            .get(i)
            .ok_or_else(|| Error::NotFound(format!("index row {i}")))
    }
    /// A half-open range of rows `[start, end)`.
    pub fn range(&self, start: usize, end: usize) -> Result<&[IndexRow]> {
        if start > end || end > self.rows.len() {
            return Err(Error::NotFound(format!("range {start}..{end}")));
        }
        Ok(&self.rows[start..end])
    }
    pub fn rows(&self) -> &[IndexRow] {
        &self.rows
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
    fn write_read_roundtrip() {
        let rows: Vec<_> = (0..5).map(row).collect();
        let mut p = std::env::temp_dir();
        p.push(format!("ferroload_idx_{}.json", std::process::id()));
        JsonIndex.write(&p, &rows).unwrap();
        let back = JsonIndex.read(&p).unwrap();
        assert_eq!(rows, back);
        std::fs::remove_file(&p).ok();
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

    #[test]
    fn reader_get_and_range() {
        let rows: Vec<_> = (0..10).map(row).collect();
        let r = IndexReader::new(rows);
        assert_eq!(r.len(), 10);
        assert_eq!(r.get(3).unwrap().sample_id, 3);
        assert_eq!(r.range(2, 5).unwrap().len(), 3);
        assert!(r.range(8, 100).is_err());
    }
}
