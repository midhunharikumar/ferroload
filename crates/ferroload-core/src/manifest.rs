//! The dataset manifest: the single source of truth for a Ferroload dataset.
//!
//! Designed to be **extensible**: unknown top-level keys and the `extensions`
//! namespace are preserved on round-trip (preserve-unknown), so a newer
//! capability (e.g. a vector search index over an embedding column) can be added
//! without a format change and without an older tool dropping it.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Structural version of the manifest core.
pub const FORMAT_VERSION: u32 = 1;
/// The reader version this build implements.
pub const READER_VERSION: u32 = 1;

fn one() -> u32 {
    1
}

/// A per-modality declaration. `kind` is one of `tensor` | `annotation` | `scalar`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Modality {
    pub ext: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default = "default_codec")]
    pub codec: String,
    /// Free-form, forward-compatible attributes (e.g. embedding dim/metric).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: BTreeMap<String, Value>,
}

fn default_kind() -> String {
    "tensor".into()
}
fn default_codec() -> String {
    "raw".into()
}

impl Modality {
    pub fn tensor(ext: &str, codec: &str) -> Self {
        Modality {
            ext: ext.into(),
            kind: "tensor".into(),
            codec: codec.into(),
            attrs: BTreeMap::new(),
        }
    }
    pub fn scalar(ext: &str) -> Self {
        Modality {
            ext: ext.into(),
            kind: "scalar".into(),
            codec: "passthrough".into(),
            attrs: BTreeMap::new(),
        }
    }
}

/// A column in the global index schema. `semantic` + `attrs` let extensions
/// attach meaning later (e.g. mark a column as an embedding).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct IndexRef {
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub rows: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ShardsRef {
    #[serde(default)]
    pub dir: String,
    #[serde(default)]
    pub count: u32,
    #[serde(default)]
    pub shard_bytes_target: u64,
}

/// An enrichment layer: an additive set of modalities/columns produced by a
/// `map` pass, joined to the base dataset on `sample_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerRef {
    pub name: String,
    /// Path (relative to root) of the layer's index fragment.
    pub index: String,
    /// Path (relative to root) of the layer's shards dir.
    pub shards_dir: String,
    /// Modalities this layer contributes (tensor blobs in its shards).
    #[serde(default)]
    pub modalities: BTreeMap<String, Modality>,
    #[serde(default)]
    pub rows: u64,
}

/// The manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    #[serde(default = "one")]
    pub min_reader_version: u32,
    pub name: String,
    #[serde(default)]
    pub created_utc: String,
    #[serde(default = "one")]
    pub version: u32,
    #[serde(default)]
    pub modalities: BTreeMap<String, Modality>,
    #[serde(default)]
    pub index: IndexRef,
    #[serde(default)]
    pub shards: ShardsRef,
    #[serde(default)]
    pub schema: Vec<Column>,
    /// Additive enrichment layers (from `map`), joined on `sample_id`. Empty for
    /// a base-only dataset (back-compat: older datasets simply omit the field).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<LayerRef>,
    /// Reserved namespace for optional, versioned capabilities.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, Value>,
    /// Preserve-unknown: any top-level key this build doesn't recognize is kept
    /// here and round-tripped verbatim on save.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Manifest {
    pub fn new(name: &str) -> Self {
        Manifest {
            format_version: FORMAT_VERSION,
            min_reader_version: 1,
            name: name.into(),
            created_utc: String::new(),
            version: 1,
            modalities: BTreeMap::new(),
            index: IndexRef::default(),
            shards: ShardsRef::default(),
            schema: Vec::new(),
            layers: Vec::new(),
            extensions: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }

    /// Fail if this build's reader is too old to safely read the dataset.
    pub fn check_reader_compat(&self) -> Result<()> {
        if self.min_reader_version > READER_VERSION {
            return Err(Error::ReaderTooOld {
                required: self.min_reader_version,
                have: READER_VERSION,
            });
        }
        Ok(())
    }

    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_json(&s)
    }

    /// Register (or replace) a capability extension block.
    pub fn put_extension(&mut self, name: &str, payload: Value) {
        self.extensions.insert(name.to_string(), payload);
    }

    pub fn get_extension(&self, name: &str) -> Option<&Value> {
        self.extensions.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let mut m = Manifest::new("demo");
        m.modalities
            .insert("image".into(), Modality::tensor("jpg", "image"));
        m.modalities.insert("text".into(), Modality::scalar("json"));
        let json = m.to_json().unwrap();
        let back = Manifest::from_json(&json).unwrap();
        assert_eq!(m, back);
        assert_eq!(back.format_version, FORMAT_VERSION);
    }

    #[test]
    fn preserves_unknown_top_level_keys() {
        // Simulate a manifest written by a NEWER tool with a key we don't model.
        let src = r#"{
            "format_version": 1,
            "name": "demo",
            "future_capability": {"hello": "world", "n": 7}
        }"#;
        let m = Manifest::from_json(src).unwrap();
        // The unknown key landed in `extra` ...
        assert!(m.extra.contains_key("future_capability"));
        // ... and survives a round-trip verbatim (preserve-unknown).
        let out = m.to_json().unwrap();
        assert!(out.contains("future_capability"));
        assert!(out.contains("\"hello\""));
    }

    #[test]
    fn extensions_namespace_roundtrip() {
        let mut m = Manifest::new("demo");
        m.put_extension(
            "vector_index",
            serde_json::json!([{
                "name": "clip_emb_hnsw", "ext_version": 1,
                "column": "image_embedding", "kind": "hnsw",
                "dim": 768, "metric": "cosine",
                "path": "indexes/clip_emb_hnsw/",
                "built_over_dataset_version": 4, "stale": false
            }]),
        );
        let back = Manifest::from_json(&m.to_json().unwrap()).unwrap();
        let ext = back.get_extension("vector_index").unwrap();
        assert_eq!(ext[0]["column"], "image_embedding");
        assert_eq!(ext[0]["dim"], 768);
    }

    #[test]
    fn reader_compat_floor() {
        let mut m = Manifest::new("demo");
        m.min_reader_version = 1;
        assert!(m.check_reader_compat().is_ok());
        m.min_reader_version = 999;
        assert!(matches!(
            m.check_reader_compat(),
            Err(Error::ReaderTooOld { .. })
        ));
    }

    #[test]
    fn column_semantic_attrs() {
        let mut m = Manifest::new("demo");
        let mut attrs = BTreeMap::new();
        attrs.insert("dim".to_string(), serde_json::json!(768));
        attrs.insert("metric".to_string(), serde_json::json!("cosine"));
        m.schema.push(Column {
            name: "image_embedding".into(),
            dtype: "list<float32>[768]".into(),
            semantic: Some("embedding".into()),
            attrs,
        });
        let back = Manifest::from_json(&m.to_json().unwrap()).unwrap();
        assert_eq!(back.schema[0].semantic.as_deref(), Some("embedding"));
        assert_eq!(back.schema[0].attrs["dim"], 768);
    }
}
