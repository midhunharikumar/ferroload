//! # ferroload-core
//!
//! Pure-Rust **format core** for the Ferroload multimodal dataset format.
//! This crate owns the on-disk format and is independent of PyTorch, decoding,
//! and cloud storage (those live in sibling crates). It provides:
//!
//! - [`manifest`] — the self-contained, **extensible** manifest (preserve-unknown
//!   top-level keys + a versioned `extensions` namespace).
//! - [`shard`] — a WebDataset-compatible USTAR tar writer with exact byte-offset
//!   tracking, plus random-access member reads.
//! - [`sideindex`] — per-shard `member -> (offset, len)` side-index.
//! - [`index`] — the global sample index (rows, projection, pluggable backend).
//! - [`dataset`] — [`dataset::DatasetWriter`] / [`dataset::Dataset`] glue with
//!   atomic, versioned commits.
//!
//! See `DESIGN.md` for the full specification and `USAGE.md` / `EXAMPLES.md` for
//! worked examples.
//!
//! ```
//! use ferroload_core::dataset::{Dataset, DatasetWriter};
//! use ferroload_core::manifest::Modality;
//! use std::collections::BTreeMap;
//!
//! let root = std::env::temp_dir().join("ferroload_doctest");
//! let _ = std::fs::remove_dir_all(&root);
//!
//! let mut w = DatasetWriter::create(&root, "demo").unwrap()
//!     .declare("image", Modality::tensor("jpg", "image"));
//! let mut blobs = BTreeMap::new();
//! blobs.insert("image".to_string(), b"JPEGBYTES".to_vec());
//! w.add("s0000", &blobs, &BTreeMap::new()).unwrap();
//! w.close().unwrap();
//!
//! let ds = Dataset::open(&root).unwrap();
//! assert_eq!(ds.len(), 1);
//! let s = ds.get(0, None).unwrap();
//! assert_eq!(s.blobs["image"], b"JPEGBYTES".to_vec());
//! # std::fs::remove_dir_all(&root).ok();
//! ```

pub mod dataset;
pub mod error;
pub mod index;
pub mod index_parquet;
pub mod manifest;
pub mod sampler;
pub mod shard;
pub mod sideindex;
pub mod subset;

pub use dataset::{Dataset, DatasetWriter, LayerWriter, Sample};
pub use error::{Error, Result};
pub use index_parquet::ParquetIndex;
pub use manifest::{Column, LayerRef, Manifest, Modality, FORMAT_VERSION, READER_VERSION};
pub use sampler::{Sampler, Topology};
pub use subset::{subset_ids, Predicate};
