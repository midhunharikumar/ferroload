//! # ferroload-io
//!
//! Storage layer over [`object_store`], giving one async API across local FS,
//! in-memory, and (feature-gated) S3 / GCS / Azure — plus a content-addressed
//! local cache for fast multi-epoch reads (DESIGN §14.2).
//!
//! The core primitive is the **byte-range read** (`get_range`), since both
//! "load index i" and "load a range of indices" reduce to ranged GETs.

use bytes::Bytes;
use object_store::path::Path as OsPath;
use object_store::{ObjectStore, PutPayload};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod cache;
pub use cache::CachedStorage;

pub type IoResult<T> = std::result::Result<T, object_store::Error>;

/// A thin handle over an `ObjectStore` with the reads/writes Ferroload needs.
#[derive(Clone)]
pub struct Storage {
    store: Arc<dyn ObjectStore>,
}

impl Storage {
    pub fn from_store(store: Arc<dyn ObjectStore>) -> Self {
        Storage { store }
    }

    /// Local filesystem rooted at `root`.
    pub fn local(root: impl AsRef<Path>) -> IoResult<Self> {
        let lfs = object_store::local::LocalFileSystem::new_with_prefix(root)?;
        Ok(Storage { store: Arc::new(lfs) })
    }

    /// In-memory store (tests / ephemeral).
    pub fn memory() -> Self {
        Storage {
            store: Arc::new(object_store::memory::InMemory::new()),
        }
    }

    pub fn store(&self) -> Arc<dyn ObjectStore> {
        self.store.clone()
    }

    pub async fn put(&self, path: &str, data: impl Into<Bytes>) -> IoResult<()> {
        let payload = PutPayload::from_bytes(data.into());
        self.store.put(&OsPath::from(path), payload).await?;
        Ok(())
    }

    /// Full object.
    pub async fn get(&self, path: &str) -> IoResult<Bytes> {
        self.store.get(&OsPath::from(path)).await?.bytes().await
    }

    /// Single byte range — the workhorse for indexed/range reads.
    pub async fn get_range(&self, path: &str, range: Range<u64>) -> IoResult<Bytes> {
        let r = (range.start as usize)..(range.end as usize);
        self.store.get_range(&OsPath::from(path), r).await
    }

    /// Coalesced multi-range read (fewer, larger GETs).
    pub async fn get_ranges(&self, path: &str, ranges: &[Range<u64>]) -> IoResult<Vec<Bytes>> {
        let rs: Vec<Range<usize>> = ranges
            .iter()
            .map(|r| (r.start as usize)..(r.end as usize))
            .collect();
        self.store.get_ranges(&OsPath::from(path), &rs).await
    }

    pub async fn exists(&self, path: &str) -> bool {
        self.store.head(&OsPath::from(path)).await.is_ok()
    }
}

/// Stable content-addressed cache key for `(path, optional range)`.
pub fn cache_key(path: &str, range: Option<Range<u64>>) -> String {
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    if let Some(r) = range {
        r.start.hash(&mut h);
        r.end.hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

/// Default per-user cache directory.
pub fn default_cache_dir() -> PathBuf {
    std::env::temp_dir().join("ferroload-cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_put_get_range() {
        let s = Storage::memory();
        s.put("shards/a.tar", Bytes::from_static(b"0123456789")).await.unwrap();
        assert_eq!(&s.get("shards/a.tar").await.unwrap()[..], b"0123456789");
        // ranged read = the core random-access primitive
        let r = s.get_range("shards/a.tar", 2..5).await.unwrap();
        assert_eq!(&r[..], b"234");
        // coalesced multi-range
        let rs = s.get_ranges("shards/a.tar", &[0..2, 8..10]).await.unwrap();
        assert_eq!(&rs[0][..], b"01");
        assert_eq!(&rs[1][..], b"89");
    }

    #[tokio::test]
    async fn local_fs_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let s = Storage::local(dir.path()).unwrap();
        s.put("x.bin", Bytes::from_static(b"hello")).await.unwrap();
        assert!(s.exists("x.bin").await);
        assert_eq!(&s.get_range("x.bin", 1..4).await.unwrap()[..], b"ell");
    }

    #[test]
    fn cache_key_is_stable_and_range_sensitive() {
        assert_eq!(cache_key("p", None), cache_key("p", None));
        assert_ne!(cache_key("p", Some(0..10)), cache_key("p", Some(0..11)));
        assert_ne!(cache_key("p", None), cache_key("q", None));
    }
}
