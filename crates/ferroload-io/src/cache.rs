//! Content-addressed local cache (cache-aside) over a [`Storage`].
//!
//! Epoch 1 fills the cache; later epochs read from local disk at NVMe speed
//! (DESIGN §14.2). Entries are addressed by `(path, range)` hash, so they are
//! immutable and safe to share.

use crate::{cache_key, IoResult, Storage};
use bytes::Bytes;
use std::ops::Range;
use std::path::PathBuf;

/// Wraps a backing [`Storage`] with an on-disk cache directory.
#[derive(Clone)]
pub struct CachedStorage {
    backing: Storage,
    dir: PathBuf,
}

impl CachedStorage {
    pub fn new(backing: Storage, dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(CachedStorage { backing, dir })
    }

    fn entry_path(&self, path: &str, range: Option<Range<u64>>) -> PathBuf {
        self.dir.join(cache_key(path, range))
    }

    /// Ranged read, served from cache on hit; otherwise fetched and stored.
    pub async fn get_range(&self, path: &str, range: Range<u64>) -> IoResult<Bytes> {
        let entry = self.entry_path(path, Some(range.clone()));
        if let Ok(bytes) = std::fs::read(&entry) {
            return Ok(Bytes::from(bytes)); // HIT
        }
        let bytes = self.backing.get_range(path, range).await?; // MISS -> fetch
        // write-through (best-effort; cache errors must not fail the read)
        let tmp = entry.with_extension("tmp");
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &entry);
        }
        Ok(bytes)
    }

    /// True if an entry for `(path, range)` is already cached.
    pub fn is_cached(&self, path: &str, range: Option<Range<u64>>) -> bool {
        self.entry_path(path, range).exists()
    }

    pub fn backing(&self) -> &Storage {
        &self.backing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn miss_then_hit_survives_backing_deletion() {
        let dir = tempfile::tempdir().unwrap();
        let s = Storage::memory();
        s.put("a.tar", Bytes::from_static(b"ABCDEFGHIJ")).await.unwrap();
        let cached = CachedStorage::new(s.clone(), dir.path()).unwrap();

        assert!(!cached.is_cached("a.tar", Some(2..5)));
        let first = cached.get_range("a.tar", 2..5).await.unwrap(); // MISS -> fetch + store
        assert_eq!(&first[..], b"CDE");
        assert!(cached.is_cached("a.tar", Some(2..5)));

        // wipe the backing object; cache must still serve the range
        let empty = Storage::memory();
        let cached2 = CachedStorage::new(empty, dir.path()).unwrap();
        let hit = cached2.get_range("a.tar", 2..5).await.unwrap(); // served from disk
        assert_eq!(&hit[..], b"CDE");
    }
}
