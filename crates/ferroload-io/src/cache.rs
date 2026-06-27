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

    /// Full-object read, served from cache on hit; otherwise fetched and stored.
    /// Used for small, immutable control files (e.g. a sharded index fragment),
    /// keyed by `(path, None)` so it never collides with ranged entries.
    pub async fn get(&self, path: &str) -> IoResult<Bytes> {
        let entry = self.entry_path(path, None);
        if let Ok(bytes) = std::fs::read(&entry) {
            return Ok(Bytes::from(bytes)); // HIT
        }
        let bytes = self.backing.get(path).await?; // MISS -> fetch
        let tmp = entry.with_extension("tmp");
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &entry);
        }
        Ok(bytes)
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

    /// Coalesced multi-range read: cache hits are served from disk, and the misses
    /// are fetched in a **single** backing `get_ranges` (object_store coalesces +
    /// parallelizes them), then written through. Order is preserved.
    pub async fn get_ranges(&self, path: &str, ranges: &[Range<u64>]) -> IoResult<Vec<Bytes>> {
        let mut out: Vec<Option<Bytes>> = vec![None; ranges.len()];
        let mut miss_idx = Vec::new();
        let mut miss_ranges = Vec::new();
        for (i, r) in ranges.iter().enumerate() {
            match std::fs::read(self.entry_path(path, Some(r.clone()))) {
                Ok(b) => out[i] = Some(Bytes::from(b)),
                Err(_) => {
                    miss_idx.push(i);
                    miss_ranges.push(r.clone());
                }
            }
        }
        if !miss_ranges.is_empty() {
            let fetched = self.backing.get_ranges(path, &miss_ranges).await?;
            for (i, bytes) in miss_idx.into_iter().zip(fetched) {
                let entry = self.entry_path(path, Some(ranges[i].clone()));
                let tmp = entry.with_extension("tmp");
                if std::fs::write(&tmp, &bytes).is_ok() {
                    let _ = std::fs::rename(&tmp, &entry);
                }
                out[i] = Some(bytes);
            }
        }
        Ok(out.into_iter().map(|b| b.expect("range filled")).collect())
    }

    /// Blocking variants (drive the async reads on the shared runtime), used by
    /// the synchronous, GIL-released core read path.
    pub fn get_blocking(&self, path: &str) -> IoResult<Bytes> {
        crate::runtime().block_on(self.get(path))
    }

    pub fn get_range_blocking(&self, path: &str, range: Range<u64>) -> IoResult<Bytes> {
        crate::runtime().block_on(self.get_range(path, range))
    }

    pub fn get_ranges_blocking(&self, path: &str, ranges: &[Range<u64>]) -> IoResult<Vec<Bytes>> {
        crate::runtime().block_on(self.get_ranges(path, ranges))
    }

    /// Object size in bytes (delegates to the backing store; immutable per
    /// version). Used to locate the Parquet footer for ranged reads.
    pub async fn size(&self, path: &str) -> IoResult<u64> {
        self.backing.size(path).await
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
