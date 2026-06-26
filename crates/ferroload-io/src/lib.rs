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
use std::sync::{Arc, OnceLock};

pub mod cache;
pub use cache::CachedStorage;

pub type IoResult<T> = std::result::Result<T, object_store::Error>;

fn generic(msg: impl Into<String>) -> object_store::Error {
    let s: String = msg.into();
    object_store::Error::Generic {
        store: "ferroload-io",
        source: s.into(),
    }
}

/// A process-wide multi-thread Tokio runtime used to drive `object_store`'s async
/// reads from Ferroload's **synchronous** (GIL-released) read path. The runtime's
/// own worker threads do the network I/O, so blocking on it from a rayon/decoder
/// thread never touches the Python GIL.
pub fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("ferroload-io")
            .build()
            .expect("build ferroload-io tokio runtime")
    })
}

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

    /// Multipart upload in `part_size` chunks — for large objects that a single
    /// PUT would time out on. Parts are uploaded sequentially.
    pub async fn put_chunked(&self, path: &str, data: &[u8], part_size: usize) -> IoResult<()> {
        let mut mp = self.store.put_multipart(&OsPath::from(path)).await?;
        let mut off = 0;
        while off < data.len() {
            let end = (off + part_size).min(data.len());
            mp.put_part(PutPayload::from(data[off..end].to_vec())).await?;
            off = end;
        }
        mp.complete().await?;
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

    /// Object size in bytes (a HEAD). Needed to locate a Parquet footer for
    /// ranged column reads.
    pub async fn size(&self, path: &str) -> IoResult<u64> {
        Ok(self.store.head(&OsPath::from(path)).await?.size as u64)
    }

    /// One directory level under `prefix`: `(common-prefixes, (object, size))`.
    /// Encapsulates `object_store` types so callers stay backend-agnostic.
    pub async fn list_dir(&self, prefix: &str) -> IoResult<(Vec<String>, Vec<(String, u64)>)> {
        let p = if prefix.is_empty() { None } else { Some(OsPath::from(prefix)) };
        let res = self.store.list_with_delimiter(p.as_ref()).await?;
        let prefixes = res.common_prefixes.iter().map(|p| p.to_string()).collect();
        let objs = res.objects.iter().map(|o| (o.location.to_string(), o.size as u64)).collect();
        Ok((prefixes, objs))
    }

    /// Build a `Storage` from a URL and return it plus the **in-store key prefix**
    /// for the dataset root (everything after the bucket/container). Credentials
    /// and region come from the environment (the standard `AWS_*` / `GOOGLE_*` /
    /// `AZURE_*` variables), so callers don't pass secrets.
    ///
    /// Schemes: `file://…`, `memory://`, and (feature-gated) `s3://bucket/prefix`,
    /// `gs://bucket/prefix`, `az://container/prefix`.
    pub fn from_url(url_str: &str) -> IoResult<(Self, String)> {
        let u = url::Url::parse(url_str)
            .map_err(|e| generic(format!("bad url {url_str:?}: {e}")))?;
        let path = u.path().trim_matches('/').to_string();
        let missing = || generic(format!("{url_str:?} has no bucket/container"));
        // For object stores rooted at a bucket/container, the URL path is the
        // in-store key prefix. For a local FS we root the store at the path itself,
        // so keys are relative and the prefix is empty.
        let (store, prefix): (Arc<dyn ObjectStore>, String) = match u.scheme() {
            "file" => (
                Arc::new(object_store::local::LocalFileSystem::new_with_prefix(u.path())?),
                String::new(),
            ),
            "memory" | "mem" => (Arc::new(object_store::memory::InMemory::new()), path),
            #[cfg(feature = "aws")]
            "s3" => (
                Arc::new(
                    object_store::aws::AmazonS3Builder::from_env()
                        .with_bucket_name(u.host_str().ok_or_else(missing)?)
                        .build()?,
                ),
                path,
            ),
            #[cfg(feature = "gcp")]
            "gs" => (
                Arc::new(
                    object_store::gcp::GoogleCloudStorageBuilder::from_env()
                        .with_bucket_name(u.host_str().ok_or_else(missing)?)
                        .build()?,
                ),
                path,
            ),
            #[cfg(feature = "azure")]
            "az" | "azure" | "abfs" => (
                Arc::new(
                    object_store::azure::MicrosoftAzureBuilder::from_env()
                        .with_container_name(u.host_str().ok_or_else(missing)?)
                        .build()?,
                ),
                path,
            ),
            other => {
                return Err(generic(format!(
                    "unsupported or disabled scheme {other:?} (build with the \
                     aws/gcp/azure feature to enable cloud backends)"
                )))
            }
        };
        Ok((Storage { store }, prefix))
    }

    /// Blocking full-object read (for small control files: manifest, index).
    pub fn get_blocking(&self, path: &str) -> IoResult<Bytes> {
        runtime().block_on(self.get(path))
    }

    /// Blocking ranged read (drives the async `get_range` on the shared runtime).
    pub fn get_range_blocking(&self, path: &str, range: Range<u64>) -> IoResult<Bytes> {
        runtime().block_on(self.get_range(path, range))
    }

    /// Blocking coalesced multi-range read.
    pub fn get_ranges_blocking(&self, path: &str, ranges: &[Range<u64>]) -> IoResult<Vec<Bytes>> {
        runtime().block_on(self.get_ranges(path, ranges))
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
