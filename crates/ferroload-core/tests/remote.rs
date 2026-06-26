//! End-to-end remote-read path via the local object-store backend (`file://`).
//!
//! Exercises `Dataset::open_url`, the cached ranged reads, the coalesced batch
//! read, full-sample materialization, and a warm-cache reopen — all without any
//! cloud credentials. Run with:  `cargo test -p ferroload-core --features remote`.
#![cfg(feature = "remote")]

use ferroload_core::dataset::{Dataset, DatasetWriter};
use ferroload_core::manifest::Modality;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn tmp(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ferroload_remote_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&p);
    p
}

#[test]
fn file_url_round_trip() {
    let root = tmp("ds");
    let mut w = DatasetWriter::create(&root, "remote-demo")
        .unwrap()
        .shard_bytes_target(4096)
        .declare("image", Modality::tensor("jpg", "image"))
        .declare("depth", Modality::tensor("png", "depth16"));
    let n = 10u64;
    for i in 0..n {
        let mut blobs = BTreeMap::new();
        blobs.insert("image".to_string(), format!("IMG-{i}").into_bytes());
        if i % 2 == 0 {
            blobs.insert("depth".to_string(), format!("DEPTH-{i}").into_bytes());
        }
        let mut meta = BTreeMap::new();
        meta.insert("label".to_string(), serde_json::json!(i % 3));
        w.add(&format!("s{i:05}"), &blobs, &meta).unwrap();
    }
    w.close().unwrap();

    // Open through the object-store path: file:// uses object_store's
    // LocalFileSystem, so the entire remote read plumbing is exercised locally.
    let url = format!("file://{}", root.display());
    let cache = tmp("cache");
    let ds = Dataset::open_url(&url, &cache).unwrap();
    assert_eq!(ds.len() as u64, n);

    // open is lazy: only manifest + the index directory were fetched, no shard.
    assert!(ds.manifest().index_shards.len() >= 1, "remote dataset should be sharded");
    assert_eq!(ds.index_shard_loads(), 0, "open_url must not fetch any index shard");

    // single ranged read + sparse modality; touching one sample fetches exactly
    // one index shard (through the on-disk cache).
    assert_eq!(ds.read_blob(7, "image").unwrap().unwrap(), b"IMG-7");
    assert_eq!(ds.index_shard_loads(), 1, "one sample touched -> one index shard");
    assert!(ds.read_blob(1, "depth").unwrap().is_none());
    assert_eq!(ds.read_blob(2, "depth").unwrap().unwrap(), b"DEPTH-2");

    // coalesced batch read (one get_ranges per shard)
    let (buf, spans) = ds.read_blobs_contig(&[0, 1, 2, 3], "image").unwrap();
    for (k, (off, len)) in spans.iter().enumerate() {
        assert_eq!(&buf[*off..*off + *len], format!("IMG-{k}").as_bytes());
    }

    // full sample materialization (blobs + merged meta) through the remote path
    let s = ds.get(4, None).unwrap();
    assert_eq!(s.blobs["image"], b"IMG-4");
    assert_eq!(s.meta["label"], serde_json::json!(4 % 3));

    // a second open reuses the warm on-disk cache
    let ds2 = Dataset::open_url(&url, &cache).unwrap();
    assert_eq!(ds2.read_blob(9, "image").unwrap().unwrap(), b"IMG-9");

    assert_eq!(ds.verify().unwrap(), n as usize);
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&cache);
}

/// Remote `subset` goes through the async, ranged Parquet reader (footer +
/// projected column chunks via cached ranged GETs), and must match the data.
#[test]
fn remote_subset_ranged_reader() {
    let root = tmp("subset_ds");
    let mut w = DatasetWriter::create(&root, "remote-subset")
        .unwrap()
        .shard_bytes_target(4096) // several index shards
        .declare("image", Modality::tensor("bin", "raw"));
    let n = 40u64;
    for i in 0..n {
        let mut blobs = BTreeMap::new();
        blobs.insert("image".to_string(), format!("IMG-{i}").into_bytes());
        let animal = if i % 2 == 0 { "cat" } else { "dog" };
        let mut meta = BTreeMap::new();
        meta.insert("caption".to_string(), serde_json::json!(format!("a photo of a {animal}")));
        meta.insert("width".to_string(), serde_json::json!(64 + i as i64));
        w.add(&format!("s{i:05}"), &blobs, &meta).unwrap();
    }
    w.close().unwrap();

    let url = format!("file://{}", root.display());
    let cache = tmp("subset_cache");
    let ds = Dataset::open_url(&url, &cache).unwrap();

    // compound predicate over a string + numeric meta column, through the
    // remote ranged reader (projection skips the fat caption column for the
    // numeric clause's row groups; pruning skips non-matching groups).
    let cats = ds.subset("caption = 'a photo of a cat' AND width >= 100").unwrap();
    let expect: Vec<u64> = (0..n).filter(|&i| i % 2 == 0 && 64 + i >= 100).collect();
    assert_eq!(cats, expect);

    // the ranged reads populated the on-disk cache (so re-subset is local)
    let again = ds.subset("width >= 100").unwrap();
    assert_eq!(again, (0..n).filter(|&i| 64 + i >= 100).collect::<Vec<u64>>());

    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&cache);
}
