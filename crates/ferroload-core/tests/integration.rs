//! End-to-end integration test on a synthetic multimodal dataset:
//! write video+audio+text(+sparse depth) -> read back -> verify bytes, projection,
//! presence masks, ranges, versioning, and an extension (vector index) round-trip.

use ferroload_core::dataset::{commit_manifest, Dataset, DatasetWriter};
use ferroload_core::manifest::{Manifest, Modality};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn root(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ferroload_it_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn b(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

#[test]
fn synthetic_multimodal_end_to_end() {
    let root = root("e2e");

    let mut w = DatasetWriter::create(&root, "synthetic-av")
        .unwrap()
        .shard_bytes_target(4096)
        .declare("video", Modality::tensor("mp4", "video"))
        .declare("audio", Modality::tensor("flac", "audio"))
        .declare("depth", Modality::tensor("png", "depth16"))
        .declare("text", Modality::scalar("json"));

    let n = 12u64;
    for i in 0..n {
        let mut blobs = BTreeMap::new();
        blobs.insert("video".to_string(), b(&format!("VIDEO-{i}")));
        blobs.insert("audio".to_string(), b(&format!("AUDIO-{i}")));
        // depth only on every 3rd sample -> sparse modality
        if i % 3 == 0 {
            blobs.insert("depth".to_string(), b(&format!("DEPTH-{i}")));
        }
        let mut meta = BTreeMap::new();
        meta.insert("caption".to_string(), serde_json::json!(format!("clip {i}")));
        meta.insert("lang".to_string(), serde_json::json!("en"));
        meta.insert("duration_s".to_string(), serde_json::json!(2 + (i % 5)));
        w.add(&format!("clip{i:05}"), &blobs, &meta).unwrap();
    }
    let manifest = w.close().unwrap();
    assert_eq!(manifest.index.rows, n);
    assert!(manifest.shards.count >= 1);

    // --- read back ---
    let ds = Dataset::open(&root).unwrap();
    assert_eq!(ds.len() as u64, n);

    // exact byte round-trip on a random index
    let s = ds.get(7, None).unwrap();
    assert_eq!(s.blobs["video"], b("VIDEO-7"));
    assert_eq!(s.blobs["audio"], b("AUDIO-7"));
    assert_eq!(s.meta["caption"], serde_json::json!("clip 7"));

    // sparse depth: present on 0,3,6,9 ; absent (mask=false, no fetch) elsewhere
    assert_eq!(ds.get(6, None).unwrap().present["depth"], true);
    assert_eq!(ds.get(7, None).unwrap().present["depth"], false);
    assert!(!ds.get(7, None).unwrap().blobs.contains_key("depth"));

    // projection: text-only read fetches no media
    let only = vec!["text".to_string()];
    let p = ds.get(0, Some(&only)).unwrap();
    assert!(!p.blobs.contains_key("video"));
    assert!(!p.blobs.contains_key("audio"));

    // full verification of all shards/members
    assert_eq!(ds.verify().unwrap() as u64, n);

    // --- extension + versioning: register a vector index, bump + commit ---
    let mut m2 = Manifest::load(&root.join("manifest.json")).unwrap();
    m2.version += 1;
    m2.put_extension(
        "vector_index",
        serde_json::json!([{
            "name": "clip_emb_hnsw", "ext_version": 1,
            "column": "caption_embedding", "kind": "hnsw",
            "dim": 768, "metric": "cosine",
            "path": "indexes/clip_emb_hnsw/",
            "built_over_dataset_version": 1, "covers_rows": n, "stale": false
        }]),
    );
    commit_manifest(&root, &m2).unwrap();

    // old snapshot preserved (time-travel), new one written
    assert!(root.join("versions/v1.json").exists());
    assert!(root.join("versions/v2.json").exists());

    // reopen: extension survives, data still reads
    let ds2 = Dataset::open(&root).unwrap();
    let ext = ds2.manifest().get_extension("vector_index").unwrap();
    assert_eq!(ext[0]["column"], "caption_embedding");
    assert_eq!(ds2.get(11, None).unwrap().blobs["video"], b("VIDEO-11"));

    std::fs::remove_dir_all(&root).ok();
}
