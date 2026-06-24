//! Worked example: build a small synthetic video+audio+text(+sparse depth)
//! dataset, then demonstrate full reads, projection, presence masks, ranges,
//! and registering a vector-index extension.
//!
//! Run with:  CARGO_TARGET_DIR=/tmp/ferro-target cargo run --example synthetic_av

use ferroload_core::dataset::{commit_manifest, Dataset, DatasetWriter};
use ferroload_core::manifest::{Column, Manifest, Modality};
use std::collections::BTreeMap;

fn b(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

fn main() -> ferroload_core::Result<()> {
    let root = std::env::temp_dir().join("ferroload_example_av");
    let _ = std::fs::remove_dir_all(&root);

    // 1) write
    let mut w = DatasetWriter::create(&root, "synthetic-av")?
        .shard_bytes_target(8192)
        .declare("video", Modality::tensor("mp4", "video"))
        .declare("audio", Modality::tensor("flac", "audio"))
        .declare("depth", Modality::tensor("png", "depth16"))
        .declare("text", Modality::scalar("json"));

    for i in 0..8u64 {
        let mut blobs = BTreeMap::new();
        blobs.insert("video".into(), b(&format!("VIDEO-{i}")));
        blobs.insert("audio".into(), b(&format!("AUDIO-{i}")));
        if i % 3 == 0 {
            blobs.insert("depth".into(), b(&format!("DEPTH-{i}")));
        }
        let mut meta = BTreeMap::new();
        meta.insert("caption".into(), serde_json::json!(format!("clip {i}")));
        meta.insert("duration_s".into(), serde_json::json!(2 + i % 4));
        w.add(&format!("clip{i:05}"), &blobs, &meta)?;
    }
    let m = w.close()?;
    println!("wrote {} samples across {} shard(s)", m.index.rows, m.shards.count);

    // 2) read
    let ds = Dataset::open(&root)?;
    let s = ds.get(5, None)?;
    println!(
        "sample 5: video={:?} present={:?}",
        String::from_utf8_lossy(&s.blobs["video"]),
        s.present
    );

    // 3) projection (text only -> no media fetched)
    let only_text = ["text".to_string()];
    let p = ds.get(5, Some(&only_text))?;
    println!("projected sample 5 fetched modalities: {:?}", p.blobs.keys().collect::<Vec<_>>());

    // 4) presence mask for sparse depth
    let present_depth: Vec<u64> = (0..ds.len())
        .filter_map(|i| {
            let s = ds.get(i, None).ok()?;
            if *s.present.get("depth").unwrap_or(&false) { Some(s.sample_id) } else { None }
        })
        .collect();
    println!("samples WITH depth: {present_depth:?}");

    // 5) verify integrity
    println!("verified {} samples", ds.verify()?);

    // 6) register a vector-index extension + version bump
    let mut m2 = Manifest::load(&root.join("manifest.json"))?;
    m2.version += 1;
    m2.schema.push(Column {
        name: "caption_embedding".into(),
        dtype: "list<float32>[768]".into(),
        semantic: Some("embedding".into()),
        attrs: BTreeMap::from([("dim".into(), serde_json::json!(768))]),
    });
    m2.put_extension(
        "vector_index",
        serde_json::json!([{ "name": "cap_hnsw", "column": "caption_embedding",
            "kind": "hnsw", "dim": 768, "metric": "cosine", "stale": false }]),
    );
    commit_manifest(&root, &m2)?;
    let ds2 = Dataset::open(&root)?;
    println!(
        "reopened: version={}, extensions={:?}",
        ds2.manifest().version,
        ds2.manifest().extensions.keys().collect::<Vec<_>>()
    );

    std::fs::remove_dir_all(&root).ok();
    Ok(())
}
