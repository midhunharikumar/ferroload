//! Flexibility tests: the format must handle arbitrary modality combinations —
//! multiple images, multiple video streams, mixed video+image+audio+metadata,
//! and sparse/heterogeneous samples — all in one dataset.

use ferroload_core::dataset::{Dataset, DatasetWriter};
use ferroload_core::manifest::Modality;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn root(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ferroload_combo_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&p);
    p
}
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

#[test]
fn multiple_image_streams() {
    let root = root("multi_img");
    let mut w = DatasetWriter::create(&root, "multi-img")
        .unwrap()
        .declare("image", Modality::tensor("jpg", "image"))
        .declare("thumb", Modality::tensor("jpg", "image"))
        .declare("mask", Modality::tensor("png", "image"));
    for i in 0..4u64 {
        let mut b = BTreeMap::new();
        b.insert("image".into(), bytes(&format!("IMG{i}")));
        b.insert("thumb".into(), bytes(&format!("TH{i}")));
        b.insert("mask".into(), bytes(&format!("MASK{i}")));
        w.add(&format!("s{i}"), &b, &BTreeMap::new()).unwrap();
    }
    w.close().unwrap();

    let ds = Dataset::open(&root).unwrap();
    let s = ds.get(2, None).unwrap();
    assert_eq!(s.blobs["image"], bytes("IMG2"));
    assert_eq!(s.blobs["thumb"], bytes("TH2"));
    assert_eq!(s.blobs["mask"], bytes("MASK2"));
    // projection: only two of three image streams
    let only = vec!["image".to_string(), "mask".to_string()];
    let p = ds.get(2, Some(&only)).unwrap();
    assert!(p.blobs.contains_key("image") && p.blobs.contains_key("mask"));
    assert!(!p.blobs.contains_key("thumb"));
    assert_eq!(ds.verify().unwrap(), 4);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn multiple_video_streams_plus_audio() {
    let root = root("multi_vid");
    let mut w = DatasetWriter::create(&root, "multi-vid")
        .unwrap()
        .declare("video", Modality::tensor("mp4", "video"))
        .declare("video_depth", Modality::tensor("mp4", "video"))
        .declare("audio", Modality::tensor("flac", "audio"));
    for i in 0..3u64 {
        let mut b = BTreeMap::new();
        b.insert("video".into(), bytes(&format!("V{i}")));
        b.insert("video_depth".into(), bytes(&format!("VD{i}")));
        b.insert("audio".into(), bytes(&format!("A{i}")));
        w.add(&format!("clip{i}"), &b, &BTreeMap::new()).unwrap();
    }
    w.close().unwrap();

    let ds = Dataset::open(&root).unwrap();
    let s = ds.get(1, None).unwrap();
    assert_eq!(s.blobs["video"], bytes("V1"));
    assert_eq!(s.blobs["video_depth"], bytes("VD1"));
    assert_eq!(s.blobs["audio"], bytes("A1"));
    assert_eq!(ds.verify().unwrap(), 3);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn mixed_video_image_audio_text_metadata() {
    let root = root("mixed");
    let mut w = DatasetWriter::create(&root, "mixed")
        .unwrap()
        .declare("video", Modality::tensor("mp4", "video"))
        .declare("keyframe", Modality::tensor("jpg", "image"))
        .declare("audio", Modality::tensor("flac", "audio"));
    for i in 0..3u64 {
        let mut b = BTreeMap::new();
        b.insert("video".into(), bytes(&format!("V{i}")));
        b.insert("keyframe".into(), bytes(&format!("K{i}")));
        b.insert("audio".into(), bytes(&format!("A{i}")));
        let mut meta = BTreeMap::new();
        meta.insert("caption".into(), serde_json::json!(format!("c{i}")));
        meta.insert("label".into(), serde_json::json!(i as i64));
        meta.insert("boxes".into(), serde_json::json!([[1, 2, 3, 4]]));
        w.add(&format!("s{i}"), &b, &meta).unwrap();
    }
    w.close().unwrap();

    let ds = Dataset::open(&root).unwrap();
    let s = ds.get(0, None).unwrap();
    assert_eq!(s.blobs["video"], bytes("V0"));
    assert_eq!(s.blobs["keyframe"], bytes("K0"));
    assert_eq!(s.meta["caption"], serde_json::json!("c0"));
    assert_eq!(s.meta["label"], serde_json::json!(0));
    assert_eq!(s.meta["boxes"], serde_json::json!([[1, 2, 3, 4]]));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn sparse_heterogeneous_samples() {
    // every sample has video; only some have a keyframe; only some have audio
    let root = root("sparse");
    let mut w = DatasetWriter::create(&root, "sparse")
        .unwrap()
        .declare("video", Modality::tensor("mp4", "video"))
        .declare("keyframe", Modality::tensor("jpg", "image"))
        .declare("audio", Modality::tensor("flac", "audio"));
    for i in 0..6u64 {
        let mut b = BTreeMap::new();
        b.insert("video".into(), bytes(&format!("V{i}")));
        if i % 2 == 0 {
            b.insert("keyframe".into(), bytes(&format!("K{i}")));
        }
        if i % 3 == 0 {
            b.insert("audio".into(), bytes(&format!("A{i}")));
        }
        w.add(&format!("s{i}"), &b, &BTreeMap::new()).unwrap();
    }
    w.close().unwrap();

    let ds = Dataset::open(&root).unwrap();
    let s1 = ds.get(1, None).unwrap(); // odd: no keyframe, no audio
    assert!(s1.present["video"]);
    assert!(!s1.present["keyframe"]);
    assert!(!s1.present["audio"]);
    assert!(!s1.blobs.contains_key("keyframe"));
    let s0 = ds.get(0, None).unwrap(); // has all three
    assert!(s0.present["video"] && s0.present["keyframe"] && s0.present["audio"]);
    assert_eq!(ds.verify().unwrap(), 6);
    std::fs::remove_dir_all(&root).ok();
}
