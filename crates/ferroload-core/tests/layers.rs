//! Enrichment-layer end-to-end: write a base dataset, add a layer via
//! `LayerWriter` (tensor + scalar/annotation outputs), reopen and read the
//! enriched modalities + merged metadata back. Plus resume/append.

use ferroload_core::dataset::{Dataset, DatasetWriter, LayerWriter};
use ferroload_core::manifest::Modality;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn root(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ferroload_layer_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&p);
    p
}
fn b(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

fn build_base(root: &std::path::Path, n: u64) {
    let mut w = DatasetWriter::create(root, "base")
        .unwrap()
        .declare("image", Modality::tensor("jpg", "image"));
    for i in 0..n {
        let mut blobs = BTreeMap::new();
        blobs.insert("image".to_string(), b(&format!("IMG{i}")));
        let mut meta = BTreeMap::new();
        meta.insert("label".to_string(), serde_json::json!(i as i64));
        w.add(&format!("s{i:04}"), &blobs, &meta).unwrap();
    }
    w.close().unwrap();
}

#[test]
fn enrich_tensor_and_annotation_layers() {
    let root = root("enrich");
    build_base(&root, 5);

    // tensor-output layer: "depth"
    let mut depth = LayerWriter::create(
        &root,
        "depth",
        BTreeMap::from([("depth".to_string(), Modality::tensor("npy", "npy"))]),
    )
    .unwrap();
    for sid in 0..5u64 {
        let mut blobs = BTreeMap::new();
        blobs.insert("depth".to_string(), b(&format!("DEPTH{sid}")));
        depth.add(sid, &blobs, &BTreeMap::new()).unwrap();
    }
    depth.close().unwrap();

    // annotation-output layer: "caption" (no shards, just inline meta)
    let mut cap = LayerWriter::create(&root, "caption", BTreeMap::new()).unwrap();
    for sid in 0..5u64 {
        let mut meta = BTreeMap::new();
        meta.insert("caption".to_string(), serde_json::json!(format!("c{sid}")));
        cap.add(sid, &BTreeMap::new(), &meta).unwrap();
    }
    cap.close().unwrap();

    // reopen: base + both layers visible
    let ds = Dataset::open(&root).unwrap();
    assert_eq!(ds.len(), 5);
    let s = ds.get(2, None).unwrap();
    assert_eq!(s.blobs["image"], b("IMG2")); // base
    assert_eq!(s.blobs["depth"], b("DEPTH2")); // tensor layer
    assert_eq!(s.present["depth"], true);
    assert_eq!(s.meta["label"], serde_json::json!(2)); // base meta
    assert_eq!(s.meta["caption"], serde_json::json!("c2")); // annotation layer meta

    // projection across base+layer
    let only = vec!["depth".to_string()];
    let p = ds.get(3, Some(&only)).unwrap();
    assert_eq!(p.blobs["depth"], b("DEPTH3"));
    assert!(!p.blobs.contains_key("image"));

    // read_blob direct
    assert_eq!(ds.read_blob(4, "depth").unwrap().unwrap(), b("DEPTH4"));
    assert_eq!(ds.verify().unwrap(), 5);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn layer_append_resume() {
    let root = root("resume");
    build_base(&root, 6);

    // first pass: only even sample_ids
    let mut w = LayerWriter::create(
        &root,
        "emb",
        BTreeMap::from([("emb".to_string(), Modality::tensor("npy", "npy"))]),
    )
    .unwrap();
    for sid in (0..6u64).step_by(2) {
        let mut blobs = BTreeMap::new();
        blobs.insert("emb".to_string(), b(&format!("E{sid}")));
        w.add(sid, &blobs, &BTreeMap::new()).unwrap();
    }
    w.close().unwrap();

    // resume: re-create sees existing ids; add the missing odd ones
    let mut w2 = LayerWriter::create(
        &root,
        "emb",
        BTreeMap::from([("emb".to_string(), Modality::tensor("npy", "npy"))]),
    )
    .unwrap();
    let done: std::collections::HashSet<u64> = w2.existing_ids().into_iter().collect();
    assert_eq!(done, [0, 2, 4].into_iter().collect());
    for sid in 0..6u64 {
        if !done.contains(&sid) {
            let mut blobs = BTreeMap::new();
            blobs.insert("emb".to_string(), b(&format!("E{sid}")));
            w2.add(sid, &blobs, &BTreeMap::new()).unwrap();
        }
    }
    w2.close().unwrap();

    let ds = Dataset::open(&root).unwrap();
    for sid in 0..6u64 {
        assert_eq!(ds.read_blob(sid as usize, "emb").unwrap().unwrap(), b(&format!("E{sid}")));
    }
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn partitioned_map_then_commit() {
    // Simulate a distributed map: 3 workers each enrich a disjoint sample range
    // into their OWN shards/fragment, then a single commit merges + registers.
    let root = root("partition");
    build_base(&root, 9);
    let mods = || BTreeMap::from([("feat".to_string(), Modality::tensor("npy", "npy"))]);

    let nparts = 3u32;
    for part in 0..nparts {
        let mut w = LayerWriter::create_partition(&root, "feat", mods(), part).unwrap();
        // worker `part` handles sample_ids where sid % nparts == part
        for sid in (0..9u64).filter(|s| (*s % nparts as u64) as u32 == part) {
            let mut blobs = BTreeMap::new();
            blobs.insert("feat".to_string(), b(&format!("F{sid}")));
            w.add(sid, &blobs, &BTreeMap::new()).unwrap();
        }
        w.close().unwrap();
    }
    // before commit: layer is NOT yet registered (workers touched no shared state)
    assert!(Dataset::open(&root).unwrap().all_modalities().iter().all(|m| m != "feat"));

    // commit merges all fragments into one layer + registers atomically
    let n = LayerWriter::commit(&root, "feat", mods()).unwrap();
    assert_eq!(n, 9);

    let ds = Dataset::open(&root).unwrap();
    assert!(ds.all_modalities().iter().any(|m| m == "feat"));
    for sid in 0..9u64 {
        assert_eq!(ds.read_blob(sid as usize, "feat").unwrap().unwrap(), b(&format!("F{sid}")));
    }
    assert_eq!(ds.verify().unwrap(), 9);
    // fragments were consumed
    let leftover: Vec<_> = std::fs::read_dir(root.join("index")).unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".part-"))
        .collect();
    assert!(leftover.is_empty(), "part fragments should be consumed by commit");

    // commit is idempotent (re-run with no new parts keeps the same rows)
    let n2 = LayerWriter::commit(&root, "feat", mods()).unwrap();
    assert_eq!(n2, 9);

    // a later partitioned pass can add a NEW sparse sample and re-commit (append)
    let mut w = LayerWriter::create_partition(&root, "feat2", BTreeMap::from(
        [("flag".to_string(), Modality::tensor("npy", "npy"))]), 0).unwrap();
    w.add(4, &BTreeMap::from([("flag".to_string(), b("Y"))]), &BTreeMap::new()).unwrap();
    w.close().unwrap();
    LayerWriter::commit(&root, "feat2",
        BTreeMap::from([("flag".to_string(), Modality::tensor("npy", "npy"))])).unwrap();
    let ds = Dataset::open(&root).unwrap();
    assert_eq!(ds.read_blob(4, "flag").unwrap().unwrap(), b("Y"));
    assert!(ds.read_blob(5, "flag").unwrap().is_none());     // sparse: only sid 4
    std::fs::remove_dir_all(&root).ok();
}
