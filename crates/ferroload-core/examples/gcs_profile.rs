//! Profile a remote ferroload dataset: cold/warm open, first read, random-read
//! throughput, and a subset scan (ranged columnar reader). Read-only.
//!
//! Run: cargo run --release -p ferroload-core --example gcs_profile --features gcp \
//!        -- gs://ferroload-datasets/laion-pop-ferro/ ["WHERE predicate"]

#[cfg(feature = "remote")]
fn trim(v: &serde_json::Value) -> String {
    let s = v.to_string();
    let t: String = s.chars().take(70).collect();
    if t.len() < s.len() { format!("{t}… ({} chars)", s.chars().count()) } else { t }
}

#[cfg(feature = "remote")]
fn main() {
    use ferroload_core::dataset::Dataset;
    use ferroload_io::Storage;
    use std::time::Instant;

    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "gs://ferroload-datasets/laion-pop-ferro/".to_string());
    let where_sql = std::env::args().nth(2);

    let cache = std::env::temp_dir().join(format!("ferro-profile-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache); // start COLD

    // index shard object size (vs the old 1.15 GB monolithic index.json)
    if let Ok((storage, prefix)) = Storage::from_url(&url) {
        let key = if prefix.is_empty() { "index/part-00000.parquet".into() } else { format!("{}/index/part-00000.parquet", prefix.trim_end_matches('/')) };
        if let Ok(sz) = ferroload_io::runtime().block_on(storage.size(&key)) {
            println!("index shard object   : {key}  = {sz} bytes ({:.1} KB)", sz as f64 / 1024.0);
        }
    }

    // ---- cold open: manifest + directory only ----
    let t = Instant::now();
    let ds = Dataset::open_url(&url, &cache).expect("open_url");
    println!("\nopen (COLD cache)    : {:?}", t.elapsed());
    println!("  len               : {}", ds.len());
    println!("  index shards loaded: {} (expect 0 — open is O(manifest))", ds.index_shard_loads());

    // schema peek
    let r0 = ds.row(0).expect("row 0");
    println!("\nrow 0: sample_id={} basename={:?}", r0.sample_id, r0.basename);
    println!("  modalities/offsets: {:?}", r0.offsets.keys().collect::<Vec<_>>());
    println!("  meta columns      : {:?}", r0.meta.keys().collect::<Vec<_>>());
    for (k, v) in r0.meta.iter().take(10) {
        println!("    {k:<18} = {}", trim(v));
    }
    println!("  index shards loaded after row(0): {}", ds.index_shard_loads());

    // ---- first blob read (one index-shard load + one data range read) ----
    let mods = ds.all_modalities();
    let m = mods.first().cloned().unwrap_or_else(|| "image".into());
    let t = Instant::now();
    let b = ds.read_blob(0, &m).expect("read_blob");
    println!("\nread_blob(0,{m})    : {:?}  -> {} bytes", t.elapsed(), b.as_ref().map(|x| x.len()).unwrap_or(0));

    // ---- per-call random reads (one ranged GET each — latency bound) ----
    let len = ds.len().max(1);
    let n = 50usize;
    let idxs: Vec<usize> = (0..n).map(|i| (i * 7919) % len).collect();
    let t = Instant::now();
    let mut bytes = 0usize;
    for &idx in &idxs {
        if let Some(buf) = ds.read_blob(idx, &m).unwrap() {
            bytes += buf.len();
        }
    }
    let el = t.elapsed();
    println!("\n{n} per-call read_blob : {:?}  ({:?}/read, {:.1} MB)", el, el / n as u32, bytes as f64 / 1e6);

    // ---- batched read of the SAME indices (coalesced get_ranges per shard) ----
    let t = Instant::now();
    let (buf, spans) = ds.read_blobs_contig(&idxs, &m).expect("read_blobs_contig");
    let el = t.elapsed();
    let got: usize = spans.iter().map(|(_, l)| *l).sum();
    println!("{n} batched read_batch : {:?}  ({:?}/read, {:.1} MB)  <- coalesced", el, el / n as u32, got as f64 / 1e6);
    let _ = buf;

    // ---- subset scan (ranged columnar reader: projection + row-group pruning) ----
    let pred = where_sql.unwrap_or_else(|| format!("{m}_present"));
    let t = Instant::now();
    match ds.subset(&pred) {
        Ok(ids) => println!("\nsubset({pred:?})\n  time    : {:?}\n  matched : {} / {}", t.elapsed(), ids.len(), ds.len()),
        Err(e) => println!("\nsubset({pred:?}) ERROR: {e}"),
    }

    // ---- warm re-open (cache populated on disk) ----
    let t = Instant::now();
    let ds2 = Dataset::open_url(&url, &cache).expect("reopen");
    println!("\nopen (WARM cache)    : {:?}  len={}", t.elapsed(), ds2.len());

    let _ = std::fs::remove_dir_all(&cache);
}

#[cfg(not(feature = "remote"))]
fn main() {
    eprintln!("build with --features gcp (or cloud)");
}
