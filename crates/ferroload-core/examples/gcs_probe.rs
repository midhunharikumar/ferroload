//! Probe a remote ferroload dataset's manifest to learn its on-disk format
//! (sharded-parquet vs old monolithic) before profiling. Read-only.
//!
//! Run: cargo run -p ferroload-core --example gcs_probe --features gcp -- <url>

#[cfg(feature = "remote")]
fn main() {
    use ferroload_io::Storage;
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "gs://ferroload-datasets/laion-pop-ferro/".to_string());

    let (storage, prefix) = Storage::from_url(&url).expect("from_url");
    let key = if prefix.is_empty() {
        "manifest.json".to_string()
    } else {
        format!("{}/manifest.json", prefix.trim_end_matches('/'))
    };

    let t = std::time::Instant::now();
    let bytes = match storage.get_blocking(&key) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAILED to fetch {key}: {e}");
            std::process::exit(2);
        }
    };
    eprintln!("fetched manifest.json: {} bytes in {:?}", bytes.len(), t.elapsed());

    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("parse manifest");
    let shards = v.get("index_shards").and_then(|s| s.as_array()).map(|a| a.len()).unwrap_or(0);
    println!("name              = {:?}", v.get("name"));
    println!("format_version    = {:?}", v.get("format_version"));
    println!("min_reader_version= {:?}", v.get("min_reader_version"));
    println!("index             = {:?}", v.get("index"));
    println!("index_shards      = {shards} entries");
    println!("shards            = {:?}", v.get("shards"));
    println!("modalities keys   = {:?}", v.get("modalities").and_then(|m| m.as_object()).map(|o| o.keys().collect::<Vec<_>>()));
    println!("layers            = {:?}", v.get("layers").and_then(|l| l.as_array()).map(|a| a.len()));
    println!("index_shards[..] = {}", serde_json::to_string_pretty(v.get("index_shards").unwrap_or(&serde_json::Value::Null)).unwrap());
    println!(
        "\nFORMAT = {}",
        if shards > 0 { "NEW (sharded index) -> readable" } else { "OLD (monolithic index.path) -> NOT readable by new reader" }
    );
}

#[cfg(not(feature = "remote"))]
fn main() {
    eprintln!("build with --features gcp (or cloud)");
}
