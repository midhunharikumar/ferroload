//! List one directory level under a gs:// prefix. Read-only.
//! Run: cargo run -p ferroload-core --example gcs_ls --features gcp -- gs://bucket/prefix/

#[cfg(feature = "remote")]
fn main() {
    use ferroload_io::Storage;

    let url = std::env::args().nth(1).unwrap_or_else(|| "gs://ferroload-datasets/".to_string());
    let (storage, prefix) = Storage::from_url(&url).expect("from_url");

    let (prefixes, objs) = ferroload_io::runtime()
        .block_on(storage.list_dir(&prefix))
        .expect("list");

    println!("== folders ==");
    for p in &prefixes {
        println!("  {p}/");
    }
    println!("== objects (first 30) ==");
    let mut total: u64 = 0;
    for (i, (loc, size)) in objs.iter().enumerate() {
        total += size;
        if i < 30 {
            println!("  {size:>12}  {loc}");
        }
    }
    println!("  ... {} objects, {:.1} MB at this level", objs.len(), total as f64 / 1e6);
}

#[cfg(not(feature = "remote"))]
fn main() {
    eprintln!("build with --features gcp");
}
