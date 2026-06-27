//! Upload a local directory to a gs:// prefix (object PUTs). For staging
//! benchmark datasets on GCS. Writes objects — use a dedicated prefix.
//!
//! Run: cargo run --release -p ferroload-core --example gcs_put_dir --features gcp \
//!        -- <local_dir> gs://bucket/prefix/

#[cfg(feature = "remote")]
fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
}

#[cfg(feature = "remote")]
fn main() {
    use ferroload_io::Storage;
    let local = std::env::args().nth(1).expect("local dir");
    let dest = std::env::args().nth(2).expect("gs://bucket/prefix/");
    let root = std::path::PathBuf::from(&local);

    let (storage, prefix) = Storage::from_url(&dest).expect("from_url");
    let mut files = Vec::new();
    walk(&root, &mut files);
    files.sort();
    println!("uploading {} files from {} -> {}", files.len(), local, dest);

    let t = std::time::Instant::now();
    let mut total = 0u64;
    ferroload_io::runtime().block_on(async {
        for f in &files {
            let rel = f.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
            let key = if prefix.is_empty() {
                rel.clone()
            } else {
                format!("{}/{}", prefix.trim_end_matches('/'), rel)
            };
            let bytes = std::fs::read(f).expect("read");
            total += bytes.len() as u64;
            // large objects (e.g. a 187 MB data shard) need multipart, else a
            // single PUT times out on a slow link.
            if bytes.len() > 8 * 1024 * 1024 {
                storage.put_chunked(&key, &bytes, 8 * 1024 * 1024).await.expect("put_chunked");
            } else {
                storage.put(&key, bytes).await.expect("put");
            }
        }
    });
    println!("uploaded {:.1} MB in {:?} ({:.1} MB/s)",
        total as f64 / 1e6, t.elapsed(), total as f64 / 1e6 / t.elapsed().as_secs_f64());
}

#[cfg(not(feature = "remote"))]
fn main() {
    eprintln!("build with --features gcp");
}
