# Data-loading benchmarks: Ferroload vs WebDataset vs HF `datasets` (Arrow)

3-way data-loading comparison on CIFAR-10, Stanford-Cars (BLIP), and FFHQ-256,
plus a GCS streaming benchmark. **Read [`REPORT.md`](REPORT.md) for results.**

## Setup
```bash
python -m venv /tmp/venv && source /tmp/venv/bin/activate
pip install torch torchvision datasets webdataset pyarrow pillow numpy gcsfs
# + a local `ferroload` build:  (cd ../crates/ferroload-py && maturin develop --release --features gcp)
```

## Run (local 3-way)
```bash
cd benchmarks
python build.py cifar10 50000        # acquire + build HF Arrow / WebDataset / Ferroload
python run_all.py cifar10 20000      # sweep loaders × workers -> results/cifar10.json
# repeat for: stanford_cars  (no limit, ~8.4k) and  ffhq256 10000
```

## Run (GCS streaming)
```bash
# stage one dataset to GCS in both formats (Ferroload + WebDataset):
cargo run --release -p ferroload-core --example gcs_put_dir --features gcp -- \
    bench_data/ffhq256/ferro gs://<bucket>/bench/ffhq256-ferro/
cargo run --release -p ferroload-core --example gcs_put_dir --features gcp -- \
    bench_data/ffhq256/wds   gs://<bucket>/bench/ffhq256-wds/
GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json \
    python streaming_gcs.py
```

## Design notes
- **Fairness:** all three formats are built from the *same encoded image bytes*
  (`formats.py`), and every loader decodes to the *same target* (`loaders.py`), so
  per-sample CPU work is identical. The decoder library differs by design (Rust vs
  PIL) and is isolated separately (REPORT §5).
- Each config runs in its **own process** (`bench_one.py`) for clean peak-RSS.
- The "diffusers" baseline = **HF `datasets` (Arrow)**, the loader diffusers
  training scripts use via `load_dataset`.
