# Ferroload

**A pure-Rust multimodal dataset format + dataloader for PyTorch.**

Ferroload stores images, video, audio, tensors, and rich metadata in a
self-contained, shardable on-disk format, and serves them to training loops with
parallel decode in Rust (the GIL released), SQL-style subsetting, and a one-call
loader that drops straight into PyTorch.

```python
from ferroload import make_loader

dl = make_loader("/data/ds", batch_size=64,
                 columns=["image", "video", "label"],   # kinds resolved from the manifest
                 resize=(224, 224), out="torch")
for epoch in range(epochs):
    dl.set_epoch(epoch)                                  # reshuffle (DDP-aware)
    for batch in dl:
        train_step(batch)                                # batch["image"], batch["video"], batch["label"]
```

## Why Ferroload

- **Self-contained format.** A dataset is a directory: sharded `.tar` blobs +
  an index + a manifest. Copy it, mount it, version it — no external services.
- **Fast reads.** Decode of images/video runs in parallel across cores in Rust
  with the GIL released, and batched reads are zero-copy into NumPy.
- **SQL subsetting.** Filter by metadata (`ds.subset("split = 'train' AND duration < 16")`)
  to get a lightweight, index-remapped view — no data copied.
- **Additive enrichment (`map`).** Run a function over the dataset and store the
  results as a new layer joined on `sample_id`. The base data is never rewritten;
  the pass is idempotent and resumable. Functions bind to inputs **positionally**
  and run **once per sample** by default, so they're generic and reusable.
- **Deterministic, DDP-aware sampling.** A `DistributedSampler` drop-in backed by
  a Rust sampler, with block-shuffle and exact resume.
- **One-call loader.** `make_loader` bundles open + dataset + sampler + background
  prefetch; or compose the pieces yourself.

## Where to next

- [Installation](installation.md) — build the extension (and the optional video feature).
- [Python quickstart](python/quickstart.md) — import a HuggingFace dataset and feed a DataLoader.
- [Python API reference](python/api.md) — `Writer`, `Dataset`, `map`, the loader, the sampler.
- [Rust core usage](rust/usage.md) and a [worked example](rust/examples.md).
- [Benchmarks](benchmarks.md) — local read throughput vs HuggingFace `datasets`.
