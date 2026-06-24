# ferroload

**A pure-Rust multimodal dataset format + dataloader for PyTorch.**

Ferroload stores images, video, audio, tensors, and rich metadata in a
self-contained, shardable on-disk format and serves them to training loops with
parallel decode in Rust (the GIL released), SQL-style subsetting, and a one-call
loader that drops straight into PyTorch.

## Install

```bash
pip install ferroload
```

Optional extras:

```bash
pip install "ferroload[hf]"      # HuggingFace import tooling (datasets, pillow, hub)
pip install "ferroload[torch]"   # torch + numpy, for the DataLoader glue
```

In-Rust video decode is feature-gated (it needs system ffmpeg) and is **not** in
the published wheel — build from source for it:

```bash
maturin develop --release --features video
```

## Quickstart

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

Enrich a dataset with an additive, resumable layer — functions bind to inputs
positionally and run once per sample by default, so they're generic:

```python
import ferroload

def mean_color(img):                                     # img <- inputs=["image"]
    return img.mean(axis=(0, 1)).astype("float32")

ds = ferroload.Dataset.open("/data/ds")
ds = ds.map(mean_color, inputs=["image"],
            outputs={"emb": ferroload.Modality("npy")}, name="emb")
ds.read_array(0, "emb")
```

## Documentation

Full docs (Python API, quickstart, Rust core, benchmarks) are built with MkDocs
and published via GitHub Pages. See the project repository for links.

## License

Apache-2.0.
