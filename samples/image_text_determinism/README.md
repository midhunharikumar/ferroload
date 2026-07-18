# Image+text training determinism

Trains a small CLIP-style contrastive model (image tower: 4-block CNN; text
tower: hashed bag-of-words MLP; symmetric InfoNCE) on **Flickr30k** — 31k
photos, each with 5 crowd-sourced captions — packed into the Ferroload format,
and shows that **independent training runs produce bitwise-identical loss
curves**.

Ferroload's contribution to that claim is the data path: `make_loader(seed=…)`
uses a deterministic sampler (`set_epoch` reshuffles it identically every run,
DDP-aware), and the parallel in-Rust decode returns batches in a fixed order.
The rest is standard torch hygiene, handled in `common.py`: seeded RNGs,
`torch.use_deterministic_algorithms(True)`, a crc32 (not salted-`hash`) text
featurizer, and a model built only from ops with deterministic CUDA kernels.

![Loss determinism](loss_determinism.png)

Top panel: per-step loss for every run, overlaid (different linewidths /
linestyles, so coincident curves stay visible). Bottom panel: the max absolute
loss difference across all run pairs at each step — a flat zero line.

## Run it on Modal (recommended)

The Modal harness is also the test: it fans the runs out to *separate*
containers on the same GPU model (T4), asserts the curves are bit-identical,
and exits non-zero if they diverge.

```bash
pip install modal && modal setup            # once
modal run samples/image_text_determinism/modal_app.py                    # 3 runs, 200 steps, 20k pairs
modal run samples/image_text_determinism/modal_app.py --runs 5 --steps 300 --limit 31000
modal run samples/image_text_determinism/modal_app.py --limit 2000 --steps 60   # quick smoke (~5 min)
```

The first invocation streams Flickr30k from HuggingFace and packs it once into
a Modal Volume (`ferroload-samples`); later runs reuse it. The plot and raw
loss curves land next to this README as `loss_determinism.png` / `.json`.

## Run it locally

Needs `ferroload torch datasets pillow matplotlib` installed:

```bash
python samples/image_text_determinism/train_local.py --limit 2000 --steps 60
```

## Files

- `common.py` — determinism setup, mini-CLIP model, training loop, plot
- `prepare_data.py` — stream an HF image+text dataset into the Ferroload format
  (auto-detects image/caption columns; also a standalone CLI)
- `modal_app.py` — Modal app: pack → N parallel GPU runs → assert + plot
- `train_local.py` — same demo on the local machine, no Modal

## Notes

- Bitwise reproducibility is defined *per hardware + library configuration*:
  identical GPU model, torch, CUDA, and ferroload versions. Curves from a T4
  will not match curves from an A100 bit-for-bit — but N runs on T4s do.
- Swap in any HF dataset with an image and a text column via
  `--hf-dataset/--split/--limit`; columns are auto-detected
  (`prepare_data.py --image-col/--caption-col` to override).
