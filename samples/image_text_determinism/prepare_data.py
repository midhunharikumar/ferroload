"""Pack an image+text HuggingFace dataset into the Ferroload format.

Streams the HF dataset (only `limit` rows are pulled regardless of total size),
auto-detects the image column (PIL values) and a caption column (str or
list-of-str, preferring caption/captions/text/sentence names), re-encodes
images as JPEG shards, and stores the caption as queryable index metadata.

Default source is Flickr30k — 31k photos with 5 crowd-sourced captions each.

Usage (local):
    python prepare_data.py /data/flickr30k --limit 20000
"""
from __future__ import annotations

import io
import os

CAPTION_NAMES = ("caption", "captions", "text", "txt", "sentence", "sentences")


def _is_pil(v):
    return hasattr(v, "save") and hasattr(v, "convert") and hasattr(v, "size")


def _pick_columns(example, image_col=None, caption_col=None):
    if image_col is None:
        image_col = next((k for k, v in example.items() if _is_pil(v)), None)
    if caption_col is None:
        def is_texty(v):
            return isinstance(v, str) or (isinstance(v, list) and v
                                          and isinstance(v[0], str))
        named = [k for k in example if k.lower() in CAPTION_NAMES and is_texty(example[k])]
        anytext = [k for k, v in example.items()
                   if k != image_col and not k.startswith("__") and is_texty(v)]
        caption_col = (named or anytext or [None])[0]
    if image_col is None or caption_col is None:
        raise ValueError(f"could not find image+caption columns in {list(example)}")
    return image_col, caption_col


def build_dataset(hf_id: str, out_root: str, *, limit: int = 20000,
                  split: str = "test", name: str | None = None,
                  image_col: str | None = None, caption_col: str | None = None,
                  jpeg_quality: int = 90) -> str:
    """Stream `hf_id` and write a Ferroload dataset at `out_root`. Idempotent:
    returns immediately if `out_root` already holds a manifest."""
    if os.path.exists(os.path.join(out_root, "manifest.json")):
        print(f"dataset already packed at {out_root}, skipping")
        return out_root

    import shutil

    import ferroload
    from datasets import load_dataset

    print(f"streaming {hf_id} split={split} limit={limit}", flush=True)
    ds = load_dataset(hf_id, name=name, split=split, streaming=True)

    tmp = out_root + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)  # leftover from an interrupted pack
    w = ferroload.Writer(tmp, os.path.basename(os.path.abspath(out_root)))
    w.declare("image", "jpg", "tensor", "image")

    n = 0
    for ex in ds:
        if n >= limit:
            break
        if n == 0:
            image_col, caption_col = _pick_columns(ex, image_col, caption_col)
            print(f"image column: {image_col!r}, caption column: {caption_col!r}")
        img, cap = ex.get(image_col), ex.get(caption_col)
        if isinstance(cap, list):
            cap = cap[0] if cap else None
        if img is None or not cap:
            continue
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
        w.add(f"sample{n:07d}", {"image": buf.getvalue()}, {"caption": str(cap)})
        n += 1
        if n % 1000 == 0:
            print(f"  packed {n}/{limit}", flush=True)
    w.close()
    os.rename(tmp, out_root)

    rd = ferroload.Dataset.open(out_root)
    print(f"packed {len(rd)} samples, {rd.num_shards()} shard(s) -> {out_root}")
    return out_root


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output dataset root directory")
    ap.add_argument("--hf-dataset", default="lmms-lab/flickr30k")
    ap.add_argument("--split", default="test")  # flickr30k ships one split
    ap.add_argument("--name", default=None, help="HF config name")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--image-col", default=None)
    ap.add_argument("--caption-col", default=None)
    a = ap.parse_args()
    build_dataset(a.hf_dataset, a.out, limit=a.limit, split=a.split, name=a.name,
                  image_col=a.image_col, caption_col=a.caption_col)
