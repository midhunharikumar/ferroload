#!/usr/bin/env python3
"""Import a (small slice of a) HuggingFace dataset into the Ferroload format
using the `ferroload` Rust extension.

It streams the dataset (so it only pulls `--limit` rows, regardless of total
size), auto-detects image columns at runtime, stores images as PNG members and
everything else as scalar metadata, then reads the result back and verifies it.

Usage:
    python import_hf.py <hf_dataset> <out_root> [--split train] [--limit 50]
                        [--name CONFIG] [--image-col COL]

Examples:
    python import_hf.py cifar10        /tmp/ds_cifar  --limit 32
    python import_hf.py mnist          /tmp/ds_mnist  --limit 32
    python import_hf.py rotten_tomatoes /tmp/ds_rt    --limit 50   # text-only
"""
import argparse
import io
import os

import ferroload  # the Rust extension (build it first; see python/README_HF.md)
from datasets import load_dataset


def is_pil_image(v):
    return hasattr(v, "save") and hasattr(v, "convert") and hasattr(v, "size")


def to_png_bytes(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="HF dataset id, e.g. cifar10")
    ap.add_argument("out", help="output dataset root directory")
    ap.add_argument("--split", default="train")
    ap.add_argument("--name", default=None, help="dataset config name")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--image-col", default=None, help="force a column to be the image")
    args = ap.parse_args()

    print(f"loading (streaming) {args.dataset} split={args.split} limit={args.limit}")
    ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)

    w = ferroload.Writer(args.out, os.path.basename(os.path.abspath(args.out)))
    w.declare("image", "png", "tensor", "image")  # harmless if unused

    n = 0
    for ex in ds:
        if n >= args.limit:
            break
        blobs, meta = {}, {}
        for k, v in ex.items():
            if (args.image_col and k == args.image_col) or (args.image_col is None and is_pil_image(v)):
                if v is not None:
                    blobs["image"] = to_png_bytes(v)
            elif isinstance(v, (bool, int, float, str)):
                meta[k] = v
            elif isinstance(v, (bytes, bytearray)):
                blobs[k] = bytes(v)
            else:
                meta[k] = str(v)  # lists/dicts -> stringified metadata
        w.add(f"sample{n:06d}", blobs, meta)
        n += 1
    w.close()
    print(f"packed {n} samples -> {args.out}")

    # read back + verify
    rd = ferroload.Dataset.open(args.out)
    print(f"reopened: {len(rd)} samples, {rd.num_shards()} shard(s)")
    s0 = rd.get(0)
    has_img = "image" in s0
    print(f"sample0: image_bytes={len(s0['image']) if has_img else 0}, meta={s0['meta']}")
    print(f"verify: {rd.verify()} samples OK")


if __name__ == "__main__":
    main()
