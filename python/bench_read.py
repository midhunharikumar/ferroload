#!/usr/bin/env python3
"""Benchmark local sequential read throughput: HuggingFace `datasets` (Arrow,
on-disk) vs the Ferroload format — using the *same image bytes* for both so the
comparison is apples-to-apples.

Two regimes:
  * raw    : return the encoded image bytes + label (no pixel decode)
  * decode : also PNG-decode to pixels (PIL) — the real training cost

Usage:
    python bench_read.py [--dataset ylecun/mnist] [--n 3000] [--repeats 3]
"""
import argparse
import io
import time

import ferroload
from datasets import Image, load_dataset


def timed(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        n, nbytes = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, n, nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ylecun/mnist")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/tmp/bench_ferro")
    args = ap.parse_args()

    # --- load HF dataset locally (downloads once, then on-disk Arrow) ---
    print(f"loading {args.dataset} train[:{args.n}] ...")
    hf = load_dataset(args.dataset, split=f"train[:{args.n}]")
    img_col = next((k for k, v in hf.features.items() if v.__class__.__name__ == "Image"), None)
    label_col = next((k for k in hf.column_names if k != img_col), None)
    # raw-bytes view of the SAME data (no decode)
    hf_raw = hf.cast_column(img_col, Image(decode=False))

    # --- build a ferroload dataset from the identical encoded bytes ---
    w = ferroload.Writer(args.out, "bench")
    w.declare("image", "img", "tensor", "image")
    total_bytes = 0
    for ex in hf_raw:
        b = ex[img_col]["bytes"]
        total_bytes += len(b)
        meta = {label_col: ex[label_col]} if label_col else {}
        w.add(f"s{_next():06d}", {"image": b}, meta)
    w.close()
    fd = ferroload.Dataset.open(args.out)
    n = len(fd)

    # capability check: a stale extension lacks read()/read_many()
    missing = [m for m in ("read", "read_batch", "decode_many") if not hasattr(fd, m)]
    if missing:
        raise SystemExit(
            f"\nLoaded `ferroload` {getattr(ferroload, '__version__', '?')} is STALE "
            f"(missing {missing}).\nRebuild it:\n"
            f"    cd crates/ferroload-py && maturin develop --release\n"
            f"(need >= 0.3.0)"
        )
    print(f"ferroload {ferroload.__version__}; prepared {n} samples; "
          f"{total_bytes/1e6:.2f} MB of image bytes\n")

    # ---------- raw-bytes read ----------
    def hf_raw_read():
        nb = 0
        for ex in hf_raw:
            nb += len(ex[img_col]["bytes"])
            _ = ex[label_col]
        return n, nb

    def ferro_raw_get():
        # full dict path (bytes + present flags + meta) — most Python overhead
        nb = 0
        for i in range(n):
            s = fd.get(i)
            nb += len(s["image"])
            _ = s["meta"]
        return n, nb

    def ferro_raw_read1():
        # minimal per-call path: just the modality bytes
        nb = 0
        for i in range(n):
            nb += len(fd.read(i, "image"))
        return n, nb

    def ferro_raw_batched():
        # contiguous batched read: one buffer + (offset,len) spans per minibatch
        B, nb = 256, 0
        for s in range(0, n, B):
            _buf, spans = fd.read_batch(list(range(s, min(s + B, n))), "image")
            nb += sum(l for _o, l in spans)
        return n, nb

    # ---------- decode-inclusive read ----------
    hf_dec = hf  # decode=True by default -> yields PIL images

    def hf_decode_read():
        nb = 0
        for ex in hf_dec:
            im = ex[img_col]
            im.load()
            nb += im.size[0] * im.size[1]
        return n, nb

    def ferro_decode_read():
        from PIL import Image as PImage
        nb = 0
        for i in range(n):
            b = fd.read(i, "image")
            im = PImage.open(io.BytesIO(b))
            im.load()
            nb += im.size[0] * im.size[1]
        return n, nb

    def ferro_decode_rust():
        # read + decode in Rust, parallel across cores, GIL released.
        # decode_many returns zero-copy NumPy uint8 arrays [H, W, C].
        B, nb = 256, 0
        for s in range(0, n, B):
            for arr in fd.decode_many(list(range(s, min(s + B, n))), "image"):
                nb += arr.shape[0] * arr.shape[1]
        return n, nb

    print(f"{'benchmark':<26}{'time (s)':>10}{'samples/s':>14}{'MB/s':>10}")
    print("-" * 60)
    for name, fn, mb in [
        ("HF  raw bytes", hf_raw_read, total_bytes / 1e6),
        ("ferro raw get()", ferro_raw_get, total_bytes / 1e6),
        ("ferro raw read()", ferro_raw_read1, total_bytes / 1e6),
        ("ferro raw read_many", ferro_raw_batched, total_bytes / 1e6),
        ("HF  decode (PIL)", hf_decode_read, total_bytes / 1e6),
        ("ferro decode (PIL)", ferro_decode_read, total_bytes / 1e6),
        ("ferro decode (Rust//)", ferro_decode_rust, total_bytes / 1e6),
    ]:
        dt, cnt, _ = timed(fn, args.repeats)
        print(f"{name:<26}{dt:>10.4f}{cnt/dt:>14.0f}{mb/dt:>10.1f}")


_counter = -1


def _next():
    global _counter
    _counter += 1
    return _counter


if __name__ == "__main__":
    main()
