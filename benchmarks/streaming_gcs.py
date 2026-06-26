"""Stream FFHQ-256 from GCS and measure throughput: Ferroload vs WebDataset.
Both decode JPEG -> 224. Over a real network this is dominated by I/O + format
efficiency. Also streams the production laion-pop-ferro (Ferroload).

Run with GOOGLE_APPLICATION_CREDENTIALS set.
"""
import io
import json
import os
import time

import numpy as np
from PIL import Image

FERRO = "gs://ferroload-datasets/bench/ffhq256-ferro/"
WDS = "gs://ferroload-datasets/bench/ffhq256-wds/"
LAION = "gs://ferroload-datasets/laion-pop-ferro/"
TARGET = 224
BS = 64
N = 3000


def resize_arr(im):
    im = im.convert("RGB")
    if im.size != (TARGET, TARGET):
        im = im.resize((TARGET, TARGET), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def time_stream(name, batch_iter, n_target=N):
    t0 = time.perf_counter()
    it = iter(batch_iter)
    first = next(it)
    first_s = time.perf_counter() - t0
    n = first.shape[0] if hasattr(first, "shape") else len(first)
    t = time.perf_counter()
    while n < n_target:
        try:
            b = next(it)
        except StopIteration:
            break
        n += b.shape[0] if hasattr(b, "shape") else len(b)
    dt = time.perf_counter() - t
    sps = n / dt if dt > 0 else 0
    print(f"{name:28} first={first_s:6.2f}s  {sps:8.1f} samp/s  ({n} samples / {dt:.2f}s)")
    return dict(name=name, first_s=round(first_s, 3), samples_per_s=round(sps, 1), n=n, seconds=round(dt, 3))


def ferro_stream(url, target):
    import ferroload
    cache = f"/tmp/ferro-stream-bench-{abs(hash(url))%9999}"
    import shutil; shutil.rmtree(cache, ignore_errors=True)
    dl = ferroload.make_loader(url, batch_size=BS, images=["image"],
                               resize=(target, target) if target else None, out="numpy",
                               streaming=True, shuffle=False, block_size=512, cache_dir=cache)
    for batch in dl:
        yield batch["image"]


def wds_stream(url):
    import webdataset as wds
    import webdataset.gopen as gop
    import fsspec

    # wds defaults gs:// to `gsutil cat` (absent here) — route through gcsfs/fsspec.
    def gopen_fsspec(u, mode="rb", bufsize=8192, **kw):
        return fsspec.open(u, mode).open()
    for d in (getattr(gop, "gopen_schemes", {}), getattr(wds, "gopen_schemes", {})):
        d["gs"] = gopen_fsspec

    shards = [f"{url}shard-{i:05d}.tar" for i in range(16)]

    def decode(sample):
        for k in ("jpg", "jpeg", "png"):
            if k in sample:
                return resize_arr(Image.open(io.BytesIO(sample[k])))
        raise KeyError
    ds = (wds.WebDataset(shards, shardshuffle=False, empty_check=False)
          .map(decode).batched(BS, collation_fn=np.stack, partial=True))
    return wds.WebLoader(ds, batch_size=None, num_workers=0)


def main():
    results = []
    results.append(time_stream("ferroload  (FFHQ gcs)", ferro_stream(FERRO, TARGET)))
    try:
        results.append(time_stream("webdataset (FFHQ gcs)", wds_stream(WDS)))
    except Exception as e:
        print(f"webdataset FAILED: {type(e).__name__}: {str(e)[:160]}")
    results.append(time_stream("ferroload  (laion-pop)", ferro_stream(LAION, TARGET)))
    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/streaming_gcs.json", "w"), indent=2)


if __name__ == "__main__":
    main()
