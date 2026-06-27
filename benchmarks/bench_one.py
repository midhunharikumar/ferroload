"""Run ONE benchmark config in its own process (clean peak-RSS + isolation).

  python bench_one.py <name> <loader> <num_workers> <batch_size> <max_samples>
  loader in {hf_arrow, webdataset, ferro_native}

Prints a single line:  RESULT {json}
"""
import multiprocessing as mp

try:
    mp.set_start_method("fork", force=True)  # fork-safe closures; avoids spawn pickling
except RuntimeError:
    pass

import json
import os
import sys

import common
import formats
import loaders

DATA = os.path.join(os.path.dirname(__file__), "bench_data")
FMT = {"hf_arrow": "hf_arrow", "webdataset": "webdataset", "ferro_native": "ferroload",
       "ferro_dl": "ferroload"}


def main():
    name, loader, nw, bs, maxs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    base = os.path.join(DATA, name)
    meta = json.load(open(os.path.join(base, "meta.json")))
    target = meta["target"]

    if loader == "hf_arrow":
        it = loaders.hf_arrow_loader(os.path.join(base, "hf_arrow"), target, bs, nw)
    elif loader == "webdataset":
        it = loaders.webdataset_loader(formats.wds_shards(os.path.join(base, "wds")), target, bs, nw)
    elif loader == "ferro_native":
        it = loaders.ferroload_native_loader(os.path.join(base, "ferro"), target, bs)
    elif loader == "ferro_dl":
        it = loaders.ferroload_dl_loader(os.path.join(base, "ferro"), target, bs, nw)
    else:
        raise ValueError(loader)

    n, secs, first = common.run_loader(it, warmup_batches=3, max_samples=maxs,
                                       sample_count_fn=loaders.batch_n)
    res = common.RunResult(
        dataset=name, fmt=FMT[loader], loader=loader, num_workers=nw, batch_size=bs,
        target_px=target, n_samples=n, seconds=round(secs, 4),
        samples_per_s=round(n / secs, 1) if secs > 0 else 0.0,
        first_batch_s=round(first, 4), peak_rss_mb=round(common.peak_rss_bytes() / 1e6, 1),
    )
    print("RESULT " + json.dumps(res.row()))


if __name__ == "__main__":
    main()
