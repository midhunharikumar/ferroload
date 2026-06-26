"""Acquire a dataset and build all three formats once; record sizes. Reused by
every benchmark config so they all read identical encoded bytes.

  python build.py <cifar10|stanford_cars|ffhq256> [limit]
"""
import json
import os
import sys

import acquire
import common
import formats

DATA = os.path.join(os.path.dirname(__file__), "bench_data")


def main():
    name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = acquire.acquire(name, limit)
    recs = cfg["records"]
    base = os.path.join(DATA, name)
    hf_dir = os.path.join(base, "hf_arrow")
    wds_dir = os.path.join(base, "wds")
    fr_dir = os.path.join(base, "ferro")

    print(f"[{name}] {len(recs)} records; building formats ...", flush=True)
    formats.build_hf_arrow(recs, hf_dir, cfg["meta_features"])
    print("  hf_arrow done", flush=True)
    formats.build_webdataset(recs, wds_dir, cfg["wds_maxcount"])
    print("  webdataset done", flush=True)
    formats.build_ferroload(recs, fr_dir, cfg["ext"], name)
    print("  ferroload done", flush=True)

    enc_bytes = sum(len(r["img"]) for r in recs)
    sizes = {
        "encoded_raw": enc_bytes,
        "hf_arrow": common.dir_size_bytes(hf_dir),
        "webdataset": common.dir_size_bytes(wds_dir),
        "ferroload": common.dir_size_bytes(fr_dir),
    }
    meta = dict(name=name, n=len(recs), ext=cfg["ext"], target=cfg["target"],
                wds_maxcount=cfg["wds_maxcount"], sizes=sizes)
    json.dump(meta, open(os.path.join(base, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
