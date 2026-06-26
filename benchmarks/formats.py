"""Build the three on-disk formats from one in-memory list of records.

A record = {"key": str, "img": bytes (encoded), "ext": "jpg"|"png", "meta": dict}.
The SAME encoded `img` bytes go into all three formats, so decode cost is identical.
"""
import json
import os
import shutil

import datasets as hfds
import webdataset as wds


def _fresh(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def build_hf_arrow(records, out_dir, meta_features: dict):
    """HF `datasets` Arrow (the loader diffusers training uses via load_dataset).
    The Image() feature stores the encoded bytes in Arrow; __getitem__ decodes to PIL."""
    _fresh(out_dir)
    feats = hfds.Features({"image": hfds.Image(), **meta_features})

    def gen():
        for r in records:
            row = {"image": {"bytes": r["img"], "path": None}}
            row.update(r["meta"])
            yield row

    ds = hfds.Dataset.from_generator(gen, features=feats)
    ds.save_to_disk(out_dir)
    return out_dir


def build_webdataset(records, out_dir, maxcount=10000):
    """WebDataset tar shards: key.<ext> = image bytes, key.json = meta."""
    _fresh(out_dir)
    pattern = os.path.join(out_dir, "shard-%05d.tar")
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for r in records:
            sample = {"__key__": r["key"], r["ext"]: r["img"],
                      "json": json.dumps(r["meta"]).encode()}
            sink.write(sample)
    return out_dir


def build_ferroload(records, out_dir, ext, name="bench"):
    """Ferroload dataset: image modality + inline meta (queryable Parquet index)."""
    import ferroload
    _fresh(out_dir)
    w = ferroload.Writer(out_dir, name)
    w.declare("image", ext, "tensor", "image")
    for r in records:
        w.add(r["key"], {"image": r["img"]}, r["meta"])
    w.close()
    return out_dir


def wds_shards(out_dir):
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".tar")
    )
