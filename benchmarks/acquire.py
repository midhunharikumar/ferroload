"""Acquire datasets via HF streaming and materialize a list of records with
identical encoded image bytes for all three formats. Streaming avoids keeping
HF's Arrow cache on disk — we build our own HF-Arrow copy from the same bytes.

record = {"key", "img" (bytes), "ext", "meta"}
"""
import io

import datasets as hfds
from datasets import Value
from PIL import Image

hfds.disable_progress_bars()


def _png(im):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG", optimize=False)
    return buf.getvalue()


def _jpg(im, max_side=256, q=90):
    im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return buf.getvalue()


def _take(repo, split, limit):
    ds = hfds.load_dataset(repo, split=split, streaming=True)
    for i, s in enumerate(ds):
        if limit and i >= limit:
            break
        yield i, s


def acquire(name, limit=None):
    if name == "cifar10":
        recs = []
        for i, s in _take("cifar10", "train", limit or 50000):
            recs.append({"key": f"{i:06d}", "img": _png(s["img"]), "ext": "png",
                         "meta": {"label": int(s["label"])}})
        return dict(name="cifar10", records=recs, ext="png", target=None,
                    meta_features={"label": Value("int64")},
                    wds_maxcount=max(1, len(recs) // 16))

    if name == "stanford_cars":
        recs = []
        for i, s in _take("roskyluo/stanford_cars_blip", "train", limit):
            recs.append({"key": f"{i:06d}", "img": _jpg(s["image"]), "ext": "jpg",
                         "meta": {"caption": s["text"]}})
        return dict(name="stanford_cars", records=recs, ext="jpg", target=224,
                    meta_features={"caption": Value("string")},
                    wds_maxcount=max(1, len(recs) // 16))

    if name == "ffhq256":
        recs = []
        for i, s in _take("merkol/ffhq-256", "train", limit or 10000):
            recs.append({"key": f"{i:06d}", "img": _jpg(s["image"]), "ext": "jpg",
                         "meta": {"idx": i}})
        return dict(name="ffhq256", records=recs, ext="jpg", target=224,
                    meta_features={"idx": Value("int64")},
                    wds_maxcount=max(1, len(recs) // 16))

    raise ValueError(name)
