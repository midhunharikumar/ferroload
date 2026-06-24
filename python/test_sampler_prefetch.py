"""Tests for the distributed sampler and async prefetch loader.

Run (after `maturin develop`):  python python/test_sampler_prefetch.py
"""
import io
import shutil
import tempfile

import numpy as np
from PIL import Image

import ferroload
from ferroload import loader as fl


def test_sampler_disjoint_deterministic_resumable():
    S = ferroload.Sampler
    total, W = 1000, 4
    allidx = []
    for r in range(W):
        allidx += S(total, W, r, 1, 0, 7, True, 64).indices(0, 0)
    assert sorted(allidx) == list(range(total))                      # disjoint + complete
    assert S(total, 1, 0, 1, 0, 7).indices(3, 0) == S(total, 1, 0, 1, 0, 7).indices(3, 0)  # deterministic
    assert S(total, 1, 0, 1, 0, 7).indices(0, 0) != S(total, 1, 0, 1, 0, 7).indices(1, 0)  # epoch varies
    full = S(total, 2, 0, 1, 0, 1).indices(0, 0)
    assert S(total, 2, 0, 1, 0, 1).indices(0, 10) == full[10:]        # resumable
    print("ok: sampler disjoint/deterministic/epoch/resumable")


def test_ferrosampler_set_epoch():
    fs = fl.FerroSampler(1000, world_size=2, rank=1, seed=1)
    e0 = list(fs)
    fs.set_epoch(1)
    e1 = list(fs)
    assert len(fs) == 500 and e0 != e1
    print("ok: FerroSampler set_epoch reshuffles")


def test_prefetch_loader_covers_all():
    root = tempfile.mkdtemp(prefix="pf_")

    def jpg(h, w):
        b = io.BytesIO()
        Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8")).save(b, "JPEG")
        return b.getvalue()

    w = ferroload.Writer(root, "pf")
    w.declare("image", "jpg", "tensor", "image")
    n = 37
    for i in range(n):
        w.add(f"s{i:03d}", {"image": jpg(20, 20)}, {"label": i})
    w.close()

    ds = ferroload.Dataset.open(root)
    tds = fl.FerroTorchDataset(ds, images=["image"], meta=["label"], resize=(16, 16))
    sampler = fl.FerroSampler(len(ds), shuffle=False)

    seen = []
    for batch in fl.PrefetchLoader(tds, fl.batched(sampler, 8),
                                   collate_fn=fl.numpy_collate, depth=3):
        seen += batch["label"].tolist()
        assert batch["image"].shape[1:] == (16, 16, 3)
    assert sorted(seen) == list(range(n))
    shutil.rmtree(root, ignore_errors=True)
    print("ok: PrefetchLoader covers all samples")


if __name__ == "__main__":
    test_sampler_disjoint_deterministic_resumable()
    test_ferrosampler_set_epoch()
    test_prefetch_loader_covers_all()
    print("\nALL SAMPLER+PREFETCH TESTS PASSED")
