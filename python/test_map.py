"""End-to-end tests for `Dataset.map` enrichment (the additive-layer path).

Covers: tensor output (depth-like array) decoded from an image input, scalar +
text annotation outputs, read-back via get()/read_array(), idempotent resume,
subset-scoped map, and the multi-output case.

Run:  source /tmp/venv/bin/activate && python python/test_map.py
"""
import io
import shutil
import tempfile

import numpy as np
from PIL import Image

import ferroload


def jpg(h, w, fill):
    arr = np.full((h, w, 3), fill, dtype="uint8")
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG")
    return b.getvalue()


def build(root, n=10):
    shutil.rmtree(root, ignore_errors=True)
    w = ferroload.Writer(root, "base")
    w.declare("image", "jpg", "tensor", "image")
    for i in range(n):
        w.add(f"s{i:04d}", {"image": jpg(32, 32, i * 10 % 256)}, {"label": i % 3})
    w.close()
    return ferroload.Dataset.open(root)


def test_tensor_and_annotation_outputs():
    root = tempfile.mkdtemp(prefix="ferro_map_")
    ds = build(root, 10)

    def enrich(im, label):                          # per-sample, positional inputs
        depth = im.mean(axis=2).astype("float32")   # [H,W] array
        bright = float(im.mean())                   # scalar
        tag = f"class{label}"                       # text
        return depth, bright, tag                   # tuple in `outputs` order

    out = ds.map(
        enrich,
        inputs=["image", "label"],
        outputs={"depth": "array", "brightness": "scalar", "tag": "text"},
        name="features",
        batch_size=4,
    )

    # new modality visible
    assert "depth" in out.modalities(), out.modalities()
    # tensor output reads back as the right-shaped array
    d0 = out.read_array(0, "depth")
    assert d0.shape == (32, 32) and d0.dtype == np.float32
    # annotation outputs merged into meta
    s5 = out.get(5)
    assert "brightness" in s5["meta"] and "tag" in s5["meta"]
    assert s5["meta"]["tag"] == "class2"          # label 5%3==2
    assert s5["depth_present"] is True
    # base modality + base meta still intact
    assert len(out.get(5)["image"]) > 0
    assert s5["meta"]["label"] == 2
    # batched array read-back
    arrs = out.read_arrays([0, 1, 2], "depth")
    assert all(a.shape == (32, 32) for a in arrs)
    shutil.rmtree(root, ignore_errors=True)
    print("ok: tensor_and_annotation_outputs")


def test_resume_idempotent():
    root = tempfile.mkdtemp(prefix="ferro_mapres_")
    ds = build(root, 12)
    calls = {"n": 0}

    def emb(im):                                    # per-sample
        calls["n"] += 1
        return im.reshape(-1)[:8].astype("float32")

    # first pass over first 6 via a subset, then full pass should only do the rest
    sub = ds.subset("label >= 0")          # all, but exercise subset path
    sub.map(emb, inputs="image", outputs=["emb"], name="emb", batch_size=5)
    first = calls["n"]
    assert first == 12

    # re-run on full ds with resume: nothing new to compute
    calls["n"] = 0
    out = ds.map(emb, inputs="image", outputs=["emb"], name="emb", batch_size=5, resume=True)
    assert calls["n"] == 0, "resume should skip already-enriched samples"
    # values are all present and correct length
    e = out.read_array(3, "emb")
    assert e.shape == (8,) and e.dtype == np.float32

    # resume=False recomputes everything
    calls["n"] = 0
    ds.map(emb, inputs="image", outputs=["emb"], name="emb", batch_size=5, resume=False)
    assert calls["n"] == 12
    shutil.rmtree(root, ignore_errors=True)
    print("ok: resume_idempotent")


def test_subset_scoped_map():
    root = tempfile.mkdtemp(prefix="ferro_mapsub_")
    ds = build(root, 9)

    def f(im):
        return im[:16, :16].mean(axis=2).astype("float32")

    only = ds.subset("label = 0")          # ids 0,3,6
    n_sub = len(only)
    assert n_sub == 3
    out = only.map(f, inputs="image", outputs=["half"], name="half", batch_size=2)
    # the returned view keeps the subset; all its rows are enriched
    assert len(out) == 3
    for i in range(3):
        assert out.read_array(i, "half").shape == (16, 16)
    # on the full dataset, only the subset ids are present (sparse layer)
    full = ferroload.Dataset.open(root)
    assert full.get(0)["half_present"] is True
    assert full.get(1)["half_present"] is False     # label 1, not in subset
    shutil.rmtree(root, ignore_errors=True)
    print("ok: subset_scoped_map")


def test_chained_maps_and_verify():
    root = tempfile.mkdtemp(prefix="ferro_mapchain_")
    ds = build(root, 6)
    ds = ds.map(lambda im: im.mean(2).astype("float32"),
                inputs="image", outputs=["gray"], name="gray", batch_size=3)
    # a second map consumes the first layer's tensor output (auto-loaded as arrays)
    ds2 = ds.map(lambda g: float(g.mean()),
                 inputs="gray", outputs={"gmean": "scalar"}, name="gmean", batch_size=3)
    assert "gray" in ds2.modalities()
    # chained value is correct: gmean == mean of the gray array
    assert abs(ds2.get(0)["meta"]["gmean"] - float(ds2.read_array(0, "gray").mean())) < 1e-3
    # integrity of all shards (base + both layers)
    assert ds2.verify() == 6
    shutil.rmtree(root, ignore_errors=True)
    print("ok: chained_maps_and_verify")


def test_bytes_output_download_pattern():
    """A URL column -> download -> store the raw bytes as a new `video` modality.
    Uses a fake fetcher (local bytes) so the test needs no network; the only
    difference in real use is `fn` does an HTTP GET."""
    root = tempfile.mkdtemp(prefix="ferro_mapdl_")
    shutil.rmtree(root, ignore_errors=True)
    w = ferroload.Writer(root, "with_urls")
    w.declare("image", "jpg", "tensor", "image")
    fake_cdn = {}
    for i in range(7):
        url = f"https://cdn.example/{i}.mp4"
        fake_cdn[url] = f"MP4-BYTES-{i}".encode() * 3
        w.add(f"s{i:04d}", {"image": jpg(16, 16, i)}, {"video_url": url})
    w.close()
    ds = ferroload.Dataset.open(root)

    def fetch(url):                       # in real life: requests.get(url).content
        return fake_cdn[url]

    def download(url):                    # per-sample, generic (no column names)
        return fetch(url)                 # raw mp4 bytes

    out = ds.map(download, inputs=["video_url"],
                 outputs={"video": "video"}, name="video", batch_size=3)

    # the new modality is declared with the right ext/codec, stored as raw bytes
    vm = out.modalities()["video"]
    assert vm["ext"] == "mp4" and vm["codec"] == "video", vm
    assert out.read(0, "video") == fake_cdn["https://cdn.example/0.mp4"]
    s3 = out.get(3)
    assert s3["video"] == fake_cdn["https://cdn.example/3.mp4"]
    assert s3["video_present"] is True
    assert len(s3["image"]) > 0           # base modality intact

    # explicit ext/codec via dict form + sparse (skip a sample by returning None)
    def download2(url):                   # None -> that sample is skipped (sparse)
        return None if url.endswith("2.mp4") else fetch(url)
    out2 = ds.map(download2, inputs=["video_url"],
                  outputs={"clip": {"type": "bytes", "ext": "mp4", "codec": "video"}},
                  name="clip", batch_size=4)
    assert out2.get(2)["clip_present"] is False      # None -> absent
    assert out2.get(1)["clip_present"] is True
    shutil.rmtree(root, ignore_errors=True)
    print("ok: bytes_output_download_pattern")


def test_typed_output_objects():
    """DESIGN §13.3 typed output declarations: Modality(...) / Annotation()."""
    from ferroload import Modality, Annotation
    root = tempfile.mkdtemp(prefix="ferro_typed_")
    ds = build(root, 6)

    def enrich(im):                                       # per-sample, 3 outputs
        return (im.mean(2).astype("float32"),             # array
                jpg(8, 8, 1),                             # raw png/jpg bytes
                "x")                                      # annotation

    out = ds.map(
        enrich, inputs=["image"],
        outputs={"depth": Modality("npy"),                       # array -> .npy modality
                 "thumb": Modality("jpg", codec="image"),        # raw bytes modality
                 "tag":   Annotation()},                         # -> metadata
        name="typed",
    )
    assert out.read_array(0, "depth").shape == (32, 32)
    assert out.modalities()["thumb"] == {"ext": "jpg", "kind": "tensor", "codec": "image"}
    assert out.read(0, "thumb") == jpg(8, 8, 1)
    assert out.get(0)["meta"]["tag"] == "x"
    shutil.rmtree(root, ignore_errors=True)
    print("ok: typed_output_objects")


def test_topology_detection():
    """Auto-detect topology + executor selection from launcher env (DESIGN §14.4)."""
    import os
    import ferroload as fe

    saved = dict(os.environ)
    try:
        # bare -> single node -> LocalExecutor
        for k in list(os.environ):
            if k in ("WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "SLURM_NTASKS",
                     "SLURM_NNODES", "SLURM_PROCID", "RAY_ADDRESS", "FERROLOAD_EXECUTOR"):
                os.environ.pop(k, None)
        assert fe.detect_topology().num_nodes == 1
        assert type(fe.select_executor()).__name__ == "LocalExecutor"

        # torchrun single node (2 procs, 1 node) -> still Local
        os.environ.update(WORLD_SIZE="2", RANK="0", LOCAL_WORLD_SIZE="2")
        t = fe.detect_topology()
        assert t.source == "torchrun" and t.world_size == 2 and t.num_nodes == 1
        assert type(fe.select_executor(t)).__name__ == "LocalExecutor"

        # torchrun 2 nodes x 2 procs -> distributed -> StaticPartitionExecutor
        os.environ.update(WORLD_SIZE="4", RANK="2", LOCAL_WORLD_SIZE="2", NNODES="2")
        t = fe.detect_topology()
        assert t.num_nodes == 2 and t.world_size == 4 and t.rank == 2
        assert type(fe.select_executor(t)).__name__ == "StaticPartitionExecutor"

        # explicit override always wins
        assert type(fe.select_executor(t, override="local")).__name__ == "LocalExecutor"
    finally:
        os.environ.clear()
        os.environ.update(saved)
    print("ok: topology_detection")


def test_static_partition_executor():
    """Simulate a 3-rank torchrun job in-process: each rank runs map with a
    StaticPartitionExecutor (writing its own fragment), then a manual commit
    merges them (no torch.distributed barrier available here)."""
    from ferroload import StaticPartitionExecutor, Topology, commit_layer
    root = tempfile.mkdtemp(prefix="ferro_static_")
    ds = build(root, 9)

    def enrich(label):
        return int(label) ** 2                                  # annotation output

    world = 3
    for rank in range(world):
        topo = Topology(num_nodes=2, node_rank=0, local_size=1, world_size=world, rank=rank)
        ex = StaticPartitionExecutor(topo)
        # each rank processes only sample_ids where sid % world == rank
        ferroload.Dataset.open(root).map(
            enrich, inputs=["label"], outputs={"sq": "scalar"}, name="sq", executor=ex)

    # not committed yet (multi-rank, no barrier) -> run the manual commit step
    commit_layer(root, "sq", None)

    out = ferroload.Dataset.open(root)
    for i in range(9):
        assert out.get(i)["meta"]["sq"] == (i % 3) ** 2, i
    assert out.verify() == 9
    shutil.rmtree(root, ignore_errors=True)
    print("ok: static_partition_executor")


if __name__ == "__main__":
    test_tensor_and_annotation_outputs()
    test_resume_idempotent()
    test_subset_scoped_map()
    test_chained_maps_and_verify()
    test_bytes_output_download_pattern()
    test_typed_output_objects()
    test_topology_detection()
    test_static_partition_executor()
    print("\nALL MAP TESTS PASSED")
