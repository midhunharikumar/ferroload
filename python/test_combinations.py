"""Flexibility tests for the loader across modality combinations:
multiple images, multiple video streams, mixed video+image+metadata, and sparse
samples. Verifies the per-sample-dict contract and that a standard (default or
custom) collate_fn can batch the result.

Run:  PYTHONPATH=<dir-with-ferroload.so>:<this dir> python3 test_combinations.py
"""
import io
import shutil
import tempfile

import numpy as np
from PIL import Image

import ferroload
import ferroload_loader as fl


def jpg(h, w):
    arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG")
    return b.getvalue()


def build(root, decls, rows):
    shutil.rmtree(root, ignore_errors=True)
    w = ferroload.Writer(root, "t")
    for name, (ext, kind, codec) in decls.items():
        w.declare(name, ext, kind, codec)
    for key, (blobs, meta) in rows:
        w.add(key, blobs, meta)
    w.close()
    return ferroload.Dataset.open(root)


def default_collate(samples):
    """Minimal stand-in for torch's default_collate (numpy)."""
    out = {}
    keys = samples[0].keys()
    for k in keys:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], np.ndarray):
            out[k] = np.stack(vals)            # uniform tensors -> [B,...]
        elif isinstance(vals[0], (int, float, bool, np.integer, np.floating, np.bool_)):
            out[k] = np.array(vals)
        else:
            out[k] = vals                      # strings / bytes / None -> list
    return out


def test_multiple_image_streams():
    root = tempfile.mkdtemp(prefix="ferro_imgs_")
    ds = build(
        root,
        {"image": ("jpg", "tensor", "image"), "thumb": ("jpg", "tensor", "image")},
        [(f"s{i}", ({"image": jpg(40, 50), "thumb": jpg(20, 20)}, {"label": i})) for i in range(5)],
    )
    tds = fl.FerroTorchDataset(ds, images=["image", "thumb"], meta=["label"], resize=(32, 32))
    samples = tds.__getitems__([0, 1, 2])
    assert len(samples) == 3
    assert samples[0]["image"].shape == (32, 32, 3)
    assert samples[0]["thumb"].shape == (32, 32, 3)
    batch = default_collate(samples)
    assert batch["image"].shape == (3, 32, 32, 3)
    assert batch["thumb"].shape == (3, 32, 32, 3)
    assert batch["label"].tolist() == [0, 1, 2]
    shutil.rmtree(root, ignore_errors=True)
    print("ok: multiple_image_streams")


def test_multiple_video_streams_plus_meta():
    root = tempfile.mkdtemp(prefix="ferro_vids_")
    ds = build(
        root,
        {"video": ("mp4", "tensor", "video"), "video_depth": ("mp4", "tensor", "video")},
        [(f"c{i}", ({"video": f"V{i}".encode(), "video_depth": f"D{i}".encode()},
                    {"task": "qa", "duration": i})) for i in range(4)],
    )
    tds = fl.FerroTorchDataset(ds, raw=["video", "video_depth"], meta=["task", "duration"])
    samples = tds.__getitems__([0, 1, 2, 3])
    assert samples[2]["video"] == b"V2"
    assert samples[2]["video_depth"] == b"D2"
    assert samples[2]["video_present"] and samples[2]["video_depth_present"]
    batch = default_collate(samples)
    assert batch["duration"].tolist() == [0, 1, 2, 3]
    assert batch["task"] == ["qa", "qa", "qa", "qa"]
    shutil.rmtree(root, ignore_errors=True)
    print("ok: multiple_video_streams_plus_meta")


def test_mixed_video_image_audio_metadata():
    root = tempfile.mkdtemp(prefix="ferro_mixed_")
    ds = build(
        root,
        {"video": ("mp4", "tensor", "video"),
         "keyframe": ("jpg", "tensor", "image"),
         "audio": ("flac", "tensor", "audio")},
        [(f"s{i}", ({"video": f"V{i}".encode(), "keyframe": jpg(64, 64), "audio": f"A{i}".encode()},
                    {"caption": f"cap{i}", "label": i})) for i in range(3)],
    )
    tds = fl.FerroTorchDataset(ds, images=["keyframe"], raw=["video", "audio"],
                               meta=["caption", "label"], resize=(48, 48))
    samples = tds.__getitems__([0, 1, 2])
    s = samples[1]
    assert s["keyframe"].shape == (48, 48, 3)
    assert s["video"] == b"V1" and s["audio"] == b"A1"
    assert s["caption"] == "cap1" and s["label"] == 1
    batch = default_collate(samples)
    assert batch["keyframe"].shape == (3, 48, 48, 3)
    assert batch["caption"] == ["cap0", "cap1", "cap2"]
    shutil.rmtree(root, ignore_errors=True)
    print("ok: mixed_video_image_audio_metadata")


def test_sparse_samples():
    root = tempfile.mkdtemp(prefix="ferro_sparse_")
    rows = []
    for i in range(6):
        blobs = {"video": f"V{i}".encode()}
        if i % 2 == 0:
            blobs["keyframe"] = jpg(30, 30)        # only even have a keyframe
        rows.append((f"s{i}", (blobs, {"label": i})))
    ds = build(
        root,
        {"video": ("mp4", "tensor", "video"), "keyframe": ("jpg", "tensor", "image")},
        rows,
    )
    tds = fl.FerroTorchDataset(ds, images=["keyframe"], raw=["video"], meta=["label"], resize=(16, 16))
    samples = tds.__getitems__([0, 1, 2, 3])
    assert samples[0]["keyframe"].shape == (16, 16, 3)   # present
    assert samples[0]["keyframe_present"] is True
    assert samples[1]["keyframe"] is None                 # absent -> None
    assert samples[1]["keyframe_present"] is False
    assert all(s["video_present"] for s in samples)
    shutil.rmtree(root, ignore_errors=True)
    print("ok: sparse_samples")


def test_projection_only_requested_read():
    # a dataset with 3 modalities; a loader requesting 1 must not error on others
    root = tempfile.mkdtemp(prefix="ferro_proj_")
    ds = build(
        root,
        {"video": ("mp4", "tensor", "video"),
         "keyframe": ("jpg", "tensor", "image"),
         "audio": ("flac", "tensor", "audio")},
        [(f"s{i}", ({"video": f"V{i}".encode(), "keyframe": jpg(20, 20), "audio": b"A"}, {})) for i in range(3)],
    )
    tds = fl.FerroTorchDataset(ds, raw=["video"])     # only video requested
    samples = tds.__getitems__([0, 1, 2])
    assert set(samples[0].keys()) == {"video", "video_present"}
    shutil.rmtree(root, ignore_errors=True)
    print("ok: projection_only_requested_read")


if __name__ == "__main__":
    test_multiple_image_streams()
    test_multiple_video_streams_plus_meta()
    test_mixed_video_image_audio_metadata()
    test_sparse_samples()
    test_projection_only_requested_read()
    print("\nALL COMBINATION TESTS PASSED")
