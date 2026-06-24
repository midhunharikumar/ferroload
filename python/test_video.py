"""Test in-Rust video decode on a real video dataset.

Prereq: build the extension WITH the video feature (needs system ffmpeg):
    brew install ffmpeg pkg-config       # macOS
    cd crates/ferroload-py
    maturin develop --release --features video

Then import a few clips and run this:
    # one-time: pack some clips into ferroload format
    python import_hf_files.py MiG-NJU/OmniVideo-Test /tmp/ds_omni \
        --jsonl test_505.jsonl --media-field video_path --modality video --ext mp4 --limit 6
    # decode them:
    python test_video.py /tmp/ds_omni --modality video --num-frames 8
"""
import argparse
import time

import ferroload
import ferroload_loader as fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="ferroload dataset root with a video modality")
    ap.add_argument("--modality", default="video")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    ds = ferroload.Dataset.open(args.root)
    if not hasattr(ds, "decode_video"):
        raise SystemExit(
            "This `ferroload` build has no decode_video — rebuild with:\n"
            "  cd crates/ferroload-py && maturin develop --release --features video"
        )

    n = min(args.n, len(ds))
    idx = list(range(n))
    print(f"ferroload {ferroload.__version__}; dataset has {len(ds)} samples; decoding {n}")

    # direct API
    t0 = time.perf_counter()
    clips = ds.decode_video(idx, args.modality, args.num_frames)
    dt = time.perf_counter() - t0
    for i, c in enumerate(clips):
        shape = None if c is None else c.shape
        print(f"  clip {i}: {shape}")     # expect [T, H, W, 3]
    print(f"decoded {n} clips ({args.num_frames} frames each) in {dt*1000:.0f} ms")

    # via the torch Dataset (per-sample dicts -> your DataLoader/collate)
    tds = fl.FerroTorchDataset(ds, videos=[args.modality], num_frames=args.num_frames)
    samples = tds.__getitems__(idx)
    s0 = samples[0]
    print("sample keys:", sorted(s0.keys()))
    print(f"sample0[{args.modality}] shape:",
          None if s0[args.modality] is None else s0[args.modality].shape)


if __name__ == "__main__":
    main()
