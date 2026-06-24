#!/usr/bin/env python3
"""Import a JSONL-manifest + referenced-media-files HF dataset (e.g.
MiG-NJU/OmniVideo-Test: a test_*.jsonl plus videos/*.mp4) into Ferroload.

Unlike Arrow datasets, here each row references a media file by path; we pull
only the first `--limit` media files (not the whole repo), pack each as a tensor
modality blob with all other JSONL fields as metadata, then read back + verify.

Usage:
    python import_hf_files.py <repo_id> <out_root> \
        --jsonl test_505.jsonl --media-field video_path --modality video --ext mp4 --limit 8
"""
import argparse
import json
import os

import ferroload
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("out")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--media-field", default="video_path")
    ap.add_argument("--modality", default="video")
    ap.add_argument("--ext", default="mp4")
    ap.add_argument("--codec", default="video")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    man = hf_hub_download(args.repo_id, args.jsonl, repo_type="dataset")
    rows = [json.loads(l) for l in open(man) if l.strip()]
    print(f"manifest: {len(rows)} rows; importing first {args.limit}")

    w = ferroload.Writer(args.out, os.path.basename(os.path.abspath(args.out)))
    w.declare(args.modality, args.ext, "tensor", args.codec)

    total = 0
    n = 0
    for row in rows[: args.limit]:
        rel = row.get(args.media_field)
        if not rel:
            continue
        local = hf_hub_download(args.repo_id, rel, repo_type="dataset")
        data = open(local, "rb").read()
        total += len(data)
        meta = {}
        for k, v in row.items():
            if k == args.media_field:
                continue
            meta[k] = v if isinstance(v, (bool, int, float, str)) else json.dumps(v)
        key = str(row.get("question_id") or row.get("video_id") or f"sample{n:06d}")
        w.add(key, {args.modality: data}, meta)
        n += 1
        print(f"  [{n}/{args.limit}] {rel}  ({len(data)/1e6:.2f} MB)")
    w.close()
    print(f"packed {n} samples, {total/1e6:.1f} MB of {args.modality} -> {args.out}")

    ds = ferroload.Dataset.open(args.out)
    print(f"reopened: {len(ds)} samples, {ds.num_shards()} shard(s)")
    s = ds.get(0)
    vlen = len(s.get(args.modality, b""))
    print(f"sample0: {args.modality}_bytes={vlen}, present={s.get(args.modality + '_present')}")
    print(f"  question: {str(s['meta'].get('question'))[:90]}...")
    print(f"  task={s['meta'].get('task')} answer={s['meta'].get('answer')} dur={s['meta'].get('duration')}")
    # projection: metadata-only read fetches no video bytes
    only_meta = ds.get(1, [])
    print(f"projection (no modalities): fetched video? {'video' in only_meta}")
    print(f"verify: {ds.verify()} samples OK")


if __name__ == "__main__":
    main()
