#!/usr/bin/env python3
"""`ferroload` command-line interface (entry point: ferroload.cli:main).

Subcommands: inspect, verify, import-hf, import-files, subset, list, add.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

CATALOG = Path(os.environ.get("FERROLOAD_HOME", Path.home() / ".ferroload")) / "catalog.json"


# ----------------------------- catalog --------------------------------------
def _load_catalog():
    if CATALOG.exists():
        return json.loads(CATALOG.read_text())
    return {"datasets": []}


def _save_catalog(cat):
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps(cat, indent=2))


def _is_remote(uri):
    return "://" in uri and not uri.startswith("file://")


def catalog_register(uri, name=None):
    """Add/update a dataset in the catalog, reading its manifest if local."""
    cat = _load_catalog()
    entry = {"uri": uri, "location": "remote" if _is_remote(uri) else "local",
             "last_seen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not _is_remote(uri):
        mpath = Path(uri) / "manifest.json"
        if mpath.exists():
            m = json.loads(mpath.read_text())
            entry["name"] = name or m.get("name")
            entry["rows"] = m.get("index", {}).get("rows")
            entry["modalities"] = sorted((m.get("modalities") or {}).keys())
            entry["version"] = m.get("version")
    entry["name"] = entry.get("name") or name or os.path.basename(uri.rstrip("/"))
    cat["datasets"] = [d for d in cat["datasets"] if d.get("uri") != uri]
    cat["datasets"].append(entry)
    _save_catalog(cat)
    return entry


# ----------------------------- commands -------------------------------------
def cmd_inspect(args):
    import ferroload
    ds = ferroload.Dataset.open(args.root)           # use binding introspection
    m = ds.manifest()
    print(f"name:            {ds.name}")
    print(f"format_version:  {m.get('format_version')}  (min_reader {m.get('min_reader_version')})")
    print(f"version:         {ds.version}")
    print(f"rows:            {len(ds)}")
    print(f"shards:          {ds.num_shards()}")
    mod_strs = [f"{k}({v.get('kind')}/{v.get('codec')})" for k, v in ds.modalities().items()]
    print(f"modalities:      {', '.join(mod_strs)}")
    sch = ds.schema()
    if sch:
        print(f"schema columns:  {', '.join(c['name'] for c in sch)}")
    if m.get("extensions"):
        print(f"extensions:      {', '.join(m['extensions'].keys())}")
    stats = Path(args.root) / "index" / "_stats.json"   # shard_bytes lives outside the manifest
    if stats.exists():
        print(f"shard bytes:     {json.loads(stats.read_text()).get('shard_bytes')}")


def cmd_verify(args):
    import ferroload
    ds = ferroload.Dataset.open(args.root)
    print(f"OK: verified {ds.verify()} samples across {ds.num_shards()} shard(s)")


def cmd_subset(args):
    import ferroload
    ds = ferroload.Dataset.open(args.root)
    ids = ds.subset(args.where, return_indices=True)
    print(f"matched {len(ids)} / {len(ds)} samples")
    if args.out:
        outdir = Path(args.root) / "subsets"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{args.out}.json"
        path.write_text(json.dumps({"where": args.where, "sample_ids": ids}))
        print(f"wrote {path}")


def cmd_import_hf(args):
    import io
    import ferroload
    from datasets import load_dataset

    def is_pil(v):
        return hasattr(v, "save") and hasattr(v, "convert")

    ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)
    w = ferroload.Writer(args.out, os.path.basename(os.path.abspath(args.out)))
    w.declare("image", "png", "tensor", "image")
    n = 0
    for ex in ds:
        if n >= args.limit:
            break
        blobs, meta = {}, {}
        for k, v in ex.items():
            if (args.image_col and k == args.image_col) or (not args.image_col and is_pil(v)):
                if v is not None:
                    b = io.BytesIO(); v.convert("RGB").save(b, "PNG"); blobs["image"] = b.getvalue()
            elif isinstance(v, (bool, int, float, str)):
                meta[k] = v
            else:
                meta[k] = str(v)
        w.add(f"sample{n:06d}", blobs, meta)
        n += 1
    w.close()
    entry = catalog_register(os.path.abspath(args.out))
    print(f"packed {n} -> {args.out}; registered as '{entry['name']}'")


def cmd_import_files(args):
    import ferroload
    from huggingface_hub import hf_hub_download

    man = hf_hub_download(args.dataset, args.jsonl, repo_type="dataset")
    rows = [json.loads(l) for l in open(man) if l.strip()]
    w = ferroload.Writer(args.out, os.path.basename(os.path.abspath(args.out)))
    w.declare(args.modality, args.ext, "tensor", args.codec)
    n = 0
    for row in rows[: args.limit]:
        rel = row.get(args.media_field)
        if not rel:
            continue
        local = hf_hub_download(args.dataset, rel, repo_type="dataset")
        data = open(local, "rb").read()
        meta = {k: (v if isinstance(v, (bool, int, float, str)) else json.dumps(v))
                for k, v in row.items() if k != args.media_field}
        key = str(row.get("question_id") or row.get("video_id") or f"sample{n:06d}")
        w.add(key, {args.modality: data}, meta)
        n += 1
    w.close()
    entry = catalog_register(os.path.abspath(args.out))
    print(f"packed {n} -> {args.out}; registered as '{entry['name']}'")


def cmd_list(args):
    cat = _load_catalog()
    if not cat["datasets"]:
        print("(catalog empty — import a dataset or `ferroload add <path>`)")
        return
    print(f"{'NAME':<24}{'LOC':<8}{'ROWS':>10}  MODALITIES        URI")
    for d in cat["datasets"]:
        print(f"{(d.get('name') or '?'):<24}{d.get('location',''):<8}"
              f"{str(d.get('rows','-')):>10}  {','.join(d.get('modalities', [])):<18}{d.get('uri')}")


def cmd_add(args):
    entry = catalog_register(args.uri, name=args.name)
    print(f"registered '{entry['name']}' ({entry['location']})")


# ----------------------------- argparse -------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="ferroload", description="Ferroload dataset CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect"); s.add_argument("root"); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("verify"); s.add_argument("root"); s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("subset")
    s.add_argument("root"); s.add_argument("--where", required=True)
    s.add_argument("--out", help="name to materialize subsets/<name>.json")
    s.set_defaults(fn=cmd_subset)

    s = sub.add_parser("import-hf")
    s.add_argument("dataset"); s.add_argument("out")
    s.add_argument("--split", default="train"); s.add_argument("--name")
    s.add_argument("--limit", type=int, default=100); s.add_argument("--image-col")
    s.set_defaults(fn=cmd_import_hf)

    s = sub.add_parser("import-files")
    s.add_argument("dataset"); s.add_argument("out")
    s.add_argument("--jsonl", required=True); s.add_argument("--media-field", default="video_path")
    s.add_argument("--modality", default="video"); s.add_argument("--ext", default="mp4")
    s.add_argument("--codec", default="video"); s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_import_files)

    s = sub.add_parser("list"); s.set_defaults(fn=cmd_list)
    s = sub.add_parser("add"); s.add_argument("uri"); s.add_argument("--name"); s.set_defaults(fn=cmd_add)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
