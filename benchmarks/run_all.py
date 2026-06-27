"""Orchestrate all benchmark configs for a dataset (each in a subprocess) and
aggregate results.

  python run_all.py <name> [max_samples]

Writes results/<name>.json and prints a summary table.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

BATCH = {"cifar10": 256, "stanford_cars": 128, "ffhq256": 128}
WORKERS = [0, 4, 8]


def configs(name):
    cfgs = []
    for nw in WORKERS:
        cfgs.append(("hf_arrow", nw))
        cfgs.append(("webdataset", nw))
    cfgs.append(("ferro_native", 0))  # in-process rayon, uses all cores
    return cfgs


def main():
    name = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    bs = BATCH[name]
    os.makedirs(RESULTS, exist_ok=True)
    meta = json.load(open(os.path.join(HERE, "bench_data", name, "meta.json")))

    results = []
    for loader, nw in configs(name):
        cmd = [sys.executable, os.path.join(HERE, "bench_one.py"), name, loader,
               str(nw), str(bs), str(max_samples)]
        print(f"-> {loader:12} nw={nw} ...", flush=True, end=" ")
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line is None:
                print("FAILED")
                print(out.stdout[-500:]); print(out.stderr[-1500:])
                continue
            r = json.loads(line[len("RESULT "):])
            results.append(r)
            print(f"{r['samples_per_s']:>8.1f} samp/s  first={r['first_batch_s']:.2f}s  rss={r['peak_rss_mb']:.0f}MB")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")

    out = {"meta": meta, "results": results}
    json.dump(out, open(os.path.join(RESULTS, f"{name}.json"), "w"), indent=2)

    # summary table
    print(f"\n=== {name}  (n={meta['n']}, target={meta['target']}, bs={bs}) ===")
    print(f"{'loader':14}{'nw':>4}{'samp/s':>11}{'first(s)':>10}{'rss(MB)':>9}")
    for r in sorted(results, key=lambda r: -r["samples_per_s"]):
        print(f"{r['loader']:14}{r['num_workers']:>4}{r['samples_per_s']:>11.1f}"
              f"{r['first_batch_s']:>10.2f}{r['peak_rss_mb']:>9.0f}")
    sz = meta["sizes"]
    print("\non-disk size (MB):  "
          + "  ".join(f"{k}={v/1e6:.1f}" for k, v in sz.items()))


if __name__ == "__main__":
    main()
