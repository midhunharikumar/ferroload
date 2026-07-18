"""Run the determinism demo locally (no Modal): pack a small slice of the
dataset, do N identical training runs in-process, assert bit-identical loss
curves, and write the comparison plot.

Local caveat: bitwise determinism holds per hardware+library configuration —
all N runs here share one machine, so this is the weaker (single-host) form of
the claim. The Modal harness (`modal_app.py`) runs each training run in a
separate container on the same GPU model, which is the stronger demo.

Usage:
    python train_local.py --data /tmp/ferro-flickr --limit 2000 --steps 60
"""
import argparse
import json
import os

from common import max_pairwise_diff, render_plot, train_one_run
from prepare_data import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="/tmp/ferro-flickr30k")
    ap.add_argument("--hf-dataset", default="lmms-lab/flickr30k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    root = build_dataset(a.hf_dataset, a.data, limit=a.limit, split=a.split)

    results = [train_one_run(root, steps=a.steps, batch_size=a.batch_size,
                             lr=a.lr, seed=a.seed, run_id=i)
               for i in range(a.runs)]

    _, overall = max_pairwise_diff(results)
    subtitle = (f"{a.hf_dataset} ({a.limit} image–caption pairs) · mini-CLIP · "
                f"batch {a.batch_size} · {results[0]['device']} · torch {results[0]['torch']}")
    plot_path = os.path.join(a.out, "loss_determinism.png")
    with open(plot_path, "wb") as f:
        f.write(render_plot(results, subtitle))
    with open(os.path.join(a.out, "loss_determinism.json"), "w") as f:
        json.dump({"config": vars(a), "results": results,
                   "max_abs_diff": overall}, f, indent=2)
    print(f"wrote {plot_path}")

    if overall == 0.0:
        print(f"PASS: {a.runs} runs x {a.steps} steps — loss curves are bitwise identical")
    else:
        raise SystemExit(f"FAIL: loss curves diverged, max |Δloss| = {overall:.3e}")


if __name__ == "__main__":
    main()
