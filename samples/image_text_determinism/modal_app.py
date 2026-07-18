"""Modal harness for the image+text determinism sample.

Runs the whole demo in the cloud and asserts the result, so it doubles as an
end-to-end determinism test for Ferroload's loader:

  1. `prepare`   — stream Flickr30k from HuggingFace and pack it into the
                   Ferroload format on a shared Modal Volume (one-time, cached).
  2. `train_run` — N identical training runs fan out to N *separate* T4
                   containers (same GPU model — bitwise determinism is only
                   defined per hardware+library configuration).
  3. `main`      — collects the loss curves, asserts they are bit-identical,
                   renders the comparison plot, and writes it locally. Exits
                   non-zero on any divergence, so `modal run` is the test.

Usage:
    modal run samples/image_text_determinism/modal_app.py            # defaults
    modal run samples/image_text_determinism/modal_app.py --runs 5 --steps 300
    modal run samples/image_text_determinism/modal_app.py --limit 2000 --steps 60  # quick smoke
"""
import json
import os

import modal

GPU = os.environ.get("FERROLOAD_SAMPLE_GPU", "T4")
VOL_PATH = "/vol"

app = modal.App("ferroload-determinism-sample")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "ferroload", "datasets", "pillow", "matplotlib", "numpy")
    # cuBLAS determinism knob must be set before the first CUDA matmul
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HOME": f"{VOL_PATH}/hf-cache"})
    .add_local_python_source("common", "prepare_data")
)

vol = modal.Volume.from_name("ferroload-samples", create_if_missing=True)


@app.function(image=image, volumes={VOL_PATH: vol}, timeout=3600)
def prepare(hf_dataset: str, split: str, limit: int) -> str:
    from prepare_data import build_dataset
    slug = hf_dataset.replace("/", "--")
    root = f"{VOL_PATH}/{slug}-{split}-{limit}"
    build_dataset(hf_dataset, root, limit=limit, split=split)
    vol.commit()
    return root


@app.function(image=image, volumes={VOL_PATH: vol}, gpu=GPU, timeout=3600)
def train_run(run_id: int, cfg: dict) -> dict:
    from common import train_one_run
    return train_one_run(cfg["root"], steps=cfg["steps"], batch_size=cfg["batch_size"],
                         lr=cfg["lr"], seed=cfg["seed"], run_id=run_id)


@app.function(image=image)
def render(results: list, subtitle: str) -> bytes:
    from common import render_plot
    return render_plot(results, subtitle)


@app.local_entrypoint()
def main(runs: int = 3, steps: int = 200, batch_size: int = 64, lr: float = 1e-3,
         seed: int = 0, limit: int = 20000, hf_dataset: str = "lmms-lab/flickr30k",
         split: str = "test", out: str = ""):
    out_dir = out or os.path.dirname(os.path.abspath(__file__))

    root = prepare.remote(hf_dataset, split, limit)
    print(f"dataset ready at {root} (Modal volume 'ferroload-samples')")

    cfg = {"root": root, "steps": steps, "batch_size": batch_size,
           "lr": lr, "seed": seed}
    results = list(train_run.map(range(runs), kwargs={"cfg": cfg}))
    results.sort(key=lambda r: r["run_id"])

    from statistics import fmean  # stdlib only — the entrypoint runs locally
    curves = [r["losses"] for r in results]
    n = min(map(len, curves))
    per_step_max = [max(abs(a[i] - b[i]) for a in curves for b in curves)
                    for i in range(n)]
    overall = max(per_step_max) if per_step_max else 0.0
    for r in results:
        print(f"run {r['run_id']}: {len(r['losses'])} steps on {r['device']} "
              f"(torch {r['torch']}), mean loss {fmean(r['losses']):.6f}")

    subtitle = (f"{hf_dataset} ({limit} image–caption pairs) · mini-CLIP · "
                f"batch {batch_size} · {results[0]['device']} · torch {results[0]['torch']}")
    png = render.remote(results, subtitle)
    plot_path = os.path.join(out_dir, "loss_determinism.png")
    with open(plot_path, "wb") as f:
        f.write(png)
    json_path = os.path.join(out_dir, "loss_determinism.json")
    with open(json_path, "w") as f:
        json.dump({"config": {**cfg, "runs": runs, "hf_dataset": hf_dataset,
                              "split": split, "limit": limit},
                   "results": results, "max_abs_diff": overall}, f, indent=2)
    print(f"wrote {plot_path} and {json_path}")

    if overall == 0.0:
        print(f"PASS: {runs} runs x {n} steps — loss curves are bitwise identical")
    else:
        raise SystemExit(f"FAIL: loss curves diverged, max |Δloss| = {overall:.3e}")
