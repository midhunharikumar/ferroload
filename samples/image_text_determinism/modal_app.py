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
import modal.experimental

GPU = os.environ.get("FERROLOAD_SAMPLE_GPU", "T4")
# single-node multi-GPU DDP (any GPU type/count, e.g. "T4:2", "L4:4")
DDP_GPU = os.environ.get("FERROLOAD_SAMPLE_GPU_DDP", "T4:2")
# multi-node is opt-in: setting FERROLOAD_SAMPLE_NODES registers a clustered
# function of that size. Kept out of the default app because Modal validates
# cluster GPU count against the workspace's concurrent-GPU cap at build time.
MULTINODE_GPU = os.environ.get("FERROLOAD_SAMPLE_GPU_MULTINODE", "H100:8")
MULTINODE_NODES = int(os.environ.get("FERROLOAD_SAMPLE_NODES", "0"))
VOL_PATH = "/vol"

app = modal.App("ferroload-determinism-sample")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "ferroload", "datasets", "pillow", "matplotlib", "numpy")
    # cuBLAS determinism knob must be set before the first CUDA matmul; pin
    # NCCL to ring all-reduce so its autotuner can't vary the reduction order.
    # FERROLOAD_SAMPLE_NODES is baked in so the container-side import of this
    # module defines the same (gated) functions the local build registered.
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_ALGO": "Ring",
          "HF_HOME": f"{VOL_PATH}/hf-cache",
          "FERROLOAD_SAMPLE_NODES": os.environ.get("FERROLOAD_SAMPLE_NODES", "0"),
          "FERROLOAD_SAMPLE_GPU_MULTINODE": MULTINODE_GPU})
    .add_local_python_source("common", "prepare_data", "train_ddp")
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


def _render_png(results: list, subtitle: str) -> bytes | None:
    """Render locally when matplotlib is installed (no extra remote call),
    else remotely; a plot failure must not fail an otherwise-passed test."""
    try:
        import matplotlib  # noqa: F401
        from common import render_plot
        return render_plot(results, subtitle)
    except ImportError:
        pass
    try:
        return render.remote(results, subtitle)
    except Exception as e:
        print(f"warning: plot render failed ({e}); losses JSON still written")
        return None


def _torchrun_ddp(cfg: dict, nnodes: int = 1, node_rank: int = 0,
                  master_addr: str | None = None) -> dict | None:
    """Launch train_ddp.py under torchrun with one process per local GPU.
    Returns the merged result (only meaningful on node 0, where torchrun's
    rank 0 writes it)."""
    import json as _json
    import torch
    from torch.distributed.run import parse_args, run

    nproc = torch.cuda.device_count()
    out = "/tmp/ddp_result.json"
    launch = (["--standalone"] if nnodes == 1 else
              [f"--nnodes={nnodes}", f"--node-rank={node_rank}",
               f"--master-addr={master_addr}", "--master-port=29500"])
    run(parse_args(launch + [f"--nproc-per-node={nproc}", "/root/train_ddp.py",
                             "--data-root", cfg["root"],
                             "--steps", str(cfg["steps"]),
                             "--batch-size", str(cfg["batch_size"]),
                             "--lr", str(cfg["lr"]),
                             "--seed", str(cfg["seed"]),
                             "--out", out]))
    if node_rank != 0:
        return None
    with open(out) as f:
        res = _json.load(f)
    res["torch"] = torch.__version__
    res["device"] = f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}"
    return res


@app.function(image=image, volumes={VOL_PATH: vol}, gpu=DDP_GPU, timeout=3600)
def train_run_ddp(run_id: int, cfg: dict) -> dict:
    """One single-node multi-GPU DDP training run."""
    res = _torchrun_ddp(cfg)
    res["run_id"] = run_id
    return res


if MULTINODE_NODES > 1:
    @app.function(image=image, volumes={VOL_PATH: vol}, gpu=MULTINODE_GPU,
                  timeout=3600)
    @modal.experimental.clustered(size=MULTINODE_NODES)
    def train_run_multinode(run_id: int, cfg: dict) -> dict | None:
        """One multi-node DDP training run on a gang-scheduled Modal cluster.
        Every container hosts its node's ranks; the caller receives node 0's
        return value (the merged result)."""
        cluster = modal.experimental.get_cluster_info()
        res = _torchrun_ddp(cfg, nnodes=MULTINODE_NODES, node_rank=cluster.rank,
                            master_addr=cluster.container_ips[0])
        if res is not None:
            res["run_id"] = run_id
            res["nodes"] = MULTINODE_NODES
        return res


@app.local_entrypoint()
def ddp(runs: int = 3, steps: int = 60, batch_size: int = 32, lr: float = 1e-3,
        seed: int = 0, limit: int = 20000, hf_dataset: str = "lmms-lab/flickr30k",
        split: str = "test", nodes: int = 1, out: str = ""):
    """Distributed determinism test. nodes=1 -> single-node multi-GPU
    (FERROLOAD_SAMPLE_GPU_DDP, default T4:2), runs in parallel containers.
    nodes>1 -> Modal clustered multi-node (FERROLOAD_SAMPLE_GPU_MULTINODE,
    default H100:8 x FERROLOAD_SAMPLE_NODES) — full-node pricing, opt in.
    `batch_size` is per rank; the global batch is batch_size x world_size."""
    out_dir = out or os.path.dirname(os.path.abspath(__file__))

    root = prepare.remote(hf_dataset, split, limit)
    print(f"dataset ready at {root}")

    cfg = {"root": root, "steps": steps, "batch_size": batch_size,
           "lr": lr, "seed": seed}
    if nodes == 1:
        results = list(train_run_ddp.map(range(runs), kwargs={"cfg": cfg}))
    else:
        if nodes != MULTINODE_NODES:
            raise SystemExit(
                f"--nodes {nodes} needs FERROLOAD_SAMPLE_NODES={nodes} (currently "
                f"{MULTINODE_NODES or 'unset'}) — cluster size is fixed at import."
                f" e.g.: FERROLOAD_SAMPLE_NODES={nodes} "
                f"FERROLOAD_SAMPLE_GPU_MULTINODE=H100:8 modal run …::ddp --nodes {nodes}")
        # sequential: one gang-scheduled cluster at a time, and run-to-run
        # (not just container-to-container) reproducibility is the claim
        results = [train_run_multinode.remote(i, cfg) for i in range(runs)]
    results.sort(key=lambda r: r["run_id"])

    world = results[0]["world_size"]
    curves = [r["losses"] for r in results]
    n = min(map(len, curves))
    overall = max((abs(a[i] - b[i]) for a in curves for b in curves
                   for i in range(n)), default=0.0)
    rank_ok = all(r["per_rank_local"] == results[0]["per_rank_local"]
                  for r in results)
    for r in results:
        print(f"run {r['run_id']}: world_size={r['world_size']} on {r['device']} "
              f"(torch {r['torch']}), final global loss {r['losses'][-1]:.6f}")

    label = (f"{nodes} node(s) · {results[0]['device']} · {world} ranks")
    subtitle = (f"DDP {label} · {hf_dataset} ({limit} pairs) · "
                f"batch {batch_size}/rank · torch {results[0]['torch']}")
    png = _render_png(results, subtitle)
    plot_path = os.path.join(out_dir, "ddp_loss_determinism.png")
    if png:
        with open(plot_path, "wb") as f:
            f.write(png)
    with open(os.path.join(out_dir, "ddp_loss_determinism.json"), "w") as f:
        json.dump({"config": {**cfg, "runs": runs, "nodes": nodes,
                              "world_size": world, "hf_dataset": hf_dataset,
                              "limit": limit},
                   "results": results, "max_abs_diff": overall,
                   "per_rank_identical": rank_ok}, f, indent=2)
    print(f"wrote {plot_path}")

    if overall == 0.0 and rank_ok:
        print(f"PASS: {runs} DDP runs ({label}) x {n} steps — global and "
              f"per-rank loss curves are bitwise identical")
    else:
        raise SystemExit(f"FAIL: max global |Δloss| = {overall:.3e}, "
                         f"per-rank curves identical: {rank_ok}")


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
    png = _render_png(results, subtitle)
    plot_path = os.path.join(out_dir, "loss_determinism.png")
    if png:
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
