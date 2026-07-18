"""Shared pieces for the image+text determinism sample.

A small CLIP-style contrastive model trained on (image, caption) pairs read
through `ferroload.make_loader`. Everything here is chosen to be *bitwise
deterministic* end to end:

  - the Ferroload sampler is seeded (`seed=`) and reshuffled via `set_epoch`,
    so the sample order is identical across runs and machines;
  - torch runs with `use_deterministic_algorithms(True)` (any op without a
    deterministic implementation raises instead of silently diverging);
  - the text featurizer hashes words with crc32, not Python's salted `hash()`;
  - the contrastive loss is written as log_softmax + diagonal so it never
    touches `nll_loss` (whose CUDA kernel is not deterministic everywhere).

Given the same GPU model, N runs produce bit-identical loss curves.
"""
from __future__ import annotations

import os
import random
import re
import zlib

VOCAB_SIZE = 4096
EMBED_DIM = 256
IMAGE_SIZE = 128

_WORD = re.compile(r"[a-z0-9']+")


def set_determinism(seed: int):
    """Seed every RNG and force deterministic kernels. Call before building
    the model or loader. Safe to call once per run in a fresh process, or
    repeatedly in one process (each call fully re-seeds)."""
    # cuBLAS reads this at first use; must be set before the first CUDA matmul
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def captions_to_bow(captions, vocab_size: int = VOCAB_SIZE):
    """Hash captions into a dense bag-of-words tensor [B, vocab_size].

    crc32 is stable across processes and platforms (unlike builtin `hash`,
    which is salted per interpreter), so the featurization itself is part of
    the determinism story. Dense multi-hot + Linear keeps the text tower free
    of EmbeddingBag/scatter ops that lack deterministic CUDA kernels.
    """
    import torch
    out = torch.zeros(len(captions), vocab_size)
    for i, text in enumerate(captions):
        for w in _WORD.findall(text.lower()):
            out[i, zlib.crc32(w.encode()) % vocab_size] += 1.0
    return out.log1p_()


def build_model(vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM):
    import torch
    from torch import nn

    class ImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            chans = [3, 32, 64, 128, 256]
            blocks = []
            for cin, cout in zip(chans, chans[1:]):
                blocks += [nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                           nn.GroupNorm(8, cout), nn.ReLU(inplace=True)]
            # 128 -> 8 spatial after four stride-2 convs; fixed AvgPool2d (not
            # adaptive — adaptive_avg_pool2d has no deterministic CUDA backward)
            self.net = nn.Sequential(*blocks, nn.AvgPool2d(IMAGE_SIZE // 16),
                                     nn.Flatten(), nn.Linear(chans[-1], embed_dim))

        def forward(self, x):
            return self.net(x)

    class TextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(vocab_size, 512), nn.ReLU(inplace=True),
                                     nn.Linear(512, embed_dim))

        def forward(self, x):
            return self.net(x)

    class MiniCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.image = ImageEncoder()
            self.text = TextEncoder()
            self.logit_scale = nn.Parameter(torch.tensor(2.6593))  # ln(1/0.07)

        def forward(self, images, bows):
            import torch.nn.functional as F
            zi = F.normalize(self.image(images), dim=-1)
            zt = F.normalize(self.text(bows), dim=-1)
            logits = self.logit_scale.exp() * (zi @ zt.t())
            # symmetric InfoNCE; positives on the diagonal. Written with
            # log_softmax + diagonal (not cross_entropy) — see module docstring.
            li = -F.log_softmax(logits, dim=1).diagonal().mean()
            lt = -F.log_softmax(logits, dim=0).diagonal().mean()
            return (li + lt) / 2

    return MiniCLIP()


def train_one_run(data_root: str, *, steps: int = 200, batch_size: int = 64,
                  lr: float = 1e-3, seed: int = 0, run_id: int = 0,
                  cache_dir: str | None = None) -> dict:
    """One full training run; returns the per-step loss curve plus an
    environment fingerprint. `run_id` labels the run — the *seed* is shared
    across runs on purpose, so their curves should be bit-identical."""
    set_determinism(seed)
    import ferroload
    import torch

    dl = ferroload.make_loader(
        data_root, batch_size=batch_size, images=["image"], meta=["caption"],
        resize=(IMAGE_SIZE, IMAGE_SIZE), out="torch",
        shuffle=True, seed=seed, drop_last=True, cache_dir=cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses, step, epoch = [], 0, 0
    while step < steps:
        dl.set_epoch(epoch)  # deterministic reshuffle, same for every run
        for batch in dl:
            x = batch["image"].permute(0, 3, 1, 2).float().div_(255).to(device)
            t = captions_to_bow(batch["caption"]).to(device)
            loss = model(x, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 25 == 0 or step == steps:
                print(f"[run {run_id}] step {step}/{steps} loss {losses[-1]:.6f}",
                      flush=True)
            if step >= steps:
                break
        epoch += 1

    return {
        "run_id": run_id,
        "losses": losses,
        "torch": torch.__version__,
        "device": (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
    }


def max_pairwise_diff(runs: list[dict]):
    """Per-step max |loss difference| across all run pairs, and its overall max."""
    curves = [r["losses"] for r in runs]
    n = min(len(c) for c in curves)
    per_step = [max(abs(a[i] - b[i]) for a in curves for b in curves)
                for i in range(n)]
    return per_step, (max(per_step) if per_step else 0.0)


# --- plot -------------------------------------------------------------------
# Palette validated with the standard six checks (light surface #fcfcfb).
_CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
_SURFACE, _INK, _INK2, _MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def render_plot(runs: list[dict], subtitle: str = "") -> bytes:
    """Overlaid per-step loss curves for all runs + a max-|Δ| panel underneath.
    Identical runs coincide exactly — differing linewidths/linestyles keep every
    run visible on top of the others. Returns PNG bytes."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_step, overall = max_pairwise_diff(runs)
    steps = range(len(per_step))

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1],
        layout="constrained", facecolor=_SURFACE)
    for a in (ax, axd):
        a.set_facecolor(_SURFACE)
        a.grid(axis="y", color=_GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(_GRID)
        a.tick_params(colors=_MUTED, labelsize=9)

    widths, styles = [2.6, 1.8, 1.1], ["-", "--", ":"]
    for i, r in enumerate(runs):
        ax.plot(range(len(r["losses"])), r["losses"],
                color=_CAT[i % len(_CAT)], linewidth=widths[i % 3],
                linestyle=styles[i % 3], label=f"run {r['run_id']}")
    ax.set_ylabel("InfoNCE loss", color=_INK2, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(_INK2)

    axd.plot(steps, per_step, color=_INK2, linewidth=1.6)
    axd.set_ylabel("max |Δloss|", color=_INK2, fontsize=10)
    axd.set_xlabel("training step", color=_INK2, fontsize=10)
    verdict = ("bitwise identical — max |Δ| = 0"
               if overall == 0.0 else f"max |Δ| = {overall:.3e}")
    axd.annotate(verdict, xy=(0.02, 0.82), xycoords="axes fraction",
                 color=_INK, fontsize=9)

    fig.suptitle(f"Ferroload loss determinism — {len(runs)} independent runs",
                 color=_INK, fontsize=13, fontweight="bold")
    if subtitle:
        ax.set_title(subtitle, color=_INK2, fontsize=9, loc="left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor=_SURFACE)
    plt.close(fig)
    return buf.getvalue()
