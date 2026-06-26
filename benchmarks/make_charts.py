"""Generate the benchmark bar charts (PNG) embedded in BENCHMARKS.md / README.

Numbers are the measured results (see REPORT.md / results/*.json). Run:
    python make_charts.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT, exist_ok=True)

FERRO = "#2563eb"   # blue
FERRO2 = "#93c5fd"  # light blue (before / variant)
WDS = "#ea580c"     # orange
HF = "#16a34a"      # green
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 140})


def _labels(ax, bars, fmt="{:.0f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8.5,
                    xytext=(0, 1.5), textcoords="offset points")


def chart_throughput():
    datasets = ["CIFAR-10\n(32px PNG)", "Stanford-Cars\n(JPEG→224)", "FFHQ-256\n(JPEG→224)"]
    hf = [95879, 6452, 6010]          # best num_workers
    wds = [37454, 8930, 8698]         # best num_workers
    ferro = [129051, 6356, 6139]      # native (turbojpeg on JPEG sets)
    x = np.arange(len(datasets)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    b1 = ax.bar(x - w, hf, w, label="HF datasets / Arrow (best workers)", color=HF)
    b2 = ax.bar(x, wds, w, label="WebDataset (best workers)", color=WDS)
    b3 = ax.bar(x + w, ferro, w, label="Ferroload (native, 1 process)", color=FERRO)
    ax.set_yscale("log")
    ax.set_ylabel("samples / s  (log scale, higher = better)")
    ax.set_title("Local data-loading throughput — best config per loader")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.legend(fontsize=9, loc="upper right")
    for b in (b1, b2, b3):
        _labels(ax, b)
    ax.set_ylim(top=max(ferro) * 2.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/throughput.png"); plt.close(fig)


def chart_storage():
    datasets = ["CIFAR-10", "Stanford-Cars", "FFHQ-256"]
    hf = [114.4, 146.9, 179.5]
    wds = [304.7, 178.9, 217.8]
    ferro = [154.2, 153.5, 187.6]
    x = np.arange(len(datasets)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    b1 = ax.bar(x - w, hf, w, label="HF Arrow", color=HF)
    b2 = ax.bar(x, wds, w, label="WebDataset", color=WDS)
    b3 = ax.bar(x + w, ferro, w, label="Ferroload", color=FERRO)
    ax.set_ylabel("on-disk size (MB, lower = better)")
    ax.set_title("Storage footprint per format (same encoded images)")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.legend(fontsize=9)
    for b in (b1, b2, b3):
        _labels(ax, b, "{:.0f}")
    fig.tight_layout(); fig.savefig(f"{OUT}/storage.png"); plt.close(fig)


def chart_gcs():
    labels = ["Ferroload\n(before fix)", "WebDataset", "Ferroload\n(after fix)"]
    vals = [113.6, 510.0, 1008.3]
    colors = [FERRO2, WDS, FERRO]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_ylabel("samples / s  (higher = better)")
    ax.set_title("GCS streaming throughput (FFHQ-256, single client)")
    _labels(ax, bars, "{:.0f}")
    ax.annotate("8.9× after coalescing\nremote reads", (2, 1008.3),
                ha="center", va="bottom", fontsize=9, color=FERRO,
                xytext=(0, 16), textcoords="offset points")
    ax.set_ylim(top=1250)
    fig.tight_layout(); fig.savefig(f"{OUT}/gcs_streaming.png"); plt.close(fig)


def chart_jpeg():
    datasets = ["Stanford-Cars", "FFHQ-256"]
    ferro_zune = [5359, 5223]
    ferro_turbo = [6356, 6139]
    hf_best = [6452, 6010]
    wds_best = [8930, 8698]
    x = np.arange(len(datasets)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - 1.5 * w, ferro_zune, w, label="Ferroload (zune-jpeg)", color=FERRO2)
    b2 = ax.bar(x - 0.5 * w, ferro_turbo, w, label="Ferroload (turbojpeg)", color=FERRO)
    ax.bar(x + 0.5 * w, hf_best, w, label="HF Arrow (best)", color=HF)
    ax.bar(x + 1.5 * w, wds_best, w, label="WebDataset (best)", color=WDS)
    ax.set_ylabel("samples / s  (higher = better)")
    ax.set_title("JPEG decode-bound throughput (1 process for Ferroload)")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.legend(fontsize=9)
    _labels(ax, b2, "{:.0f}")
    fig.tight_layout(); fig.savefig(f"{OUT}/jpeg_decode.png"); plt.close(fig)


if __name__ == "__main__":
    chart_throughput(); chart_storage(); chart_gcs(); chart_jpeg()
    print("wrote charts to", OUT, ":", sorted(os.listdir(OUT)))
