"""Create final architecture-only experiment plots from saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "accuracy_research" / "experiments" / "structured_ordering" / "runs"


def finish(ax, percent=True):
    ax.grid(axis="y", color="#dbe5ef", linewidth=0.8)
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.figsize": (11, 6), "font.size": 10, "axes.titleweight": "bold"})

    labels = [
        "Original\nbaseline",
        "Pointer\nassignment",
        "Word + char\npairwise",
        "Learned\nadjacency",
        "Candidate\nseed 42",
        "Candidate\nseed 43",
        "Candidate\nseed 44",
        "Frozen\nensemble",
    ]
    values = [
        0.5697469747,
        0.5407774111,
        0.5830583060,
        0.6468646865,
        0.6809680968,
        0.6963696370,
        0.6809680968,
        0.7370737074,
    ]
    colors = ["#718096", "#A0AEC0", "#63B3ED", "#4299E1", "#2B6CB0", "#234E8A", "#3182CE", "#00A38C"]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(values[0], color="#718096", linestyle="--", linewidth=1.3, label="Original baseline")
    ax.set(title="Validation exact-match accuracy by architecture", ylabel="Exact-match accuracy", ylim=(0.50, 0.77))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.2%}", ha="center", fontsize=9)
    ax.legend(frameon=False)
    finish(ax)
    fig.savefig(args.output_dir / "architecture_validation_comparison.png", dpi=180)
    plt.close(fig)

    histories = {
        "Candidate seed 42": RUNS / "candidate_sequence_flatten_v3" / "history.json",
        "Candidate seed 43": RUNS / "candidate_sequence_seed43_v4" / "history.json",
        "Candidate seed 44": RUNS / "candidate_sequence_seed44_v6" / "history.json",
    }
    fig, ax = plt.subplots()
    for label, path in histories.items():
        history = json.loads(path.read_text(encoding="utf-8"))
        ax.plot(
            [row["epoch"] for row in history],
            [row["exact_match_accuracy"] for row in history],
            linewidth=1.8,
            label=label,
        )
    ax.axhline(0.5697469747, color="#718096", linestyle="--", label="Original baseline")
    ax.set(title="Candidate-Transformer validation learning curves", xlabel="Epoch", ylabel="Exact-match accuracy", ylim=(0, 0.73))
    ax.legend(frameon=False)
    finish(ax)
    fig.savefig(args.output_dir / "candidate_training_curves.png", dpi=180)
    plt.close(fig)

    bands = ["Seen once", "Seen twice", "Seen 3–5", "Seen 6–10", "Seen >10"]
    band_accuracy = [0.68, 2 / 3, 0.7578475336, 0.7394366197, 0.7649769585]
    fig, ax = plt.subplots()
    bars = ax.bar(bands, band_accuracy, color=["#E07A5F", "#E89B6C", "#F2CC8F", "#81B29A", "#3D8D7A"])
    ax.set(
        title="Frozen ensemble accuracy by rarest word in each sentence",
        xlabel="Minimum number of appearances in unique training targets",
        ylabel="Exact-match accuracy",
        ylim=(0.60, 0.80),
    )
    for bar, value in zip(bars, band_accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.1%}", ha="center")
    finish(ax)
    fig.savefig(args.output_dir / "accuracy_by_training_word_frequency.png", dpi=180)
    plt.close(fig)

    categories = ["Validation exact", "Validation position", "Test exact", "Test position"]
    baseline = [0.5697469747, 0.7223065640, 30598 / 53790, 0.7303]
    final = [0.7370737074, 0.8400440044, 0.6826547685, 0.8065811489]
    x = range(len(categories))
    width = 0.36
    fig, ax = plt.subplots()
    ax.bar([value - width / 2 for value in x], baseline, width, label="Original", color="#94A3B8")
    ax.bar([value + width / 2 for value in x], final, width, label="Architecture-only final", color="#00A38C")
    ax.set_xticks(list(x), categories)
    ax.set(title="Original versus final reconstruction accuracy", ylabel="Accuracy", ylim=(0.50, 0.88))
    ax.legend(frameon=False)
    finish(ax)
    fig.savefig(args.output_dir / "baseline_vs_final.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
