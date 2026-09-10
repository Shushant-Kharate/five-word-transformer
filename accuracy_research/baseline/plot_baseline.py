"""Create authoritative plots from the full 20-epoch baseline history."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = ROOT / "experiments" / "permutations_4581_grouped_epoch7" / "training_history.json"
ERROR_PATH = ROOT / "accuracy_research" / "analysis" / "error_analysis.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "graphs"


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.25, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    error = json.loads(ERROR_PATH.read_text(encoding="utf-8"))

    epochs = [row["epoch"] + 1 for row in history]
    train_loss = [row["training_loss"] for row in history]
    val_loss = [row["validation_loss"] for row in history]
    exact = [100.0 * row["exact_match_accuracy"] for row in history]
    position = [100.0 * row["word_position_accuracy"] for row in history]
    best_index = max(range(len(history)), key=lambda i: history[i]["exact_match_accuracy"])

    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    axis.plot(epochs, train_loss, marker="o", markersize=3.5, label="Training token loss")
    axis.plot(epochs, val_loss, marker="o", markersize=3.5, label="Validation token loss")
    axis.axvline(epochs[best_index], color="#d62728", linestyle="--", alpha=0.75, label="Best exact match (epoch 15)")
    axis.set(title="Baseline loss across all 20 epochs", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.set_xticks(epochs)
    axis.legend(frameon=False)
    style_axis(axis)
    figure.savefig(OUTPUT_DIR / "baseline_loss_20_epochs.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    axis.plot(epochs, exact, marker="o", markersize=3.5, label="Exact sentence match")
    axis.plot(epochs, position, marker="o", markersize=3.5, label="Word-position accuracy")
    axis.scatter([epochs[best_index]], [exact[best_index]], color="#d62728", zorder=5)
    axis.annotate(
        f"best: {exact[best_index]:.2f}%",
        xy=(epochs[best_index], exact[best_index]),
        xytext=(epochs[best_index] - 5.5, exact[best_index] - 5.5),
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
    )
    axis.set(title="Baseline validation accuracy across all 20 epochs", xlabel="Epoch", ylabel="Accuracy (%)")
    axis.set_xticks(epochs)
    axis.set_ylim(40, 80)
    axis.legend(frameon=False)
    style_axis(axis)
    figure.savefig(OUTPUT_DIR / "baseline_accuracy_20_epochs.png", dpi=180)
    plt.close(figure)

    frequency = error["minimum_training_word_frequency_bins"]
    labels = list(frequency)
    values = [100.0 * frequency[label]["record_weighted_exact_accuracy"] for label in labels]
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    bars = axis.bar(labels, values, color=["#c44e52", "#dd8452", "#55a868", "#4c72b0"])
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set(title="Validation exact match by rarest target-word frequency", xlabel="Minimum training-sentence frequency", ylabel="Exact match (%)")
    axis.set_ylim(0, max(values) + 12)
    style_axis(axis)
    figure.savefig(OUTPUT_DIR / "baseline_exact_by_word_frequency.png", dpi=180)
    plt.close(figure)

    hamming = error["error_distance"]["hamming_wrong_positions"]
    distance_labels = list(hamming)
    counts = [hamming[key] for key in distance_labels]
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    bars = axis.bar(distance_labels, counts, color="#4c72b0")
    axis.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3, fontsize=9)
    axis.set(title="Baseline validation predictions by wrong word positions", xlabel="Wrong positions in the five-word output", ylabel="Validation records")
    style_axis(axis)
    figure.savefig(OUTPUT_DIR / "baseline_hamming_distribution.png", dpi=180)
    plt.close(figure)

    summary = {
        "history_source": str(HISTORY_PATH),
        "epochs_plotted": len(history),
        "best_epoch": epochs[best_index],
        "best_validation_exact_accuracy": history[best_index]["exact_match_accuracy"],
        "output_files": sorted(path.name for path in OUTPUT_DIR.glob("*.png")),
    }
    (OUTPUT_DIR / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
