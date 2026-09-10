"""Create diagnostic learning-curve graphs from saved epoch history."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from config import get_config


def main() -> None:
    config = get_config()
    history_path = Path(config["training_history_file"])
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history does not exist: {history_path}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError("Training history is empty")

    output_directory = Path(config["experiment_root"]) / "diagnostics"
    output_directory.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) + 1 for row in history]
    training_loss = [float(row["training_loss"]) for row in history]
    validation_loss = [float(row["validation_loss"]) for row in history]
    exact_accuracy = [float(row["exact_match_accuracy"]) for row in history]
    position_accuracy = [float(row["word_position_accuracy"]) for row in history]
    loss_gap = [validation - training for training, validation in zip(training_loss, validation_loss)]

    colors = {
        "navy": "#17324D",
        "teal": "#008C95",
        "orange": "#F28E2B",
        "red": "#D1495B",
        "grid": "#D9E2EC",
    }

    def finish(ax: plt.Axes) -> None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", color=colors["grid"], alpha=0.75)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, training_loss, marker="o", linewidth=2.5, color=colors["navy"], label="Training loss")
    ax.plot(epochs, validation_loss, marker="o", linewidth=2.5, color=colors["orange"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and validation loss versus epoch", weight="bold")
    finish(ax)
    fig.tight_layout()
    fig.savefig(output_directory / "loss_vs_epoch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, exact_accuracy, marker="o", linewidth=2.5, color=colors["teal"], label="Exact-match accuracy")
    ax.plot(epochs, position_accuracy, marker="o", linewidth=2.5, color=colors["navy"], label="Word-position accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1)
    ax.set_title("Validation accuracy versus epoch", weight="bold")
    finish(ax)
    fig.tight_layout()
    fig.savefig(output_directory / "validation_accuracy_vs_epoch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="#64748B", linewidth=1)
    ax.plot(epochs, loss_gap, marker="o", linewidth=2.5, color=colors["red"], label="Validation loss − training loss")
    ax.fill_between(epochs, 0, loss_gap, color=colors["red"], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generalization gap")
    ax.set_title("Generalization gap versus epoch", weight="bold")
    finish(ax)
    fig.tight_layout()
    fig.savefig(output_directory / "generalization_gap_vs_epoch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    best = max(
        history,
        key=lambda row: (
            float(row["exact_match_accuracy"]),
            -float(row["validation_loss"]),
        ),
    )
    latest = history[-1]
    summary = {
        "epochs_available": len(history),
        "best_epoch": int(best["epoch"]) + 1,
        "best_validation_exact_match_accuracy": float(best["exact_match_accuracy"]),
        "best_validation_word_position_accuracy": float(best["word_position_accuracy"]),
        "best_validation_loss": float(best["validation_loss"]),
        "latest_training_loss": float(latest["training_loss"]),
        "latest_validation_loss": float(latest["validation_loss"]),
        "latest_generalization_gap": float(latest["validation_loss"]) - float(latest["training_loss"]),
        "interpretation_guide": {
            "both_losses_decrease": "The architecture is learning the available patterns.",
            "training_decreases_validation_increases": "Overfitting signal; more epochs alone are unlikely to help, and more diverse base sentences or stronger regularization may be needed.",
            "both_losses_high_and_flat": "Underfitting signal; optimization settings or model capacity may be insufficient.",
            "important_limit": "Epoch curves suggest a failure mode but cannot alone prove whether data or architecture is the sole cause. A dataset-size learning-curve experiment is needed for stronger separation.",
        },
        "graphs": [
            "loss_vs_epoch.png",
            "validation_accuracy_vs_epoch.png",
            "generalization_gap_vs_epoch.png",
        ],
    }
    (output_directory / "training_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
