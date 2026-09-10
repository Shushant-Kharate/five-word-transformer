"""Central configuration for the five-word reconstruction Transformer.

This file intentionally follows the structure of config.py from
https://github.com/hkproj/pytorch-transformer. Translation-specific settings
have been replaced with local dataset paths and compact model dimensions.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "permutations_4581_grouped_epoch7"


def get_config() -> dict:
    """Return the fixed initial experiment configuration."""
    return {
        # Data paths
        # Every permutation of a target remains in exactly one split.
        "train_file": str(
            PROJECT_ROOT / "data" / "permutations_4581" / "splits" / "train.jsonl"
        ),
        "validation_file": str(
            PROJECT_ROOT / "data" / "permutations_4581" / "splits" / "validation.jsonl"
        ),
        "test_file": str(
            PROJECT_ROOT / "data" / "permutations_4581" / "splits" / "test.jsonl"
        ),
        "tokenizer_source_files": [
            str(
                PROJECT_ROOT
                / "data"
                / "permutations_4581"
                / "splits"
                / "train.jsonl"
            ),
        ],
        "targets_file": str(PROJECT_ROOT / "data" / "targets.txt"),
        # Training settings
        "batch_size": 32,
        "validation_decode_batch_size": 32,
        "test_decode_batch_size": 32,
        # One augmented epoch contains 433,680 records (13,553 batches),
        # already more optimizer updates than the complete earlier run.
        "num_epochs": 20,
        "lr": 10**-4,
        "seed": 42,
        # This experiment is intentionally being extended to 20 total epochs.
        # A patience of 20 prevents early stopping before that requested limit.
        "early_stopping_patience": 20,
        "show_progress": False,
        "label_smoothing": 0.1,
        # Transformer settings
        "seq_len": 7,
        "d_model": 64,
        "N": 2,
        "h": 4,
        "dropout": 0.1,
        "d_ff": 256,
        # Saved artifacts
        "experiment_root": str(EXPERIMENT_ROOT),
        "manifest_file": str(EXPERIMENT_ROOT / "data_manifest.json"),
        "model_folder": str(EXPERIMENT_ROOT / "weights"),
        "model_basename": "tmodel_",
        # Continue from the most recently completed checkpoint rather than
        # restarting the model or hard-coding a particular epoch number.
        "preload": "latest",
        "tokenizer_file": str(EXPERIMENT_ROOT / "tokenizer.json"),
        "experiment_name": str(EXPERIMENT_ROOT / "runs" / "tmodel"),
        "training_history_file": str(EXPERIMENT_ROOT / "training_history.json"),
        "best_model_filename": "best_model.pt",
        "results_folder": str(EXPERIMENT_ROOT / "results"),
        "test_predictions_file": str(
            EXPERIMENT_ROOT / "results" / "test_predictions.jsonl"
        ),
        "test_summary_file": str(
            EXPERIMENT_ROOT / "results" / "test_summary.json"
        ),
    }


def get_weights_file_path(config: dict, epoch: str) -> str:
    """Return the checkpoint path for a specific epoch."""
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(config["model_folder"]) / model_filename)


def latest_weights_file_path(config: dict) -> str | None:
    """Return the latest epoch checkpoint, or None when none exists."""
    model_folder = Path(config["model_folder"])
    weights_files = list(model_folder.glob(f"{config['model_basename']}*.pt"))
    if not weights_files:
        return None
    weights_files.sort()
    return str(weights_files[-1])


def best_weights_file_path(config: dict) -> str:
    """Return the path reserved for the best-validation checkpoint."""
    return str(Path(config["model_folder"]) / config["best_model_filename"])
