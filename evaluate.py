"""Protected final-test evaluation for sentence reconstruction.

This module loads only the best validation checkpoint, evaluates the held-out
test split without gradients, and writes one prediction file plus one summary.
It refuses to overwrite existing final results.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from config import best_weights_file_path, get_config
from dataset import SentenceReconstructionDataset, get_or_build_tokenizer, load_jsonl
from model import Transformer
from train import (
    calculate_validation_loss,
    compute_reconstruction_metrics,
    decoded_token_ids_to_words,
    get_loss_function,
    get_model,
    greedy_decode,
    greedy_decode_batch,
    set_seed,
)


def get_test_dataloaders(
    config: dict, tokenizer: Tokenizer
) -> tuple[DataLoader, DataLoader]:
    """Return test loaders for batched loss and batch-one generation."""
    test_dataset = SentenceReconstructionDataset(
        load_jsonl(config["test_file"]), tokenizer, config["seq_len"]
    )
    return (
        DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
        ),
        DataLoader(
            test_dataset,
            batch_size=config.get("test_decode_batch_size", 1),
            shuffle=False,
        ),
    )


def load_model_for_evaluation(
    config: dict,
    tokenizer: Tokenizer,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Transformer, dict]:
    """Load model weights and metadata from the selected checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Best validation checkpoint does not exist: {path}. "
            "Train the model before final test evaluation."
        )

    state = torch.load(path, map_location=device)
    if "model_state_dict" not in state:
        raise ValueError(f"Checkpoint has no model_state_dict: {path}")

    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


def build_prediction_record(
    source: list[str],
    target: list[str],
    prediction: list[str],
    ended_with_eos: bool,
) -> dict[str, str | bool | int]:
    """Build one auditable per-sentence final-evaluation record."""
    correct_positions = sum(
        index < len(prediction) and prediction[index] == target_word
        for index, target_word in enumerate(target)
    )
    return {
        "input": " ".join(source),
        "target": " ".join(target),
        "prediction": " ".join(prediction),
        "ended_with_eos": ended_with_eos,
        "exact_match": prediction == target and ended_with_eos,
        "correct_word_positions": correct_positions,
        "output_valid": len(prediction) == 5 and ended_with_eos,
        "input_words_preserved": Counter(prediction) == Counter(source),
    }


def evaluate_model(
    model: Transformer,
    loss_dataloader: DataLoader,
    decoding_dataloader: DataLoader,
    tokenizer: Tokenizer,
    loss_function: torch.nn.CrossEntropyLoss,
    device: torch.device,
    max_len: int,
) -> tuple[dict[str, float | int], list[dict]]:
    """Evaluate a model without gradients and return metrics/predictions."""
    model.eval()
    test_loss = calculate_validation_loss(
        model,
        loss_dataloader,
        loss_function,
        tokenizer.get_vocab_size(),
        device,
    )

    sources: list[list[str]] = []
    targets: list[list[str]] = []
    predictions: list[list[str]] = []
    eos_flags: list[bool] = []
    prediction_records: list[dict] = []

    with torch.no_grad():
        for batch in decoding_dataloader:
            generated_batch = greedy_decode_batch(
                model,
                batch["encoder_input"].to(device),
                batch["encoder_mask"].to(device),
                tokenizer,
                max_len,
                device,
            )
            for row_index, generated_ids in enumerate(generated_batch):
                prediction, ended_with_eos = decoded_token_ids_to_words(
                    generated_ids, tokenizer
                )
                source = batch["src_text"][row_index].split()
                target = batch["tgt_text"][row_index].split()

                sources.append(source)
                targets.append(target)
                predictions.append(prediction)
                eos_flags.append(ended_with_eos)
                prediction_records.append(
                    build_prediction_record(
                        source, target, prediction, ended_with_eos
                    )
                )

    metrics = compute_reconstruction_metrics(
        sources, targets, predictions, eos_flags
    )
    return {"test_loss": test_loss, **metrics}, prediction_records


def write_evaluation_outputs(
    predictions_path: str | Path,
    summary_path: str | Path,
    prediction_records: list[dict],
    summary: dict,
) -> None:
    """Write final outputs once and refuse to replace existing results."""
    predictions_file = Path(predictions_path)
    summary_file = Path(summary_path)
    existing = [
        str(path) for path in (predictions_file, summary_file) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Final test output already exists and will not be overwritten: "
            + ", ".join(existing)
        )

    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in prediction_records
        )
        + "\n",
        encoding="utf-8",
    )
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_test_evaluation(config: dict) -> tuple[dict, list[dict]]:
    """Load the best model, evaluate the fixed test split once, and save."""
    predictions_path = Path(config["test_predictions_file"])
    summary_path = Path(config["test_summary_file"])
    if predictions_path.exists() or summary_path.exists():
        raise FileExistsError(
            "Final test outputs already exist. They are protected from "
            "accidental repeated evaluation."
        )

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_or_build_tokenizer(config)
    checkpoint_path = best_weights_file_path(config)
    model, checkpoint_state = load_model_for_evaluation(
        config, tokenizer, checkpoint_path, device
    )
    loss_dataloader, decoding_dataloader = get_test_dataloaders(
        config, tokenizer
    )
    loss_function = get_loss_function(config, tokenizer, device)
    metrics, prediction_records = evaluate_model(
        model,
        loss_dataloader,
        decoding_dataloader,
        tokenizer,
        loss_function,
        device,
        config["seq_len"],
    )

    summary = {
        "evaluation_type": "final held-out test evaluation",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_state.get("epoch"),
        "checkpoint_global_step": checkpoint_state.get("global_step"),
        "checkpoint_training_loss": checkpoint_state.get("training_loss"),
        "checkpoint_validation_metrics": checkpoint_state.get(
            "validation_metrics"
        ),
        "test_file": config["test_file"],
        "configuration": {
            "batch_size": config["batch_size"],
            "seq_len": config["seq_len"],
            "d_model": config["d_model"],
            "layers": config["N"],
            "heads": config["h"],
            "d_ff": config["d_ff"],
            "dropout": config["dropout"],
            "seed": config["seed"],
        },
        "metrics": metrics,
    }
    write_evaluation_outputs(
        predictions_path,
        summary_path,
        prediction_records,
        summary,
    )
    return summary, prediction_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the one-time final test evaluation"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate test_fixed.jsonl using weights/best_model.pt",
    )
    arguments = parser.parse_args()
    if arguments.evaluate:
        final_summary, _ = run_test_evaluation(get_config())
        print(json.dumps(final_summary["metrics"], indent=2))
    else:
        print(
            "Final test evaluation was not run. Use `python evaluate.py "
            "--evaluate` only after training has produced best_model.pt."
        )
