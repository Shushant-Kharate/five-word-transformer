"""Training and validation for the five-word reconstruction Transformer.

The forward/loss/backpropagation/checkpoint flow intentionally follows
``train.py`` from https://github.com/hkproj/pytorch-transformer. Translation
metrics are replaced with measurements for exact five-word reconstruction.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tqdm import tqdm

from config import (
    best_weights_file_path,
    get_config,
    get_weights_file_path,
    latest_weights_file_path,
)
from dataset import (
    SentenceReconstructionDataset,
    causal_mask,
    get_or_build_tokenizer,
    load_jsonl,
)
from model import Transformer, build_transformer


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select CUDA when available; otherwise use the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(config: dict, vocabulary_size: int) -> Transformer:
    """Build the configured reference-style Transformer."""
    return build_transformer(
        vocabulary_size,
        vocabulary_size,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config["N"],
        h=config["h"],
        dropout=config["dropout"],
        d_ff=config["d_ff"],
    )


def get_train_dataloader(
    config: dict, tokenizer: Tokenizer, epoch: int = 0
) -> DataLoader:
    """Return a reproducibly shuffled loader for a specific epoch."""
    # Load training only. Validation and especially the protected test split
    # are not touched while constructing an epoch's training loader.
    train_dataset = SentenceReconstructionDataset(
        load_jsonl(config["train_file"]), tokenizer, config["seq_len"]
    )
    generator = torch.Generator()
    generator.manual_seed(config["seed"] + epoch)
    return DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
    )


def get_validation_dataloaders(
    config: dict, tokenizer: Tokenizer
) -> tuple[DataLoader, DataLoader]:
    """Return loaders for batched validation loss and greedy decoding."""
    # Load validation only; the held-out test file remains unopened.
    validation_dataset = SentenceReconstructionDataset(
        load_jsonl(config["validation_file"]), tokenizer, config["seq_len"]
    )
    loss_dataloader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    decoding_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.get("validation_decode_batch_size", 1),
        shuffle=False,
    )
    return loss_dataloader, decoding_dataloader


def get_loss_function(
    config: dict, tokenizer: Tokenizer, device: torch.device
) -> nn.CrossEntropyLoss:
    """Create token cross-entropy loss while ignoring `[PAD]` labels."""
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        raise ValueError("Tokenizer is missing required token [PAD]")
    return nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=config["label_smoothing"],
    ).to(device)


def get_optimizer(config: dict, model: Transformer) -> Adam:
    """Create the Adam optimizer used by the reference implementation."""
    return Adam(model.parameters(), lr=config["lr"], eps=1e-9)


def calculate_batch_loss(
    model: Transformer,
    batch: dict,
    loss_function: nn.CrossEntropyLoss,
    vocabulary_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run encoder, decoder, and projection, then calculate token loss."""
    encoder_input = batch["encoder_input"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)
    label = batch["label"].to(device)

    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_output = model.decode(
        encoder_output,
        encoder_mask,
        decoder_input,
        decoder_mask,
    )
    projection_output = model.project(decoder_output)

    return loss_function(
        projection_output.view(-1, vocabulary_size),
        label.view(-1),
    )


def train_one_batch(
    model: Transformer,
    batch: dict,
    optimizer: Adam,
    loss_function: nn.CrossEntropyLoss,
    vocabulary_size: int,
    device: torch.device,
) -> float:
    """Perform one complete forward, backward, and optimizer-update step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = calculate_batch_loss(
        model,
        batch,
        loss_function,
        vocabulary_size,
        device,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def train_one_epoch(
    model: Transformer,
    train_dataloader: DataLoader,
    optimizer: Adam,
    loss_function: nn.CrossEntropyLoss,
    vocabulary_size: int,
    device: torch.device,
    epoch: int,
    global_step: int,
    writer: SummaryWriter | None = None,
    show_progress: bool = True,
) -> tuple[float, int]:
    """Train over every batch once and return average loss/global step."""
    batch_iterator: Iterable = tqdm(
        train_dataloader,
        desc=f"Processing Epoch {epoch:02d}",
        disable=not show_progress,
    )
    total_loss = 0.0
    example_count = 0
    batch_count = 0

    for batch in batch_iterator:
        loss_value = train_one_batch(
            model,
            batch,
            optimizer,
            loss_function,
            vocabulary_size,
            device,
        )
        current_batch_size = int(batch["encoder_input"].shape[0])
        total_loss += loss_value * current_batch_size
        example_count += current_batch_size
        batch_count += 1

        if show_progress:
            batch_iterator.set_postfix({"loss": f"{loss_value:6.3f}"})
        if writer is not None:
            writer.add_scalar("train/loss_step", loss_value, global_step)
        global_step += 1

    if batch_count == 0:
        raise ValueError("Training DataLoader produced no batches")

    average_loss = total_loss / example_count
    if writer is not None:
        writer.add_scalar("train/loss_epoch", average_loss, epoch)
        writer.flush()

    return average_loss, global_step


def save_checkpoint(
    checkpoint_path: str | Path,
    model: Transformer,
    optimizer: Adam,
    epoch: int,
    global_step: int,
    training_loss: float,
    validation_metrics: dict[str, float | int] | None = None,
    selection_state: dict[str, float | int] | None = None,
) -> None:
    """Save model, optimizer, losses, metrics, and selection state."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "training_loss": training_loss,
            "validation_metrics": validation_metrics,
            "selection_state": selection_state,
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: Transformer,
    optimizer: Adam,
    device: torch.device,
) -> tuple[int, int, float]:
    """Restore model/optimizer state and return next epoch, step, and loss."""
    state = load_training_checkpoint(checkpoint_path, model, optimizer, device)
    return (
        int(state["epoch"]) + 1,
        int(state["global_step"]),
        float(state["training_loss"]),
    )


def load_training_checkpoint(
    checkpoint_path: str | Path,
    model: Transformer,
    optimizer: Adam,
    device: torch.device,
) -> dict:
    """Restore a complete checkpoint and return its saved state."""
    state = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def greedy_decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a permutation of the source words with greedy decoding.

    The Transformer still scores the complete vocabulary at every word step,
    but selection is restricted to source-token IDs that have not yet been
    emitted. Counts are consumed rather than merely using a set, so repeated
    source words are preserved correctly. ``[EOS]`` is forced only after every
    source word has been used.
    """
    if source.size(0) != 1:
        raise ValueError("greedy_decode requires a batch size of 1")
    return greedy_decode_batch(
        model, source, source_mask, tokenizer, max_len, device
    ).squeeze(0)


def greedy_decode_batch(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Batched source-constrained greedy decoding.

    This is mathematically identical to calling ``greedy_decode`` for every
    record, but it shares Transformer forward passes across the batch.
    """

    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    if sos_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer is missing [SOS], [EOS], or [PAD]")

    # The encoder input is ``[SOS] source words [EOS]`` (and may contain
    # padding in a more general configuration). Counter preserves duplicate
    # words, unlike a vocabulary set or a simple attention mask.
    special_ids = {sos_id, eos_id, pad_id}
    remaining_by_record = [
        Counter(
            int(token_id)
            for token_id in row
            if int(token_id) not in special_ids
        )
        for row in source.detach().cpu().tolist()
    ]
    source_word_counts = [sum(counter.values()) for counter in remaining_by_record]
    if not source_word_counts or min(source_word_counts) == 0:
        raise ValueError("Source contains no word tokens to reconstruct")
    if len(set(source_word_counts)) != 1:
        raise ValueError("All records in a decoding batch must have equal word counts")
    required_length = source_word_counts[0] + 2  # [SOS], words, [EOS]
    if max_len < required_length:
        raise ValueError(
            f"max_len must be at least {required_length} for "
            f"{source_word_counts[0]} source words"
        )

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full(
        (source.size(0), 1), sos_id, dtype=source.dtype, device=device
    )

    while decoder_input.size(1) < max_len:
        if any(remaining_by_record):
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(
                source_mask
            ).to(device)
            decoder_output = model.decode(
                encoder_output,
                source_mask,
                decoder_input,
                decoder_mask,
            )
            logits = model.project(decoder_output[:, -1])

        next_token_ids: list[int] = []
        for row_index, remaining in enumerate(remaining_by_record):
            if not remaining:
                next_token_ids.append(eos_id)
                continue
            allowed_ids = torch.tensor(
                list(remaining.keys()), dtype=torch.long, device=logits.device
            )
            allowed_logits = logits[row_index].index_select(0, allowed_ids)
            next_token_id = int(allowed_ids[torch.argmax(allowed_logits)].item())
            next_token_ids.append(next_token_id)
            remaining[next_token_id] -= 1
            if remaining[next_token_id] == 0:
                del remaining[next_token_id]

        next_token = torch.tensor(
            next_token_ids, dtype=source.dtype, device=device
        ).unsqueeze(1)
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if all(token_id == eos_id for token_id in next_token_ids):
            break

    return decoder_input


def decoded_token_ids_to_words(
    token_ids: torch.Tensor, tokenizer: Tokenizer
) -> tuple[list[str], bool]:
    """Convert generated IDs to words and report whether `[EOS]` occurred."""
    words: list[str] = []
    ended_with_eos = False

    for token_id in token_ids.detach().cpu().tolist():
        token = tokenizer.id_to_token(int(token_id))
        if token is None:
            token = "[UNK]"
        if token == "[EOS]":
            ended_with_eos = True
            break
        if token in {"[SOS]", "[PAD]"}:
            continue
        words.append(token)

    return words, ended_with_eos


def compute_reconstruction_metrics(
    sources: list[list[str]],
    targets: list[list[str]],
    predictions: list[list[str]],
    ended_with_eos: list[bool],
) -> dict[str, float | int]:
    """Calculate metrics that directly measure five-word reconstruction."""
    lengths = {
        len(sources), len(targets), len(predictions), len(ended_with_eos)
    }
    if len(lengths) != 1:
        raise ValueError("Metric inputs must contain the same number of records")
    total_records = len(targets)
    if total_records == 0:
        raise ValueError("Cannot calculate metrics for an empty validation set")

    exact_matches = 0
    correct_positions = 0
    target_positions = 0
    valid_outputs = 0
    preserved_inputs = 0

    for source, target, prediction, has_eos in zip(
        sources, targets, predictions, ended_with_eos
    ):
        if prediction == target and has_eos:
            exact_matches += 1
        correct_positions += sum(
            index < len(prediction) and prediction[index] == target_word
            for index, target_word in enumerate(target)
        )
        target_positions += len(target)
        if len(prediction) == 5 and has_eos:
            valid_outputs += 1
        if Counter(prediction) == Counter(source):
            preserved_inputs += 1

    return {
        "records": total_records,
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_matches / total_records,
        "word_position_accuracy": correct_positions / target_positions,
        "output_validity_rate": valid_outputs / total_records,
        "input_word_preservation_rate": preserved_inputs / total_records,
    }


def calculate_validation_loss(
    model: Transformer,
    validation_dataloader: DataLoader,
    loss_function: nn.CrossEntropyLoss,
    vocabulary_size: int,
    device: torch.device,
) -> float:
    """Calculate record-weighted validation loss without changing weights."""
    model.eval()
    total_loss = 0.0
    record_count = 0

    with torch.no_grad():
        for batch in validation_dataloader:
            loss = calculate_batch_loss(
                model,
                batch,
                loss_function,
                vocabulary_size,
                device,
            )
            current_batch_size = int(batch["encoder_input"].shape[0])
            total_loss += float(loss.item()) * current_batch_size
            record_count += current_batch_size

    if record_count == 0:
        raise ValueError("Validation DataLoader produced no records")
    return total_loss / record_count


def run_validation(
    model: Transformer,
    validation_loss_dataloader: DataLoader,
    validation_decoding_dataloader: DataLoader,
    tokenizer: Tokenizer,
    loss_function: nn.CrossEntropyLoss,
    device: torch.device,
    max_len: int,
    epoch: int,
    writer: SummaryWriter | None = None,
    num_examples: int = 2,
) -> dict:
    """Run validation loss, greedy decoding, metrics, and example capture."""
    model.eval()
    validation_loss = calculate_validation_loss(
        model,
        validation_loss_dataloader,
        loss_function,
        tokenizer.get_vocab_size(),
        device,
    )

    source_words: list[list[str]] = []
    target_words: list[list[str]] = []
    predicted_words: list[list[str]] = []
    eos_flags: list[bool] = []
    examples: list[dict[str, str | bool]] = []

    with torch.no_grad():
        for batch in validation_decoding_dataloader:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            generated_batch = greedy_decode_batch(
                model,
                encoder_input,
                encoder_mask,
                tokenizer,
                max_len,
                device,
            )
            for row_index, generated_ids in enumerate(generated_batch):
                prediction, has_eos = decoded_token_ids_to_words(
                    generated_ids, tokenizer
                )
                source = batch["src_text"][row_index].split()
                target = batch["tgt_text"][row_index].split()

                source_words.append(source)
                target_words.append(target)
                predicted_words.append(prediction)
                eos_flags.append(has_eos)

                if len(examples) < num_examples:
                    examples.append(
                        {
                            "source": " ".join(source),
                            "target": " ".join(target),
                            "prediction": " ".join(prediction),
                            "ended_with_eos": has_eos,
                        }
                    )

    metrics = compute_reconstruction_metrics(
        source_words,
        target_words,
        predicted_words,
        eos_flags,
    )
    result: dict = {
        "validation_loss": validation_loss,
        **metrics,
        "examples": examples,
    }

    if writer is not None:
        for name in (
            "validation_loss",
            "exact_match_accuracy",
            "word_position_accuracy",
            "output_validity_rate",
            "input_word_preservation_rate",
        ):
            writer.add_scalar(f"validation/{name}", result[name], epoch)
        writer.flush()

    return result


def is_better_validation_result(
    exact_match_accuracy: float,
    validation_loss: float,
    best_exact_match_accuracy: float,
    best_validation_loss: float,
) -> bool:
    """Prefer exact match; use lower validation loss to break a tie."""
    return exact_match_accuracy > best_exact_match_accuracy or (
        exact_match_accuracy == best_exact_match_accuracy
        and validation_loss < best_validation_loss
    )


def build_training_components(
    config: dict,
) -> tuple[
    Tokenizer,
    DataLoader,
    Transformer,
    Adam,
    nn.CrossEntropyLoss,
    torch.device,
]:
    """Construct training components without starting any epochs."""
    set_seed(config["seed"])
    device = get_device()
    tokenizer = get_or_build_tokenizer(config)
    train_dataloader = get_train_dataloader(config, tokenizer)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    optimizer = get_optimizer(config, model)
    loss_function = get_loss_function(config, tokenizer, device)
    return (
        tokenizer,
        train_dataloader,
        model,
        optimizer,
        loss_function,
        device,
    )


def train_model(config: dict) -> list[dict]:
    """Train with validation, best-checkpoint selection, and early stopping."""
    (
        tokenizer,
        train_dataloader,
        model,
        optimizer,
        loss_function,
        device,
    ) = build_training_components(config)
    validation_loss_loader, validation_decode_loader = (
        get_validation_dataloaders(config, tokenizer)
    )

    initial_epoch = 0
    global_step = 0
    best_exact_match_accuracy = -1.0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    preload = config.get("preload")
    if preload:
        if preload == "latest":
            checkpoint_path = latest_weights_file_path(config)
        else:
            supplied_path = Path(str(preload))
            checkpoint_path = (
                str(supplied_path)
                if supplied_path.exists()
                else get_weights_file_path(config, str(preload))
            )
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found for preload={preload}")

        state = load_training_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        initial_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        selection_state = state.get("selection_state") or {}
        best_exact_match_accuracy = float(
            selection_state.get("best_exact_match_accuracy", -1.0)
        )
        best_validation_loss = float(
            selection_state.get("best_validation_loss", float("inf"))
        )
        epochs_without_improvement = int(
            selection_state.get("epochs_without_improvement", 0)
        )

    history: list[dict] = []
    history_path = Path(config["training_history_file"])
    if preload and history_path.is_file():
        loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_history, list):
            raise ValueError("Existing training history must be a JSON list")
        if loaded_history and int(loaded_history[-1]["epoch"]) >= initial_epoch:
            raise ValueError(
                "Training history already contains the epoch that would be resumed"
            )
        history = loaded_history
    writer = SummaryWriter(config["experiment_name"])

    try:
        for epoch in range(initial_epoch, config["num_epochs"]):
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Recreate the shuffle from seed + epoch so resumed training uses
            # the same order it would have used in an uninterrupted run.
            train_dataloader = get_train_dataloader(
                config, tokenizer, epoch=epoch
            )
            training_loss, global_step = train_one_epoch(
                model,
                train_dataloader,
                optimizer,
                loss_function,
                tokenizer.get_vocab_size(),
                device,
                epoch,
                global_step,
                writer=writer,
                show_progress=config.get("show_progress", True),
            )
            validation_result = run_validation(
                model,
                validation_loss_loader,
                validation_decode_loader,
                tokenizer,
                loss_function,
                device,
                config["seq_len"],
                epoch,
                writer=writer,
            )

            improved = is_better_validation_result(
                float(validation_result["exact_match_accuracy"]),
                float(validation_result["validation_loss"]),
                best_exact_match_accuracy,
                best_validation_loss,
            )
            if improved:
                best_exact_match_accuracy = float(
                    validation_result["exact_match_accuracy"]
                )
                best_validation_loss = float(
                    validation_result["validation_loss"]
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            numeric_validation_metrics = {
                key: value
                for key, value in validation_result.items()
                if key != "examples"
            }
            selection_state = {
                "best_exact_match_accuracy": best_exact_match_accuracy,
                "best_validation_loss": best_validation_loss,
                "epochs_without_improvement": epochs_without_improvement,
            }

            save_checkpoint(
                get_weights_file_path(config, f"{epoch:02d}"),
                model,
                optimizer,
                epoch,
                global_step,
                training_loss,
                numeric_validation_metrics,
                selection_state,
            )
            if improved:
                save_checkpoint(
                    best_weights_file_path(config),
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    training_loss,
                    numeric_validation_metrics,
                    selection_state,
                )

            epoch_summary = {
                "epoch": epoch,
                "training_loss": training_loss,
                **validation_result,
                "is_best": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
            history.append(epoch_summary)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(history, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"Epoch {epoch:02d} | train loss {training_loss:.4f} | "
                f"validation loss {validation_result['validation_loss']:.4f} | "
                f"exact match {validation_result['exact_match_accuracy']:.4f}"
            )
            for example in validation_result["examples"]:
                print(f"  SOURCE:    {example['source']}")
                print(f"  TARGET:    {example['target']}")
                print(f"  PREDICTED: {example['prediction']}")

            if epochs_without_improvement >= config[
                "early_stopping_patience"
            ]:
                print(
                    "Early stopping: validation did not improve for "
                    f"{config['early_stopping_patience']} epochs."
                )
                break
    finally:
        writer.close()

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the five-word sentence reconstruction Transformer"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="start the real multi-epoch training experiment",
    )
    arguments = parser.parse_args()
    if arguments.train:
        train_model(get_config())
    else:
        print(
            "Validation-enabled training is ready. Run `python train.py "
            "--train` only when you intentionally want to start the real "
            "experiment."
        )
