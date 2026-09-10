"""Train a permutation-aware Transformer without touching the protected test set."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "permutations_4581" / "splits" / "train.jsonl"
VALIDATION_PATH = ROOT / "data" / "permutations_4581" / "splits" / "validation.jsonl"
TOKENIZER_PATH = ROOT / "experiments" / "permutations_4581_grouped_epoch7" / "tokenizer.json"
CACHE_DIR = EXPERIMENT_DIR / "cache"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.10
    train_batch_size: int = 2048
    validation_batch_size: int = 4096
    epochs: int = 18
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    early_stopping_patience: int = 6
    num_workers: int = 0


class PointerAssignmentTransformer(nn.Module):
    def __init__(self, vocab_size: int, settings: Settings) -> None:
        super().__init__()
        d_model = settings.d_model
        self.scale = math.sqrt(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.source_position = nn.Embedding(5, d_model)
        self.output_queries = nn.Embedding(5, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=settings.nhead,
            dim_feedforward=settings.dim_feedforward,
            dropout=settings.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=settings.encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=settings.nhead,
            dim_feedforward=settings.dim_feedforward,
            dropout=settings.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=settings.decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.assignment_bias = nn.Parameter(torch.zeros(5, 5))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, source_ids: torch.Tensor) -> torch.Tensor:
        batch_size = source_ids.shape[0]
        positions = torch.arange(5, device=source_ids.device)
        source = self.token_embedding(source_ids) * self.scale
        source = source + self.source_position(positions).unsqueeze(0)
        memory = self.encoder(source)

        queries = self.output_queries(positions).unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        query = self.query_projection(decoded)
        key = self.key_projection(memory)
        return torch.einsum("bod,bsd->bos", query, key) / self.scale + self.assignment_bias


def load_vocabulary() -> dict[str, int]:
    tokenizer = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    vocabulary = tokenizer["model"]["vocab"]
    if not isinstance(vocabulary, dict):
        raise TypeError("Tokenizer vocabulary is not a word-to-id mapping")
    return {str(word): int(identifier) for word, identifier in vocabulary.items()}


def tensorize_split(path: Path, vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    source_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            source_words = record["input"].split()
            target_words = record["target"].split()
            if len(source_words) != 5 or len(target_words) != 5:
                raise ValueError(f"{path}:{line_number} is not a five-word record")
            try:
                source_rows.append([vocabulary[word] for word in source_words])
                target_rows.append([vocabulary[word] for word in target_words])
            except KeyError as error:
                raise ValueError(f"{path}:{line_number} contains out-of-vocabulary word {error}") from error
    return torch.tensor(source_rows, dtype=torch.long), torch.tensor(target_rows, dtype=torch.long)


def load_or_create_cache(name: str, path: Path, vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}_tensors.pt"
    source_stat = path.stat()
    fingerprint = {
        "source_path": str(path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_size": TOKENIZER_PATH.stat().st_size,
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") == fingerprint:
            return payload["source_ids"], payload["target_ids"]
    source_ids, target_ids = tensorize_split(path, vocabulary)
    torch.save(
        {"fingerprint": fingerprint, "source_ids": source_ids, "target_ids": target_ids},
        cache_path,
    )
    return source_ids, target_ids


def duplicate_aware_loss(logits: torch.Tensor, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Marginalize over source positions containing the correct target word."""
    log_probabilities = torch.log_softmax(logits, dim=-1)
    valid_positions = source_ids[:, None, :].eq(target_ids[:, :, None])
    masked = log_probabilities.masked_fill(~valid_positions, -torch.inf)
    return -torch.logsumexp(masked, dim=-1).mean()


def exhaustive_assign(logits: torch.Tensor, permutations: torch.Tensor) -> torch.Tensor:
    """Return the highest-scoring one-to-one output-to-source assignment."""
    batch_size = logits.shape[0]
    permutation_count = permutations.shape[0]
    expanded_logits = logits[:, None, :, :].expand(-1, permutation_count, -1, -1)
    indices = permutations[None, :, :, None].expand(batch_size, -1, -1, -1)
    scores = expanded_logits.gather(dim=3, index=indices).squeeze(-1).sum(dim=-1)
    return permutations[scores.argmax(dim=-1)]


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    permutations: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    records = 0
    exact_matches = 0
    correct_positions = 0
    for source_ids, target_ids in loader:
        source_ids = source_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(source_ids)
            loss = duplicate_aware_loss(logits, source_ids, target_ids)
        assignment = exhaustive_assign(logits.float(), permutations)
        predicted_ids = source_ids.gather(1, assignment)
        matches = predicted_ids.eq(target_ids)
        batch_size = source_ids.shape[0]
        total_loss += float(loss) * batch_size
        records += batch_size
        exact_matches += int(matches.all(dim=1).sum())
        correct_positions += int(matches.sum())
    return {
        "records": records,
        "validation_loss": total_loss / records,
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_matches / records,
        "word_position_accuracy": correct_positions / (records * 5),
        "output_validity_rate": 1.0,
        "input_word_preservation_rate": 1.0,
    }


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one short train/validation batch without saving a candidate")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    settings_values = asdict(Settings())
    for argument_name, setting_name in (
        ("epochs", "epochs"),
        ("train_batch_size", "train_batch_size"),
        ("validation_batch_size", "validation_batch_size"),
    ):
        value = getattr(arguments, argument_name)
        if value is not None:
            settings_values[setting_name] = value
    settings = Settings(**settings_values)
    set_reproducible_seed(settings.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the separately validated CUDA environment")
    device = torch.device("cuda")
    use_amp = torch.cuda.is_bf16_supported()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    vocabulary = load_vocabulary()
    source_train, target_train = load_or_create_cache("train", TRAIN_PATH, vocabulary)
    source_validation, target_validation = load_or_create_cache("validation", VALIDATION_PATH, vocabulary)
    print(
        f"data train={len(source_train):,} validation={len(source_validation):,} "
        f"vocab={len(vocabulary):,} device={torch.cuda.get_device_name(0)} amp_bf16={use_amp}",
        flush=True,
    )

    train_generator = torch.Generator().manual_seed(settings.seed)
    train_loader = DataLoader(
        TensorDataset(source_train, target_train),
        batch_size=settings.train_batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=settings.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    validation_loader = DataLoader(
        TensorDataset(source_validation, target_validation),
        batch_size=settings.validation_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = PointerAssignmentTransformer(len(vocabulary), settings).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.epochs,
        eta_min=settings.minimum_learning_rate,
    )
    permutations = torch.tensor(list(itertools.permutations(range(5))), dtype=torch.long, device=device)

    if arguments.smoke:
        train_loader = [next(iter(train_loader))]
        validation_loader = [next(iter(validation_loader))]

    history: list[dict[str, float | int | bool]] = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    run_started = time.perf_counter()

    for epoch in range(1, settings.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        running_loss = 0.0
        records_seen = 0
        for batch_index, (source_ids, target_ids) in enumerate(train_loader, start=1):
            source_ids = source_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(source_ids)
                loss = duplicate_aware_loss(logits, source_ids, target_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
            batch_size = source_ids.shape[0]
            running_loss += float(loss.detach()) * batch_size
            records_seen += batch_size
            if batch_index % 50 == 0:
                print(
                    f"epoch={epoch:02d} batch={batch_index:03d}/{len(train_loader):03d} "
                    f"train_loss={running_loss / records_seen:.5f}",
                    flush=True,
                )
        scheduler.step()

        metrics = evaluate(model, validation_loader, permutations, device, use_amp)
        accuracy = float(metrics["exact_match_accuracy"])
        improved = accuracy > best_accuracy
        if improved:
            best_accuracy = accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            if not arguments.smoke:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "settings": asdict(settings),
                        "vocab_size": len(vocabulary),
                        "parameter_count": parameter_count,
                        "epoch": epoch,
                        "validation_metrics": metrics,
                    },
                    ARTIFACT_DIR / "best_model.pt",
                )
        else:
            epochs_without_improvement += 1

        row: dict[str, float | int | bool] = {
            "epoch": epoch,
            "training_loss": running_loss / records_seen,
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "is_best": improved,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['training_loss']:.5f} "
            f"val_loss={metrics['validation_loss']:.5f} "
            f"exact={100.0 * accuracy:.4f}% position={100.0 * float(metrics['word_position_accuracy']):.4f}% "
            f"seconds={row['elapsed_seconds']:.1f} best_epoch={best_epoch}",
            flush=True,
        )
        if not arguments.smoke:
            (ARTIFACT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epochs_without_improvement >= settings.early_stopping_patience:
            print(f"early stopping after {epochs_without_improvement} unimproved epochs", flush=True)
            break

    summary = {
        "experiment": "pointer_assignment_v1",
        "status": "smoke" if arguments.smoke else "complete",
        "selection_split": "validation only",
        "protected_test_opened": False,
        "settings": asdict(settings),
        "parameter_count": parameter_count,
        "train_records": len(source_train),
        "validation_records": len(source_validation),
        "best_epoch": best_epoch,
        "best_validation_exact_accuracy": best_accuracy,
        "baseline_validation_exact_accuracy": 0.5697469746974697,
        "absolute_improvement": best_accuracy - 0.5697469746974697,
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not arguments.smoke:
        (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
