"""Train the architecture-only 20,500-target ordering ensemble from scratch.

This file intentionally has no automatic test evaluation.  Model selection is
validation-only; the protected test is a separate, explicit final operation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREPARED = ROOT / "data" / "extended_4581" / "prepared_20500"
TRAIN_PATH = PREPARED / "splits" / "train.jsonl"
VALIDATION_PATH = PREPARED / "splits" / "validation.jsonl"
TEST_PATH = PREPARED / "splits" / "test.jsonl"
TOKENIZER_PATH = PREPARED / "tokenizer.json"
LEGACY_COMPONENTS = HERE.parent / "structured_ordering"
sys.path.insert(0, str(LEGACY_COMPONENTS))
base = importlib.import_module("train")
candidate = importlib.import_module("candidate_sequence_train")


class DynamicPermutationDataset(Dataset):
    """Generate a deterministic fresh source order for each epoch and record."""

    def __init__(self, source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, seed: int):
        self.source = source
        self.target = target
        self.weights = weights
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * len(self.source) + index
        )
        order = torch.randperm(5, generator=generator)
        return self.source[index, order], self.target[index], self.weights[index]


def load_vocabulary() -> tuple[dict[str, int], dict[int, str]]:
    tokenizer = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    word_to_id = {
        str(word): int(identifier)
        for word, identifier in tokenizer["model"]["vocab"].items()
    }
    return word_to_id, {identifier: word for word, identifier in word_to_id.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_scheduler(optimizer, epochs: int, warmup_epochs: int, steps: int, start: float, floor: float):
    total_steps = epochs * steps
    warmup_steps = warmup_epochs * steps

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return floor / start + (1.0 - floor / start) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("candidate", "structured"), default="candidate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run; a new run is created when the directory is absent.",
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("run prepare_data.py once before training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    word_to_id, id_to_word = load_vocabulary()
    train_source, train_target, train_weights = base.load_unique_groups(TRAIN_PATH, word_to_id)
    validation_source, validation_target, validation_weights = base.load_unique_groups(
        VALIDATION_PATH, word_to_id
    )

    if args.component == "candidate":
        values = asdict(candidate.Settings())
        values["seed"] = args.seed
        settings = candidate.Settings(**values)
        model = candidate.CandidateSequenceTransformer(
            len(word_to_id),
            base.build_character_table(id_to_word),
            base.build_structural_prior_tables(train_target, len(word_to_id)),
            settings,
        )
        evaluate = candidate.evaluate
    else:
        values = asdict(base.Settings())
        values.update(seed=args.seed, use_adjacency=True, use_structural_priors=True)
        settings = base.Settings(**values)
        model = base.StructuredOrderingTransformer(
            len(word_to_id),
            base.build_character_table(id_to_word),
            settings,
            base.build_structural_prior_tables(train_target, len(word_to_id)),
        )
        evaluate = base.evaluate

    dynamic_train = DynamicPermutationDataset(
        train_source, train_target, train_weights, args.seed
    )
    train_loader = DataLoader(
        dynamic_train,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(validation_source, validation_target, validation_weights),
        batch_size=settings.validation_batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = model.to(device)
    permutations = base.PERMUTATIONS.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = cosine_warmup_scheduler(
        optimizer,
        settings.epochs,
        settings.warmup_epochs,
        len(train_loader),
        settings.learning_rate,
        settings.minimum_learning_rate,
    )
    run_dir = HERE / "runs" / args.name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"run already exists: {run_dir}; pass --resume to preserve and continue it"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_position = -1.0
    stale_epochs = 0
    history = []
    start_epoch = 1
    resume_mode = "new"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    best_checkpoint_path = run_dir / "best_model.pt"

    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["component"] != args.component or int(checkpoint["seed"]) != args.seed:
            raise ValueError("resume checkpoint component or seed does not match this task")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = list(checkpoint["history"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_position = float(checkpoint["best_position"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        resume_mode = "exact_last_checkpoint"
    elif args.resume and best_checkpoint_path.exists():
        # Checkpoints created before exact-resume support contain the best model
        # but not AdamW moments. Recover from that best epoch without discarding it.
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("component", args.component) != args.component:
            raise ValueError("best checkpoint component does not match this task")
        model.load_state_dict(checkpoint["model_state_dict"])
        recovered_epoch = int(checkpoint["epoch"])
        metrics = checkpoint["validation_metrics"]
        best_accuracy = float(metrics["exact_match_accuracy"])
        best_position = float(metrics["word_position_accuracy"])
        history_path = run_dir / "history.json"
        if history_path.exists():
            saved_history = json.loads(history_path.read_text(encoding="utf-8"))
            history = [row for row in saved_history if int(row["epoch"]) <= recovered_epoch]
        start_epoch = recovered_epoch + 1
        completed_steps = recovered_epoch * len(train_loader)
        scheduler.last_epoch = completed_steps - 1
        current_multiplier = scheduler.lr_lambdas[0](completed_steps)
        for parameter_group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            parameter_group["lr"] = base_lr * current_multiplier
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        resume_mode = "best_checkpoint_optimizer_reinitialized"

    if start_epoch > settings.epochs:
        raise RuntimeError(
            f"run already reached epoch {start_epoch - 1}, configured maximum is {settings.epochs}"
        )
    print(
        f"run={args.name} component={args.component} device={device} "
        f"resume={resume_mode} start_epoch={start_epoch}",
        flush=True,
    )
    started = time.perf_counter()

    for epoch in range(start_epoch, settings.epochs + 1):
        dynamic_train.set_epoch(epoch)
        model.train()
        loss_sum = examples = 0
        for source_ids, target_ids, _ in train_loader:
            source_ids = source_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                valid = base.valid_permutation_mask(source_ids, target_ids, permutations)
                if args.component == "candidate":
                    energies = model(source_ids, permutations)
                    loss = base.listwise_loss(energies.float(), valid)
                else:
                    unary, pair, adjacency = model(source_ids)
                    energies = base.permutation_energies(unary, pair, adjacency, permutations)
                    loss = (
                        base.listwise_loss(energies.float(), valid)
                        + settings.unary_auxiliary_weight
                        * base.unary_auxiliary_loss(unary.float(), source_ids, target_ids)
                        + settings.pairwise_auxiliary_weight
                        * base.pairwise_auxiliary_loss(pair.float(), source_ids, target_ids)
                        + settings.adjacency_auxiliary_weight
                        * base.adjacency_auxiliary_loss(adjacency.float(), source_ids, target_ids)
                    )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(source_ids)
            examples += len(source_ids)

        metrics = evaluate(model, validation_loader, permutations, device, use_amp)
        accuracy = float(metrics["exact_match_accuracy"])
        position = float(metrics["word_position_accuracy"])
        improved = accuracy > best_accuracy or (accuracy == best_accuracy and position > best_position)
        if improved:
            best_accuracy, best_position, stale_epochs = accuracy, position, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "component": args.component,
                    "settings": asdict(settings),
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "tokenizer_path": str(TOKENIZER_PATH),
                },
                best_checkpoint_path,
            )
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / examples,
                **metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "component": args.component,
                "seed": args.seed,
                "settings": asdict(settings),
                "epoch": epoch,
                "history": history,
                "best_accuracy": best_accuracy,
                "best_position": best_position,
                "stale_epochs": stale_epochs,
            },
            last_checkpoint_path,
        )
        if stale_epochs >= settings.early_stopping_patience:
            break

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "component": args.component,
                "seed": args.seed,
                "resume_mode": resume_mode,
                "best_validation_exact_accuracy": best_accuracy,
                "best_validation_word_position_accuracy": best_position,
                "elapsed_seconds": time.perf_counter() - started,
                "pretrained_weights": False,
                "test_opened": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
