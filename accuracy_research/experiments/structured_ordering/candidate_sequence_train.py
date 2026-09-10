"""Train a higher-order Transformer that directly scores all 120 candidate orders."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train as base


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    d_model: int = 96
    word_embedding_dim: int = 64
    character_embedding_dim: int = 24
    character_channels_per_width: int = 24
    set_encoder_layers: int = 2
    candidate_encoder_layers: int = 2
    nhead: int = 8
    dim_feedforward: int = 256
    dropout: float = 0.20
    input_dropout: float = 0.10
    word_identity_dropout: float = 0.10
    batch_size: int = 32
    validation_batch_size: int = 64
    epochs: int = 180
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    warmup_epochs: int = 8
    early_stopping_patience: int = 30
    use_structural_priors: bool = True
    position_prior_scale: float = 0.50
    precedence_prior_scale: float = 0.25
    adjacency_prior_scale: float = 2.0
    use_distributional_features: bool = False
    distributional_feature_dim: int = 32
    distributional_top_words: int = 64


def build_distributional_feature_table(
    target_ids: torch.Tensor,
    vocabulary_size: int,
    top_word_count: int,
) -> torch.Tensor:
    """Build train-only syntactic-role features from word order and neighbors."""
    frequency = torch.bincount(target_ids.flatten(), minlength=vocabulary_size).float()
    top_words = frequency.argsort(descending=True)[:top_word_count]
    top_index = {int(token): index for index, token in enumerate(top_words)}
    position = torch.zeros(vocabulary_size, 5)
    before = torch.zeros(vocabulary_size, top_word_count)
    after = torch.zeros_like(before)
    next_to = torch.zeros_like(before)
    previous_to = torch.zeros_like(before)
    for target in target_ids.tolist():
        for index, token in enumerate(target):
            position[token, index] += 1.0
            for other_index, other in enumerate(target):
                feature_index = top_index.get(other)
                if feature_index is None or other_index == index:
                    continue
                if index < other_index:
                    before[token, feature_index] += 1.0
                else:
                    after[token, feature_index] += 1.0
                if other_index == index + 1:
                    next_to[token, feature_index] += 1.0
                elif other_index == index - 1:
                    previous_to[token, feature_index] += 1.0
    denominator = frequency.clamp_min(1.0)[:, None]
    position = (position + 0.25) / (denominator + 1.25)
    relations = [matrix / denominator for matrix in (before, after, next_to, previous_to)]
    log_frequency = torch.log1p(frequency) / torch.log1p(frequency.max())
    return torch.cat((position, *relations, log_frequency[:, None]), dim=1)


class CandidateSequenceTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        character_table: torch.Tensor,
        structural_priors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        settings: Settings,
        distributional_features: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.word_embedding = nn.Embedding(vocabulary_size, settings.word_embedding_dim)
        self.character_encoder = base.CharacterWordEncoder(character_table, settings)
        input_dim = settings.word_embedding_dim + self.character_encoder.output_dim
        if settings.use_distributional_features:
            if distributional_features is None:
                raise ValueError("Train-only distributional features are required")
            self.register_buffer(
                "distributional_features", distributional_features, persistent=False
            )
            self.distributional_projection = nn.Sequential(
                nn.Linear(distributional_features.shape[1], settings.distributional_feature_dim),
                nn.GELU(),
                nn.LayerNorm(settings.distributional_feature_dim),
            )
            input_dim += settings.distributional_feature_dim
        self.input_dropout = nn.Dropout(settings.input_dropout)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, settings.d_model),
            nn.GELU(),
            nn.LayerNorm(settings.d_model),
        )

        def encoder(layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=settings.d_model,
                nhead=settings.nhead,
                dim_feedforward=settings.dim_feedforward,
                dropout=settings.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(
                layer,
                num_layers=layers,
                norm=nn.LayerNorm(settings.d_model),
                enable_nested_tensor=False,
            )

        self.set_encoder = encoder(settings.set_encoder_layers)
        self.candidate_encoder = encoder(settings.candidate_encoder_layers)
        self.output_positions = nn.Parameter(torch.empty(1, 5, settings.d_model))
        self.score = nn.Sequential(
            nn.Linear(5 * settings.d_model, settings.d_model),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(settings.d_model, 1),
        )
        self.neural_scale = nn.Parameter(torch.tensor(0.0))
        position, precedence, adjacency = structural_priors
        self.register_buffer("position_prior", position, persistent=False)
        self.register_buffer("precedence_prior", precedence, persistent=False)
        self.register_buffer("adjacency_prior", adjacency, persistent=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.output_positions, std=0.02)

    def forward(self, source_ids: torch.Tensor, permutations: torch.Tensor) -> torch.Tensor:
        word_features = self.word_embedding(source_ids)
        if self.training and self.settings.word_identity_dropout > 0:
            keep = torch.rand(
                (*word_features.shape[:-1], 1),
                device=word_features.device,
                dtype=word_features.dtype,
            ).ge(self.settings.word_identity_dropout)
            word_features = word_features * keep / (1.0 - self.settings.word_identity_dropout)
        feature_parts = [word_features, self.character_encoder(source_ids)]
        if self.settings.use_distributional_features:
            feature_parts.append(
                self.distributional_projection(self.distributional_features[source_ids])
            )
        features = torch.cat(feature_parts, dim=-1)
        contextual = self.set_encoder(self.input_projection(self.input_dropout(features)))

        batch_size = len(source_ids)
        permutation_count = len(permutations)
        ordered = contextual[:, permutations]
        candidates = ordered + self.output_positions[:, None]
        encoded = self.candidate_encoder(
            candidates.reshape(batch_size * permutation_count, 5, self.settings.d_model)
        )
        neural = self.score(encoded.flatten(1)).reshape(batch_size, permutation_count)
        energy = functional.softplus(self.neural_scale) * neural

        if self.settings.use_structural_priors:
            candidate_ids = source_ids[:, permutations]
            position = self.position_prior[candidate_ids, torch.arange(5, device=source_ids.device)].sum(-1)
            precedence = torch.zeros_like(position)
            for left, right in base.OUTPUT_PAIRS:
                precedence = precedence + self.precedence_prior[
                    candidate_ids[:, :, left], candidate_ids[:, :, right]
                ]
            adjacency = torch.zeros_like(position)
            for left in range(4):
                adjacency = adjacency + self.adjacency_prior[
                    candidate_ids[:, :, left], candidate_ids[:, :, left + 1]
                ]
            energy = (
                energy
                + self.settings.position_prior_scale * position
                + self.settings.precedence_prior_scale * precedence
                + self.settings.adjacency_prior_scale * adjacency
            )
        return energy


@torch.inference_mode()
def evaluate(model, loader, permutations, device, use_amp):
    model.eval()
    exact_weight = position_weight = total_weight = group_exact = group_count = 0
    loss_sum = 0.0
    for source_ids, target_ids, weights in loader:
        source_ids = source_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            energies = model(source_ids, permutations)
        valid = base.valid_permutation_mask(source_ids, target_ids, permutations)
        loss = base.listwise_loss(energies.float(), valid)
        selected = permutations[energies.argmax(-1)]
        predictions = source_ids.gather(1, selected)
        matches = predictions.eq(target_ids).cpu()
        exact = matches.all(-1)
        weights = weights.long()
        exact_weight += int((exact.long() * weights).sum())
        position_weight += int((matches.long() * weights[:, None]).sum())
        total_weight += int(weights.sum())
        group_exact += int(exact.sum())
        group_count += len(exact)
        loss_sum += float(loss) * len(exact)
    return {
        "records": total_weight,
        "target_groups": group_count,
        "validation_loss": loss_sum / group_count,
        "exact_matches": exact_weight,
        "exact_match_accuracy": exact_weight / total_weight,
        "group_exact_match_accuracy": group_exact / group_count,
        "word_position_accuracy": position_weight / (5 * total_weight),
        "output_validity_rate": 1.0,
        "input_word_preservation_rate": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--no-structural-priors", action="store_true")
    parser.add_argument("--candidate-layers", type=int, default=None)
    parser.add_argument("--word-identity-dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--distributional-features", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    values = asdict(Settings())
    if arguments.seed is not None:
        values["seed"] = arguments.seed
    values["use_structural_priors"] = not arguments.no_structural_priors
    values["use_distributional_features"] = arguments.distributional_features
    if arguments.candidate_layers is not None:
        values["candidate_encoder_layers"] = arguments.candidate_layers
    if arguments.word_identity_dropout is not None:
        values["word_identity_dropout"] = arguments.word_identity_dropout
    settings = Settings(**values)

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    device = torch.device("cuda")
    use_amp = torch.cuda.is_bf16_supported()
    word_to_id, id_to_word = base.load_vocabulary()
    train_source, train_target, train_weights = base.load_unique_groups(base.TRAIN_PATH, word_to_id)
    validation_source, validation_target, validation_weights = base.load_unique_groups(
        base.VALIDATION_PATH, word_to_id
    )
    train_loader = DataLoader(
        TensorDataset(train_source, train_target, train_weights),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(settings.seed),
        pin_memory=True,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_source, validation_target, validation_weights),
        batch_size=settings.validation_batch_size,
        pin_memory=True,
    )
    if arguments.smoke:
        train_loader = [next(iter(train_loader))]
        validation_loader = [next(iter(validation_loader))]
        values["epochs"] = 2
        settings = Settings(**values)

    model = CandidateSequenceTransformer(
        len(word_to_id),
        base.build_character_table(id_to_word),
        base.build_structural_prior_tables(train_target, len(word_to_id)),
        settings,
        build_distributional_feature_table(
            train_target, len(word_to_id), settings.distributional_top_words
        )
        if settings.use_distributional_features
        else None,
    ).to(device)
    permutations = base.PERMUTATIONS.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    total_steps = settings.epochs * len(train_loader)
    warmup_steps = settings.warmup_epochs * len(train_loader)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        floor = settings.minimum_learning_rate / settings.learning_rate
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    artifact_dir = base.EXPERIMENT_ROOT / "runs" / arguments.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_accuracy = best_position = -1.0
    best_epoch = no_improvement = 0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.perf_counter()
    print(f"parameters={parameter_count:,} train_groups={len(train_source):,}", flush=True)

    for epoch in range(1, settings.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        loss_sum = examples = 0
        for source_ids, target_ids, _ in train_loader:
            source_ids = source_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                energies = model(source_ids, permutations)
                valid = base.valid_permutation_mask(source_ids, target_ids, permutations)
                loss = base.listwise_loss(energies.float(), valid)
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
            best_accuracy, best_position, best_epoch = accuracy, position, epoch
            no_improvement = 0
            if not arguments.smoke:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "settings": asdict(settings),
                        "parameter_count": parameter_count,
                        "epoch": epoch,
                        "validation_metrics": metrics,
                    },
                    artifact_dir / "best_model.pt",
                )
        else:
            no_improvement += 1
        row = {
            "epoch": epoch,
            "training_loss": loss_sum / examples,
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "is_best": improved,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={row['training_loss']:.4f} val={metrics['validation_loss']:.4f} "
            f"exact={100*accuracy:.3f}% position={100*position:.3f}% "
            f"best={100*best_accuracy:.3f}%@{best_epoch} seconds={row['elapsed_seconds']:.2f}",
            flush=True,
        )
        if not arguments.smoke:
            (artifact_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if no_improvement >= settings.early_stopping_patience:
            break

    summary = {
        "experiment": arguments.name,
        "status": "smoke" if arguments.smoke else "complete",
        "architecture_only": True,
        "pretrained_weights": False,
        "external_data": False,
        "selection_split": "validation only",
        "protected_test_opened": False,
        "settings": asdict(settings),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_exact_accuracy": best_accuracy,
        "best_validation_word_position_accuracy": best_position,
        "baseline_validation_exact_accuracy": 0.5697469746974697,
        "absolute_improvement": best_accuracy - 0.5697469746974697,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not arguments.smoke:
        (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
