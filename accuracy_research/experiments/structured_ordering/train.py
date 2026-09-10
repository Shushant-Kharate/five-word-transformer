"""Train a permutation-invariant structured Transformer from scratch."""

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
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "permutations_4581" / "splits" / "train.jsonl"
VALIDATION_PATH = ROOT / "data" / "permutations_4581" / "splits" / "validation.jsonl"
TOKENIZER_PATH = ROOT / "experiments" / "permutations_4581_grouped_epoch7" / "tokenizer.json"
PERMUTATIONS = torch.tensor(list(itertools.permutations(range(5))), dtype=torch.long)
OUTPUT_PAIRS = tuple(itertools.combinations(range(5), 2))


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    d_model: int = 128
    word_embedding_dim: int = 96
    character_embedding_dim: int = 24
    character_channels_per_width: int = 32
    encoder_layers: int = 3
    nhead: int = 8
    dim_feedforward: int = 384
    dropout: float = 0.15
    batch_size: int = 128
    validation_batch_size: int = 256
    epochs: int = 240
    learning_rate: float = 4e-4
    minimum_learning_rate: float = 2e-5
    weight_decay: float = 1e-3
    gradient_clip: float = 1.0
    warmup_epochs: int = 8
    early_stopping_patience: int = 30
    pairwise_auxiliary_weight: float = 0.30
    unary_auxiliary_weight: float = 0.20
    adjacency_auxiliary_weight: float = 0.10
    input_dropout: float = 0.10
    word_identity_dropout: float = 0.0
    use_word_embedding: bool = True
    use_character_encoder: bool = True
    use_adjacency: bool = False
    use_structural_priors: bool = False
    position_prior_scale: float = 1.907
    precedence_prior_scale: float = 0.334
    adjacency_prior_scale: float = 6.184


def load_vocabulary() -> tuple[dict[str, int], dict[int, str]]:
    tokenizer = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    word_to_id = {str(word): int(identifier) for word, identifier in tokenizer["model"]["vocab"].items()}
    id_to_word = {identifier: word for word, identifier in word_to_id.items()}
    return word_to_id, id_to_word


def load_unique_groups(
    path: Path,
    word_to_id: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped: dict[tuple[tuple[str, ...], str], int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source_words = record["input"].split()
            target_words = record["target"].split()
            canonical_words = tuple(sorted(source_words))
            if len(canonical_words) != 5 or sorted(target_words) != list(canonical_words):
                raise ValueError(f"Invalid five-word permutation record in {path}")
            key = (canonical_words, record["target"])
            grouped[key] = grouped.get(key, 0) + 1

    source_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    weights: list[int] = []
    for (canonical_words, target), count in grouped.items():
        source_rows.append([word_to_id[word] for word in canonical_words])
        target_rows.append([word_to_id[word] for word in target.split()])
        weights.append(count)
    return (
        torch.tensor(source_rows, dtype=torch.long),
        torch.tensor(target_rows, dtype=torch.long),
        torch.tensor(weights, dtype=torch.long),
    )


def build_character_table(id_to_word: dict[int, str]) -> torch.Tensor:
    maximum_length = max(max(len(word), 4) for word in id_to_word.values())
    table = torch.zeros((max(id_to_word) + 1, maximum_length), dtype=torch.long)
    for identifier, word in id_to_word.items():
        for position, character in enumerate(word.lower()[:maximum_length]):
            if "a" <= character <= "z":
                table[identifier, position] = ord(character) - ord("a") + 1
    return table


def build_structural_prior_tables(
    target_ids: torch.Tensor,
    vocabulary_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate word-position and directed-neighbor priors from training targets only."""
    position = torch.zeros((vocabulary_size, 5), dtype=torch.float64)
    precedence = torch.zeros((vocabulary_size, vocabulary_size), dtype=torch.float64)
    adjacency = torch.zeros((vocabulary_size, vocabulary_size), dtype=torch.float64)
    cooccurrence = torch.zeros((vocabulary_size, vocabulary_size), dtype=torch.float64)
    for target in target_ids:
        for output_position, token in enumerate(target.tolist()):
            position[token, output_position] += 1.0
        for left_position, right_position in itertools.combinations(range(5), 2):
            left = int(target[left_position])
            right = int(target[right_position])
            if left != right:
                precedence[left, right] += 1.0
                cooccurrence[left, right] += 1.0
                cooccurrence[right, left] += 1.0
        for output_position in range(4):
            left = int(target[output_position])
            right = int(target[output_position + 1])
            if left != right:
                adjacency[left, right] += 1.0
    position_alpha = 0.5
    position_prior = torch.log(
        (position + position_alpha)
        / (position.sum(dim=1, keepdim=True) + 5.0 * position_alpha)
    )
    precedence_alpha = 0.5
    precedence_prior = torch.log(
        (precedence + precedence_alpha)
        / (precedence.T + precedence_alpha)
    )
    adjacency_alpha = 0.5
    adjacency_prior = torch.log(
        (adjacency + adjacency_alpha)
        / (cooccurrence + 2.0 * adjacency_alpha)
    )
    return position_prior.float(), precedence_prior.float(), adjacency_prior.float()


class CharacterWordEncoder(nn.Module):
    def __init__(self, character_table: torch.Tensor, settings: Settings) -> None:
        super().__init__()
        self.register_buffer("character_table", character_table, persistent=True)
        self.embedding = nn.Embedding(27, settings.character_embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                settings.character_embedding_dim,
                settings.character_channels_per_width,
                kernel_size=width,
                bias=False,
            )
            for width in (2, 3, 4)
        )

    @property
    def output_dim(self) -> int:
        return sum(convolution.out_channels for convolution in self.convolutions)

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        characters = self.character_table[word_ids]
        original_shape = characters.shape[:-1]
        embedded = self.embedding(characters.reshape(-1, characters.shape[-1])).transpose(1, 2)
        features = [
            functional.gelu(convolution(embedded)).amax(dim=-1)
            for convolution in self.convolutions
        ]
        return torch.cat(features, dim=-1).reshape(*original_shape, -1)


class StructuredOrderingTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        character_table: torch.Tensor,
        settings: Settings,
        structural_priors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        if settings.use_structural_priors and structural_priors is None:
            raise ValueError("Training-only structural prior tables are required")
        position_prior, precedence_prior, adjacency_prior = structural_priors or (
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        )
        self.register_buffer("position_prior", position_prior, persistent=False)
        self.register_buffer("fixed_precedence_prior", precedence_prior, persistent=False)
        self.register_buffer("fixed_adjacency_prior", adjacency_prior, persistent=False)
        self.word_embedding = nn.Embedding(vocabulary_size, settings.word_embedding_dim)
        self.character_encoder = CharacterWordEncoder(character_table, settings)
        input_dim = settings.word_embedding_dim if settings.use_word_embedding else 0
        if settings.use_character_encoder:
            input_dim += self.character_encoder.output_dim
        if input_dim == 0:
            raise ValueError("At least one word representation must be enabled")
        self.input_dropout = nn.Dropout(settings.input_dropout)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, settings.d_model),
            nn.GELU(),
            nn.LayerNorm(settings.d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=settings.d_model,
            nhead=settings.nhead,
            dim_feedforward=settings.dim_feedforward,
            dropout=settings.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=settings.encoder_layers,
            norm=nn.LayerNorm(settings.d_model),
            enable_nested_tensor=False,
        )
        self.output_queries = nn.Parameter(torch.empty(5, settings.d_model))
        self.unary_query = nn.Linear(settings.d_model, settings.d_model, bias=False)
        self.unary_key = nn.Linear(settings.d_model, settings.d_model, bias=False)
        self.pair_left = nn.Linear(settings.d_model, settings.d_model, bias=False)
        self.pair_right = nn.Linear(settings.d_model, settings.d_model, bias=False)
        self.adjacency_left = nn.Linear(settings.d_model, settings.d_model, bias=False)
        self.adjacency_right = nn.Linear(settings.d_model, settings.d_model, bias=False)
        self.unary_scale = nn.Parameter(torch.tensor(0.0))
        self.pair_scale = nn.Parameter(torch.tensor(0.0))
        self.adjacency_scale = nn.Parameter(torch.tensor(0.0))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.output_queries, mean=0.0, std=0.02)

    def forward(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = []
        if self.settings.use_word_embedding:
            word_features = self.word_embedding(source_ids)
            if self.training and self.settings.word_identity_dropout > 0.0:
                keep = torch.rand(
                    (*word_features.shape[:-1], 1),
                    device=word_features.device,
                    dtype=word_features.dtype,
                ).ge(self.settings.word_identity_dropout)
                word_features = (
                    word_features
                    * keep
                    / (1.0 - self.settings.word_identity_dropout)
                )
            features.append(word_features)
        if self.settings.use_character_encoder:
            features.append(self.character_encoder(source_ids))
        contextual = self.set_encoder(
            self.input_projection(self.input_dropout(torch.cat(features, dim=-1)))
        )

        queries = self.unary_query(self.output_queries)
        keys = self.unary_key(contextual)
        unary = torch.einsum("od,bsd->bos", queries, keys) / math.sqrt(self.settings.d_model)

        left = self.pair_left(contextual)
        right = self.pair_right(contextual)
        raw_pair = torch.einsum("bid,bjd->bij", left, right) / math.sqrt(self.settings.d_model)
        pair = raw_pair - raw_pair.transpose(1, 2)

        adjacency_left = self.adjacency_left(contextual)
        adjacency_right = self.adjacency_right(contextual)
        adjacency = torch.einsum(
            "bid,bjd->bij", adjacency_left, adjacency_right
        ) / math.sqrt(self.settings.d_model)
        if not self.settings.use_adjacency:
            adjacency = adjacency * 0.0
        unary = functional.softplus(self.unary_scale) * unary
        pair = functional.softplus(self.pair_scale) * pair
        adjacency = functional.softplus(self.adjacency_scale) * adjacency
        if self.settings.use_structural_priors:
            unary = unary + self.settings.position_prior_scale * self.position_prior[
                source_ids
            ].transpose(1, 2)
            pair = pair + self.settings.precedence_prior_scale * self.fixed_precedence_prior[
                source_ids[:, :, None], source_ids[:, None, :]
            ]
            adjacency = adjacency + self.settings.adjacency_prior_scale * self.fixed_adjacency_prior[
                source_ids[:, :, None], source_ids[:, None, :]
            ]
        return unary, pair, adjacency


def permutation_energies(
    unary: torch.Tensor,
    pair: torch.Tensor,
    adjacency: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    batch_size = unary.shape[0]
    permutation_count = permutations.shape[0]
    expanded_unary = unary[:, None, :, :].expand(-1, permutation_count, -1, -1)
    unary_indices = permutations[None, :, :, None].expand(batch_size, -1, -1, -1)
    energy = expanded_unary.gather(3, unary_indices).squeeze(-1).sum(dim=-1)
    for left_output, right_output in OUTPUT_PAIRS:
        left_source = permutations[:, left_output]
        right_source = permutations[:, right_output]
        energy = energy + pair[:, left_source, right_source]
    for left_output in range(4):
        left_source = permutations[:, left_output]
        right_source = permutations[:, left_output + 1]
        energy = energy + adjacency[:, left_source, right_source]
    return energy


def valid_permutation_mask(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    candidates = source_ids[:, permutations]
    return candidates.eq(target_ids[:, None, :]).all(dim=-1)


def listwise_loss(energies: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    positive = torch.logsumexp(energies.masked_fill(~valid_mask, -torch.inf), dim=-1)
    normalizer = torch.logsumexp(energies, dim=-1)
    return (normalizer - positive).mean()


def unary_auxiliary_loss(
    unary: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    correct = source_ids[:, None, :].eq(target_ids[:, :, None])
    log_probabilities = torch.log_softmax(unary, dim=-1)
    return -torch.logsumexp(log_probabilities.masked_fill(~correct, -torch.inf), dim=-1).mean()


def pairwise_auxiliary_loss(
    pair: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for left_source, right_source in itertools.combinations(range(5), 2):
        left_token = source_ids[:, left_source]
        right_token = source_ids[:, right_source]
        distinct = left_token.ne(right_token)
        if not distinct.any():
            continue
        positions = torch.arange(5, device=source_ids.device)
        left_target_position = target_ids.eq(left_token[:, None]).long().argmax(dim=1)
        right_target_position = target_ids.eq(right_token[:, None]).long().argmax(dim=1)
        label = left_target_position.lt(right_target_position).float()
        loss = functional.binary_cross_entropy_with_logits(
            pair[:, left_source, right_source],
            label,
            reduction="none",
        )
        losses.append(loss[distinct])
    return torch.cat(losses).mean()


def adjacency_auxiliary_loss(
    adjacency: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Teach directed next-word preferences on rows whose five words are distinct."""
    multiplicities = source_ids[:, :, None].eq(source_ids[:, None, :]).sum(dim=-1)
    distinct_rows = multiplicities.eq(1).all(dim=-1)
    if not distinct_rows.any():
        return adjacency.sum() * 0.0
    selected_source = source_ids[distinct_rows]
    selected_target = target_ids[distinct_rows]
    selected_adjacency = adjacency[distinct_rows]
    target_source_positions = selected_target[:, :, None].eq(
        selected_source[:, None, :]
    ).long().argmax(dim=-1)
    batch_indices = torch.arange(len(selected_source), device=source_ids.device)
    losses: list[torch.Tensor] = []
    for output_position in range(4):
        current_source = target_source_positions[:, output_position]
        next_source = target_source_positions[:, output_position + 1]
        logits = selected_adjacency[batch_indices, current_source].clone()
        logits[batch_indices, current_source] = -torch.inf
        losses.append(functional.cross_entropy(logits, next_source))
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    permutations: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float | int]:
    model.eval()
    exact_weight = 0
    position_weight = 0
    total_weight = 0
    group_exact = 0
    group_count = 0
    loss_sum = 0.0
    for source_ids, target_ids, weights in loader:
        source_ids = source_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            unary, pair, adjacency = model(source_ids)
        energies = permutation_energies(
            unary.float(), pair.float(), adjacency.float(), permutations
        )
        valid_mask = valid_permutation_mask(source_ids, target_ids, permutations)
        loss = listwise_loss(energies, valid_mask)
        best = permutations[energies.argmax(dim=-1)]
        predictions = source_ids.gather(1, best)
        matches = predictions.eq(target_ids).cpu()
        weights = weights.long()
        exact = matches.all(dim=1)
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
        "word_position_accuracy": position_weight / (total_weight * 5),
        "output_validity_rate": 1.0,
        "input_word_preservation_rate": 1.0,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--no-character-encoder", action="store_true")
    parser.add_argument("--no-word-embedding", action="store_true")
    parser.add_argument("--adjacency", action="store_true")
    parser.add_argument("--structural-priors", action="store_true")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--word-embedding-dim", type=int, default=None)
    parser.add_argument("--character-channels", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--input-dropout", type=float, default=None)
    parser.add_argument("--word-identity-dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--pairwise-auxiliary-weight", type=float, default=None)
    parser.add_argument("--unary-auxiliary-weight", type=float, default=None)
    parser.add_argument("--adjacency-auxiliary-weight", type=float, default=None)
    parser.add_argument("--prior-position-scale", type=float, default=None)
    parser.add_argument("--prior-precedence-scale", type=float, default=None)
    parser.add_argument("--prior-adjacency-scale", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()

    values = asdict(Settings())
    values["use_character_encoder"] = not arguments.no_character_encoder
    values["use_word_embedding"] = not arguments.no_word_embedding
    values["use_adjacency"] = arguments.adjacency
    values["use_structural_priors"] = arguments.structural_priors
    if arguments.epochs is not None:
        values["epochs"] = arguments.epochs
    for argument_name, setting_name in (
        ("d_model", "d_model"),
        ("word_embedding_dim", "word_embedding_dim"),
        ("character_channels", "character_channels_per_width"),
        ("encoder_layers", "encoder_layers"),
        ("dropout", "dropout"),
        ("input_dropout", "input_dropout"),
        ("word_identity_dropout", "word_identity_dropout"),
        ("weight_decay", "weight_decay"),
        ("learning_rate", "learning_rate"),
        ("pairwise_auxiliary_weight", "pairwise_auxiliary_weight"),
        ("unary_auxiliary_weight", "unary_auxiliary_weight"),
        ("adjacency_auxiliary_weight", "adjacency_auxiliary_weight"),
        ("prior_position_scale", "position_prior_scale"),
        ("prior_precedence_scale", "precedence_prior_scale"),
        ("prior_adjacency_scale", "adjacency_prior_scale"),
    ):
        value = getattr(arguments, argument_name)
        if value is not None:
            values[setting_name] = value
    settings = Settings(**values)
    set_seed(settings.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for controlled architecture experiments")
    device = torch.device("cuda")
    use_amp = torch.cuda.is_bf16_supported()

    artifact_dir = EXPERIMENT_ROOT / "runs" / arguments.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    word_to_id, id_to_word = load_vocabulary()
    train_source, train_target, train_weights = load_unique_groups(TRAIN_PATH, word_to_id)
    validation_source, validation_target, validation_weights = load_unique_groups(VALIDATION_PATH, word_to_id)
    print(
        f"train_groups={len(train_source):,} validation_groups={len(validation_source):,} "
        f"validation_records={int(validation_weights.sum()):,} characters={settings.use_character_encoder}",
        flush=True,
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
        shuffle=False,
        pin_memory=True,
    )
    if arguments.smoke:
        train_loader = [next(iter(train_loader))]
        validation_loader = [next(iter(validation_loader))]

    model = StructuredOrderingTransformer(
        len(word_to_id),
        build_character_table(id_to_word),
        settings,
        build_structural_prior_tables(train_target, len(word_to_id))
        if settings.use_structural_priors
        else None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    total_steps = settings.epochs * len(train_loader)
    warmup_steps = settings.warmup_epochs * len(train_loader)

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        floor = settings.minimum_learning_rate / settings.learning_rate
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    permutations = PERMUTATIONS.to(device)
    history: list[dict[str, float | int | bool]] = []
    best_accuracy = -1.0
    best_position = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.perf_counter()

    for epoch in range(1, settings.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        total_loss = 0.0
        total_examples = 0
        for source_ids, target_ids, _ in train_loader:
            source_ids = source_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                unary, pair, adjacency = model(source_ids)
                energies = permutation_energies(unary, pair, adjacency, permutations)
                valid_mask = valid_permutation_mask(source_ids, target_ids, permutations)
                primary = listwise_loss(energies.float(), valid_mask)
                pair_auxiliary = pairwise_auxiliary_loss(pair.float(), source_ids, target_ids)
                unary_auxiliary = unary_auxiliary_loss(unary.float(), source_ids, target_ids)
                adjacency_auxiliary = (
                    adjacency_auxiliary_loss(adjacency.float(), source_ids, target_ids)
                    if settings.use_adjacency
                    else adjacency.sum() * 0.0
                )
                loss = (
                    primary
                    + settings.pairwise_auxiliary_weight * pair_auxiliary
                    + settings.unary_auxiliary_weight * unary_auxiliary
                    + settings.adjacency_auxiliary_weight * adjacency_auxiliary
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
            scheduler.step()
            batch_size = len(source_ids)
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size

        metrics = evaluate(model, validation_loader, permutations, device, use_amp)
        accuracy = float(metrics["exact_match_accuracy"])
        position_accuracy = float(metrics["word_position_accuracy"])
        improved = accuracy > best_accuracy or (
            accuracy == best_accuracy and position_accuracy > best_position
        )
        if improved:
            best_accuracy = accuracy
            best_position = position_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
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
            epochs_without_improvement += 1
        row: dict[str, float | int | bool] = {
            "epoch": epoch,
            "training_loss": total_loss / total_examples,
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "is_best": improved,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={row['training_loss']:.4f} val={metrics['validation_loss']:.4f} "
            f"exact={100 * accuracy:.3f}% position={100 * position_accuracy:.3f}% "
            f"best={100 * best_accuracy:.3f}%@{best_epoch} seconds={row['elapsed_seconds']:.2f}",
            flush=True,
        )
        if not arguments.smoke:
            (artifact_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epochs_without_improvement >= settings.early_stopping_patience:
            print(f"early_stop={epochs_without_improvement}", flush=True)
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
        "train_unique_groups": len(train_source),
        "validation_unique_groups": len(validation_source),
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
