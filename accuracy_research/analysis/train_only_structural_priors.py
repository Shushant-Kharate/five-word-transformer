"""Measure transparent structural priors estimated only from unique training targets.

The protected test split is intentionally never loaded by this script.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "permutations_4581" / "splits" / "train.jsonl"
VALIDATION_PATH = ROOT / "data" / "permutations_4581" / "splits" / "validation.jsonl"
TOKENIZER_PATH = ROOT / "experiments" / "permutations_4581_grouped_epoch7" / "tokenizer.json"
OUTPUT_PATH = Path(__file__).with_suffix(".json")
PERMUTATIONS = np.asarray(list(itertools.permutations(range(5))), dtype=np.int64)


def vocabulary() -> dict[str, int]:
    tokenizer = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    return {word: int(identifier) for word, identifier in tokenizer["model"]["vocab"].items()}


def unique_rows(path: Path, word_to_id: dict[str, int]) -> list[tuple[np.ndarray, np.ndarray, int]]:
    groups: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source = tuple(sorted(record["input"].split()))
            target = tuple(record["target"].split())
            groups[(source, target)] = groups.get((source, target), 0) + 1
    return [
        (
            np.asarray([word_to_id[word] for word in source], dtype=np.int64),
            np.asarray([word_to_id[word] for word in target], dtype=np.int64),
            weight,
        )
        for (source, target), weight in groups.items()
    ]


def estimate_priors(
    rows: list[tuple[np.ndarray, np.ndarray, int]], vocabulary_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.zeros((vocabulary_size, 5), dtype=np.float64)
    precedence = np.zeros((vocabulary_size, vocabulary_size), dtype=np.float64)
    adjacency = np.zeros((vocabulary_size, vocabulary_size), dtype=np.float64)
    cooccurrence = np.zeros((vocabulary_size, vocabulary_size), dtype=np.float64)
    for _, target, _ in rows:
        for output_position, token in enumerate(target):
            position[token, output_position] += 1.0
        for left_position, right_position in itertools.combinations(range(5), 2):
            left = target[left_position]
            right = target[right_position]
            if left != right:
                precedence[left, right] += 1.0
                cooccurrence[left, right] += 1.0
                cooccurrence[right, left] += 1.0
        for output_position in range(4):
            left = target[output_position]
            right = target[output_position + 1]
            if left != right:
                adjacency[left, right] += 1.0

    position_alpha = 0.5
    position_prior = np.log(
        (position + position_alpha)
        / (position.sum(axis=1, keepdims=True) + 5.0 * position_alpha)
    )
    precedence_alpha = 0.5
    precedence_prior = np.log(
        (precedence + precedence_alpha)
        / (precedence.T + precedence_alpha)
    )
    adjacency_alpha = 0.5
    adjacency_prior = np.log(
        (adjacency + adjacency_alpha)
        / (cooccurrence + 2.0 * adjacency_alpha)
    )
    return position_prior, precedence_prior, adjacency_prior


def components(
    source: np.ndarray,
    position: np.ndarray,
    precedence: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = source[PERMUTATIONS]
    unary_scores = position[candidates, np.arange(5)].sum(axis=1)
    precedence_scores = np.zeros(len(PERMUTATIONS), dtype=np.float64)
    for left, right in itertools.combinations(range(5), 2):
        precedence_scores += precedence[candidates[:, left], candidates[:, right]]
    adjacency_scores = np.zeros(len(PERMUTATIONS), dtype=np.float64)
    for left in range(4):
        adjacency_scores += adjacency[candidates[:, left], candidates[:, left + 1]]
    return unary_scores, precedence_scores, adjacency_scores


def main() -> None:
    word_to_id = vocabulary()
    train = unique_rows(TRAIN_PATH, word_to_id)
    validation = unique_rows(VALIDATION_PATH, word_to_id)
    priors = estimate_priors(train, len(word_to_id))
    features = [components(source, *priors) for source, _, _ in validation]
    weights = np.asarray([weight for _, _, weight in validation], dtype=np.int64)
    targets = [target for _, target, _ in validation]
    sources = [source for source, _, _ in validation]

    def evaluate(scales: tuple[float, float, float]) -> tuple[float, float]:
        exact = 0
        positions = 0
        total = int(weights.sum())
        for source, target, weight, feature in zip(sources, targets, weights, features):
            energy = sum(scale * values for scale, values in zip(scales, feature))
            prediction = source[PERMUTATIONS[int(energy.argmax())]]
            matches = prediction == target
            exact += int(matches.all()) * int(weight)
            positions += int(matches.sum()) * int(weight)
        return exact / total, positions / (5 * total)

    scale_values = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    results = []
    for scales in itertools.product(scale_values, repeat=3):
        if scales == (0.0, 0.0, 0.0):
            continue
        exact, word_position = evaluate(scales)
        results.append(
            {"scales": scales, "exact_accuracy": exact, "word_position_accuracy": word_position}
        )
    results.sort(key=lambda row: (row["exact_accuracy"], row["word_position_accuracy"]), reverse=True)
    report = {
        "architecture_only": True,
        "external_data": False,
        "protected_test_opened": False,
        "train_unique_groups": len(train),
        "validation_unique_groups": len(validation),
        "single_components": {
            "position": dict(zip(("exact_accuracy", "word_position_accuracy"), evaluate((1, 0, 0)))),
            "precedence": dict(zip(("exact_accuracy", "word_position_accuracy"), evaluate((0, 1, 0)))),
            "adjacency": dict(zip(("exact_accuracy", "word_position_accuracy"), evaluate((0, 0, 1)))),
        },
        "best_grid_results": results[:20],
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
