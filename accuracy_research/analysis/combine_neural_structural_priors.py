"""Validation-only complementarity analysis for neural and train-only structural scores."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = ROOT / "accuracy_research" / "experiments" / "structured_ordering" / "train.py"
PRIOR_PATH = ROOT / "accuracy_research" / "analysis" / "train_only_structural_priors.py"
CHECKPOINT_PATH = (
    ROOT
    / "accuracy_research"
    / "experiments"
    / "structured_ordering"
    / "runs"
    / "structured_word_char_adjacency_v3"
    / "best_model.pt"
)
OUTPUT_PATH = Path(__file__).with_name("combine_adjacency_structural_priors.json")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    trainer = load_module("structured_trainer", TRAINER_PATH)
    priors_module = load_module("structural_priors", PRIOR_PATH)
    word_to_id, id_to_word = trainer.load_vocabulary()
    validation = priors_module.unique_rows(trainer.VALIDATION_PATH, word_to_id)
    train = priors_module.unique_rows(trainer.TRAIN_PATH, word_to_id)
    priors = priors_module.estimate_priors(train, len(word_to_id))
    prior_features = np.stack(
        [priors_module.components(source, *priors) for source, _, _ in validation], axis=1
    )
    # [component, group, candidate]

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    settings = trainer.Settings(**checkpoint["settings"])
    model = trainer.StructuredOrderingTransformer(
        len(word_to_id), trainer.build_character_table(id_to_word), settings
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {
        "adjacency_scale",
        "adjacency_left.weight",
        "adjacency_right.weight",
    }
    if set(incompatible.missing_keys) not in (set(), allowed_missing) or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint mismatch: {incompatible}")
    device = torch.device("cuda")
    model.to(device).eval()
    permutations = trainer.PERMUTATIONS.to(device)
    neural_batches = []
    with torch.inference_mode():
        for start in range(0, len(validation), 256):
            source = torch.tensor(
                np.stack([row[0] for row in validation[start : start + 256]]),
                dtype=torch.long,
                device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                unary, pair, adjacency = model(source)
            neural_batches.append(
                trainer.permutation_energies(
                    unary.float(), pair.float(), adjacency.float(), permutations
                ).cpu().numpy()
            )
    neural = np.concatenate(neural_batches)
    weights = np.asarray([row[2] for row in validation], dtype=np.int64)
    valid = np.stack(
        [source[priors_module.PERMUTATIONS] == target for source, target, _ in validation]
    ).all(axis=-1)

    def accuracy(scores: np.ndarray) -> tuple[float, float]:
        choices = scores.argmax(axis=1)
        correct = valid[np.arange(len(valid)), choices]
        exact = float((correct * weights).sum() / weights.sum())
        position_total = 0
        for group, (source, target, weight) in enumerate(validation):
            prediction = source[priors_module.PERMUTATIONS[choices[group]]]
            position_total += int((prediction == target).sum()) * weight
        return exact, float(position_total / (5 * weights.sum()))

    neural_std = max(float(neural.std()), 1e-8)
    normalized_neural = neural / neural_std
    normalized_priors = prior_features / np.maximum(
        prior_features.std(axis=(1, 2), keepdims=True), 1e-8
    )
    values = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    results = []
    for scales in itertools.product(values, repeat=4):
        if not any(scales):
            continue
        scores = scales[0] * normalized_neural
        for scale, component in zip(scales[1:], normalized_priors):
            scores = scores + scale * component
        exact, position = accuracy(scores)
        results.append(
            {
                "scales_neural_position_precedence_adjacency": scales,
                "exact_accuracy": exact,
                "word_position_accuracy": position,
            }
        )
    results.sort(key=lambda row: (row["exact_accuracy"], row["word_position_accuracy"]), reverse=True)
    report = {
        "architecture_only": True,
        "external_data": False,
        "protected_test_opened": False,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
        "score_standard_deviations": {
            "neural": neural_std,
            "position": float(prior_features[0].std()),
            "precedence": float(prior_features[1].std()),
            "adjacency": float(prior_features[2].std()),
        },
        "standalone_neural": dict(
            zip(("exact_accuracy", "word_position_accuracy"), accuracy(neural))
        ),
        "best_combinations": results[:30],
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
