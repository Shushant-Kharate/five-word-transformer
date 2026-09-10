"""Combine the two architecture-only neural scorers and train-only structural priors."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
STRUCTURED_DIR = ROOT / "accuracy_research" / "experiments" / "structured_ordering"
ADJACENCY_CHECKPOINT = STRUCTURED_DIR / "runs" / "structured_word_char_adjacency_v3" / "best_model.pt"
CANDIDATE_CHECKPOINT = STRUCTURED_DIR / "runs" / "candidate_sequence_flatten_v3" / "best_model.pt"
SECOND_CANDIDATE_CHECKPOINT = STRUCTURED_DIR / "runs" / "candidate_sequence_seed43_v4" / "best_model.pt"
THIRD_CANDIDATE_CHECKPOINT = STRUCTURED_DIR / "runs" / "candidate_sequence_seed44_v6" / "best_model.pt"
OUTPUT_PATH = Path(__file__).with_suffix(".json")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.path.insert(0, str(STRUCTURED_DIR))
    base = load_module("train", STRUCTURED_DIR / "train.py")
    candidate_module = load_module("candidate_sequence_train", STRUCTURED_DIR / "candidate_sequence_train.py")
    priors_module = load_module(
        "structural_priors", ROOT / "accuracy_research" / "analysis" / "train_only_structural_priors.py"
    )
    word_to_id, id_to_word = base.load_vocabulary()
    train = priors_module.unique_rows(base.TRAIN_PATH, word_to_id)
    validation = priors_module.unique_rows(base.VALIDATION_PATH, word_to_id)
    statistical_priors = priors_module.estimate_priors(train, len(word_to_id))
    prior_features = np.stack(
        [priors_module.components(source, *statistical_priors) for source, _, _ in validation], axis=1
    )
    source_tensor = torch.tensor(np.stack([row[0] for row in validation]), dtype=torch.long)
    target_array = np.stack([row[1] for row in validation])
    weights = np.asarray([row[2] for row in validation], dtype=np.int64)
    candidate_ids = np.stack([row[0][priors_module.PERMUTATIONS] for row in validation])
    valid = (candidate_ids == target_array[:, None]).all(-1)
    positions = (candidate_ids == target_array[:, None]).sum(-1)

    device = torch.device("cuda")
    permutations = base.PERMUTATIONS.to(device)
    character_table = base.build_character_table(id_to_word)
    adjacency_checkpoint = torch.load(ADJACENCY_CHECKPOINT, map_location="cpu", weights_only=False)
    adjacency_settings = base.Settings(**adjacency_checkpoint["settings"])
    adjacency_model = base.StructuredOrderingTransformer(
        len(word_to_id), character_table, adjacency_settings
    )
    adjacency_model.load_state_dict(adjacency_checkpoint["model_state_dict"])
    adjacency_model.to(device).eval()

    candidate_checkpoint = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    candidate_settings = candidate_module.Settings(**candidate_checkpoint["settings"])
    train_target = torch.tensor(np.stack([row[1] for row in train]), dtype=torch.long)
    candidate_model = candidate_module.CandidateSequenceTransformer(
        len(word_to_id),
        character_table,
        base.build_structural_prior_tables(train_target, len(word_to_id)),
        candidate_settings,
    )
    candidate_model.load_state_dict(candidate_checkpoint["model_state_dict"])
    candidate_model.to(device).eval()

    second_candidate_checkpoint = torch.load(
        SECOND_CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    second_candidate_settings = candidate_module.Settings(**second_candidate_checkpoint["settings"])
    second_candidate_model = candidate_module.CandidateSequenceTransformer(
        len(word_to_id),
        character_table,
        base.build_structural_prior_tables(train_target, len(word_to_id)),
        second_candidate_settings,
    )
    second_candidate_model.load_state_dict(second_candidate_checkpoint["model_state_dict"])
    second_candidate_model.to(device).eval()

    third_candidate_checkpoint = torch.load(
        THIRD_CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    third_candidate_settings = candidate_module.Settings(**third_candidate_checkpoint["settings"])
    third_candidate_model = candidate_module.CandidateSequenceTransformer(
        len(word_to_id),
        character_table,
        base.build_structural_prior_tables(train_target, len(word_to_id)),
        third_candidate_settings,
    )
    third_candidate_model.load_state_dict(third_candidate_checkpoint["model_state_dict"])
    third_candidate_model.to(device).eval()

    adjacency_scores = []
    candidate_scores = []
    second_candidate_scores = []
    third_candidate_scores = []
    with torch.inference_mode():
        for start in range(0, len(source_tensor), 32):
            source = source_tensor[start : start + 32].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                unary, pair, adjacency = adjacency_model(source)
                sequence_energy = candidate_model(source, permutations)
                second_sequence_energy = second_candidate_model(source, permutations)
                third_sequence_energy = third_candidate_model(source, permutations)
            adjacency_scores.append(
                base.permutation_energies(
                    unary.float(), pair.float(), adjacency.float(), permutations
                ).cpu().numpy()
            )
            candidate_scores.append(sequence_energy.float().cpu().numpy())
            second_candidate_scores.append(second_sequence_energy.float().cpu().numpy())
            third_candidate_scores.append(third_sequence_energy.float().cpu().numpy())
    components = (
        np.concatenate(adjacency_scores),
        np.concatenate(candidate_scores),
        np.concatenate(second_candidate_scores),
        np.concatenate(third_candidate_scores),
        *prior_features,
    )
    standard_deviations = np.asarray([max(float(component.std()), 1e-8) for component in components])
    normalized = [component / deviation for component, deviation in zip(components, standard_deviations)]

    def metrics(scores: np.ndarray) -> tuple[float, float]:
        choices = scores.argmax(1)
        row = np.arange(len(choices))
        exact = float((valid[row, choices] * weights).sum() / weights.sum())
        word_position = float((positions[row, choices] * weights).sum() / (5 * weights.sum()))
        return exact, word_position

    results_by_scales = {}

    def record(scales) -> None:
        scales = tuple(float(value) for value in scales)
        score = sum(scale * component for scale, component in zip(scales, normalized))
        exact, word_position = metrics(score)
        results_by_scales[scales] = {
            "scales_adjacency_candidate42_candidate43_candidate44_position_precedence_fixed_adjacency": scales,
            "exact_accuracy": exact,
            "word_position_accuracy": word_position,
        }

    search_values = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    starts = [
        [0.5, 2.0, 4.0, 0.0, 2.0, 0.5, 2.0],
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0] * 7,
    ]
    for scales in starts:
        for _ in range(5):
            for dimension in range(7):
                candidates = []
                for value in search_values:
                    trial = list(scales)
                    trial[dimension] = value
                    record(trial)
                    candidates.append(results_by_scales[tuple(float(item) for item in trial)])
                best = max(candidates, key=lambda row: (row["exact_accuracy"], row["word_position_accuracy"]))
                scales = list(
                    best["scales_adjacency_candidate42_candidate43_candidate44_position_precedence_fixed_adjacency"]
                )
    rng = np.random.default_rng(42)
    for _ in range(5000):
        scales = rng.lognormal(mean=0.0, sigma=1.0, size=7)
        scales[rng.random(7) < 0.18] = 0.0
        scales = 4.0 * scales / max(scales.max(), 1e-8)
        record(np.round(scales, 3))
    results = list(results_by_scales.values())
    results.sort(key=lambda row: (row["exact_accuracy"], row["word_position_accuracy"]), reverse=True)
    neural_choices = np.stack([component.argmax(1) for component in components[:4]])
    group_indices = np.arange(len(validation))
    neural_correct = np.stack(
        [valid[group_indices, choices] for choices in neural_choices]
    )
    oracle_correct = neural_correct.any(axis=0)
    oracle_exact = float((oracle_correct * weights).sum() / weights.sum())
    unanimous = (neural_choices == neural_choices[0:1]).all(axis=0)
    unanimous_accuracy = float(
        (valid[group_indices, neural_choices[0]] * weights * unanimous).sum()
        / max((weights * unanimous).sum(), 1)
    )

    best_scales = results[0][
        "scales_adjacency_candidate42_candidate43_candidate44_position_precedence_fixed_adjacency"
    ]
    best_score = sum(scale * component for scale, component in zip(best_scales, normalized))
    best_choices = best_score.argmax(1)
    best_correct = valid[group_indices, best_choices]
    train_frequency = np.bincount(
        np.concatenate([row[1] for row in train]), minlength=len(word_to_id)
    )
    minimum_frequency = np.asarray(
        [train_frequency[source].min() for source, _, _ in validation]
    )
    frequency_bands = {}
    lower = 0
    for upper in (1, 2, 5, 10, 10**9):
        selected = (minimum_frequency > lower) & (minimum_frequency <= upper)
        label = f"{lower + 1}-{upper}" if upper < 10**9 else f">{lower}"
        selected_weight = int(weights[selected].sum())
        frequency_bands[label] = {
            "target_groups": int(selected.sum()),
            "records": selected_weight,
            "exact_accuracy": float(
                (best_correct[selected] * weights[selected]).sum() / max(selected_weight, 1)
            ),
        }
        lower = upper
    report = {
        "architecture_only": True,
        "external_data": False,
        "protected_test_opened": False,
        "checkpoint_epochs": {
            "adjacency": adjacency_checkpoint["epoch"],
            "candidate": candidate_checkpoint["epoch"],
            "candidate_seed43": second_candidate_checkpoint["epoch"],
            "candidate_seed44": third_candidate_checkpoint["epoch"],
        },
        "component_standard_deviations": standard_deviations.tolist(),
        "standalone": {
            "adjacency": dict(zip(("exact_accuracy", "word_position_accuracy"), metrics(components[0]))),
            "candidate": dict(zip(("exact_accuracy", "word_position_accuracy"), metrics(components[1]))),
            "candidate_seed43": dict(
                zip(("exact_accuracy", "word_position_accuracy"), metrics(components[2]))
            ),
            "candidate_seed44": dict(
                zip(("exact_accuracy", "word_position_accuracy"), metrics(components[3]))
            ),
        },
        "neural_model_oracle_exact_accuracy": oracle_exact,
        "unanimous_prediction": {
            "target_groups": int(unanimous.sum()),
            "exact_accuracy": unanimous_accuracy,
        },
        "best_ensemble_by_minimum_training_word_frequency": frequency_bands,
        "best_combinations": results[:40],
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
