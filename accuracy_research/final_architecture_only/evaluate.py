"""Evaluate the frozen architecture-only ensemble without changing its scales."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = Path(__file__).resolve().parent
STRUCTURED_DIR = ROOT / "accuracy_research" / "experiments" / "structured_ordering"
CONFIG_PATH = FINAL_DIR / "frozen_config.json"
TEST_PATH = ROOT / "data" / "permutations_4581" / "splits" / "test.jsonl"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_path(config: dict, name: str) -> Path:
    return ROOT / config["checkpoints"][name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--confirm-final-test", action="store_true")
    arguments = parser.parse_args()
    output_path = FINAL_DIR / f"{arguments.split}_metrics.json"
    if arguments.split == "test":
        if not arguments.confirm_final_test:
            raise RuntimeError("Formal test evaluation requires --confirm-final-test")
        if output_path.exists():
            raise RuntimeError(f"Formal test evaluation was already completed: {output_path}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sys.path.insert(0, str(STRUCTURED_DIR))
    base = load_module("train", STRUCTURED_DIR / "train.py")
    candidate_module = load_module(
        "candidate_sequence_train", STRUCTURED_DIR / "candidate_sequence_train.py"
    )
    word_to_id, id_to_word = base.load_vocabulary()
    train_source, train_target, _ = base.load_unique_groups(base.TRAIN_PATH, word_to_id)
    split_path = base.VALIDATION_PATH if arguments.split == "validation" else TEST_PATH
    source_ids, target_ids, weights = base.load_unique_groups(split_path, word_to_id)
    loader = DataLoader(
        TensorDataset(source_ids, target_ids, weights),
        batch_size=32,
        shuffle=False,
        pin_memory=True,
    )

    device = torch.device("cuda")
    use_amp = torch.cuda.is_bf16_supported()
    character_table = base.build_character_table(id_to_word)
    structural_priors = base.build_structural_prior_tables(train_target, len(word_to_id))
    position_prior, precedence_prior, adjacency_prior = (
        prior.to(device) for prior in structural_priors
    )

    adjacency_checkpoint = torch.load(
        checkpoint_path(config, "adjacency"), map_location="cpu", weights_only=False
    )
    adjacency_model = base.StructuredOrderingTransformer(
        len(word_to_id),
        character_table,
        base.Settings(**adjacency_checkpoint["settings"]),
    )
    adjacency_model.load_state_dict(adjacency_checkpoint["model_state_dict"])
    adjacency_model.to(device).eval()

    candidate_models = []
    candidate_checkpoints = []
    for name in ("candidate_seed42", "candidate_seed43", "candidate_seed44"):
        checkpoint = torch.load(checkpoint_path(config, name), map_location="cpu", weights_only=False)
        settings = candidate_module.Settings(**checkpoint["settings"])
        model = candidate_module.CandidateSequenceTransformer(
            len(word_to_id), character_table, structural_priors, settings
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        candidate_models.append(model)
        candidate_checkpoints.append(checkpoint)

    scales = torch.tensor(config["component_scales"], device=device)
    deviations = torch.tensor(
        config["component_validation_standard_deviations"], device=device
    )
    permutations = base.PERMUTATIONS.to(device)
    exact_weight = position_weight = total_weight = group_exact = group_count = 0
    loss_sum = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for source, target, batch_weights in loader:
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                unary, pair, adjacency = adjacency_model(source)
                learned_adjacency_energy = base.permutation_energies(
                    unary, pair, adjacency, permutations
                )
                candidate_energies = [model(source, permutations) for model in candidate_models]

            candidate_tokens = source[:, permutations]
            position_energy = position_prior[
                candidate_tokens, torch.arange(5, device=device)
            ].sum(-1)
            precedence_energy = torch.zeros_like(position_energy)
            for left, right in base.OUTPUT_PAIRS:
                precedence_energy += precedence_prior[
                    candidate_tokens[:, :, left], candidate_tokens[:, :, right]
                ]
            fixed_adjacency_energy = torch.zeros_like(position_energy)
            for left in range(4):
                fixed_adjacency_energy += adjacency_prior[
                    candidate_tokens[:, :, left], candidate_tokens[:, :, left + 1]
                ]
            components = [
                learned_adjacency_energy.float(),
                *(energy.float() for energy in candidate_energies),
                position_energy.float(),
                precedence_energy.float(),
                fixed_adjacency_energy.float(),
            ]
            ensemble_energy = sum(
                scale * component / deviation
                for scale, deviation, component in zip(scales, deviations, components)
            )
            valid = base.valid_permutation_mask(source, target, permutations)
            loss = base.listwise_loss(ensemble_energy, valid)
            selected = permutations[ensemble_energy.argmax(-1)]
            prediction = source.gather(1, selected)
            matches = prediction.eq(target).cpu()
            exact = matches.all(-1)
            batch_weights = batch_weights.long()
            exact_weight += int((exact.long() * batch_weights).sum())
            position_weight += int((matches.long() * batch_weights[:, None]).sum())
            total_weight += int(batch_weights.sum())
            group_exact += int(exact.sum())
            group_count += len(exact)
            loss_sum += float(loss) * len(exact)

    parameter_count = int(adjacency_checkpoint["parameter_count"]) + sum(
        int(checkpoint["parameter_count"]) for checkpoint in candidate_checkpoints
    )
    metrics = {
        "status": "formal_final" if arguments.split == "test" else "frozen_validation_check",
        "split": arguments.split,
        "architecture_only": True,
        "pretrained_weights": False,
        "external_models": False,
        "external_data": False,
        "frozen_config": str(CONFIG_PATH),
        "records": total_weight,
        "target_groups": group_count,
        "exact_matches": exact_weight,
        "exact_match_accuracy": exact_weight / total_weight,
        "group_exact_match_accuracy": group_exact / group_count,
        "word_position_accuracy": position_weight / (5 * total_weight),
        "listwise_loss": loss_sum / group_count,
        "output_validity_rate": 1.0,
        "input_word_preservation_rate": 1.0,
        "parameter_count": parameter_count,
        "inference_seconds": time.perf_counter() - started,
        "protected_test_evaluation_count": 1 if arguments.split == "test" else 0,
    }
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
