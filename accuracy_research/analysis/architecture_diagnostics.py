"""Quantify baseline objective/decoding mismatch on validation only."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import best_weights_file_path, get_config  # noqa: E402
from dataset import SentenceReconstructionDataset, get_or_build_tokenizer, load_jsonl  # noqa: E402
from evaluate import load_model_for_evaluation  # noqa: E402


OUTPUT_PATH = Path(__file__).resolve().parent / "architecture_diagnostics.json"
BASELINE_PREDICTIONS = ROOT / "accuracy_research" / "baseline" / "validation_predictions.jsonl"


def percentile_summary(values: list[int]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "maximum": int(array.max()),
    }


def main() -> None:
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_or_build_tokenizer(config)
    model, checkpoint = load_model_for_evaluation(
        config,
        tokenizer,
        best_weights_file_path(config),
        device,
    )
    validation_records = load_jsonl(config["validation_file"])
    dataset = SentenceReconstructionDataset(validation_records, tokenizer, config["seq_len"])
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    vocabulary_size = tokenizer.get_vocab_size()

    word_predictions_full: list[torch.Tensor] = []
    word_predictions_oracle_source: list[torch.Tensor] = []
    word_targets: list[torch.Tensor] = []
    legal_probability_mass: list[float] = []
    target_probability: list[float] = []
    target_ranks: list[int] = []
    unrestricted_top1_is_illegal = 0
    total_word_positions = 0
    target_beats_all_illegal = 0
    target_beats_other_legal = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            targets = batch["label"][:, :5].to(device)
            encoded = model.encode(encoder_input, encoder_mask)
            decoded = model.decode(encoded, encoder_mask, decoder_input, decoder_mask)
            logits = model.project(decoded)[:, :5, :].float()
            probabilities = torch.softmax(logits, dim=-1)
            full_top1 = logits.argmax(dim=-1)

            batch_size = targets.shape[0]
            oracle_predictions = torch.empty_like(targets)
            for output_position in range(5):
                remaining_targets = targets[:, output_position:]
                legal_mask = torch.zeros(
                    (batch_size, vocabulary_size),
                    dtype=torch.bool,
                    device=device,
                )
                legal_mask.scatter_(1, remaining_targets, True)
                step_logits = logits[:, output_position, :]
                constrained_logits = step_logits.masked_fill(~legal_mask, -torch.inf)
                oracle_predictions[:, output_position] = constrained_logits.argmax(dim=-1)

                step_probabilities = probabilities[:, output_position, :]
                legal_probability_mass.extend(
                    step_probabilities.masked_fill(~legal_mask, 0.0).sum(dim=-1).cpu().tolist()
                )
                step_targets = targets[:, output_position]
                target_scores = step_logits.gather(1, step_targets[:, None]).squeeze(1)
                target_probability.extend(
                    step_probabilities.gather(1, step_targets[:, None]).squeeze(1).cpu().tolist()
                )
                target_ranks.extend(
                    (step_logits.gt(target_scores[:, None]).sum(dim=-1) + 1).cpu().tolist()
                )
                unrestricted_top1_is_illegal += int(
                    (~legal_mask.gather(1, full_top1[:, output_position, None]).squeeze(1)).sum()
                )
                illegal_best = step_logits.masked_fill(legal_mask, -torch.inf).max(dim=-1).values
                target_beats_all_illegal += int(target_scores.gt(illegal_best).sum())
                other_legal = legal_mask.clone()
                other_legal.scatter_(1, step_targets[:, None], False)
                other_legal_best = step_logits.masked_fill(~other_legal, -torch.inf).max(dim=-1).values
                target_beats_other_legal += int(target_scores.ge(other_legal_best).sum())
                total_word_positions += batch_size

            word_predictions_full.append(full_top1.cpu())
            word_predictions_oracle_source.append(oracle_predictions.cpu())
            word_targets.append(targets.cpu())

    full = torch.cat(word_predictions_full)
    oracle = torch.cat(word_predictions_oracle_source)
    targets = torch.cat(word_targets)
    full_matches = full.eq(targets)
    oracle_matches = oracle.eq(targets)
    full_exact = full_matches.all(dim=1)
    oracle_exact = oracle_matches.all(dim=1)

    greedy_rows = [json.loads(line) for line in BASELINE_PREDICTIONS.read_text(encoding="utf-8").splitlines()]
    greedy_exact = torch.tensor([bool(row["exact_match"]) for row in greedy_rows])
    if len(greedy_exact) != len(oracle_exact):
        raise ValueError("Baseline prediction count does not match validation records")

    parameter_groups = {
        "source_embedding": sum(parameter.numel() for parameter in model.src_embed.parameters()),
        "target_embedding": sum(parameter.numel() for parameter in model.tgt_embed.parameters()),
        "vocabulary_projection": sum(parameter.numel() for parameter in model.projection_layer.parameters()),
        "encoder": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "decoder": sum(parameter.numel() for parameter in model.decoder.parameters()),
    }
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    lexical_parameters = (
        parameter_groups["source_embedding"]
        + parameter_groups["target_embedding"]
        + parameter_groups["vocabulary_projection"]
    )

    result = {
        "evaluation_split": "validation only",
        "protected_test_opened": False,
        "checkpoint_epoch_human": int(checkpoint["epoch"]) + 1,
        "records": len(targets),
        "word_positions": total_word_positions,
        "parameter_allocation": {
            "total": total_parameters,
            **parameter_groups,
            "three_untied_lexical_matrices": lexical_parameters,
            "lexical_fraction": lexical_parameters / total_parameters,
        },
        "teacher_forced_unrestricted": {
            "word_accuracy": float(full_matches.float().mean()),
            "five_word_exact_accuracy": float(full_exact.float().mean()),
            "top1_illegal_under_remaining_source_rate": unrestricted_top1_is_illegal / total_word_positions,
            "target_rank_in_full_vocabulary": percentile_summary(target_ranks),
            "mean_target_probability": float(np.mean(target_probability)),
        },
        "teacher_forced_oracle_remaining_source": {
            "word_accuracy": float(oracle_matches.float().mean()),
            "five_word_exact_accuracy": float(oracle_exact.float().mean()),
            "mean_probability_mass_on_legal_remaining_words": float(np.mean(legal_probability_mass)),
            "target_beats_every_illegal_token_rate": target_beats_all_illegal / total_word_positions,
            "target_beats_other_legal_words_rate": target_beats_other_legal / total_word_positions,
        },
        "autoregressive_source_constrained": {
            "five_word_exact_accuracy": float(greedy_exact.float().mean()),
        },
        "error_attribution": {
            "records_oracle_teacher_forced_exact_but_greedy_wrong": int((oracle_exact & ~greedy_exact).sum()),
            "rate_oracle_teacher_forced_exact_but_greedy_wrong": float((oracle_exact & ~greedy_exact).float().mean()),
            "records_oracle_teacher_forced_wrong": int((~oracle_exact).sum()),
            "rate_oracle_teacher_forced_wrong": float((~oracle_exact).float().mean()),
            "records_greedy_correct_despite_oracle_teacher_forced_not_exact": int((~oracle_exact & greedy_exact).sum()),
        },
        "architectural_facts": {
            "source_positional_encoding_applied_to_shuffled_input": True,
            "source_and_target_embeddings_tied": False,
            "target_embedding_and_projection_tied": False,
            "training_output_classes": vocabulary_size,
            "inference_legal_word_choices": "at most 5 and decreases each step",
            "label_smoothing_probability_assigned_across_illegal_vocabulary": config["label_smoothing"],
            "layer_norm_uses_sample_standard_deviation": True,
            "layer_norm_epsilon_added_after_standard_deviation": True,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
