"""Analyze saved baseline validation predictions without touching test data."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = PROJECT_ROOT / "accuracy_research" / "baseline" / "validation_predictions.jsonl"
TRAIN_PATH = PROJECT_ROOT / "data" / "permutations_4581" / "splits" / "train.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "error_analysis.json"


NEGATIONS = {"not", "never", "no", "neither", "nor"}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
AUXILIARIES = {
    "am", "are", "is", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
}
PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "by", "despite", "down", "during", "except", "for", "from", "in", "inside",
    "into", "near", "of", "off", "on", "onto", "out", "outside", "over", "past",
    "since", "through", "throughout", "to", "toward", "under", "until", "up",
    "upon", "with", "within", "without",
}
PRONOUNS = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "we", "us", "our", "ours", "ourselves", "they", "them", "their",
    "theirs", "themselves", "who", "whom", "whose", "which", "that",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def unique_permutation_count(words: list[str]) -> int:
    denominator = math.prod(math.factorial(count) for count in Counter(words).values())
    return math.factorial(len(words)) // denominator


def canonical_position_permutation(words: list[str], target: list[str]) -> list[int]:
    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, word in enumerate(words):
        positions[word].append(index)
    return [positions[word].popleft() for word in target]


def kendall_distance(target: list[str], prediction: list[str]) -> int:
    target_positions = canonical_position_permutation(target, prediction)
    return sum(
        target_positions[left] > target_positions[right]
        for left in range(5)
        for right in range(left + 1, 5)
    )


def training_word_frequencies() -> Counter[str]:
    target_rows = Counter(record["target"] for record in read_jsonl(TRAIN_PATH))
    frequencies = Counter()
    for target, rows in target_rows.items():
        words = target.split()
        base_occurrences = rows // unique_permutation_count(words)
        for word in set(words):
            frequencies[word] += base_occurrences
    return frequencies


def category_flags(target: list[str]) -> dict[str, bool]:
    word_set = set(target)
    return {
        "contains_negation": bool(word_set & NEGATIONS),
        "contains_modal": bool(word_set & MODALS),
        "contains_auxiliary": bool(word_set & AUXILIARIES),
        "contains_preposition": bool(word_set & PREPOSITIONS),
        "contains_pronoun": bool(word_set & PRONOUNS),
        "contains_ly_adverb": any(word.endswith("ly") for word in target),
        "contains_repeated_word": len(word_set) < 5,
    }


def summarize_bucket(rows: list[dict]) -> dict:
    records = sum(row["records"] for row in rows)
    exact = sum(row["exact_records"] for row in rows)
    return {
        "target_groups": len(rows),
        "records": records,
        "exact_records": exact,
        "record_weighted_exact_accuracy": exact / records if records else 0.0,
        "mean_target_exact_accuracy": (
            sum(row["exact_rate"] for row in rows) / len(rows) if rows else 0.0
        ),
    }


def main() -> None:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for record in read_jsonl(PREDICTIONS_PATH):
        grouped[tuple(sorted(record["input"].split()))].append(record)

    train_frequency = training_word_frequencies()
    group_rows = []
    hamming_distribution = Counter()
    kendall_distribution = Counter()
    wrong_pair_counts = Counter()

    for key, records in grouped.items():
        targets = {record["target"] for record in records}
        if len(targets) != 1:
            raise ValueError("Validation multiset has conflicting targets")
        target = next(iter(targets))
        target_words = target.split()
        exact_records = sum(bool(record["exact_match"]) for record in records)
        prediction_counts = Counter(record["prediction"] for record in records)
        modal_prediction, modal_count = sorted(
            prediction_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        canonical_input = " ".join(sorted(key))
        canonical_records = [record for record in records if record["input"] == canonical_input]
        if len(canonical_records) != 1:
            raise ValueError("Expected one lexicographically canonical input permutation")
        canonical_prediction = canonical_records[0]["prediction"]

        for record in records:
            prediction_words = record["prediction"].split()
            hamming = sum(
                predicted != expected
                for predicted, expected in zip(prediction_words, target_words)
            )
            hamming_distribution[hamming] += 1
            kendall_distribution[kendall_distance(target_words, prediction_words)] += 1
            if not record["exact_match"]:
                wrong_pair_counts[(target, record["prediction"])] += 1

        min_frequency = min(train_frequency[word] for word in set(target_words))
        group_rows.append(
            {
                "multiset": " ".join(key),
                "target": target,
                "records": len(records),
                "exact_records": exact_records,
                "exact_rate": exact_records / len(records),
                "distinct_predictions": len(prediction_counts),
                "modal_prediction": modal_prediction,
                "modal_share": modal_count / len(records),
                "modal_is_exact": modal_prediction == target,
                "canonical_prediction": canonical_prediction,
                "canonical_is_exact": canonical_prediction == target,
                "any_input_permutation_is_exact": exact_records > 0,
                "min_training_base_sentence_frequency": min_frequency,
                **category_flags(target_words),
            }
        )

    total_records = sum(row["records"] for row in group_rows)
    total_exact = sum(row["exact_records"] for row in group_rows)
    modal_exact_records = sum(
        row["records"] for row in group_rows if row["modal_is_exact"]
    )
    canonical_exact_records = sum(
        row["records"] for row in group_rows if row["canonical_is_exact"]
    )
    oracle_exact_records = sum(
        row["records"]
        for row in group_rows
        if row["any_input_permutation_is_exact"]
    )

    categories = {}
    for category in category_flags(["placeholder"]):
        positive = [row for row in group_rows if row[category]]
        negative = [row for row in group_rows if not row[category]]
        categories[category] = {
            "present": summarize_bucket(positive),
            "absent": summarize_bucket(negative),
        }

    rare_bins = {
        "1": [row for row in group_rows if row["min_training_base_sentence_frequency"] == 1],
        "2_to_5": [row for row in group_rows if 2 <= row["min_training_base_sentence_frequency"] <= 5],
        "6_to_20": [row for row in group_rows if 6 <= row["min_training_base_sentence_frequency"] <= 20],
        "over_20": [row for row in group_rows if row["min_training_base_sentence_frequency"] > 20],
    }

    top_error_pairs = [
        {"target": target, "prediction": prediction, "records": count}
        for (target, prediction), count in wrong_pair_counts.most_common(30)
    ]
    report = {
        "validation_records": total_records,
        "validation_target_groups": len(group_rows),
        "baseline_exact_records": total_exact,
        "baseline_exact_accuracy": total_exact / total_records,
        "target_group_outcomes": {
            "all_permutations_correct": sum(row["exact_rate"] == 1.0 for row in group_rows),
            "no_permutations_correct": sum(row["exact_rate"] == 0.0 for row in group_rows),
            "partially_correct": sum(0.0 < row["exact_rate"] < 1.0 for row in group_rows),
            "single_prediction_for_all_input_permutations": sum(
                row["distinct_predictions"] == 1 for row in group_rows
            ),
            "mean_distinct_predictions": sum(row["distinct_predictions"] for row in group_rows) / len(group_rows),
            "mean_modal_share": sum(row["modal_share"] for row in group_rows) / len(group_rows),
        },
        "decoding_only_opportunities": {
            "lexicographically_canonical_source_exact_accuracy": canonical_exact_records / total_records,
            "all_source_permutations_modal_vote_exact_accuracy": modal_exact_records / total_records,
            "oracle_if_any_source_permutation_prediction_is_correct": oracle_exact_records / total_records,
            "note": "canonical and modal methods use no target at inference; oracle is diagnostic only",
        },
        "error_distance": {
            "hamming_wrong_positions": dict(sorted(hamming_distribution.items())),
            "kendall_inversions": dict(sorted(kendall_distribution.items())),
        },
        "linguistic_heuristic_categories": categories,
        "minimum_training_word_frequency_bins": {
            name: summarize_bucket(rows) for name, rows in rare_bins.items()
        },
        "top_repeated_validation_error_pairs": top_error_pairs,
        "protected_test_opened": False,
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
