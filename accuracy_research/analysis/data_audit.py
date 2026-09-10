"""Audit the authoritative five-word data without modifying it.

The report focuses on properties that can limit exact-match accuracy:
ambiguous word multisets, split leakage, repeated words, vocabulary coverage,
permutation-class balance, and malformed records. Test targets are used only
for aggregate integrity checks; no examples or test-derived tuning features
are emitted.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
BASE_FILE = DATA_ROOT / "extended_4581" / "dataset_4581.jsonl"
SPLIT_ROOT = DATA_ROOT / "permutations_4581" / "splits"
OUTPUT_PATH = Path(__file__).resolve().parent / "data_audit.json"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def multiset_key(words: list[str]) -> tuple[str, ...]:
    return tuple(sorted(words))


def unique_permutation_count(words: list[str]) -> int:
    denominator = math.prod(math.factorial(count) for count in Counter(words).values())
    return math.factorial(len(words)) // denominator


def pointer_permutation(source: list[str], target: list[str]) -> tuple[int, ...]:
    """Return a deterministic target-to-source-position assignment."""
    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, word in enumerate(source):
        positions[word].append(index)
    result = []
    for word in target:
        if not positions[word]:
            raise ValueError("Target is not a source-position permutation")
        result.append(positions[word].popleft())
    return tuple(result)


def audit_base() -> tuple[dict, dict[tuple[str, ...], Counter[str]]]:
    records = 0
    invalid_fields = 0
    invalid_lengths = 0
    non_permutation_pairs = 0
    nonalpha_tokens = 0
    uppercase_tokens = 0
    targets = Counter()
    multisets: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    repeat_patterns = Counter()
    word_source_frequency = Counter()
    article_records = 0

    for _, record in read_jsonl(BASE_FILE):
        records += 1
        if set(record) != {"input", "target"}:
            invalid_fields += 1
            continue
        source = record["input"].split()
        target = record["target"].split()
        if len(source) != 5 or len(target) != 5:
            invalid_lengths += 1
        if Counter(source) != Counter(target):
            non_permutation_pairs += 1
        nonalpha_tokens += sum(not token.isalpha() for token in source + target)
        uppercase_tokens += sum(token != token.lower() for token in source + target)
        if {"a", "an", "the"} & set(target):
            article_records += 1
        target_text = " ".join(target)
        targets[target_text] += 1
        multisets[multiset_key(target)][target_text] += 1
        repeat_patterns[tuple(sorted(Counter(target).values(), reverse=True))] += 1
        word_source_frequency.update(set(target))

    ambiguous = {key: counts for key, counts in multisets.items() if len(counts) > 1}
    unavoidable_misses = sum(sum(counts.values()) - max(counts.values()) for counts in ambiguous.values())
    return (
        {
            "records": records,
            "unique_targets": len(targets),
            "duplicate_target_rows": sum(count - 1 for count in targets.values()),
            "duplicate_target_strings": sum(count > 1 for count in targets.values()),
            "invalid_fields": invalid_fields,
            "invalid_lengths": invalid_lengths,
            "non_permutation_pairs": non_permutation_pairs,
            "nonalpha_token_occurrences": nonalpha_tokens,
            "uppercase_token_occurrences": uppercase_tokens,
            "records_with_articles_a_an_the": article_records,
            "vocabulary_size_without_special_tokens": len(word_source_frequency),
            "words_in_one_base_record": sum(count == 1 for count in word_source_frequency.values()),
            "words_in_at_most_five_base_records": sum(count <= 5 for count in word_source_frequency.values()),
            "repeat_pattern_distribution": {
                "-".join(map(str, pattern)): count
                for pattern, count in sorted(repeat_patterns.items())
            },
            "ambiguous_multiset_groups": len(ambiguous),
            "base_rows_in_ambiguous_multisets": sum(sum(counts.values()) for counts in ambiguous.values()),
            "label_consistency_ceiling_base_rows": (
                (records - unavoidable_misses) / records if records else 0.0
            ),
        },
        multisets,
    )


def audit_split(path: Path) -> dict:
    records = 0
    targets = Counter()
    multisets: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    vocabulary = set()
    pointer_classes = Counter()
    exact_pair_hashes = set()
    invalid = Counter()

    for line_number, record in read_jsonl(path):
        records += 1
        if set(record) != {"input", "target"}:
            invalid["fields"] += 1
            continue
        source = record["input"].split()
        target = record["target"].split()
        if len(source) != 5 or len(target) != 5:
            invalid["length"] += 1
        if Counter(source) != Counter(target):
            invalid["not_permutation"] += 1
            continue
        target_text = " ".join(target)
        targets[target_text] += 1
        key = multiset_key(source)
        multisets[key][target_text] += 1
        vocabulary.update(target)
        pointer_classes[pointer_permutation(source, target)] += 1
        exact_pair_hashes.add((record["input"], target_text))

    completeness_errors = 0
    source_record_estimates = Counter()
    for target, row_count in targets.items():
        expected = unique_permutation_count(target.split())
        if row_count % expected:
            completeness_errors += 1
        source_record_estimates[target] = row_count // expected

    ambiguous = {key: counts for key, counts in multisets.items() if len(counts) > 1}
    unavoidable_rows = sum(sum(counts.values()) - max(counts.values()) for counts in ambiguous.values())
    return {
        "records": records,
        "unique_input_target_pairs": len(exact_pair_hashes),
        "unique_targets": len(targets),
        "estimated_base_source_records": sum(source_record_estimates.values()),
        "vocabulary": vocabulary,
        "multisets": multisets,
        "target_row_counts": targets,
        "pointer_classes": pointer_classes,
        "invalid": dict(invalid),
        "target_permutation_completeness_errors": completeness_errors,
        "ambiguous_multiset_groups": len(ambiguous),
        "rows_in_ambiguous_multisets": sum(sum(counts.values()) for counts in ambiguous.values()),
        "label_consistency_ceiling_rows": (
            (records - unavoidable_rows) / records if records else 0.0
        ),
    }


def compact_split(split: dict) -> dict:
    class_counts = split["pointer_classes"]
    return {
        key: value
        for key, value in split.items()
        if key not in {"vocabulary", "multisets", "target_row_counts", "pointer_classes"}
    } | {
        "vocabulary_size": len(split["vocabulary"]),
        "pointer_permutation_classes_observed": len(class_counts),
        "pointer_class_min_records": min(class_counts.values()),
        "pointer_class_max_records": max(class_counts.values()),
        "pointer_class_mean_records": sum(class_counts.values()) / len(class_counts),
    }


def main() -> None:
    base_report, _ = audit_base()
    splits = {
        name: audit_split(SPLIT_ROOT / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    names = tuple(splits)
    target_overlap = {}
    multiset_overlap = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pair_name = f"{left}__{right}"
            target_overlap[pair_name] = len(
                set(splits[left]["target_row_counts"])
                & set(splits[right]["target_row_counts"])
            )
            multiset_overlap[pair_name] = len(
                set(splits[left]["multisets"]) & set(splits[right]["multisets"])
            )

    report = {
        "base_file": str(BASE_FILE),
        "base": base_report,
        "splits": {name: compact_split(split) for name, split in splits.items()},
        "split_integrity": {
            "target_overlap_groups": target_overlap,
            "source_multiset_overlap_groups": multiset_overlap,
            "validation_words_missing_from_train": len(
                splits["validation"]["vocabulary"] - splits["train"]["vocabulary"]
            ),
            "test_words_missing_from_train": len(
                splits["test"]["vocabulary"] - splits["train"]["vocabulary"]
            ),
        },
        "interpretation": {
            "pointer_class": "zero-based source positions emitted in target order",
            "label_consistency_ceiling": "maximum exact accuracy possible for a deterministic model if identical source word multisets have conflicting target strings; this is not a model estimate",
            "test_use": "test targets were used only for aggregate integrity counts; no examples or test-derived features were emitted",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
