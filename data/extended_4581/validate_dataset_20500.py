"""Independently validate the expanded five-word base dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARTICLES = {"a", "an", "the"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=HERE / "dataset_20500.jsonl")
    parser.add_argument("--expected-records", type=int, default=20_500)
    parser.add_argument("--report", type=Path, default=HERE / "dataset_20500_validation.json")
    args = parser.parse_args()

    errors: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    multisets: Counter[tuple[str, ...]] = Counter()
    vocabulary: Counter[str] = Counter()
    records = 0
    repeated_word_records = 0

    with args.dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors["blank_lines"] += 1
                continue
            records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors["invalid_json"] += 1
                continue
            if set(record) != {"input", "target"}:
                errors["invalid_fields"] += 1
                continue
            source = record["input"].split()
            target = record["target"].split()
            if len(source) != 5 or len(target) != 5:
                errors["invalid_word_count"] += 1
            if any(not token.isascii() or not token.isalpha() for token in source + target):
                errors["non_ascii_alphabetic_tokens"] += 1
            if any(token != token.lower() for token in source + target):
                errors["uppercase_tokens"] += 1
            if ARTICLES & set(target):
                errors["article_records"] += 1
            if Counter(source) != Counter(target):
                errors["non_permutation_pairs"] += 1
            if source == target:
                errors["unshuffled_inputs"] += 1
            target_text = " ".join(target)
            targets[target_text] += 1
            multisets[tuple(sorted(target))] += 1
            vocabulary.update(target)
            repeated_word_records += len(set(target)) < len(target)

    errors["wrong_record_total"] = int(records != args.expected_records)
    duplicate_target_rows = sum(count - 1 for count in targets.values())
    conflicting_multiset_rows = sum(count - 1 for count in multisets.values())
    errors["duplicate_target_rows"] = duplicate_target_rows
    errors["conflicting_multiset_rows"] = conflicting_multiset_rows
    errors = Counter({key: value for key, value in errors.items() if value})

    report = {
        "dataset": str(args.dataset.resolve()),
        "sha256": sha256(args.dataset),
        "records": records,
        "unique_targets": len(targets),
        "unique_unordered_word_multisets": len(multisets),
        "vocabulary_size": len(vocabulary),
        "records_with_repeated_words": repeated_word_records,
        "errors": dict(errors),
        "passed": not errors,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
