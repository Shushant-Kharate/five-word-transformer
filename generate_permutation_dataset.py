"""Generate every unique word-order permutation for each five-word target.

The training output remains compatible with the existing project: every JSONL
record contains only ``input`` and ``target``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import permutations
from math import factorial
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "extended_4581" / "dataset_4581.jsonl"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "permutations_4581"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_unique_permutations(words: list[str]) -> int:
    """Return n! / product(count(word)!) for a multiset of words."""
    denominator = 1
    for count in Counter(words).values():
        denominator *= factorial(count)
    return factorial(len(words)) // denominator


def unique_permutations(words: list[str]) -> list[tuple[str, ...]]:
    """Return deterministic unique permutations, including original order.

    ``itertools.permutations`` treats equal words at different positions as
    separate input elements. Converting its small five-word result to a set
    removes those duplicate tuples; sorting makes the output reproducible.
    """
    return sorted(set(permutations(words)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create all unique word permutations for every target sentence."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_path = output_directory / "dataset_all_permutations.jsonl"
    index_path = output_directory / "permutation_index.jsonl"
    report_path = output_directory / "permutation_report.json"

    source_records = 0
    output_records = 0
    sentences_with_repeated_words = 0
    count_distribution: Counter[int] = Counter()
    target_order_records = 0

    with (
        source.open("r", encoding="utf-8") as source_handle,
        dataset_path.open("w", encoding="utf-8", newline="\n") as dataset_handle,
        index_path.open("w", encoding="utf-8", newline="\n") as index_handle,
    ):
        for source_record_id, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if set(record) != {"input", "target"}:
                raise ValueError(
                    f"Source record {source_record_id} must contain only input and target"
                )

            target = record["target"]
            words = target.split()
            if len(words) != 5:
                raise ValueError(
                    f"Source record {source_record_id} does not contain five target words"
                )

            arrangements = unique_permutations(words)
            expected = expected_unique_permutations(words)
            if len(arrangements) != expected:
                raise AssertionError(
                    f"Permutation count mismatch at source record {source_record_id}"
                )

            if len(set(words)) < len(words):
                sentences_with_repeated_words += 1

            first_output_record = output_records + 1
            for arrangement in arrangements:
                input_sentence = " ".join(arrangement)
                dataset_handle.write(
                    json.dumps(
                        {"input": input_sentence, "target": target},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_records += 1
                if input_sentence == target:
                    target_order_records += 1

            index_handle.write(
                json.dumps(
                    {
                        "source_record_id": source_record_id,
                        "target": target,
                        "word_counts": dict(Counter(words)),
                        "unique_permutations": len(arrangements),
                        "first_output_record": first_output_record,
                        "last_output_record": output_records,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            source_records += 1
            count_distribution[len(arrangements)] += 1

    report = {
        "source_file": str(source),
        "source_sha256": sha256(source),
        "source_records": source_records,
        "output_file": dataset_path.name,
        "output_records": output_records,
        "output_sha256": sha256(dataset_path),
        "index_file": index_path.name,
        "index_sha256": sha256(index_path),
        "algorithm": "sorted(set(itertools.permutations(words)))",
        "count_formula": "n! / product(frequency_of_each_distinct_word!)",
        "includes_correct_target_order": True,
        "correct_target_order_records": target_order_records,
        "sentences_with_repeated_words": sentences_with_repeated_words,
        "unique_permutation_count_distribution": {
            str(count): sentences
            for count, sentences in sorted(count_distribution.items())
        },
        "validation": {
            "all_source_records_processed": source_records == 4581,
            "one_correctly_ordered_per_source_record": target_order_records == source_records,
            "every_per_sentence_count_matches_multiset_formula": True,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
