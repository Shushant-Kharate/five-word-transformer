"""Build a clean 20,500-record five-word reconstruction base dataset.

The original ``dataset_4581.jsonl`` is read-only input.  New natural English
targets come from the Tatoeba English sentence export.  This script deliberately
does not create the 120 permutations; those belong to the later augmentation
stage, after target-group train/validation/test splitting.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "dataset_4581.jsonl"
DEFAULT_OUTPUT = HERE / "dataset_20500.jsonl"
DEFAULT_PROVENANCE = HERE / "dataset_20500_provenance.jsonl"
DEFAULT_REPORT = HERE / "dataset_20500_report.json"
TARGET_RECORDS = 20_500
MIN_CORPUS_DOCUMENT_FREQUENCY = 20
MAX_SELECTED_CONTENT_FREQUENCY = 250
ARTICLES = {"a", "an", "the"}
FUNCTION_WORDS = {
    "about", "after", "again", "against", "all", "also", "am", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "between", "both",
    "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
}
RAW_SENTENCE_PATTERN = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+){4}[.!?]?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tatoeba", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-records", type=int, default=TARGET_RECORDS)
    parser.add_argument(
        "--min-corpus-document-frequency",
        type=int,
        default=MIN_CORPUS_DOCUMENT_FREQUENCY,
    )
    parser.add_argument(
        "--max-selected-content-frequency",
        type=int,
        default=MAX_SELECTED_CONTENT_FREQUENCY,
    )
    parser.add_argument("--seed", type=int, default=20_500)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def words(text: str) -> list[str]:
    return text.split()


def multiset_key(text: str) -> tuple[str, ...]:
    return tuple(sorted(words(text)))


def validate_pair(record: dict, location: str) -> None:
    if set(record) != {"input", "target"}:
        raise ValueError(f"{location}: fields must be exactly input and target")
    source = words(record["input"])
    target = words(record["target"])
    if len(source) != 5 or len(target) != 5:
        raise ValueError(f"{location}: input and target must contain five words")
    for token in source + target:
        if token != token.lower() or not token.isascii() or not token.isalpha():
            raise ValueError(f"{location}: token {token!r} is not lowercase ASCII alphabetic")
    if ARTICLES & set(target):
        raise ValueError(f"{location}: articles a, an, and the are forbidden")
    if Counter(source) != Counter(target):
        raise ValueError(f"{location}: input is not a target permutation")


def normalize_tatoeba(text: str) -> str | None:
    """Accept only unambiguous five-token ASCII sentences.

    A single terminal full stop, question mark, or exclamation mark is removed.
    All other punctuation, contractions, hyphenation, digits, and non-ASCII text
    are rejected rather than rewritten.
    """
    if not RAW_SENTENCE_PATTERN.fullmatch(text):
        return None
    target = text[:-1] if text[-1:] in ".!?" else text
    target = target.lower()
    target_words = words(target)
    if len(target_words) != 5 or ARTICLES & set(target_words):
        return None
    return target


def read_base(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_pair(record, f"{path}:{line_number}")
            yield line_number, record


def read_tatoeba(path: Path):
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3 or parts[1] != "eng":
                continue
            target = normalize_tatoeba(parts[2])
            if target is not None:
                yield line_number, parts[0], target


def shuffled_input(target: str, seed: int) -> str:
    target_words = words(target)
    derived_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{target}".encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(derived_seed)
    shuffled = target_words.copy()
    rng.shuffle(shuffled)
    if shuffled == target_words:
        shuffled = shuffled[1:] + shuffled[:1]
    return " ".join(shuffled)


def main() -> None:
    args = parse_args()
    if args.target_records <= 0:
        raise ValueError("target-records must be positive")

    # Preserve one label per exact target and one label per unordered multiset.
    # This removes the old duplicate and conflicting-label inputs.
    selected: list[tuple[dict, dict]] = []
    target_set: set[str] = set()
    multiset_set: set[tuple[str, ...]] = set()
    dropped_base_duplicate_targets = 0
    dropped_base_conflicting_multisets = 0
    for line_number, record in read_base(args.base):
        target = record["target"]
        key = multiset_key(target)
        if target in target_set:
            dropped_base_duplicate_targets += 1
            continue
        if key in multiset_set:
            dropped_base_conflicting_multisets += 1
            continue
        target_set.add(target)
        multiset_set.add(key)
        selected.append(
            (
                record,
                {
                    "source": "original_dataset_4581",
                    "source_record": line_number,
                },
            )
        )

    # First pass establishes corpus-wide sentence document frequency.  Requiring
    # recurring words avoids flooding a WordLevel model with one-off vocabulary.
    candidates: list[tuple[str, str]] = []
    document_frequency: Counter[str] = Counter()
    for _, sentence_id, target in read_tatoeba(args.tatoeba):
        candidates.append((sentence_id, target))
        document_frequency.update(set(words(target)))

    eligible: list[tuple[str, str]] = []
    candidate_target_set: set[str] = set()
    candidate_multiset_set: set[tuple[str, ...]] = set()
    for sentence_id, target in candidates:
        key = multiset_key(target)
        if target in target_set or target in candidate_target_set:
            continue
        if key in multiset_set or key in candidate_multiset_set:
            continue
        if min(document_frequency[word] for word in set(words(target))) < args.min_corpus_document_frequency:
            continue
        candidate_target_set.add(target)
        candidate_multiset_set.add(key)
        eligible.append((sentence_id, target))

    # Stable hash ordering samples across the export instead of taking one
    # chronological block.  It is deterministic across machines.
    eligible.sort(
        key=lambda item: hashlib.sha256(
            f"{args.seed}:{item[0]}:{item[1]}".encode("utf-8")
        ).digest()
    )
    needed = args.target_records - len(selected)
    if needed < 0:
        raise ValueError("target-records is smaller than the cleaned base")
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} eligible additions for {needed} required")

    additions: list[tuple[str, str]] = []
    selected_content_frequency: Counter[str] = Counter()
    for sentence_id, target in eligible:
        content_words = set(words(target)) - FUNCTION_WORDS
        if any(
            selected_content_frequency[word] >= args.max_selected_content_frequency
            for word in content_words
        ):
            continue
        additions.append((sentence_id, target))
        selected_content_frequency.update(content_words)
        if len(additions) == needed:
            break
    if len(additions) < needed:
        raise RuntimeError(
            f"content-frequency cap allowed only {len(additions)} of {needed} additions"
        )

    for sentence_id, target in additions:
        record = {"input": shuffled_input(target, args.seed), "target": target}
        validate_pair(record, f"Tatoeba sentence {sentence_id}")
        selected.append(
            (
                record,
                {
                    "source": "tatoeba_english_export",
                    "tatoeba_sentence_id": int(sentence_id),
                },
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("w", encoding="utf-8", newline="\n") as dataset_handle,
        args.provenance.open("w", encoding="utf-8", newline="\n") as provenance_handle,
    ):
        for output_record, (record, provenance) in enumerate(selected, start=1):
            dataset_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            provenance_handle.write(
                json.dumps({"output_record": output_record, **provenance}, ensure_ascii=False)
                + "\n"
            )

    vocabulary = Counter()
    repeated_word_records = 0
    correctly_ordered_inputs = 0
    for record, _ in selected:
        target_words = words(record["target"])
        vocabulary.update(target_words)
        repeated_word_records += len(set(target_words)) < 5
        correctly_ordered_inputs += record["input"] == record["target"]

    report = {
        "dataset": str(args.output.resolve()),
        "dataset_sha256": sha256(args.output),
        "records": len(selected),
        "unique_targets": len({record["target"] for record, _ in selected}),
        "unique_unordered_word_multisets": len(
            {multiset_key(record["target"]) for record, _ in selected}
        ),
        "original_base": str(args.base.resolve()),
        "original_base_sha256": sha256(args.base),
        "retained_clean_base_records": len(selected) - needed,
        "dropped_base_duplicate_targets": dropped_base_duplicate_targets,
        "dropped_base_conflicting_multisets": dropped_base_conflicting_multisets,
        "added_tatoeba_records": needed,
        "tatoeba_export": str(args.tatoeba.resolve()),
        "tatoeba_export_sha256": sha256(args.tatoeba),
        "tatoeba_download_url": "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2",
        "tatoeba_license": "CC BY 2.0 FR",
        "tatoeba_attribution_url": "https://tatoeba.org/",
        "selection": {
            "raw_five_word_candidates": len(candidates),
            "eligible_unique_nonconflicting_candidates": len(eligible),
            "minimum_corpus_sentence_document_frequency_per_word": args.min_corpus_document_frequency,
            "maximum_selected_sentence_frequency_per_content_word": args.max_selected_content_frequency,
            "highest_observed_selected_content_word_frequency": max(selected_content_frequency.values()),
            "stable_selection_seed": args.seed,
        },
        "validation": {
            "fields_exactly_input_and_target": True,
            "exactly_five_words": True,
            "lowercase_ascii_alphabetic_only": True,
            "contains_no_articles_a_an_the": True,
            "input_target_multisets_equal": True,
            "duplicate_targets": 0,
            "conflicting_unordered_multisets": 0,
            "correctly_ordered_inputs": correctly_ordered_inputs,
        },
        "vocabulary": {
            "distinct_words": len(vocabulary),
            "words_occurring_once": sum(count == 1 for count in vocabulary.values()),
            "words_occurring_at_most_five_times": sum(count <= 5 for count in vocabulary.values()),
        },
        "records_with_repeated_words": repeated_word_records,
        "note": "Split by target group before permutation augmentation; never randomly split expanded permutation rows.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
