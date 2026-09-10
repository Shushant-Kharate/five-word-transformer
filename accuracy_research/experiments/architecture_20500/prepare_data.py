"""Prepare leakage-free base splits and a train-only tokenizer for 20,500 targets.

This script does not generate all 120 input permutations.  The architecture is
permutation-invariant and the training dataset randomizes source positions at
access time, so storing repeated permutation rows is unnecessary.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data" / "extended_4581" / "dataset_20500.jsonl"
OUTPUT_ROOT = ROOT / "data" / "extended_4581" / "prepared_20500"
SPLIT_ROOT = OUTPUT_ROOT / "splits"
TOKENIZER_PATH = OUTPUT_ROOT / "tokenizer.json"
MANIFEST_PATH = OUTPUT_ROOT / "preparation_manifest.json"
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
SEED = 42
SPLIT_COUNTS = {"train": 16_400, "validation": 2_050, "test": 2_050}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    targets: set[str] = set()
    multisets: set[tuple[str, ...]] = set()
    with SOURCE.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if set(record) != {"input", "target"}:
                raise ValueError(f"invalid fields at source line {line_number}")
            source = record["input"].split()
            target = record["target"].split()
            if len(source) != 5 or len(target) != 5 or Counter(source) != Counter(target):
                raise ValueError(f"invalid five-word permutation at source line {line_number}")
            if any(not token.isascii() or not token.isalpha() or token != token.lower() for token in source + target):
                raise ValueError(f"invalid token at source line {line_number}")
            target_text = " ".join(target)
            key = tuple(sorted(target))
            if target_text in targets or key in multisets:
                raise ValueError(f"duplicate target or unordered multiset at source line {line_number}")
            targets.add(target_text)
            multisets.add(key)
            records.append(record)
    if len(records) != sum(SPLIT_COUNTS.values()):
        raise ValueError(f"expected 20,500 records, found {len(records)}")
    return records


def assign_splits(records: list[dict[str, str]]) -> list[str]:
    """Hold out groups while keeping every holdout word represented in train."""
    word_group_frequency = Counter(
        word for record in records for word in set(record["target"].split())
    )
    order = list(range(len(records)))
    random.Random(SEED).shuffle(order)
    assignments = ["train"] * len(records)
    heldout_word_frequency: Counter[str] = Counter()
    used: set[int] = set()
    for split in ("validation", "test"):
        remaining = SPLIT_COUNTS[split]
        for index in order:
            if index in used:
                continue
            target_words = set(records[index]["target"].split())
            if any(
                heldout_word_frequency[word] + 1 >= word_group_frequency[word]
                for word in target_words
            ):
                continue
            assignments[index] = split
            used.add(index)
            heldout_word_frequency.update(target_words)
            remaining -= 1
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(f"could not assign {remaining} records to {split}")
    if Counter(assignments) != Counter(SPLIT_COUNTS):
        raise AssertionError(f"unexpected split counts: {Counter(assignments)}")
    return assignments


def main() -> None:
    records = read_records()
    assignments = assign_splits(records)
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    split_paths = {name: SPLIT_ROOT / f"{name}.jsonl" for name in SPLIT_COUNTS}
    handles = {
        name: path.open("w", encoding="utf-8", newline="\n")
        for name, path in split_paths.items()
    }
    assignment_path = OUTPUT_ROOT / "split_assignments.jsonl"
    try:
        with assignment_path.open("w", encoding="utf-8", newline="\n") as assignment_handle:
            for source_record, (record, split) in enumerate(zip(records, assignments), start=1):
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                assignment_handle.write(
                    json.dumps(
                        {
                            "source_record": source_record,
                            "target": record["target"],
                            "split": split,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        for handle in handles.values():
            handle.close()

    train_records = [record for record, split in zip(records, assignments) if split == "train"]
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        (record["target"] for record in train_records),
        WordLevelTrainer(special_tokens=SPECIAL_TOKENS, min_frequency=1),
    )
    tokenizer.save(str(TOKENIZER_PATH))

    vocabulary_by_split = {
        split: {
            word
            for record, assigned in zip(records, assignments)
            if assigned == split
            for word in record["target"].split()
        }
        for split in SPLIT_COUNTS
    }
    target_by_split = {
        split: {
            record["target"]
            for record, assigned in zip(records, assignments)
            if assigned == split
        }
        for split in SPLIT_COUNTS
    }
    manifest = {
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "method": "deterministic target-group 80:10:10 split before augmentation",
        "seed": SEED,
        "split_counts": dict(Counter(assignments)),
        "tokenizer": {
            "path": str(TOKENIZER_PATH),
            "sha256": sha256(TOKENIZER_PATH),
            "training_source": "training targets only",
            "model": "WordLevel",
            "pre_tokenizer": "Whitespace",
            "special_tokens": SPECIAL_TOKENS,
            "vocabulary_size": tokenizer.get_vocab_size(),
        },
        "files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in split_paths.items()
        },
        "assignment_file": {
            "path": str(assignment_path),
            "sha256": sha256(assignment_path),
        },
        "validation": {
            "no_target_overlap": not (
                target_by_split["train"] & target_by_split["validation"]
                or target_by_split["train"] & target_by_split["test"]
                or target_by_split["validation"] & target_by_split["test"]
            ),
            "validation_words_missing_from_train": len(
                vocabulary_by_split["validation"] - vocabulary_by_split["train"]
            ),
            "test_words_missing_from_train": len(
                vocabulary_by_split["test"] - vocabulary_by_split["train"]
            ),
            "permutations_materialized": False,
            "future_training_permutations": "dynamic per access",
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
