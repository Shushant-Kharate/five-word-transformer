"""Create leakage-free 80:10:10 splits of the permutation dataset.

Every permutation belonging to the same target sentence is assigned to the
same split. Identical targets in the source are also kept together.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "permutations_4581"
PERMUTATIONS_FILE = DATA_DIR / "dataset_all_permutations.jsonl"
INDEX_FILE = DATA_DIR / "permutation_index.jsonl"
SPLIT_DIR = DATA_DIR / "splits"
SEED = 42
EXPECTED_SOURCE_RECORDS = 4581
DESIRED_SOURCE_COUNTS = {"train": 3665, "validation": 458, "test": 458}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index() -> list[dict]:
    with INDEX_FILE.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assign_target_groups(index: list[dict]) -> dict[str, str]:
    target_groups: dict[str, list[int]] = defaultdict(list)
    for row in index:
        target_groups[row["target"]].append(row["source_record_id"])

    # A WordLevel tokenizer must be built from training only. Select holdout
    # groups while ensuring at least one group containing every holdout word
    # remains in training, preventing [UNK] collisions during reconstruction.
    word_group_counts = Counter(
        word
        for target in target_groups
        for word in set(target.split())
    )
    candidates = [
        target for target, source_ids in target_groups.items() if len(source_ids) == 1
    ]
    random.Random(SEED).shuffle(candidates)
    assignments: dict[str, str] = {}
    heldout_word_counts: Counter[str] = Counter()

    for split in ("validation", "test"):
        needed = DESIRED_SOURCE_COUNTS[split]
        for target in candidates:
            if target in assignments:
                continue
            words = set(target.split())
            if any(
                heldout_word_counts[word] + 1 >= word_group_counts[word]
                for word in words
            ):
                continue
            assignments[target] = split
            heldout_word_counts.update(words)
            needed -= 1
            if needed == 0:
                break
        if needed:
            raise RuntimeError(f"Could not fill {split}; {needed} records remain")

    for target in target_groups:
        assignments.setdefault(target, "train")

    observed = Counter()
    for target, source_ids in target_groups.items():
        observed[assignments[target]] += len(source_ids)
    if dict(observed) != DESIRED_SOURCE_COUNTS:
        raise AssertionError(f"Unexpected source split counts: {observed}")
    return assignments


def main() -> None:
    index = load_index()
    if len(index) != EXPECTED_SOURCE_RECORDS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_RECORDS} index rows, found {len(index)}")
    assignments = assign_target_groups(index)
    split_for_source = {
        row["source_record_id"]: assignments[row["target"]] for row in index
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_paths = {
        split: SPLIT_DIR / f"{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    handles = {
        split: path.open("w", encoding="utf-8", newline="\n")
        for split, path in split_paths.items()
    }
    permutation_counts = Counter()
    source_id = 1
    rows_remaining = index[0]["unique_permutations"]
    try:
        with PERMUTATIONS_FILE.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                split = split_for_source[source_id]
                handles[split].write(line)
                permutation_counts[split] += 1
                rows_remaining -= 1
                if rows_remaining == 0 and source_id < len(index):
                    source_id += 1
                    rows_remaining = index[source_id - 1]["unique_permutations"]
    finally:
        for handle in handles.values():
            handle.close()

    if source_id != len(index) or rows_remaining != 0:
        raise AssertionError("Permutation file and index did not end together")

    assignment_path = SPLIT_DIR / "split_assignments.jsonl"
    with assignment_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in index:
            handle.write(
                json.dumps(
                    {
                        "source_record_id": row["source_record_id"],
                        "target": row["target"],
                        "split": split_for_source[row["source_record_id"]],
                        "unique_permutations": row["unique_permutations"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    targets_by_split = {
        split: {target for target, assigned in assignments.items() if assigned == split}
        for split in ("train", "validation", "test")
    }
    assert not (targets_by_split["train"] & targets_by_split["validation"])
    assert not (targets_by_split["train"] & targets_by_split["test"])
    assert not (targets_by_split["validation"] & targets_by_split["test"])

    source_counts = Counter(split_for_source.values())
    manifest = {
        "method": "grouped deterministic 80:10:10 split with training-vocabulary coverage",
        "seed": SEED,
        "grouping_key": "target sentence",
        "correct_order_included": True,
        "source_dataset": str(PERMUTATIONS_FILE),
        "source_dataset_sha256": sha256(PERMUTATIONS_FILE),
        "source_sentence_records": dict(source_counts),
        "permutation_records": dict(permutation_counts),
        "unique_targets": {split: len(targets) for split, targets in targets_by_split.items()},
        "files": {
            split: {"path": str(path), "sha256": sha256(path)}
            for split, path in split_paths.items()
        },
        "assignment_file": {
            "path": str(assignment_path),
            "sha256": sha256(assignment_path),
        },
        "validation": {
            "source_counts_match_requested": dict(source_counts) == DESIRED_SOURCE_COUNTS,
            "all_permutations_written": sum(permutation_counts.values()) == 542010,
            "no_target_overlap_between_splits": True,
            "all_validation_words_present_in_training": (
                set().union(*(target.split() for target in targets_by_split["validation"]))
                <= set().union(*(target.split() for target in targets_by_split["train"]))
            ),
            "all_test_words_present_in_training": (
                set().union(*(target.split() for target in targets_by_split["test"]))
                <= set().union(*(target.split() for target in targets_by_split["train"]))
            ),
        },
    }
    manifest_path = SPLIT_DIR / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
