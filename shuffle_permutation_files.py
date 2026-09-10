"""Create reproducibly shuffled copies of the permutation JSONL files.

The canonical full permutation dataset remains untouched because its physical
order is linked to permutation_index.jsonl.  Shuffled copies are written to a
separate directory and validated with order-independent record fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "data" / "permutations_4581"
OUTPUT_ROOT = SOURCE_ROOT / "shuffled"
DEFAULT_SEED = 42

FILES = {
    "full": (
        SOURCE_ROOT / "dataset_all_permutations.jsonl",
        OUTPUT_ROOT / "dataset_all_permutations.jsonl",
    ),
    "train": (
        SOURCE_ROOT / "splits" / "train.jsonl",
        OUTPUT_ROOT / "splits" / "train.jsonl",
    ),
    "validation": (
        SOURCE_ROOT / "splits" / "validation.jsonl",
        OUTPUT_ROOT / "splits" / "validation.jsonl",
    ),
    "test": (
        SOURCE_ROOT / "splits" / "test.jsonl",
        OUTPUT_ROOT / "splits" / "test.jsonl",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def multiset_fingerprint(lines: list[str]) -> dict[str, int | str]:
    """Return an order-independent fingerprint that retains duplicates."""
    modulus = 1 << 256
    total = 0
    xor_value = 0
    for line in lines:
        value = int.from_bytes(hashlib.sha256(line.encode("utf-8")).digest(), "big")
        total = (total + value) % modulus
        xor_value ^= value
    return {
        "records": len(lines),
        "hash_sum": f"{total:064x}",
        "hash_xor": f"{xor_value:064x}",
    }


def shuffle_file(source: Path, destination: Path, seed: int) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    before = multiset_fingerprint(lines)

    random.Random(seed).shuffle(lines)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
    temporary.replace(destination)

    with destination.open("r", encoding="utf-8") as handle:
        written_lines = [line for line in handle if line.strip()]
    after = multiset_fingerprint(written_lines)
    if before != after:
        raise AssertionError(f"Records changed while shuffling {source}")

    return {
        "source": str(source),
        "destination": str(destination),
        "seed": seed,
        "records": before["records"],
        "records_unchanged": True,
        "source_sha256": sha256(source),
        "shuffled_sha256": sha256(destination),
        "order_changed": sha256(source) != sha256(destination),
        "multiset_fingerprint": before,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    results = {}
    for offset, (name, (source, destination)) in enumerate(FILES.items()):
        results[name] = shuffle_file(source, destination, args.seed + offset)

    manifest = {
        "purpose": "Randomized physical line order without changing any records or split membership",
        "base_seed": args.seed,
        "canonical_files_preserved": True,
        "files": results,
    }
    manifest_path = OUTPUT_ROOT / "shuffle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
