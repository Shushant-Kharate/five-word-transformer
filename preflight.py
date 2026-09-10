"""Read-only pre-training readiness gate.

Run this immediately before ``python train.py --train``. It verifies that the
environment, frozen data, tokenizer, model, loss path, and output state match
the approved experiment without updating model weights or writing artifacts.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import torch

from config import PROJECT_ROOT, get_config
from dataset import get_datasets, get_or_build_tokenizer, load_jsonl
from train import build_training_components, calculate_batch_loss


EXPECTED_PACKAGES = {
    "torch": "2.0.1",
    "tokenizers": "0.13.3",
    "numpy": "1.24.4",
    "tqdm": "4.66.1",
    "tensorboard": "2.13.0",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_preflight() -> dict:
    config = get_config()
    manifest_path = Path(config["manifest_file"])
    require(manifest_path.is_file(), f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    require(
        platform.python_version() == "3.11.9",
        f"Expected Python 3.11.9, found {platform.python_version()}",
    )
    installed_packages = {
        package: version(package) for package in EXPECTED_PACKAGES
    }
    require(
        installed_packages == EXPECTED_PACKAGES,
        f"Dependency mismatch: {installed_packages}",
    )

    hashed_paths = {
        "train.jsonl": Path(config["train_file"]),
        "validation.jsonl": Path(config["validation_file"]),
        "test.jsonl": Path(config["test_file"]),
        "tokenizer.json": Path(config["tokenizer_file"]),
    }
    observed_hashes = {name: sha256(path) for name, path in hashed_paths.items()}
    require(
        observed_hashes == manifest["sha256"],
        "Dataset or tokenizer content changed after the manifest was frozen",
    )

    train_records = load_jsonl(config["train_file"])
    validation_records = load_jsonl(config["validation_file"])
    test_records = load_jsonl(config["test_file"])
    observed_counts = {
        "train.jsonl": len(train_records),
        "validation.jsonl": len(validation_records),
        "test.jsonl": len(test_records),
    }
    require(observed_counts == manifest["records"], "Record counts changed")

    pair_sets = [
        {(row["input"], row["target"]) for row in records}
        for records in (train_records, validation_records, test_records)
    ]
    require(not (pair_sets[0] & pair_sets[1]), "Train/validation overlap")
    require(not (pair_sets[0] & pair_sets[2]), "Train/test overlap")
    require(not (pair_sets[1] & pair_sets[2]), "Validation/test overlap")
    require(
        sum(len(pair_set) for pair_set in pair_sets)
        == sum(len(records) for records in (train_records, validation_records, test_records)),
        "Duplicate input-target records exist inside a split",
    )

    tokenizer = get_or_build_tokenizer(config)
    require(
        tokenizer.get_vocab_size() == manifest["tokenizer"]["vocabulary_size"],
        "Vocabulary size changed",
    )
    for token, token_id in manifest["tokenizer"]["special_tokens"].items():
        require(tokenizer.token_to_id(token) == token_id, f"Changed ID for {token}")

    train_dataset, validation_dataset, test_dataset = get_datasets(
        config, tokenizer
    )
    require(
        (len(train_dataset), len(validation_dataset), len(test_dataset))
        == tuple(manifest["split_sizes"]),
        "Dataset loader sizes changed",
    )

    (
        built_tokenizer,
        train_dataloader,
        model,
        _,
        loss_function,
        device,
    ) = build_training_components(config)
    require(
        len(train_dataloader) == manifest["batches_per_epoch"],
        "Training batch count changed",
    )
    require(
        sum(parameter.numel() for parameter in model.parameters())
        == manifest["model_parameters"],
        "Model parameter count changed",
    )
    first_batch = next(iter(train_dataloader))
    model.eval()
    with torch.no_grad():
        loss = calculate_batch_loss(
            model,
            first_batch,
            loss_function,
            built_tokenizer.get_vocab_size(),
            device,
        )
    require(bool(torch.isfinite(loss)), "Pre-training loss is not finite")

    output_state = {
        "weights": Path(config["model_folder"]).exists(),
        "runs": Path(config["experiment_name"]).parent.exists(),
        "results": Path(config["results_folder"]).exists(),
    }
    require(
        not any(output_state.values()),
        f"Experiment output folders are not clean: {output_state}",
    )

    free_disk_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    require(free_disk_gb >= 1.0, "Less than 1 GB of free disk space")

    return {
        "status": "READY",
        "python": platform.python_version(),
        "packages": installed_packages,
        "device": str(device),
        "free_disk_gb": round(free_disk_gb, 2),
        "records": observed_counts,
        "vocabulary_size": tokenizer.get_vocab_size(),
        "model_parameters": manifest["model_parameters"],
        "batches_per_epoch": len(train_dataloader),
        "initial_loss_is_finite": True,
        "output_folders_clean": True,
        "manifest": str(manifest_path),
    }


if __name__ == "__main__":
    print(json.dumps(run_preflight(), indent=2))
