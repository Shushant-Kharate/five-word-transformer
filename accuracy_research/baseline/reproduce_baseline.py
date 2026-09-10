"""Reproduce the saved epoch-15 baseline on validation only.

This script never opens the protected test split. It reloads the copied
baseline checkpoint, evaluates all validation records, writes auditable
predictions under ``accuracy_research/baseline``, and records elapsed time and
environment metadata.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import best_weights_file_path, get_config  # noqa: E402
from evaluate import evaluate_model, load_model_for_evaluation  # noqa: E402
from train import (  # noqa: E402
    get_loss_function,
    get_validation_dataloaders,
    set_seed,
)
from dataset import get_or_build_tokenizer  # noqa: E402


OUTPUT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = get_config()
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_or_build_tokenizer(config)
    checkpoint_path = Path(best_weights_file_path(config))
    model, checkpoint = load_model_for_evaluation(
        config, tokenizer, checkpoint_path, device
    )
    loss_loader, decoding_loader = get_validation_dataloaders(config, tokenizer)
    loss_function = get_loss_function(config, tokenizer, device)

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    metrics, predictions = evaluate_model(
        model,
        loss_loader,
        decoding_loader,
        tokenizer,
        loss_function,
        device,
        config["seq_len"],
    )
    elapsed_seconds = time.perf_counter() - start

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    predictions_path = OUTPUT_ROOT / "validation_predictions.jsonl"
    predictions_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions)
        + "\n",
        encoding="utf-8",
    )

    result = {
        "evaluation_type": "independent validation-only baseline reproduction",
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_zero_based": checkpoint["epoch"],
        "checkpoint_global_step": checkpoint["global_step"],
        "checkpoint_saved_validation_metrics": checkpoint["validation_metrics"],
        "reproduced_validation_metrics": metrics,
        "metric_differences": {
            key: float(metrics[key])
            - float(checkpoint["validation_metrics"][key])
            for key in metrics
            if key != "test_loss"
            and key in checkpoint["validation_metrics"]
            and isinstance(metrics[key], (int, float))
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_size": tokenizer.get_vocab_size(),
        "prediction_file": str(predictions_path),
        "protected_test_opened": False,
    }
    # evaluate_model uses a generic test_loss key even when supplied validation
    # loaders; rename it in the research report for clarity.
    result["reproduced_validation_metrics"]["validation_loss"] = result[
        "reproduced_validation_metrics"
    ].pop("test_loss")
    result_path = OUTPUT_ROOT / "reproduced_validation.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
