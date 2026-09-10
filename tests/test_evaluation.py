"""Section 6 tests that do not evaluate the held-out test split."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from config import get_config
from dataset import get_datasets, get_or_build_tokenizer
from evaluate import (
    build_prediction_record,
    evaluate_model,
    load_model_for_evaluation,
    write_evaluation_outputs,
)
from train import (
    get_loss_function,
    get_model,
    get_optimizer,
    save_checkpoint,
    set_seed,
)


class FinalEvaluationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = get_config()
        cls.device = torch.device("cpu")
        cls.tokenizer = get_or_build_tokenizer(cls.config)
        _, validation_dataset, _ = get_datasets(cls.config, cls.tokenizer)
        # Validation fixtures deliberately protect test.jsonl from use.
        subset = Subset(validation_dataset, range(4))
        cls.loss_loader = DataLoader(subset, batch_size=4, shuffle=False)
        cls.decoding_loader = DataLoader(subset, batch_size=1, shuffle=False)

    def make_model(self):
        set_seed(42)
        return get_model(
            self.config, self.tokenizer.get_vocab_size()
        ).to(self.device)

    def test_prediction_record(self) -> None:
        record = build_prediction_record(
            source=["b", "a", "c", "d", "e"],
            target=["a", "b", "c", "d", "e"],
            prediction=["a", "b", "c", "d", "e"],
            ended_with_eos=True,
        )
        self.assertTrue(record["exact_match"])
        self.assertEqual(record["correct_word_positions"], 5)
        self.assertTrue(record["output_valid"])
        self.assertTrue(record["input_words_preserved"])

    def test_evaluator_does_not_change_weights(self) -> None:
        model = self.make_model()
        loss_function = get_loss_function(
            self.config, self.tokenizer, self.device
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics, records = evaluate_model(
            model,
            self.loss_loader,
            self.decoding_loader,
            self.tokenizer,
            loss_function,
            self.device,
            self.config["seq_len"],
        )
        after = [parameter.detach() for parameter in model.parameters()]

        self.assertEqual(metrics["records"], 4)
        self.assertEqual(len(records), 4)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["test_loss"])))
        for original, current in zip(before, after):
            self.assertTrue(torch.equal(original, current))

    def test_best_checkpoint_can_be_loaded_without_optimizer(self) -> None:
        model = self.make_model()
        optimizer = get_optimizer(self.config, model)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best_model.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=7,
                global_step=224,
                training_loss=1.5,
                validation_metrics={"exact_match_accuracy": 0.4},
            )
            restored_model, state = load_model_for_evaluation(
                self.config,
                self.tokenizer,
                checkpoint_path,
                self.device,
            )

        self.assertEqual(state["epoch"], 7)
        self.assertEqual(state["global_step"], 224)
        for original, restored in zip(
            model.parameters(), restored_model.parameters()
        ):
            self.assertTrue(torch.equal(original, restored))

    def test_missing_best_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "best_model.pt"
            with self.assertRaises(FileNotFoundError):
                load_model_for_evaluation(
                    self.config,
                    self.tokenizer,
                    missing_path,
                    self.device,
                )

    def test_outputs_are_written_once_and_protected(self) -> None:
        records = [
            {"input": "b a c d e", "prediction": "a b c d e"},
            {"input": "g f h i j", "prediction": "f g h i j"},
        ]
        summary = {"metrics": {"exact_match_accuracy": 1.0}}
        with tempfile.TemporaryDirectory() as directory:
            predictions_path = Path(directory) / "predictions.jsonl"
            summary_path = Path(directory) / "summary.json"
            write_evaluation_outputs(
                predictions_path, summary_path, records, summary
            )
            written_records = [
                json.loads(line)
                for line in predictions_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            written_summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            with self.assertRaises(FileExistsError):
                write_evaluation_outputs(
                    predictions_path, summary_path, records, summary
                )

        self.assertEqual(written_records, records)
        self.assertEqual(written_summary, summary)


if __name__ == "__main__":
    unittest.main()
