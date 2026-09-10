"""Section 5 tests for greedy decoding and reconstruction validation."""

from __future__ import annotations

import unittest
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset

from config import get_config
from dataset import get_datasets, get_or_build_tokenizer
from train import (
    compute_reconstruction_metrics,
    decoded_token_ids_to_words,
    get_loss_function,
    get_model,
    greedy_decode,
    is_better_validation_result,
    run_validation,
    set_seed,
)


class ValidationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = get_config()
        cls.device = torch.device("cpu")
        cls.tokenizer = get_or_build_tokenizer(cls.config)
        _, validation_dataset, _ = get_datasets(cls.config, cls.tokenizer)
        cls.validation_subset = Subset(validation_dataset, range(4))
        cls.loss_loader = DataLoader(
            cls.validation_subset, batch_size=4, shuffle=False
        )
        cls.decoding_loader = DataLoader(
            cls.validation_subset, batch_size=1, shuffle=False
        )

    def make_model_and_loss(self):
        set_seed(42)
        model = get_model(
            self.config, self.tokenizer.get_vocab_size()
        ).to(self.device)
        loss_function = get_loss_function(
            self.config, self.tokenizer, self.device
        )
        return model, loss_function

    def test_reconstruction_metric_calculation(self) -> None:
        result = compute_reconstruction_metrics(
            sources=[
                ["a", "b", "c", "d", "e"],
                ["f", "g", "h", "i", "j"],
            ],
            targets=[
                ["a", "b", "c", "d", "e"],
                ["f", "h", "g", "i", "j"],
            ],
            predictions=[
                ["a", "b", "c", "d", "e"],
                ["f", "g", "h", "i"],
            ],
            ended_with_eos=[True, False],
        )
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["exact_matches"], 1)
        self.assertAlmostEqual(result["exact_match_accuracy"], 0.5)
        self.assertAlmostEqual(result["word_position_accuracy"], 0.7)
        self.assertAlmostEqual(result["output_validity_rate"], 0.5)
        self.assertAlmostEqual(result["input_word_preservation_rate"], 0.5)

    def test_best_result_selection_rule(self) -> None:
        self.assertTrue(is_better_validation_result(0.4, 2.0, 0.3, 1.0))
        self.assertTrue(is_better_validation_result(0.3, 0.9, 0.3, 1.0))
        self.assertFalse(is_better_validation_result(0.3, 1.1, 0.3, 1.0))
        self.assertFalse(is_better_validation_result(0.2, 0.5, 0.3, 1.0))

    def test_greedy_decode_respects_maximum_length(self) -> None:
        model, _ = self.make_model_and_loss()
        model.eval()
        batch = next(iter(self.decoding_loader))
        with torch.no_grad():
            token_ids = greedy_decode(
                model,
                batch["encoder_input"],
                batch["encoder_mask"],
                self.tokenizer,
                self.config["seq_len"],
                self.device,
            )
        self.assertGreaterEqual(token_ids.size(0), 2)
        self.assertLessEqual(token_ids.size(0), 7)
        self.assertEqual(
            token_ids[0].item(), self.tokenizer.token_to_id("[SOS]")
        )
        words, ended_with_eos = decoded_token_ids_to_words(
            token_ids, self.tokenizer
        )
        self.assertIsInstance(words, list)
        self.assertTrue(ended_with_eos)
        self.assertEqual(
            Counter(words), Counter(batch["src_text"][0].split())
        )

    def test_greedy_decode_rejects_outside_word_and_keeps_duplicates(
        self,
    ) -> None:
        vocabulary_size = self.tokenizer.get_vocab_size()
        sos_id = self.tokenizer.token_to_id("[SOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")
        pad_id = self.tokenizer.token_to_id("[PAD]")
        word_ids = [
            token_id
            for token, token_id in self.tokenizer.get_vocab().items()
            if token not in {"[UNK]", "[PAD]", "[SOS]", "[EOS]"}
        ]
        repeated_id, second_id, third_id, fourth_id = word_ids[:4]
        source_word_ids = [
            repeated_id,
            repeated_id,
            second_id,
            third_id,
            fourth_id,
        ]
        source_id_set = set(source_word_ids) | {sos_id, eos_id, pad_id}
        outside_id = next(
            token_id
            for token_id in range(vocabulary_size)
            if token_id not in source_id_set
        )

        class OutsideWordFavoringModel:
            def __init__(self) -> None:
                self.project_calls = 0

            def encode(self, source, source_mask):
                return source.float().unsqueeze(-1)

            def decode(
                self,
                encoder_output,
                source_mask,
                decoder_input,
                decoder_mask,
            ):
                return torch.zeros(
                    (1, decoder_input.size(1), 1),
                    dtype=torch.float32,
                )

            def project(self, decoder_output):
                self.project_calls += 1
                logits = torch.arange(
                    vocabulary_size, dtype=torch.float32
                ).unsqueeze(0)
                logits[0, outside_id] = vocabulary_size + 10_000
                return logits

        source = torch.tensor(
            [[sos_id, *source_word_ids, eos_id]], dtype=torch.int64
        )
        source_mask = torch.ones((1, 1, 1, 7), dtype=torch.int64)
        model = OutsideWordFavoringModel()

        generated_ids = greedy_decode(
            model,
            source,
            source_mask,
            self.tokenizer,
            max_len=7,
            device=self.device,
        )

        self.assertEqual(generated_ids.size(0), 7)
        self.assertEqual(generated_ids[-1].item(), eos_id)
        self.assertNotIn(outside_id, generated_ids.tolist())
        self.assertEqual(
            Counter(generated_ids[1:-1].tolist()),
            Counter(source_word_ids),
        )
        self.assertEqual(model.project_calls, 5)

    def test_validation_does_not_update_model_weights(self) -> None:
        model, loss_function = self.make_model_and_loss()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        result = run_validation(
            model,
            self.loss_loader,
            self.decoding_loader,
            self.tokenizer,
            loss_function,
            self.device,
            self.config["seq_len"],
            epoch=0,
            writer=None,
            num_examples=2,
        )
        after = [parameter.detach() for parameter in model.parameters()]

        self.assertEqual(result["records"], 4)
        self.assertEqual(len(result["examples"]), 2)
        self.assertTrue(
            torch.isfinite(torch.tensor(result["validation_loss"]))
        )
        for metric in (
            "exact_match_accuracy",
            "word_position_accuracy",
            "output_validity_rate",
            "input_word_preservation_rate",
        ):
            self.assertGreaterEqual(result[metric], 0.0)
            self.assertLessEqual(result[metric], 1.0)
        for original, current in zip(before, after):
            self.assertTrue(torch.equal(original, current))


if __name__ == "__main__":
    unittest.main()
