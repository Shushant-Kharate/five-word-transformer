"""Section 7 inference tests using no test-set records or trained weights."""

from __future__ import annotations

import unittest
from collections import Counter

import torch

from config import get_config
from dataset import get_datasets, get_or_build_tokenizer
from train import get_model, set_seed
from translate import (
    build_encoder_input,
    reconstruct_sentence,
    validate_input_sentence,
)


class InferencePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = get_config()
        cls.device = torch.device("cpu")
        cls.tokenizer = get_or_build_tokenizer(cls.config)
        _, validation_dataset, _ = get_datasets(cls.config, cls.tokenizer)
        cls.validation_sentence = validation_dataset[0]["src_text"]

    def test_valid_input_produces_five_token_ids(self) -> None:
        token_ids = validate_input_sentence(
            self.validation_sentence, self.tokenizer
        )
        self.assertEqual(len(token_ids), 5)
        self.assertNotIn(self.tokenizer.token_to_id("[UNK]"), token_ids)

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_input_sentence("only four input words", self.tokenizer)
        with self.assertRaises(ValueError):
            validate_input_sentence(
                "these are five words 7", self.tokenizer
            )
        with self.assertRaises(ValueError):
            validate_input_sentence(
                "this contains unknownwordxyz for model", self.tokenizer
            )

    def test_encoder_input_and_mask_shapes(self) -> None:
        token_ids = validate_input_sentence(
            self.validation_sentence, self.tokenizer
        )
        encoder_input, encoder_mask = build_encoder_input(
            token_ids, self.tokenizer, self.device
        )
        self.assertEqual(tuple(encoder_input.shape), (1, 7))
        self.assertEqual(tuple(encoder_mask.shape), (1, 1, 1, 7))
        self.assertEqual(
            encoder_input[0, 0].item(),
            self.tokenizer.token_to_id("[SOS]"),
        )
        self.assertEqual(
            encoder_input[0, -1].item(),
            self.tokenizer.token_to_id("[EOS]"),
        )

    def test_inference_does_not_change_model_weights(self) -> None:
        set_seed(42)
        model = get_model(
            self.config, self.tokenizer.get_vocab_size()
        ).to(self.device)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        result = reconstruct_sentence(
            model,
            self.tokenizer,
            self.validation_sentence,
            self.device,
            self.config["seq_len"],
        )
        after = [parameter.detach() for parameter in model.parameters()]

        self.assertEqual(result["input"], self.validation_sentence)
        self.assertIn("prediction", result)
        self.assertTrue(result["ended_with_eos"])
        self.assertTrue(result["output_valid"])
        self.assertTrue(result["input_words_preserved"])
        self.assertEqual(
            Counter(result["prediction"].split()),
            Counter(self.validation_sentence.split()),
        )
        for original, current in zip(before, after):
            self.assertTrue(torch.equal(original, current))


if __name__ == "__main__":
    unittest.main()
