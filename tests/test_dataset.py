"""Section 2 tests for tokenizer, tensors, and attention masks."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset import (
    SPECIAL_TOKENS,
    causal_mask,
    get_datasets,
    get_or_build_tokenizer,
    load_jsonl,
)


class DatasetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = get_config()
        cls.tokenizer = get_or_build_tokenizer(cls.config)
        cls.train_ds, cls.validation_ds, cls.test_ds = get_datasets(
            cls.config, cls.tokenizer
        )

    def test_split_sizes(self) -> None:
        self.assertEqual(len(self.train_ds), 433_680)
        self.assertEqual(len(self.validation_ds), 54_540)
        self.assertEqual(len(self.test_ds), 53_790)

    def test_vocabulary_and_special_tokens(self) -> None:
        self.assertEqual(self.tokenizer.get_vocab_size(), 2412)
        for token in SPECIAL_TOKENS:
            self.assertIsNotNone(self.tokenizer.token_to_id(token))

    def test_all_dataset_words_are_known(self) -> None:
        unknown_id = self.tokenizer.token_to_id("[UNK]")
        for file_key in (
            "train_file",
            "validation_file",
            "test_file",
        ):
            for record in load_jsonl(self.config[file_key]):
                for field in ("input", "target"):
                    self.assertNotIn(
                        unknown_id,
                        self.tokenizer.encode(record[field]).ids,
                    )

    def test_item_tensor_shapes_and_special_tokens(self) -> None:
        item = self.train_ds[0]
        sos_id = self.tokenizer.token_to_id("[SOS]")
        eos_id = self.tokenizer.token_to_id("[EOS]")
        pad_id = self.tokenizer.token_to_id("[PAD]")

        self.assertEqual(tuple(item["encoder_input"].shape), (7,))
        self.assertEqual(tuple(item["decoder_input"].shape), (7,))
        self.assertEqual(tuple(item["label"].shape), (7,))
        self.assertEqual(tuple(item["encoder_mask"].shape), (1, 1, 7))
        self.assertEqual(tuple(item["decoder_mask"].shape), (1, 7, 7))

        self.assertEqual(item["encoder_input"][0].item(), sos_id)
        self.assertEqual(item["encoder_input"][-1].item(), eos_id)
        self.assertEqual(item["decoder_input"][0].item(), sos_id)
        self.assertEqual(item["decoder_input"][-1].item(), pad_id)
        self.assertEqual(item["label"][-2].item(), eos_id)
        self.assertEqual(item["label"][-1].item(), pad_id)

    def test_causal_mask(self) -> None:
        expected = torch.tensor(
            [[[True, False, False], [True, True, False], [True, True, True]]]
        )
        self.assertTrue(torch.equal(causal_mask(3), expected))

    def test_batch_shapes(self) -> None:
        batch = next(
            iter(DataLoader(self.train_ds, batch_size=32, shuffle=False))
        )
        self.assertEqual(tuple(batch["encoder_input"].shape), (32, 7))
        self.assertEqual(tuple(batch["decoder_input"].shape), (32, 7))
        self.assertEqual(tuple(batch["label"].shape), (32, 7))
        self.assertEqual(tuple(batch["encoder_mask"].shape), (32, 1, 1, 7))
        self.assertEqual(tuple(batch["decoder_mask"].shape), (32, 1, 7, 7))


if __name__ == "__main__":
    unittest.main()
