"""Section 3 tests for the encoder–decoder Transformer architecture."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset import get_datasets, get_or_build_tokenizer
from model import build_transformer


class TransformerArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.config = get_config()
        cls.tokenizer = get_or_build_tokenizer(cls.config)
        train_ds, _, _ = get_datasets(cls.config, cls.tokenizer)
        cls.batch = next(iter(DataLoader(train_ds, batch_size=2, shuffle=False)))

        vocabulary_size = cls.tokenizer.get_vocab_size()
        cls.model = build_transformer(
            vocabulary_size,
            vocabulary_size,
            cls.config["seq_len"],
            cls.config["seq_len"],
            d_model=cls.config["d_model"],
            N=cls.config["N"],
            h=cls.config["h"],
            dropout=cls.config["dropout"],
            d_ff=cls.config["d_ff"],
        )
        cls.model.eval()

    def test_configured_layer_counts_and_head_dimensions(self) -> None:
        self.assertEqual(len(self.model.encoder.layers), 2)
        self.assertEqual(len(self.model.decoder.layers), 2)
        for layer in self.model.encoder.layers:
            self.assertEqual(layer.self_attention_block.h, 4)
            self.assertEqual(layer.self_attention_block.d_k, 16)
        for layer in self.model.decoder.layers:
            self.assertEqual(layer.self_attention_block.h, 4)
            self.assertEqual(layer.cross_attention_block.h, 4)
            self.assertEqual(layer.self_attention_block.d_k, 16)

    def test_separate_source_and_target_embeddings(self) -> None:
        self.assertIsNot(self.model.src_embed, self.model.tgt_embed)
        vocabulary_size = self.tokenizer.get_vocab_size()
        self.assertEqual(
            tuple(self.model.src_embed.embedding.weight.shape),
            (vocabulary_size, 64),
        )
        self.assertEqual(
            tuple(self.model.tgt_embed.embedding.weight.shape),
            (vocabulary_size, 64),
        )

    def test_positional_encoding_is_fixed_buffer(self) -> None:
        self.assertEqual(tuple(self.model.src_pos.pe.shape), (1, 7, 64))
        self.assertEqual(tuple(self.model.tgt_pos.pe.shape), (1, 7, 64))
        parameter_names = dict(self.model.named_parameters())
        self.assertNotIn("src_pos.pe", parameter_names)
        self.assertNotIn("tgt_pos.pe", parameter_names)

    def test_full_forward_shapes_and_finite_values(self) -> None:
        with torch.no_grad():
            encoder_output = self.model.encode(
                self.batch["encoder_input"], self.batch["encoder_mask"]
            )
            decoder_output = self.model.decode(
                encoder_output,
                self.batch["encoder_mask"],
                self.batch["decoder_input"],
                self.batch["decoder_mask"],
            )
            logits = self.model.project(decoder_output)

        self.assertEqual(tuple(encoder_output.shape), (2, 7, 64))
        self.assertEqual(tuple(decoder_output.shape), (2, 7, 64))
        self.assertEqual(
            tuple(logits.shape), (2, 7, self.tokenizer.get_vocab_size())
        )
        self.assertTrue(torch.isfinite(logits).all())

    def test_attention_tensor_shapes(self) -> None:
        with torch.no_grad():
            encoder_output = self.model.encode(
                self.batch["encoder_input"], self.batch["encoder_mask"]
            )
            self.model.decode(
                encoder_output,
                self.batch["encoder_mask"],
                self.batch["decoder_input"],
                self.batch["decoder_mask"],
            )

        encoder_scores = self.model.encoder.layers[0].self_attention_block.attention_scores
        decoder_scores = self.model.decoder.layers[0].self_attention_block.attention_scores
        cross_scores = self.model.decoder.layers[0].cross_attention_block.attention_scores
        self.assertEqual(tuple(encoder_scores.shape), (2, 4, 7, 7))
        self.assertEqual(tuple(decoder_scores.shape), (2, 4, 7, 7))
        self.assertEqual(tuple(cross_scores.shape), (2, 4, 7, 7))

    def test_decoder_attention_cannot_see_future_positions(self) -> None:
        with torch.no_grad():
            encoder_output = self.model.encode(
                self.batch["encoder_input"], self.batch["encoder_mask"]
            )
            self.model.decode(
                encoder_output,
                self.batch["encoder_mask"],
                self.batch["decoder_input"],
                self.batch["decoder_mask"],
            )

        scores = self.model.decoder.layers[0].self_attention_block.attention_scores
        future_scores = torch.triu(scores, diagonal=1)
        self.assertTrue(torch.equal(future_scores, torch.zeros_like(future_scores)))

    def test_parameter_count(self) -> None:
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(parameter_count, 697_708)


if __name__ == "__main__":
    unittest.main()
