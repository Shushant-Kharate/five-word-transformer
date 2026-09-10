"""Dataset and tokenizer utilities for five-word sentence reconstruction.

The tensor construction and causal-mask logic intentionally follow
dataset.py from https://github.com/hkproj/pytorch-transformer. The original
bilingual data access is replaced with local JSONL ``input``/``target`` pairs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]


def load_jsonl(file_path: str | Path) -> list[dict[str, str]]:
    """Load and validate sentence-reconstruction records from a JSONL file."""
    path = Path(file_path)
    records: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc

            if set(record) != {"input", "target"}:
                raise ValueError(
                    f"Expected only 'input' and 'target' in {path} "
                    f"at line {line_number}"
                )
            if not isinstance(record["input"], str) or not isinstance(
                record["target"], str
            ):
                raise ValueError(
                    f"Input and target must be strings in {path} "
                    f"at line {line_number}"
                )
            if len(record["input"].split()) != 5 or len(
                record["target"].split()
            ) != 5:
                raise ValueError(
                    f"Input and target must each have five words in {path} "
                    f"at line {line_number}"
                )
            if any(character.isdigit() for character in record["input"] + record["target"]):
                raise ValueError(
                    f"Digits are not allowed in {path} at line {line_number}"
                )
            if Counter(record["input"].split()) != Counter(
                record["target"].split()
            ):
                raise ValueError(
                    f"Target must be a permutation of input in {path} "
                    f"at line {line_number}"
                )
            records.append(record)

    return records


def get_all_input_sentences(records: list[dict[str, str]]) -> Iterator[str]:
    """Yield source inputs used to construct the shared closed vocabulary."""
    for record in records:
        yield record["input"]


def get_or_build_tokenizer(config: dict) -> Tokenizer:
    """Load or build the shared tokenizer from authoritative split inputs."""
    tokenizer_path = Path(config["tokenizer_file"])

    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    all_records: list[dict[str, str]] = []
    for source_file in config["tokenizer_source_files"]:
        all_records.extend(load_jsonl(source_file))
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
    )
    tokenizer.train_from_iterator(
        get_all_input_sentences(all_records),
        trainer=trainer,
    )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


class SentenceReconstructionDataset(Dataset):
    """Convert five-word JSONL pairs into encoder–decoder tensors."""

    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer: Tokenizer,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.records = records
        self.tokenizer = tokenizer

        self.sos_token = torch.tensor(
            [self._required_token_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [self._required_token_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token_id = self._required_token_id("[PAD]")

    def _required_token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required token {token}")
        return token_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        source_target_pair = self.records[index]
        source_text = source_target_pair["input"]
        target_text = source_target_pair["target"]

        # Transform the text into token IDs.
        encoder_input_tokens = self.tokenizer.encode(source_text).ids
        decoder_input_tokens = self.tokenizer.encode(target_text).ids

        # Match the reference repository: encoder gets SOS and EOS; decoder
        # input gets SOS; the label gets EOS.
        encoder_padding_count = (
            self.seq_len - len(encoder_input_tokens) - 2
        )
        decoder_padding_count = (
            self.seq_len - len(decoder_input_tokens) - 1
        )

        if encoder_padding_count < 0 or decoder_padding_count < 0:
            raise ValueError("Sentence is too long for the configured sequence length")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full(
                    (encoder_padding_count,),
                    self.pad_token_id,
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.full(
                    (decoder_padding_count,),
                    self.pad_token_id,
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full(
                    (decoder_padding_count,),
                    self.pad_token_id,
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token_id)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (
                (decoder_input != self.pad_token_id).unsqueeze(0).int()
                & causal_mask(decoder_input.size(0))
            ),
            "label": label,
            "src_text": source_text,
            "tgt_text": target_text,
        }


def causal_mask(size: int) -> torch.Tensor:
    """Return a mask that prevents attention to future target positions."""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_datasets(
    config: dict, tokenizer: Tokenizer
) -> tuple[
    SentenceReconstructionDataset,
    SentenceReconstructionDataset,
    SentenceReconstructionDataset,
]:
    """Load the fixed train, validation, and test datasets."""
    return (
        SentenceReconstructionDataset(
            load_jsonl(config["train_file"]), tokenizer, config["seq_len"]
        ),
        SentenceReconstructionDataset(
            load_jsonl(config["validation_file"]), tokenizer, config["seq_len"]
        ),
        SentenceReconstructionDataset(
            load_jsonl(config["test_file"]), tokenizer, config["seq_len"]
        ),
    )
