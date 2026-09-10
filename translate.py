"""Inference command for source-constrained sentence reconstruction.

The decoder retains the reference repository's autoregressive greedy flow,
while limiting each word choice to an unused token from the five-word input.
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from tokenizers import Tokenizer

from config import best_weights_file_path, get_config
from dataset import get_or_build_tokenizer
from evaluate import load_model_for_evaluation
from model import Transformer
from train import decoded_token_ids_to_words, greedy_decode, set_seed


def validate_input_sentence(sentence: str, tokenizer: Tokenizer) -> list[int]:
    """Validate a five-word, digit-free, known-vocabulary input."""
    words = sentence.strip().split()
    if len(words) != 5:
        raise ValueError(
            f"Input must contain exactly five words; received {len(words)}"
        )
    if any(character.isdigit() for character in sentence):
        raise ValueError("Input must not contain digits")

    token_ids = tokenizer.encode(" ".join(words)).ids
    if len(token_ids) != 5:
        raise ValueError(
            "Each whitespace-separated word must map to exactly one token"
        )
    unknown_id = tokenizer.token_to_id("[UNK]")
    if unknown_id in token_ids:
        unknown_words = [
            word
            for word in words
            if unknown_id in tokenizer.encode(word).ids
        ]
        raise ValueError(
            "Input contains words outside the project vocabulary: "
            + ", ".join(unknown_words)
        )
    return token_ids


def build_encoder_input(
    token_ids: list[int], tokenizer: Tokenizer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create `[SOS] five words [EOS]` and its encoder mask."""
    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    if sos_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer is missing required special tokens")

    encoder_input = torch.tensor(
        [[sos_id, *token_ids, eos_id]],
        dtype=torch.int64,
        device=device,
    )
    encoder_mask = (
        (encoder_input != pad_id).unsqueeze(1).unsqueeze(1).int()
    )
    return encoder_input, encoder_mask


def reconstruct_sentence(
    model: Transformer,
    tokenizer: Tokenizer,
    sentence: str,
    device: torch.device,
    max_len: int,
) -> dict[str, str | bool]:
    """Reconstruct one sentence with hard input-word preservation."""
    token_ids = validate_input_sentence(sentence, tokenizer)
    source_words = sentence.strip().split()
    encoder_input, encoder_mask = build_encoder_input(
        token_ids, tokenizer, device
    )

    model.eval()
    with torch.no_grad():
        generated_ids = greedy_decode(
            model,
            encoder_input,
            encoder_mask,
            tokenizer,
            max_len,
            device,
        )
    prediction_words, ended_with_eos = decoded_token_ids_to_words(
        generated_ids, tokenizer
    )

    return {
        "input": " ".join(source_words),
        "prediction": " ".join(prediction_words),
        "ended_with_eos": ended_with_eos,
        "output_valid": len(prediction_words) == 5 and ended_with_eos,
        "input_words_preserved": Counter(prediction_words)
        == Counter(source_words),
    }


def run_inference(sentence: str, config: dict) -> dict[str, str | bool]:
    """Load the best checkpoint and reconstruct one user sentence."""
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_or_build_tokenizer(config)
    model, _ = load_model_for_evaluation(
        config,
        tokenizer,
        best_weights_file_path(config),
        device,
    )
    return reconstruct_sentence(
        model,
        tokenizer,
        sentence,
        device,
        config["seq_len"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct the order of five jumbled words"
    )
    parser.add_argument(
        "--sentence",
        type=str,
        help='five jumbled words, for example: "home his nearby winds leveled"',
    )
    arguments = parser.parse_args()
    if arguments.sentence is None:
        print(
            "Inference was not run. After training, provide exactly five "
            "words with `python translate.py --sentence \"...\"`."
        )
    else:
        inference_result = run_inference(arguments.sentence, get_config())
        print(f"INPUT:      {inference_result['input']}")
        print(f"PREDICTION: {inference_result['prediction']}")
        print(f"VALID:      {inference_result['output_valid']}")
        print(
            "WORDS KEPT:  "
            f"{inference_result['input_words_preserved']}"
        )
