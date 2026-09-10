"""Trace one requested example through the actual saved baseline."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import best_weights_file_path, get_config  # noqa: E402
from dataset import causal_mask, get_or_build_tokenizer  # noqa: E402
from evaluate import load_model_for_evaluation  # noqa: E402
from translate import build_encoder_input, validate_input_sentence  # noqa: E402


SENTENCE = "stop me she can not"
EXPECTED_TARGET = "she can not stop me"
OUTPUT_PATH = Path(__file__).resolve().parent / "trace_stop_me_she_can_not.json"


def main() -> None:
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_or_build_tokenizer(config)
    token_ids = validate_input_sentence(SENTENCE, tokenizer)
    encoder_input, encoder_mask = build_encoder_input(token_ids, tokenizer, device)
    model, checkpoint = load_model_for_evaluation(
        config, tokenizer, best_weights_file_path(config), device
    )

    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    assert sos_id is not None and eos_id is not None and pad_id is not None

    model.eval()
    with torch.no_grad():
        source_embedding = model.src_embed(encoder_input)
        source_with_position = model.src_pos(source_embedding)
        encoder_output = model.encoder(source_with_position, encoder_mask)

        remaining = Counter(token_ids)
        decoder_input = torch.tensor([[sos_id]], dtype=torch.long, device=device)
        steps = []
        while remaining:
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            logits = model.project(decoder_output[:, -1])
            allowed_ids = torch.tensor(list(remaining), dtype=torch.long, device=device)
            allowed_logits = logits[0].index_select(0, allowed_ids)
            ranked = sorted(
                (
                    {
                        "token": tokenizer.id_to_token(int(token_id)),
                        "token_id": int(token_id),
                        "logit": float(logit),
                    }
                    for token_id, logit in zip(
                        allowed_ids.detach().cpu().tolist(),
                        allowed_logits.detach().cpu().tolist(),
                    )
                ),
                key=lambda row: row["logit"],
                reverse=True,
            )
            selected_id = ranked[0]["token_id"]
            steps.append(
                {
                    "step": len(steps) + 1,
                    "decoder_input_tokens": [
                        tokenizer.id_to_token(int(token_id))
                        for token_id in decoder_input[0].detach().cpu().tolist()
                    ],
                    "decoder_input_shape": list(decoder_input.shape),
                    "decoder_mask_shape": list(decoder_mask.shape),
                    "decoder_output_shape": list(decoder_output.shape),
                    "full_vocabulary_logits_shape": list(logits.shape),
                    "allowed_source_candidates": ranked,
                    "selected_token": tokenizer.id_to_token(selected_id),
                }
            )
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.tensor([[selected_id]], dtype=torch.long, device=device),
                ],
                dim=1,
            )
            remaining[selected_id] -= 1
            if remaining[selected_id] == 0:
                del remaining[selected_id]

        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[eos_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    generated_tokens = [
        tokenizer.id_to_token(int(token_id))
        for token_id in decoder_input[0].detach().cpu().tolist()
    ]
    prediction = " ".join(generated_tokens[1:-1])
    report = {
        "device": str(device),
        "checkpoint_epoch_zero_based": checkpoint["epoch"],
        "input": SENTENCE,
        "expected_target": EXPECTED_TARGET,
        "input_word_token_ids": dict(zip(SENTENCE.split(), token_ids)),
        "encoder_tokens": ["[SOS]", *SENTENCE.split(), "[EOS]"],
        "encoder_input_shape": list(encoder_input.shape),
        "encoder_mask_shape": list(encoder_mask.shape),
        "source_embedding_shape": list(source_embedding.shape),
        "source_positional_shape": list(model.src_pos.pe.shape),
        "encoder_output_shape": list(encoder_output.shape),
        "decoding_steps": steps,
        "generated_tokens": generated_tokens,
        "prediction": prediction,
        "exact_match": prediction == EXPECTED_TARGET,
        "important_observation": "training predicts over 2412 vocabulary tokens; inference scores all 2412 and then selects only among unused source token IDs",
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
