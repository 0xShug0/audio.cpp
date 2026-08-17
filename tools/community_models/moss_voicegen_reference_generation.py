"""Dump a reference greedy generation for MOSS-VoiceGenerator.

Runs MossTTSDelayModel.generate with sampling disabled and the repetition penalty off, so
the emitted rows are a deterministic function of the weights. That makes the delay-pattern
state machine in audio.cpp checkable row for row: any divergence is a real divergence and
not an RNG difference.

    python3 tools/community_models/moss_voicegen_reference_generation.py \
        --model /path/to/MOSS-VoiceGenerator \
        --reference tests/moss_voicegen/reference/ref_prompt_en_radio_voice.json \
        --output tests/moss_voicegen/reference/ref_generation_en_radio_voice.json \
        --steps 40
"""

import argparse
import json
import pathlib

import torch
from transformers import AutoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    reference = json.loads(pathlib.Path(args.reference).read_text())
    input_ids = torch.tensor([reference["input_ids"]], dtype=torch.long)
    attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.steps,
            text_temperature=0.0,
            audio_temperature=0.0,
            audio_repetition_penalty=1.0,
        )

    prompt_rows, rows = outputs[0]
    rows = rows.tolist()
    prompt_rows = int(prompt_rows)
    generated = rows[prompt_rows:]

    pathlib.Path(args.output).write_text(
        json.dumps(
            {
                "name": reference["name"],
                "steps": len(generated),
                "greedy": True,
                "repetition_penalty": 1.0,
                "generated_rows": generated,
            }
        )
    )
    text_column = [row[0] for row in generated]
    print(f"{reference['name']}: {len(generated)} generated rows -> {args.output}")
    print(f"text column: {text_column[:12]}{' ...' if len(text_column) > 12 else ''}")


if __name__ == "__main__":
    main()
