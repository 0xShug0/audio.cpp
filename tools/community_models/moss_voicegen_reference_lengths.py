"""Measure how long the reference implementation speaks, with no length bounds.

The port adds text-derived minimum and maximum frame counts that upstream's generate()
does not have — upstream only caps total steps at 1000 and forbids the turn-end token for
the first n_vq steps. That addition needs evidence, not an assertion, so this runs the
reference itself over the same text and reports how much audio each take produced against
what the text should take to read.

    python3 tools/community_models/moss_voicegen_reference_lengths.py \
        --model /path/to/MOSS-VoiceGenerator --seeds 3 --max-new-tokens 400
"""

import argparse

import torch
from transformers import AutoModel, AutoProcessor

INSTRUCTION = "A warm male radio voice in his fifties, calm, never shrill."
TEXT = (
    "Good evening, and welcome back to the late show. Tonight we look at three stories "
    "that shaped the week, and we close with a long set of quiet music for the small hours."
)
FRAMES_PER_SECOND = 12.5
FRAMES_PER_CHARACTER = 0.95      # measured from takes that read their text in full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()

    message = processor.build_user_message(text=TEXT, instruction=INSTRUCTION, language="English")
    batch = processor([[message]], mode="generation")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    expected = len(TEXT) * FRAMES_PER_CHARACTER
    print(f"text: {len(TEXT)} characters -> expected about {expected:.0f} frames "
          f"({expected/FRAMES_PER_SECOND:.1f} s)\n")

    n_vq = model.config.n_vq
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
            )
        prompt_rows, rows = outputs[0]
        generated = rows[int(prompt_rows):]
        text_column = generated[:, 0].tolist()
        audio_rows = sum(
            1 for token in text_column
            if token in (model.config.audio_assistant_gen_slot_token_id,
                         model.config.audio_assistant_delay_slot_token_id)
        )
        frames = max(0, audio_rows - n_vq + 1)
        started = model.config.audio_start_token_id in text_column
        print(f"seed {seed}: {len(text_column)} steps, {frames} frames "
              f"({frames/FRAMES_PER_SECOND:.1f} s), {frames/expected*100:5.1f}% of expected"
              f"{'' if started else '  [never entered audio]'}")


if __name__ == "__main__":
    main()
