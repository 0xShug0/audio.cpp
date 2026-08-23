"""Dump reference voice-design prompts for MOSS-VoiceGenerator.

The checkpoint ships its own MossTTSDelayProcessor, so the prompt rows it produces are
the ground truth for the audio.cpp prompt builder. This writes one JSON file per case for
tests/moss_voicegen/prompt_parity.cpp to compare against.

    python3 tools/community_models/moss_voicegen_reference_prompt.py \
        --model /path/to/MOSS-VoiceGenerator --output parity

Needs transformers with `trust_remote_code` (the repo's processor is remote code) and
torch/torchaudio, which the processor imports even when no audio is involved.
"""

import argparse
import json
import pathlib

from transformers import AutoProcessor

# Deliberately varied: a trailing period that merges with the template suffix, text that
# ends on a letter, non-Latin script, and an absent instruction that must render "None".
CASES = [
    {
        "name": "en_radio_voice",
        "instruction": "A warm male radio voice in his fifties, calm, never shrill.",
        "text": "Good evening, and welcome back to the late show.",
        "language": "English",
    },
    {
        "name": "en_no_final_punctuation",
        "instruction": "A bright young female narrator, quick and precise",
        "text": "Tonight we look at three stories that shaped the week",
        "language": "English",
    },
    {
        "name": "zh_storyteller",
        "instruction": "一位年长的男性说书人，声音低沉，语速缓慢。",
        "text": "各位听众，晚上好，欢迎收听今天的节目。",
        "language": "Chinese",
    },
    {
        "name": "en_no_instruction",
        "instruction": None,
        "text": "This one carries no voice description at all.",
        "language": "English",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="directory for the dumped cases")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        message = processor.build_user_message(
            text=case["text"], instruction=case["instruction"], language=case["language"]
        )
        content = message["content"] if isinstance(message, dict) else message.to_dict()["content"]
        encoded = processor([[message]], add_generation_prompt=True, return_tensors="pt")
        rows = encoded["input_ids"][0].tolist()

        path = output_dir / f"ref_prompt_{case['name']}.json"
        path.write_text(
            json.dumps(
                {
                    "name": case["name"],
                    "instruction": case["instruction"] or "",
                    "text": case["text"],
                    "language": case["language"],
                    "content": content,
                    "input_ids": rows,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"{case['name']}: {len(rows)} rows x {len(rows[0])} -> {path}")


if __name__ == "__main__":
    main()
