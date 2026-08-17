"""Dump reference backbone hidden states for MOSS-VoiceGenerator.

Runs the checkpoint's own MossTTSDelayModel over a prompt produced by its own processor
and writes the final hidden state, so tests/moss_voicegen/backbone_parity.cpp can check
the audio.cpp backbone against it.

    python3 tools/community_models/moss_voicegen_reference_hidden.py \
        --model /path/to/MOSS-VoiceGenerator \
        --reference tests/moss_voicegen/reference/ref_prompt_en_radio_voice.json \
        --output tests/moss_voicegen/reference/ref_hidden_en_radio_voice.json

The weights are upcast to float32 to match how audio.cpp computes: bf16 storage, f32
math. Running the reference in bf16 instead moves the last-hidden values by roughly 1e-2,
which is noise on this scale but would make a tight tolerance meaningless.
"""

import argparse
import json
import pathlib

import torch
from transformers import AutoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True, help="prompt dump from moss_voicegen_reference_prompt.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = json.loads(pathlib.Path(args.reference).read_text())
    input_ids = torch.tensor([reference["input_ids"]], dtype=torch.long)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        embeds = model.get_input_embeddings(input_ids)
        outputs = model.language_model(inputs_embeds=embeds, use_cache=False, output_hidden_states=False)
        hidden = outputs.last_hidden_state[0]

    last = hidden[-1].tolist()
    payload = {
        "name": reference["name"],
        "rows": int(input_ids.shape[1]),
        "hidden_size": int(hidden.shape[-1]),
        "last_hidden": last,
        "last_hidden_abs_max": float(hidden[-1].abs().max()),
        # A few interior positions guard against a backbone that only happens to get the
        # final row right, e.g. through a rope or mask offset that cancels at the end.
        "probe_positions": [0, 1, int(input_ids.shape[1]) // 2],
        "probe_hidden": [hidden[i].tolist() for i in [0, 1, int(input_ids.shape[1]) // 2]],
    }
    pathlib.Path(args.output).write_text(json.dumps(payload))
    print(
        f"{reference['name']}: rows={payload['rows']} hidden={payload['hidden_size']} "
        f"absmax={payload['last_hidden_abs_max']:.4f} -> {args.output}"
    )


if __name__ == "__main__":
    main()
