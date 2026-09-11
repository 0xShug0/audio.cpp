#!/usr/bin/env python3
"""Run the Audio8-ASR-0.1B reference (HF transformers, trust_remote_code) on
repo test audio and dump golden parity artifacts.

Outputs (under --outdir):
  <name>.prompt.json      input_ids, audio token count/positions, prompt text
  <name>.logits.npy       first generated step logits (float32)
  <name>.text.txt         greedy transcript
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def load_audio_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    import librosa

    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return np.asarray(wav, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    for audio_path in args.audio:
        stem = Path(audio_path).stem
        audio = load_audio_mono(audio_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(Path(audio_path).resolve())},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            }
        ]
        batch = processor.apply_chat_template(
            conversation,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
            add_generation_prompt=True,
            audio_max_length=int(args.max_audio_seconds * 16000),
            text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
        )
        input_ids = batch["input_ids"][0].tolist()
        audio_token_id = model.config.audio_token_id
        audio_positions = [i for i, t in enumerate(input_ids) if t == audio_token_id]
        with torch.inference_mode():
            output_ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        prompt_len = int(batch["input_ids"].shape[1])
        # One forward pass to capture first-step logits over the prompt.
        with torch.inference_mode():
            logits = model(
                input_ids=batch["input_ids"],
                input_features=batch.get("input_features"),
            ).logits[0, -1, :]
        text = processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()

        prompt_record = {
            "audio": str(audio_path),
            "samples": int(len(audio)),
            "input_ids": input_ids,
            "prompt_len": prompt_len,
            "audio_token_id": audio_token_id,
            "audio_token_count": len(audio_positions),
            "audio_positions_head": audio_positions[:5],
            "generated_token_ids": output_ids[0, prompt_len:].tolist(),
        }
        (args.outdir / f"{stem}.prompt.json").write_text(json.dumps(prompt_record, indent=2))
        np.save(args.outdir / f"{stem}.logits.npy", logits.to(torch.float32).numpy())
        (args.outdir / f"{stem}.text.txt").write_text(text + "\n")
        print(f"[{stem}] audio_tokens={len(audio_positions)} text={text!r}")


if __name__ == "__main__":
    main()
