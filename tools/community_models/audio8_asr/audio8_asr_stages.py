#!/usr/bin/env python3
"""Capture staged intermediate tensors from the Audio8-ASR-0.1B reference
pipeline for C++ side parity checks: mel -> encoder -> tower -> pool -> projector.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

import librosa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    for audio_path in args.audio:
        stem = Path(audio_path).stem
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(Path(audio_path).resolve())},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            }
        ]
        batch = processor.apply_chat_template(
            conv,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
            add_generation_prompt=True,
            audio_max_length=int(args.max_audio_seconds * 16000),
            text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
        )
        feats = batch["input_features"][0]  # [mel, frames]
        with torch.inference_mode():
            # Replicate modeling_arkasr._project_audio_row stage by stage.
            features = feats
            enc_in = features.to(next(model.audio_encoder.parameters()).dtype)
            encoded = model.audio_encoder(enc_in, feature_lens=torch.tensor([features.shape[1]]))
            hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
            encoder_out = hidden.squeeze(0)  # [T, 1024]

            tower_out = model.audio_mlp_tower(encoder_out)  # [T, 1024]
            input_ids = batch["input_ids"][0].tolist()
            n_tokens = sum(1 for t in input_ids if t == model.config.audio_token_id)
            pooled = tower_out
            if pooled.shape[0] != n_tokens:
                pooled = torch.nn.functional.adaptive_avg_pool1d(
                    pooled.transpose(0, 1).float().unsqueeze(0), output_size=n_tokens
                ).squeeze(0).transpose(0, 1)
            projected = model.audio_projector(pooled)  # [N, 512]

        np.save(args.outdir / f"{stem}.mel.npy", features.to(torch.float32).numpy())
        np.save(args.outdir / f"{stem}.encoder_out.npy", encoder_out.numpy())
        np.save(args.outdir / f"{stem}.projected.npy", projected.numpy())
        info = {
            "mel_shape": list(features.shape),
            "encoder_tokens": int(encoder_out.shape[0]),
            "audio_tokens": int(n_tokens),
            "projected_shape": list(projected.shape),
        }
        (args.outdir / f"{stem}.stages.json").write_text(json.dumps(info, indent=2))
        print(stem, info)


if __name__ == "__main__":
    main()
