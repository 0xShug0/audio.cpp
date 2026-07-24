#!/usr/bin/env python3
"""Run the official GLM-TTS Whisper-VQ tokenizer for parity checks."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import safetensors
import torch
import torchaudio
from transformers import WhisperFeatureExtractor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(args.reference_dir.resolve()))
    from utils.whisper_models.configuration_whisper import WhisperVQConfig
    from utils.whisper_models.modeling_whisper import WhisperVQEncoder

    model_dir = str(args.model_dir.resolve())
    config = WhisperVQConfig.from_pretrained(model_dir)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(str(Path(model_dir) / "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as file:
            for key in file.keys():
                if not key.startswith("model.encoder."):
                    continue
                new_key = key[len("model.encoder.") :]
                if new_key.startswith("layer_norm"):
                    continue
                if new_key.startswith("layers"):
                    layer_id = int(new_key.split(".")[1])
                    if layer_id >= config.quantize_position:
                        continue
                state_dict[new_key] = file.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval().to(args.device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    audio, sample_rate = torchaudio.load(args.audio)
    audio = audio[0]
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(
            audio.to(args.device), sample_rate, 16000
        )
    features = feature_extractor(
        [audio.cpu().numpy()],
        sampling_rate=16000,
        return_attention_mask=True,
        return_tensors="pt",
        device=args.device,
        padding="longest",
        pad_to_multiple_of=640,
    ).to(args.device)
    with torch.inference_mode():
        outputs = model(**features)
    mask = features.attention_mask[:, ::2][:, ::2].bool()
    ids = outputs.quantized_token_ids[mask].cpu().tolist()
    print(json.dumps({"count": len(ids), "ids": ids}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
