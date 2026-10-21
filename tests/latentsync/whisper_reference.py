"""Export LatentSync Whisper conditioning tensors for a controlled component check."""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-repeat", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference.resolve()))
    from latentsync.whisper.whisper import load_model
    from latentsync.whisper.whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
    from latentsync.whisper.audio2feature import Audio2Feature

    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(str(args.checkpoint), device="cuda").float().eval()
    audio = np.tile(load_audio(str(args.audio)), args.audio_repeat)
    mel = log_mel_spectrogram(pad_or_trim(audio)).unsqueeze(0).cuda()
    with torch.inference_mode():
        _, embeddings = model.encoder(mel, include_embeddings=True)
        segments = model.transcribe(audio, fp16=False)["segments"]
    audio_features = torch.cat([torch.from_numpy(segment["encoder_embeddings"]).permute(0, 2, 1, 3)[0,
        :int((segment["end"] - segment["start"]) / 2)] for segment in segments])
    save_file({"whisper.encoder." + name: value.detach().cpu().contiguous()
               for name, value in model.encoder.state_dict().items()}, args.output / "weights.safetensors")
    features = torch.from_numpy(embeddings).permute(0, 2, 1, 3).contiguous()
    slicer = Audio2Feature.__new__(Audio2Feature)
    slicer.embedding_dim = 384
    slicer.audio_feat_length = [2, 2]
    windows = torch.stack([slicer.get_sliced_feature(features[0], i, 25)[0] for i in range(751)])
    save_file({"audio": torch.from_numpy(audio), "audio_features": audio_features.contiguous(),
               "mel": mel.cpu().contiguous(), "windows": windows.contiguous(),
               "embeddings": features},
              args.output / "reference.safetensors")
    (args.output / "config.json").write_text(json.dumps(vars(model.dims), indent=2) + "\n")
    print(f"conditioning_shape={embeddings.shape}")


if __name__ == "__main__":
    main()
