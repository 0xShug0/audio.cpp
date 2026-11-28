#!/usr/bin/env python3
"""Capture an independent FP32 transcript for the shared Parakeet test input."""
import argparse
import hashlib
import json
from pathlib import Path

# The native package does not include Hugging Face generation settings. Use the
# checkpoint's pinned settings, including TDT duration-token suppression.
GENERATION_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=ROOT / "assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = ParakeetForTDT.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float32).to(args.device).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", revision=GENERATION_REVISION)
    audio, rate = sf.read(args.audio, dtype="float32")
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt").to(args.device)
    with torch.no_grad():
        result = model.generate(**inputs)
    text = processor.batch_decode(result.sequences, skip_special_tokens=True)[0]
    path = args.audio.resolve()
    output = {"reference": "transformers.ParakeetForTDT", "transformers_version": transformers.__version__,
              "generation_config_revision": GENERATION_REVISION, "dtype": "float32",
              "audio": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
              "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "text": text}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
