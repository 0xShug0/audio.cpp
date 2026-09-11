#!/usr/bin/env bash
set -e

OUTPUT_DIR="$(cd "$(dirname "$0")" && pwd)/output"
mkdir -p "$OUTPUT_DIR"
MODEL_DIR="$(cd "$(dirname "$0")/../models" && pwd)"
REFERENCES_DIR="$(cd "$(dirname "$0")/../references" && pwd)"

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$MODEL_DIR:/models:ro" \
  -v "$REFERENCES_DIR:/references:ro" \
  -v "$OUTPUT_DIR:/output" \
  ghcr.io/0xshug0/audio.cpp:full-cuda12 \
  cli \
  --task tts \
  --family qwen3_tts \
  --model /models/Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf \
  --backend cuda \
  --device 0 \
  --text "You are successfully running a text-to-speech model using audio.cpp, a pure C++ inference engine for audio models." \
  --voice-ref /references/ref_audio.wav \
  --reference-text "$(cat "$REFERENCES_DIR/ref_text.txt")" \
  --out /output/speech.wav

echo "Saved to: $OUTPUT_DIR/speech.wav"
