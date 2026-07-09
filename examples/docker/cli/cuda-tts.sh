#!/usr/bin/env bash
set -e

OUTPUT_DIR="$(cd "$(dirname "$0")" && pwd)/output"
mkdir -p "$OUTPUT_DIR"
MODEL_DIR="$(cd "$(dirname "$0")/../models" && pwd)"

docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$MODEL_DIR:/models:ro" \
  -v "$OUTPUT_DIR:/output" \
  local/audiocpp:full-cuda \
  cli \
  --task tts \
  --family pocket_tts \
  --model /models/pocket-tts \
  --backend cuda \
  --device 0 \
  --text "You are successfully running a text-to-speech model using audio.cpp, a pure C++ inference engine for audio models." \
  --voice-id alba \
  --out /output/speech.wav

echo "Saved to: $OUTPUT_DIR/speech.wav"
