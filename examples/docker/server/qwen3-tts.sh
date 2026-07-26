#!/usr/bin/env bash
set -e

REFERENCES_DIR="$(cd "$(dirname "$0")/../references" && pwd)"
REF_TEXT="$(cat "$REFERENCES_DIR/ref_text.txt")"

mkdir -p output
curl -f http://localhost:8080/v1/audio/speech -H 'Content-Type: application/json' -o output/speech.wav -d '
{
  "model": "qwen3-tts",
  "input": "You are successfully running a text-to-speech model using audio.cpp, a pure C++ inference engine for audio models.",
  "voice_ref": "/references/ref_audio.wav",
  "reference_text": "'"$REF_TEXT"'"
}
'

echo "Saved to: output/speech.wav"
