#!/usr/bin/env bash
set -e

# Direct C++ CLI baseline.
# Run from anywhere. The script resolves the audio.cpp repository root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

DEVICE=0
MODEL="models/OmniVoice"

ASSET_DIR="tests/omnivoice/assets"
TEXT="আমি একটা টিটিএস মডেল, নাম OmniVoice। কিভাবে আপনাকে সাহায্য করতে পারি?"
REF_WAV="${ASSET_DIR}/ref_audio_02.wav"
REF_TEXT="$(cat "${ASSET_DIR}/ref_text_02.txt")"

LANGUAGE="Bengali"
SEED=42
STEPS=20
GUIDANCE_SCALE=2.0
SPEED=1.2

GENERATOR_WEIGHT_TYPE="f16"
AUDIO_TOKENIZER_WEIGHT_TYPE="f16"

OUTPUT="tests/omnivoice/outputs/python_cpp_simple/cpp_output.wav"

mkdir -p "$(dirname "$OUTPUT")"

CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
    --task tts \
    --family omnivoice \
    --model "$MODEL" \
    --backend cuda \
    --device "$DEVICE" \
    --text "$TEXT" \
    --voice-ref "$REF_WAV" \
    --reference-text "$REF_TEXT" \
    --language "$LANGUAGE" \
    --seed "$SEED" \
    --num-inference-steps "$STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --request-option "speed=$SPEED" \
    --session-option "omnivoice.generator_weight_type=$GENERATOR_WEIGHT_TYPE" \
    --session-option "omnivoice.audio_tokenizer_weight_type=$AUDIO_TOKENIZER_WEIGHT_TYPE" \
    --out "$OUTPUT" \
    --log

echo
echo "C++ output: $OUTPUT"
