#!/usr/bin/env bash
set -e

# Sweeps OmniVoice voice-clone single inference over every combination of
# num_inference_steps, generator weight type, and audio tokenizer weight type.
#
# Run from anywhere. The script resolves the audio.cpp repository root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

DEVICE="0"
LANG="Bengali"
SEED="42"

# Enable or disable warmup runs
WARMUP=true

# Number of warmup runs when WARMUP=true
WARMUP_QUANTITY=10

STEPS_LIST=("8" "12" "16" "20" "24" "28" "32" "36" "40" "44" "48" "52" "56" "60" "64")
WEIGHT_TYPES=("native" "f32" "f16" "bf16" "q8_0")
AUDIO_TOKENIZER_WEIGHT_TYPES=("native" "f32" "f16" "bf16" "q8_0")

ASSET_DIR="tests/omnivoice/assets"
REF_WAV="${ASSET_DIR}/ref_audio_02.wav"
REF_TEXT="$(cat "${ASSET_DIR}/ref_text_02.txt")"
TEXT="আমি একটা টিটিএস মডেল, নাম OmniVoice। কিভাবে আপনাকে সাহায্য করতে পারি?"

OUT_DIR="tests/omnivoice/outputs/clone_single"
LOG_FILE="tests/omnivoice/outputs/clone_single.log"

mkdir -p "$OUT_DIR"

# Redirect ALL output to the global log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "Started: $(date)"
echo "Running all audio.cpp OmniVoice combinations"
echo "Device: cuda:$DEVICE | Language: $LANG | Seed: $SEED"
echo "Warmup: $WARMUP | Warmup Quantity: $WARMUP_QUANTITY"
echo "Global Log: $LOG_FILE"
echo "=========================================="
echo ""

# Basic warmup using the first configured combination
if [[ "$WARMUP" == "true" ]]; then
    WARMUP_STEPS="${STEPS_LIST[0]}"
    WARMUP_WEIGHT_TYPE="${WEIGHT_TYPES[0]}"
    WARMUP_AUDIO_TOKENIZER_WEIGHT_TYPE="${AUDIO_TOKENIZER_WEIGHT_TYPES[0]}"

    echo "=========================================="
    echo "Starting $WARMUP_QUANTITY warmup run(s)"
    echo "weight_type=$WARMUP_WEIGHT_TYPE"
    echo "audio_tokenizer_weight_type=$WARMUP_AUDIO_TOKENIZER_WEIGHT_TYPE"
    echo "steps=$WARMUP_STEPS"
    echo "Started at: $(date)"
    echo "=========================================="
    echo ""

    for ((warmup_index = 1; warmup_index <= WARMUP_QUANTITY; warmup_index++)); do
        WARMUP_OUT="${OUT_DIR}/warmup_${warmup_index}.wav"

        echo "[Warmup $warmup_index/$WARMUP_QUANTITY]"
        echo "  Started at: $(date +%H:%M:%S)"

        CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
            --task tts \
            --family omnivoice \
            --model models/OmniVoice \
            --backend cuda \
            --device "$DEVICE" \
            --text "$TEXT" \
            --voice-ref "$REF_WAV" \
            --reference-text "$REF_TEXT" \
            --language "$LANG" \
            --seed "$SEED" \
            --num-inference-steps "$WARMUP_STEPS" \
            --session-option omnivoice.generator_weight_type="$WARMUP_WEIGHT_TYPE" \
            --session-option omnivoice.audio_tokenizer_weight_type="$WARMUP_AUDIO_TOKENIZER_WEIGHT_TYPE" \
            --out "$WARMUP_OUT" \
            --log

        echo "  Completed at: $(date +%H:%M:%S)"
        echo "  Output: $WARMUP_OUT"
        echo ""
    done

    echo "=========================================="
    echo "All $WARMUP_QUANTITY warmup run(s) completed: $(date)"
    echo "=========================================="
    echo ""
else
    echo "Warmup disabled."
    echo ""
fi

TOTAL=$(( ${#STEPS_LIST[@]} * ${#WEIGHT_TYPES[@]} * ${#AUDIO_TOKENIZER_WEIGHT_TYPES[@]} ))
COUNT=0

for steps in "${STEPS_LIST[@]}"; do
    for wt in "${WEIGHT_TYPES[@]}"; do
        for atwt in "${AUDIO_TOKENIZER_WEIGHT_TYPES[@]}"; do
            COUNT=$((COUNT + 1))

            OUT_NAME="audiocpp_omnivoice_clone_mdwt_${wt}_atwt_${atwt}_step${steps}.wav"

            echo "[$COUNT/$TOTAL] weight_type=$wt | audio_tokenizer_weight_type=$atwt | steps=$steps -> $OUT_NAME"
            echo "  Started at: $(date +%H:%M:%S)"

            CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
                --task tts \
                --family omnivoice \
                --model models/OmniVoice \
                --backend cuda \
                --device "$DEVICE" \
                --text "$TEXT" \
                --voice-ref "$REF_WAV" \
                --reference-text "$REF_TEXT" \
                --language "$LANG" \
                --seed "$SEED" \
                --num-inference-steps "$steps" \
                --session-option omnivoice.generator_weight_type="$wt" \
                --session-option omnivoice.audio_tokenizer_weight_type="$atwt" \
                --out "${OUT_DIR}/${OUT_NAME}" \
                --log

            echo "  Completed at: $(date +%H:%M:%S)"
            echo "  Output: ${OUT_DIR}/${OUT_NAME}"
            echo ""
        done
    done
done

echo "=========================================="
echo "All $TOTAL combinations completed!"
echo "Finished: $(date)"
echo "Outputs in: $OUT_DIR"
echo "Global Log: $LOG_FILE"
echo "=========================================="
