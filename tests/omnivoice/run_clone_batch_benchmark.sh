#!/usr/bin/env bash
set -e

# Sweeps OmniVoice voice-clone batch inference (--batch-text-file) over every
# combination of num_inference_steps, generator weight type, and audio
# tokenizer weight type.
#
# Run from anywhere. The script resolves the audio.cpp repository root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

DEVICE="0"
LANG="Bengali"
SEED="42"

# Enable or disable the warmup run
WARMUP=true

STEPS_LIST=("8" "12" "16" "20" "24" "28" "32" "36" "40" "44" "48" "52" "56" "60" "64")
WEIGHT_TYPES=("native" "f32" "f16" "bf16" "q8_0")
AUDIO_TOKENIZER_WEIGHT_TYPES=("native" "f32" "f16" "bf16" "q8_0")

ASSET_DIR="tests/omnivoice/assets"
BATCH_FILE="${ASSET_DIR}/clone_batch_prompts.txt"
REF_WAV="${ASSET_DIR}/ref_audio_02.wav"
REF_TEXT="$(cat "${ASSET_DIR}/ref_text_02.txt")"

OUT_DIR="tests/omnivoice/outputs/clone_batch"
LOG_FILE="tests/omnivoice/outputs/clone_batch.log"

mkdir -p "$OUT_DIR"

# Redirect ALL output to the global log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "Started: $(date)"
echo "Running all audio.cpp OmniVoice batch combinations"
echo "Device: cuda:$DEVICE | Language: $LANG | Seed: $SEED"
echo "Warmup: $WARMUP"
echo "Batch File: $BATCH_FILE"
echo "Global Log: $LOG_FILE"
echo "=========================================="
echo ""

# Basic warmup using the first configured combination
if [[ "$WARMUP" == "true" ]]; then
    WARMUP_STEPS="${STEPS_LIST[0]}"
    WARMUP_WEIGHT_TYPE="${WEIGHT_TYPES[0]}"
    WARMUP_AUDIO_TOKENIZER_WEIGHT_TYPE="${AUDIO_TOKENIZER_WEIGHT_TYPES[0]}"
    WARMUP_OUT_DIR="${OUT_DIR}/warmup"

    echo "=========================================="
    echo "Starting warmup run"
    echo "weight_type=$WARMUP_WEIGHT_TYPE"
    echo "audio_tokenizer_weight_type=$WARMUP_AUDIO_TOKENIZER_WEIGHT_TYPE"
    echo "steps=$WARMUP_STEPS"
    echo "Started at: $(date)"
    echo "=========================================="

    mkdir -p "$WARMUP_OUT_DIR"

    CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
        --task tts \
        --family omnivoice \
        --model models/OmniVoice \
        --backend cuda \
        --device "$DEVICE" \
        --batch-text-file "$BATCH_FILE" \
        --voice-ref "$REF_WAV" \
        --reference-text "$REF_TEXT" \
        --language "$LANG" \
        --seed "$SEED" \
        --num-inference-steps "$WARMUP_STEPS" \
        --session-option omnivoice.generator_weight_type="$WARMUP_WEIGHT_TYPE" \
        --session-option omnivoice.audio_tokenizer_weight_type="$WARMUP_AUDIO_TOKENIZER_WEIGHT_TYPE" \
        --out-dir "$WARMUP_OUT_DIR" \
        --log

    echo "=========================================="
    echo "Warmup completed: $(date)"
    echo "Warmup output: $WARMUP_OUT_DIR"
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

            CURRENT_OUT_DIR="${OUT_DIR}/audiocpp_omnivoice_clone_batch_mdwt_${wt}_atwt_${atwt}_step${steps}"

            echo "[$COUNT/$TOTAL] weight_type=$wt | audio_tokenizer_weight_type=$atwt | steps=$steps -> $CURRENT_OUT_DIR"
            echo "  Started at: $(date +%H:%M:%S)"

            mkdir -p "$CURRENT_OUT_DIR"

            CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
                --task tts \
                --family omnivoice \
                --model models/OmniVoice \
                --backend cuda \
                --device "$DEVICE" \
                --batch-text-file "$BATCH_FILE" \
                --voice-ref "$REF_WAV" \
                --reference-text "$REF_TEXT" \
                --language "$LANG" \
                --seed "$SEED" \
                --num-inference-steps "$steps" \
                --session-option omnivoice.generator_weight_type="$wt" \
                --session-option omnivoice.audio_tokenizer_weight_type="$atwt" \
                --out-dir "$CURRENT_OUT_DIR" \
                --log

            echo "  Completed at: $(date +%H:%M:%S)"
            echo "  Output: $CURRENT_OUT_DIR"
            echo ""
        done
    done
done

echo "=========================================="
echo "All $TOTAL batch combinations completed!"
echo "Finished: $(date)"
echo "Outputs in: $OUT_DIR"
echo "Global Log: $LOG_FILE"
echo "=========================================="
