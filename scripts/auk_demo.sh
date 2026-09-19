#!/usr/bin/env bash
# One invocation per capability AuK's model card claims, each writing a WAV.
#
# The parity tests answer "does this compute what the reference computes". They cannot
# answer "does the denoising denoise", because that is a judgement about audio. This
# script exists to produce the artifacts for that judgement, with the instruction that
# produced each one recorded next to it.
#
#   ./scripts/auk_demo.sh <models-dir> <out-dir> [steps]
#
# <models-dir> holds thinker.gguf, auk_dit.gguf, auk_vae_raw.gguf, omni/ and the
# fusion fixture.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${1:?usage: auk_demo.sh <models-dir> <out-dir> [steps]}"
OUT="${2:?usage: auk_demo.sh <models-dir> <out-dir> [steps]}"
STEPS="${3:-16}"
BIN="$ROOT/build-cpu/bin/auk_pipeline"
SOURCE16="$OUT/source_16k.wav"
SOURCE24="$OUT/source_24k.wav"

mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "build auk_pipeline first: cmake --build build-cpu --target auk_pipeline"; exit 1; }
for f in thinker.gguf auk_dit.gguf auk_vae_raw.gguf; do
    [ -f "$MODELS/$f" ] || { echo "missing $MODELS/$f"; exit 1; }
done

# ⚠ THESE INSTRUCTIONS ARE NOT FREE-FORM. AuK ships the templates it was trained on in
# src/auk/infer/pe.config.yaml, and its prompt enhancer exists to rewrite a user's
# phrasing INTO them. Inventing plausible-sounding English instead produces audio that
# is generated but not the task -- which is exactly what the first pass of this script
# did. Each line below is the canonical template with its placeholders filled.
#
# ⚠ Five tasks have NO English template, only Chinese. They are run in Chinese, because
# an English paraphrase of a Chinese-only instruction is a different prompt, not a
# translation of one the model knows.
#
# category|name|instruction|needs_audio|seconds
CASES=(
"generation|instruct_tts|Based on the following description: \"a calm, warm narrator\", generate speech content \"the harbour was quiet in the early morning\".|no|4"
"generation|zero_shot_tts|Say the following with the same voice: \"the tide came in before dawn\"|yes|4"
"content|content_edit_replace|Replace 'quiet' with 'busy'.|yes|4"
"content|content_edit_delete|Remove 'the'.|yes|4"
"content|vocal_edit|Change \"river\" to \"harbour\" in the vocal recording.|yes|4"
"acoustic|speed_edit|Adjust the speech speed to 1.5x.|yes|4"
"acoustic|pitch_edit|Raise the pitch by 4 semitones.|yes|4"
"acoustic|volume_edit|Decrease the volume by 6 dB.|yes|4"
"paralinguistic|emotion_edit|Change the emotion to happy.|yes|4"
"paralinguistic|voice_edit|Keep the spoken content unchanged and change the timbre to: \"a deeper male voice\".|yes|4"
"paralinguistic|nonverbal_edit|Remove all the laughter from the audio.|yes|4"
"paralinguistic|whisper_edit_zh|用小声耳语的方式把这段话说出来。|yes|4"
"paralinguistic|accent_edit_zh|请去掉这段语音里的方言口音，保持说话人音色一致。|yes|4"
"enhancement|enhance_speech_zh|请清理这段输入语音，不做说话人删除，并去除房间混响，输出干净的人声结果。|yes|4"
"enhancement|separate_speech_zh|这段音频中只保留第一个开始说话的人对应的语音，去掉其余说话人。|yes|4"
"enhancement|improve_quality_zh|请对这段语音做超分辨率/带宽扩展处理，恢复被削掉的高频成分，输出宽带纯净人声。|yes|4"
"enhancement|extract_vocals|Keep all human voices, speech and singing alike, drop everything else.|yes|4"
)

printf '%-16s %-20s %-10s %s\n' CATEGORY NAME RESULT FILE | tee "$OUT/summary.txt"
for case in "${CASES[@]}"; do
    IFS='|' read -r category name instruction needs_audio seconds <<< "$case"
    args=(--thinker "$MODELS/thinker.gguf" --dit "$MODELS/auk_dit.gguf" --vae "$MODELS/auk_vae_raw.gguf"
          --fusion "$MODELS/fusion.f32" --omni "$MODELS/omni"
          --instruction "$instruction" --out "$OUT/$name.wav"
          --seconds "$seconds" --steps "$STEPS" --cfg 2.0 --threads "$(nproc)")
    if [ "$needs_audio" = "yes" ]; then
        args+=(--source-audio "$SOURCE16" --ref-audio "$SOURCE24")
    fi
    if "$BIN" "${args[@]}" > "$OUT/$name.log" 2>&1; then
        result=ok
    else
        result=FAILED
    fi
    printf '%-16s %-20s %-10s %s\n' "$category" "$name" "$result" "$name.wav" | tee -a "$OUT/summary.txt"
    # The instruction is the only thing that distinguishes these runs, so it is kept
    # next to the audio rather than only in this script.
    echo "$instruction" > "$OUT/$name.txt"
done

echo
echo "wrote $(ls "$OUT"/*.wav 2>/dev/null | wc -l) files to $OUT"
