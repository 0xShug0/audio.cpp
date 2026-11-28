#!/usr/bin/env bash
# Builds the ZONOS2 parity oracle consumed by zonos2_prompt_test and zonos2_parity.
#   make_oracle.sh <zonos2_build_dir> <gguf_dir> <voice_wav> <out_dir>
# <zonos2_build_dir> is a Zyphra/zonos2.cpp build tree (zonos2-cli, spk-encoder-cli, dac-cli, libzonos2.a, shared
# ggml); oracle_decode.cpp is compiled against it. <gguf_dir> holds zonos2-q8_0.gguf, dac.gguf and spk-encoder.gguf.
# ffmpeg must be on PATH.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "usage: $0 <zonos2_build_dir> <gguf_dir> <voice_wav> <out_dir>" >&2
    exit 1
fi
B=$1 G=$2 V=$3 O=$4
mkdir -p "$O/val" "$O/val_spk"
TEXT='The quick brown fox jumps over the lazy dog, while forty two crows watched from the old fence.'
printf '%s' "$TEXT" > "$O/text.txt"
cp "$V" "$O/voice.wav"
M="$G/zonos2-q8_0.gguf"
ffmpeg -v error -y -i "$V" -ac 1 -ar 24000 -f f32le "$O/voice24k.f32"
"$B/spk-encoder-cli" "$G/spk-encoder.gguf" --raw "$O/voice24k.f32" "$O/spk.npy"
S=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$B/CMakeCache.txt")
RPATHS=()
for d in "$B/ggml/src" "$B"/ggml/src/ggml-*; do [ -d "$d" ] && RPATHS+=("-Wl,-rpath,$d"); done
"${CXX:-c++}" -O2 -std=c++17 -I"$S/src" -I"$S/ggml/include" "$(dirname "$0")/oracle_decode.cpp" \
    -o "$O/zonos2-oracle-decode" "$B/libzonos2.a" -L"$B/ggml/src" -lggml -lggml-base "${RPATHS[@]}"
"$B/zonos2-cli" "$M" --cpu --build-prompt "Hello." "$O/prompt_hello.npy"
"$B/zonos2-cli" "$M" --cpu --build-prompt "Hello." "$O/prompt_hello_spk.npy" --speaker "$O/spk.npy" --noisy-bg
# Teacher-forcing sequences come from seeded sampled takes: greedy decoding without the repetition penalty
# collapses into the silence token, which would leave the decode parity nothing but silence frames.
"$B/zonos2-cli" "$M" --cpu --tts "$TEXT" "$O/tts.npy" --seed 1 --max 300 --dump-ids "$O/tts_ids.npy"
"$B/zonos2-cli" "$M" --cpu --validate "$O/tts_ids.npy" "$O/val"
"$B/zonos2-cli" "$M" --cpu --tts "$TEXT" "$O/tts_spk.npy" --seed 1 --max 300 --speaker "$O/spk.npy" --noisy-bg \
    --dump-ids "$O/tts_spk_ids.npy"
"$B/zonos2-cli" "$M" --cpu --validate "$O/tts_spk_ids.npy" "$O/val_spk" --speaker "$O/spk.npy" --speaker-pos 0
"$B/zonos2-cli" "$M" --cpu --build-prompt "$TEXT" "$O/prompt.npy"
"$B/zonos2-cli" "$M" --cpu --build-prompt "$TEXT" "$O/prompt_spk.npy" --speaker "$O/spk.npy" --noisy-bg
"$O/zonos2-oracle-decode" "$M" "$O/tts_ids.npy" "$O/prompt.npy" "$O/decode_logits.npy"
"$O/zonos2-oracle-decode" "$M" "$O/tts_spk_ids.npy" "$O/prompt_spk.npy" "$O/decode_spk_logits.npy" "$O/spk.npy"
"$B/zonos2-cli" "$M" --cpu --tts "$TEXT" "$O/tts_greedy.npy" --greedy --max 300
"$B/dac-cli" "$G/dac.gguf" "$O/tts.npy" "$O/tts_audio.wav" --cpu
