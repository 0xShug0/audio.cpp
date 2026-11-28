#!/usr/bin/env bash
# Builds the ZONOS3 parity oracle consumed by zonos3_parity.
#   make_oracle.sh <zonos3_build_dir> <gguf_dir> <voice_wav> <out_dir>
# <zonos3_build_dir> holds ZONOS3's native tools (zonos3-cli, zac-cli, spk-encoder-cli, zac-encoder-cli);
# <gguf_dir> holds backbone-ptq.gguf, zac-keep2.gguf, spk-encoder.gguf, zac-encoder.gguf and ssl-teacher.gguf.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "usage: $0 <zonos3_build_dir> <gguf_dir> <voice_wav> <out_dir>" >&2
    exit 1
fi
B=$1 G=$2 V=$3 O=$4
mkdir -p "$O"
TEXT='The quick brown fox jumps over the lazy dog, while 42 crows watched from the old fence.'
printf '%s' "$TEXT" > "$O/text.txt"
cp "$V" "$O/voice.wav"
"$B/zonos3-cli" "$G/backbone-ptq.gguf" --cpu --tts "$TEXT" "$O/tts.npy" --greedy --max 120 --seed 1 \
    --zac "$G/zac-keep2.gguf" --dump-ids "$O/tts_ids.npy" --dump-logits "$O/tts_logits.npy"
"$B/zac-cli" "$G/zac-keep2.gguf" "$O/tts.npy" "$O/tts_audio.npy" --cpu
"$B/spk-encoder-cli" "$G/spk-encoder.gguf" --audio "$V" "$O/spk.npy" --cpu
"$B/zac-encoder-cli" "$G/zac-encoder.gguf" "$G/ssl-teacher.gguf" --audio "$V" "$O/ref_codes.npy" --cpu
