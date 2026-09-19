#!/usr/bin/env bash
# Transcribe every AuK fixture, so task parameters name words that are actually spoken.
#
# ⚠ Two rounds of demos were invalidated by parameters invented without checking:
# "replace 'we have'" on a recording that never says it, "delete 'the'" on a clip with
# three. The engine ships ASR; there is no reason to guess.
#
# Word timestamps come out too, which makes content edits checkable after the fact:
# transcribe the OUTPUT and see whether the word changed.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="${1:?usage: auk_transcribe_fixtures.sh <fixtures-dir> [asr-model]}"
MODEL="${2:-/mnt/data/models/audiocpp/Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf}"
CLI="$ROOT/build-cpu/bin/audiocpp_cli"

for wav in "$FIX"/*_16k.wav; do
    name="$(basename "$wav" _16k.wav)"
    out="$FIX/$name.transcript.txt"
    text=$("$CLI" --task asr --family parakeet_tdt --model "$MODEL" --backend cpu \
            --threads "$(nproc)" --audio "$wav" 2>/dev/null | sed -n 's/^text_output=//p')
    printf '%s\n' "$text" > "$out"
    printf '%-22s %s\n' "$name" "${text:-<no transcript>}"
done
