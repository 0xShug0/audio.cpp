#!/usr/bin/env bash
# All 16 of AuK's task types, each on audio it can actually act on.
#
# ⚠ PARAMETERS MUST MATCH WHAT IS ACTUALLY SAID. The first version of this suite asked
# content_edit to replace "we have" in a recording whose transcript does not contain that
# phrase, and to delete "the" from a clip containing three of them. Both ran and produced
# audio; neither tested anything. The transcript of clean_speech is:
#
#   THE ENGLISH FORWARDED TO THE FRENCH BASKETS OF FLOWERS OF WHICH THEY HAD MADE
#   A PLENTIFUL PROVISION TO GREET THE ARRIVAL OF THE YOUNG PRINCESS ...
#
# so the targets below are words that occur EXACTLY ONCE in it, and the anchors are words
# a listener can confirm are actually spoken.
#
# ⚠ THAT SECOND CONDITION IS NOT AUTOMATIC. An earlier version anchored on "forwarded",
# which ASR reports confidently and the speaker does not clearly say -- in isolation that
# word transcribes as "Voted". The model found nothing to anchor to and correctly did
# nothing, which looked like a broken feature. ASR regularizes a mumbled word into the
# one that fits the sentence, so it is reliable for checking OUTPUTS and not for choosing
# parameters.
#
# ⚠ The fixture matters as much as the instruction. "Remove the background noise" of a
# clean recording and "keep the first speaker" of a single-speaker one produce output
# that cannot be judged, because the task had nothing to do. Each case below names the
# fixture that gives the task something to act on, and what a correct result would look
# like -- so the WAVs can be listened to against an expectation rather than in the
# abstract.
#
#   ./scripts/auk_tasks.sh <models-dir> <fixtures-dir> <out-dir> [steps] [backend] [build-dir]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${1:?usage: auk_tasks.sh <models-dir> <fixtures-dir> <out-dir> [steps]}"
FIX="${2:?}"
OUT="${3:?}"
STEPS="${4:-32}"
BACKEND="${5:-cpu}"
BUILD="${6:-build-cpu}"
BIN="$ROOT/$BUILD/bin/auk_pipeline"
mkdir -p "$OUT"

# name|task|variant|params|fixture|seconds|expectation
CASES=(
"zero_shot_tts|zero_shot_tts||text=the tide came in before dawn|reference_voice|5|the new sentence, in the reference speaker's voice"
"instruct_tts|instruct_tts||style_desc=a calm, warm narrator;text=the harbour was quiet in the early morning|-|5|the sentence spoken calmly, in any voice"
"content_edit_replace|content_edit|replace|orig=baskets;new=bundles|clean_speech|6|the same voice, BASKETS becomes BUNDLES"
"content_edit_insert|content_edit|insert_before|text=fresh;anchor=flowers|clean_speech|6|FRESH appears before FLOWERS, nothing else changes"
"content_edit_delete|content_edit|delete|target=plentiful|clean_speech|6|PLENTIFUL gone, the rest intact"
"delete_after_plentiful|content_edit|delete_after|target=plentiful;anchor=made a|clean_speech|6|PLENTIFUL gone, anchored to MADE A -- both words are clearly spoken"
"delete_after_flowers|content_edit|delete_after|target=flowers;anchor=baskets of|clean_speech|6|FLOWERS gone, anchored to BASKETS OF"
"lyric_edit|vocal_edit||orig=baskets;new=bundles|voice_over_music|6|SUBSTITUTE FIXTURE (speech over music, not singing): the word changes, the music stays"
"pitch_up|pitch_edit|increase|semitones=4|clean_speech|6|the same words, audibly higher"
"pitch_down|pitch_edit|decrease|semitones=4|clean_speech|6|the same words, audibly lower"
"speed_up|speed_edit||speed_multiplier=1.5|clean_speech|4|the same words, faster, and SHORTER"
"volume_down|volume_edit|decrease|gain_db=6|clean_speech|6|the same recording, quieter"
"emotion_happy|emotion_edit||emotion=happy|clean_speech|6|the same words and voice, happier delivery"
"timbre_deeper|voice_edit||timbre_desc=a deeper male voice|clean_speech|6|the same words, a different voice"
"deaccent|accent_edit||-|accented_speech|6|the same words, accent reduced, voice kept"
"nonverbal_remove|nonverbal_edit|delete|sound=breaths|clean_speech|6|the same speech without audible breaths"
"whisper_convert|whisper_edit|to_whisper|-|clean_speech|6|the same words, whispered"
"enhance_denoise|enhance_speech||-|noisy_speech|6|the speech kept, the noise gone"
"enhance_dereverb|enhance_speech||-|reverb_speech|6|the speech kept, the room gone"
"improve_bandwidth|improve_quality|bandwidth_extension|-|bandlimited_speech|6|high frequencies restored above 3 kHz"
"separate_first|separate_speech|by_order|n_zh=一|two_speakers|7|only the first speaker, the second removed"
"extract_singing|extract_vocals|singing_only|-|voice_over_music|6|the voice kept, instruments removed (stand-in: this is speech, not singing)"
"extract_voices|extract_vocals|all_human_voices|-|voice_over_music|6|the voice kept, instruments gone -- ASR of the output should still read the words"
"target_speaker|separate_speech|by_content|text=phoebe|two_speakers|7|only the SECOND speaker (who says Phoebe), the first removed"
)

printf '%-22s %-18s %-8s %-20s %s\n' NAME TASK RESULT FIXTURE EXPECTATION | tee "$OUT/summary.txt"
for case in "${CASES[@]}"; do
    IFS='|' read -r name task variant params fixture seconds expectation <<< "$case"
    args=(--thinker "$MODELS/thinker.gguf" --dit "$MODELS/auk_dit.gguf" --vae "$MODELS/auk_vae_raw.gguf"
          --fusion "$MODELS/fusion.f32" --omni "$MODELS/omni"
          --task "$task" --out "$OUT/$name.wav" --seconds "$seconds" --steps "$STEPS"
          --cfg 2.0 --threads "$(nproc)" --backend "$BACKEND")
    [ "$variant" != "" ] && args+=(--variant "$variant")
    if [ "$params" != "-" ]; then
        IFS=';' read -ra pairs <<< "$params"
        for pair in "${pairs[@]}"; do args+=(--param "$pair"); done
    fi
    if [ "$fixture" != "-" ]; then
        args+=(--source-audio "$FIX/${fixture}_16k.wav" --ref-audio "$FIX/${fixture}_24k.wav")
    fi
    if "$BIN" "${args[@]}" > "$OUT/$name.log" 2>&1; then result=ok; else result=FAILED; fi
    printf '%-22s %-18s %-8s %-20s %s\n' "$name" "$task" "$result" "$fixture" "$expectation" | tee -a "$OUT/summary.txt"
    { echo "task:        $task${variant:+/$variant}"; echo "params:      $params";
      echo "fixture:     $fixture"; echo "expectation: $expectation";
      grep -m1 '^  "' "$OUT/$name.log" 2>/dev/null | sed 's/^  /instruction: /'; } > "$OUT/$name.txt"
done
echo; echo "wrote $(ls "$OUT"/*.wav 2>/dev/null | wc -l) files to $OUT"
