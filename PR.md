# fix(voxcpm1): auto-convert Traditional Chinese to Simplified unless Cantonese

## Summary
Fixes the same `yue`/Cantonese mis-trigger that `audio8_tts` had with Traditional Chinese. Even without `language=yue`/`cantonese`, Traditional input was being rendered with a Cantonese voice. Port the `audio8_tts` OpenCC fix ( `0eec2be feat(text): add Traditional->Simplified Chinese variant utility` ) to `voxcpm1`.

Behaviour mirrors `src/community_models/audio8_tts/session.cpp:477,504` + streaming: only keep Traditional when language is `yue`/`cantonese`/`zh-HK`/`zh-MO`; otherwise convert via shared `engine::text::chinese_variant` (3222-entry `TSCharacters.txt`).

## Root cause
`voxcpm1` had no Traditional→Simplified normalization. The model tokenizer/vocoder treats many Traditional codepoints as Cantonese-correlated features, so `"發財"` without an explicit `language` produced Yue output.

## Changes
- `src/community_models/voxcpm1/session.cpp` (`+41/-8`)
  - `#include "engine/framework/text/chinese_variant.h"`
  - Add `extract_request_language()` (checks `text_input.language` → `voice.style.language` → `language`/`lang` option) – same as `audio8_tts:313`
  - `run_offline_request()`: extract `language`, `maybe_convert = maybe_convert_traditional_to_simplified_opt(text, language)`, convert `voxcpm1.prompt_text`/`prompt_text`/`reference_text` and TTS text before `chunk_text_request` (converted `converted_request` drives chunking)
  - `run_streaming_request()`: same language extraction + convert `prompt_text` and `streaming_text` before `generate_streaming`
  - Reuses existing `engine_core` utility `src/framework/text/chinese_variant.cpp` + `chinese_variant_data.inc` (added in `0eec2be`, built via `CMakeLists.txt:380`), no new deps

## References
- Audio8 fix: `0eec2be`, `src/community_models/audio8_tts/session.cpp`, `include/engine/framework/text/chinese_variant.h:1`, `src/framework/text/chinese_variant.cpp:1`
- Prior voxcpm1 GGUF standalone fix in this branch: `2333009 fix(voxcpm1): extract tokenizer/config directly from GGUF for standalone packages` (assets/tokenizer/audiovae/minicpm)

## Verification
```bash
cmake -S . -B /tmp/build_test -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=voxcpm1 -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_METAL=OFF -DENGINE_ENABLE_VULKAN=OFF
cmake --build /tmp/build_test -j4  # engine_core + audiocpp_cli/server OK

# Traditional without language → auto converts to Simplified (Mandarin voice, not Yue)
# audiocpp_cli --task tts --family voxcpm1 --model ... --text "發財 發現 中國傳統" --out out.wav

# With yue preserves Traditional
# audiocpp_cli --task tts --family voxcpm1 --model ... --language yue --text "發財" --out yue.wav

# Prompt/clone reference text also converted
# audiocpp_cli --task tts --family voxcpm1 --model ... --text "你好" --voice-ref ref.wav --request-option reference_text="發財"
```
Matches `audio8_tts` validation: Traditional `發財...` auto → `发财...` same frames as Simplified; `yue` preserves Traditional.
