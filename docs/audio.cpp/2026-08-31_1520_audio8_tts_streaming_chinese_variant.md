# 2026-08-31 15:20 — Audio8 TTS streaming, Chinese variant, spec fixes & upstream sync

**Branch:** `dev-audio8_tts` → `0xShug0/audio.cpp:main` (`bf3315f`)

## 1. Upstream sync
- Fetched `shug0/main` (`bf3315f docs(server): explain host and port options (#381)` ahead of previous base `a0525f2`).
- `git merge --no-commit shug0/main`:
  - Kept `shug0`'s `webui/native/dist/index.html` via `git checkout --theirs webui/native/dist/index.html` as instructed, then rebuilt.
  - Kept both `soprano-tts` (upstream new model) and `audio8-tts` (our dev) in `webui/configs/models_catalog.json`.
  - Merged `webui/configs/model_params.json`: kept upstream `voxcpm2.text_chunk_mode` addition plus our `audio8_tts` entry (updated to `max_tokens` below).
- `cd webui/native && pnpm run build` → regenerated `webui/native/dist/index.html` (vite 7.3.6, svelte 5.38, 590kB bundle). Staged rebuilt `dist/index.html`.

## 2. Audio8 TTS streaming (PR 1)
- `include/engine/community_models/audio8_tts/session.h`: `Audio8TtsSession` now `IOffline+ IStreaming` (`streaming_policy/start_stream/next_stream_event/finish_stream/reset` etc., `stream_*` state, mirror `omnivoice/session.h:22/97`).
- `src/community_models/audio8_tts/session.cpp`: allow `Offline|Streaming`, `append_cross_faded_chunk` (0.3s, `omnivoice:155`), `initialize_streaming_request`/`synthesize_stream_chunk` with `chunk_text_request` parity and `previous_turn` chaining; streaming verified `audiocpp_cli --mode streaming` (multi-chunk 0.3s crossfade) and `--voice-ref` clone.
- `model_specs/audio8_tts.json`: `modes ["offline","streaming"]`, `runtime.tags ["gguf","stream"]`.

## 3. Traditional → Simplified Chinese variant utility (common)
- New: `include/engine/framework/text/chinese_variant.h`, `src/framework/text/chinese_variant.cpp`, `src/framework/text/chinese_variant_data.inc` (3222 entries from OpenCC `TSCharacters.txt`), added to `CMakeLists.txt:380`.
- API: `is_cantonese_language()`, `convert_traditional_to_simplified()`, `maybe_convert_traditional_to_simplified[_opt]()` — preserves Traditional only when language is `yue`/`cantonese`/`zh-HK`/`zh-MO`.
- Integrated in `src/community_models/audio8_tts/session.cpp:make_request` and streaming `initialize_streaming_request`/`synthesize_stream_chunk` (`stream_language_`); other families can `#include "engine/framework/text/chinese_variant.h"` and call same helper to avoid Cantonese mis-trigger.

## 4. Spec & code fixes (PR 2)
- `model_specs/audio8_tts.json`: `max_new_tokens` → `max_tokens` (request option), fixed invalid JSON (missing `,` after `repo`), corrected `package_defaults` to `huggingface_snapshot` `js-byte/Audio8-TTS-Preview-0.6b-GGUF` and `packages` to `files ["audio8-tts-preview-0.6b-q8_0.gguf"]` (was `Audio8-TTS-.../audio...` with stale `strip_prefix`). Repacked GGUF (`convert_audio8_tts.py`) so `audiocpp_cli --help` now shows `max_tokens`.
- `src/community_models/audio8_tts/session.cpp`: `parse_i64_option({"max_tokens","max_new_tokens"})` with backward compat, error messages updated.
- `webui/configs/model_params.json`: `audio8_tts.max_new_tokens` → `max_tokens`.

## 5. Validation
- `cmake --build --target engine_model_audio8_tts/audiocpp_cli/audiocpp_server` clean.
- `python -m json.tool model_specs/audio8_tts.json` valid.
- Offline `The quick brown fox…` 141k frames, streaming multi-chunk 637k frames, clone with `jfk.wav`/`rubi`/`paraformer_zh` OK.
- Traditional `發財...` auto → `发财...` 141312 frames (same as Simplified), `yue` preserves 159744 frames.
- `pnpm run build` in `webui/native` clean (vite 590kB).

## 6. PR readiness
- Merge commit staged (upstream 7 commits + our 3 feature commits + spec fix).
- Untracked local artifacts not staged: `AGENTS.md`, `Audio8_TTS...`, `out/`, `audio8_tts_streaming_server.json` remain untracked for local dev only.
- `git status` after `pnpm build` shows only intended staged changes plus rebuilt `dist/index.html`; ready to `git commit` merge and `git push` for PR.

Generated via `pnpm run build` on 2026-08-31 15:20 UTC.
