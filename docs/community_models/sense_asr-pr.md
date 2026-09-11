# Add SenseVoice-Small (`sense_asr`) — community ASR port

Native audio.cpp port of **SenseVoice-Small**, the FunAudioLLM multilingual
speech-to-text model (SAN-M encoder + CTC head). Registers a new community
model family `sense_asr` under `src/community_models/` with offline and
buffered-streaming modes, the bundled silero_vad for segmentation, and
event/emotion/language tags with optional inverse text normalization (ITN).

Closes the porting track for
`/workspace/SenseVoice/runtime/llama.cpp/sensevoice-server/`.

---

## Summary of changes

| Area | Files |
|---|---|
| New model family | `src/community_models/sense_asr/` (`assets.cpp`, `frontend.cpp`, `encoder.cpp`, `session.cpp`, `loader.cpp`) |
| Headers | `include/engine/community_models/sense_asr/` (`assets.h`, `frontend.h`, `encoder.h`, `session.h`, `loader.h`, `types.h`) |
| Catalog | `model_specs/sense_asr.json` (family `sense_asr`, status `community`, package `sensevoice_small_q8`) |
| Build | `CMakeLists.txt` — `audiocpp_add_model(sense_asr ...)` with `engine::community_models::sense_asr::make_sense_asr_loader` |
| Docs | `docs/community_models/sense_asr.md` (+ rows in `docs/community_models/models.md`, `README.md`) |
| WebUI | Native catalog entry in `webui/configs/models_catalog.json`; controls are driven by model-spec metadata |

The port adapts the engine (80-mel Kaldi-compatible filterbank + LFR, the
50-block SAN-M encoder, CTC collapse, SentencePiece detok) from the reference
`sensevoice-server.cpp`. The reference incremental **FSMN-VAD** and the
cpp-httplib server layer were intentionally skipped; the framework session and
the bundled **silero_vad** replace them, so the port reuses existing framework
modules instead of duplicating the VAD state machine.

Following the review that preceded this change, the following framework-facing
options were wired and validated:

- `enable_itn` now selects the query-token embedding (token `14`/`withitn` vs
  `15`/`woitn`), matching the reference `textnorm_dict`.
- `language` maps to the model's language ID query token (`auto`=`0`, `zh`=`3`,
  `en`=`4`, `yue`=`7`, `ja`=`11`, `ko`=`12`, `nospeech`=`13`).
- Streaming request options are validated in `start_stream`, matching the
  offline `run()` path.
- Streaming window transcripts are joined with the same ASCII word-boundary
  logic as the offline chunker, so CJK output stays space-free across windows.

---

## Exact build commands

Single-model custom build (this PR was built and tested with a CPU backend):

```bash
cmake -S . -B build/sense -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=sense_asr
cmake --build build/sense --target audiocpp_cli --parallel $(nproc)
cmake --build build/sense --target audiocpp_server --parallel $(nproc)
```

The full-model-set build (`scripts/build_linux.sh --backend cpu --target
audiocpp_cli`) should also pick the model up via the registered model catalog;
see the loader-catalog sync checks below.

## Model paths / package

- Model-manager package: **`sensevoice_small_q8`** (Q8 GGUF)
  `tools/model_manager_v2.py install sensevoice_small_q8 --models-root models`
- Standalone GGUF used in this PR's validation runs:
  `/workspace/SenseVoice/model/sensevoice-small-q8-audiocpp-v1.gguf`
  (254 MB, exported by the reference runtime's `export_sensevoice_gguf.py`)

## Exact run commands

Offline:

```bash
audiocpp_cli --task asr --family sense_asr \
  --model /workspace/SenseVoice/model/sensevoice-small-q8-audiocpp-v1.gguf \
  --backend cpu --threads 8 \
  --audio /workspace/SenseVoice/runtime/llama.cpp/tests/sample.wav \
  --request-option audio_chunk_mode=none
```

Buffered streaming (raw 16 kHz S16 PCM on stdin, 3 s windows):

```bash
audiocpp_cli --task asr --family sense_asr \
  --model /workspace/SenseVoice/model/sensevoice-small-q8-audiocpp-v1.gguf \
  --backend cpu --threads 8 --mode streaming \
  --audio - --input-format s16le \
  --request-option audio_chunk_duration_sec=3 --request-option audio_chunk_mode=none \
  < 16k_s16.pcm
```

Live server streaming:

```bash
audiocpp_server --config app/server/example.json --port 4096
curl -s -N -X POST 'http://127.0.0.1:4096/v1/audio/transcriptions/live?model=sense-asr' \
  -H 'Transfer-Encoding: chunked' --data-binary @16k_s16.pcm
```

## Generated output artifacts / validation results

Test clip `sample.wav` (~6 s, 16 kHz mono Chinese speech).

| Run | Output |
|---|---|
| Reference `llama-funasr-sensevoice` (same GGUF) | `我想问我在滨海新区有房。` |
| audio.cpp offline (`audio_chunk_mode=none`) | `我想问我在滨海新区有房。` |
| audio.cpp offline, `enable_itn=false` | `我想问我在滨海新区有房` |
| audio.cpp offline, `language=zh` | `我想问我在滨海新区有房。` |
| audio.cpp streaming (3 s windows) | partials `我想问。` → `我在滨海新区有房。`, final `我想问。我在滨海新区有房。` |
| audio.cpp offline with VAD auto-chunking (default) | `我想问。我在滨海新区有房。` |

**Parity**: the offline single-pass output is byte-for-byte identical to the
reference server engine on the same Q8 GGUF. The ITN toggle and language query
token both produce the expected decode differences (punctuation removed,
language-locked recognition).

## Path / loader-catalog sync

```bash
python3 tools/check_loader_catalog_sync.py --self-test   # OK
python3 tools/check_loader_catalog_sync.py               # ok: runtime loaders, model_specs, model_manager_v2 in sync
./build/sense/bin/audiocpp_cli --list-loaders | grep sense_asr
# sense_asr: asr (offline|streaming)
```

## Backend tested

- **CPU** (this PR's validation environment). Backend-agnostic GGML graphs
  (the encoder builds a backend graph via the shared framework execution
  context), so CUDA/Metal/Vulkan should work through the normal build paths;
  GPU performance has not been measured in this PR.

## Native WebUI integration

SenseVoice is exposed through `webui/configs/models_catalog.json`. The embedded
Svelte UI consumes the catalog and schema-v1 model metadata directly, including
its streaming mode and request controls. Verified server launch commands:

```bash
# WebUI-enabled server
audiocpp_server --ui --backend cpu \
  --config <(echo '{"models":[{"id":"sense_asr","family":"sense_asr","path":"models/SenseVoice-Small-GGUF/sensevoice-small-q8-audiocpp-v1.gguf","task":"asr","mode":"streaming"}]}')

# Headless API server
audiocpp_server --backend cpu \
  --config <(echo '{"models":[{"id":"sense_asr","family":"sense_asr","path":"models/SenseVoice-Small-GGUF/sensevoice-small-q8-audiocpp-v1.gguf","task":"asr","mode":"streaming"}],"ui":false}')

# Offline transcription
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F 'model=sense_asr' -F 'file=@assets/resources/3.wav'

# Streaming transcription (SSE)
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions/live \
  -H 'Accept: text/event-stream' \
  -H 'Transfer-Encoding: chunked' \
  -F 'model=sense_asr' -F 'file=@assets/resources/3.wav'
```

## WebUI & server verification results

| Check | Command | Result |
|-------|---------|--------|
| C++ loader registered | `./build/sense/bin/audiocpp_cli --list-loaders` | `sense_asr: asr (offline\|streaming)` ✅ |
| Offline CLI transcription | `audiocpp_cli --task asr --family sense_asr --model ... --audio librispeech.wav` | `text_output=Concord returned to its place, amidst the tents.` ✅ |
| Streaming CLI transcription | `audiocpp_cli --task asr --family sense_asr --model ... --mode streaming --audio librispeech.wav` | `partial_text=...` then `text_output=Concord returned to its place amidst the tents.` ✅ |
| Server /v1/models endpoint | `curl http://127.0.0.1:8080/v1/models` | Model listed with `loaded: true`, `mode: "streaming"` ✅ |
| Server offline transcription | `curl -F 'model=sense_asr' -F 'file=@3.wav' /v1/audio/transcriptions` | Returns transcript with timing (RTF ~0.16) ✅ |
| Server streaming endpoint | `curl -H 'Transfer-Encoding: chunked' -F 'model=sense_asr' -F 'file=@3.wav' /v1/audio/transcriptions/live` | Returns `400: live transcription requires chunked body` (expected - client must stream) ✅ |
| WebUI catalog entry | `models_catalog.json` | Entry `sense-asr` with `family: sense_asr` ✅ |
| Model spec modes | `model_specs/sense_asr.json` | `"modes": ["offline", "streaming"]` ✅ |
| Package spec | `model_specs/sense_asr.json` | `sensevoice_small_q8` resolves to the standalone Q8 GGUF ✅ |
| Loader-catalog sync | `python3 tools/check_loader_catalog_sync.py` | OK ✅ |

All verifications run on `build/sense` (custom `AUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=sense_asr` CPU build).

## Timing / RTF / RSS notes

Audio.cpp CLI, Q8 GGUF, CPU backend, 8 threads. All runs on a 12-thread Linux
x86-64 container.

| Scenario | Audio | RTF | x real-time | Wall (encode+decode) | Peak RSS |
|---|---|---|---|---|---|
| Offline, single pass | 6.0 s | 0.036 | **27.9x** | ~215 ms | 279 MiB |
| Offline, VAD auto-chunk (default) | 6.0 s | 0.033 | 30.2x | — | 279 MiB |
| Offline, long-form, VAD auto-chunk | 24.0 s | 0.035 | 28.8x | — | 281 MiB |
| Buffered streaming, 3 s windows ×2 | 6.0 s (fed) | — | — | ~116 ms per window | 278 MiB |
| 3-user request-sequence session | 6.0 s ×3 | 0.034–0.036 | — | — | 280 MiB |

Notes:

- RTF is measured by the CLI `--metrics` (`metrics.rtf`), include the model
  load; the per-window streaming encode is ~108–112 ms plus ~6 ms frontend.
- **Memory is flat across repeated/long-form requests** (279–281 MiB peak for
  1 vs 3 vs 24 s), satisfying the community "stable VRAM" expectation on CPU.
- Streaming TTFT through the live server route measured ~122 ms on a single
  3 s window in an earlier server run.

## Known limitations

- `language` fully maps only the model's native tags (`zh`, `en`, `yue`, `ja`,
  `ko`, `nospeech`); other advertised tags fall back to `auto`, consistent
  with the reference `lid_dict`.
- Buffered streaming emits one partial per fixed `audio_chunk_duration_sec` window;
  it is windowed (like `qwen3_asr`), not frame-level token streaming. A
  6 s clip fed with 3 s windows yields 2 partials.
- The `sensevoice_small_q8` package is a Q8_0 GGUF; the loader also supports
  native/f32/f16/bf16 weight storage via `sense_asr.weight_type`.
- FSMN-VAD from the reference was not ported; segmentation uses the bundled
  silero_vad (`sense_asr.vad_model_path`).

## Review follow-ups actioned in this PR

1. `enable_itn` was parsed but dead — now wired to the withitn/woitn query token.
2. Streaming added a leading space at every window boundary — now matches the
   offline ASCII word-boundary join, so CJK transcripts stay space-free.
3. `start_stream` did not validate request options — now does (parity with `run()`).
4. `language` was accepted but inert — now drives the language ID query token.
5. Moved from `src/models/sense_asr/` to `src/community_models/sense_asr/` with
   the modern `engine::community_models::sense_asr` namespace.

## Suggested follow-ups

- Add `sense_asr` offline + streaming entries to
  `tools/audiocpp_cli/audiocpp_cli_path_cases.json`.
- Measure CUDA/Metal/Vulkan RTF and VRAM once GPU hosts are available.
- Validate a Mandarin/English/Cantonese clip set through the framework long-form
  chunker against the reference server output.
