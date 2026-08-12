# SenseVoice-Small Community ASR

Default model-manager downloads use the published GGUF package when available;
the original source/conversion instructions below remain valid for manual use.

`sense_asr` is a native audio.cpp port of SenseVoice-Small, a multilingual
speech-to-text model from FunAudioLLM that reads audio through a 50-block SAN-M
encoder and a CTC head. It recognizes 23 language tags and emits rich
`<|event|>/<|emotion|>/<|language|>` meta tags, with optional inverse text
normalization (ITN). The port adapts the SenseVoice llama.cpp runtime
(`sensevoice-server.cpp`) into the framework layout and uses the bundled
silero_vad for audio segmentation instead of the reference FSMN-VAD.

## Capabilities

| Field | Value |
|---|---|
| Task | `asr` |
| Modes | `offline`, buffered `streaming` |
| Languages | `auto`, `zh`, `en`, `yue`, `ja`, `ko`, `pt`, `ru`, `es`, `it`, `fr`, `de`, `nl`, `pl`, `tr`, `ar`, `hi`, `vi`, `th`, `id`, `ms`, `fa`, `nospeech` |
| Input | WAV; audio.cpp converts to 16 kHz mono |
| Output | Transcript, detected language, event/emotion tags |
| Frontend | 80-mel filterbank + LFR (140ms context), Kaldi-compatible |
| Encoder | SAN-M, 1 stem + 49 main blocks + 20 timestamp-prediction blocks, 4 heads, 512-wide, kernel 11 |
| Decoding | CTC greedy collapse with SentencePiece detok |
| ITN | Query token 14 (`withitn`) on, 15 (`woitn`) off; on by default |
| Package | `sensevoice_small_q8` Q8 GGUF |

The `language` request option hard-codes a language ID query token when one of
the model's native tags (`zh`, `en`, `yue`, `ja`, `ko`, `nospeech`) is given;
the rest of the tags fall back to `auto`. `keep_tags` keeps the
`<|event|>/<|emotion|>/<|language|>` tags inline in the output text.

## Install

Install the default package:

Download the standalone GGUF directly from
[FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp)
(such as `sensevoice-small-q8-audiocpp-v1.gguf`) and point `--model` at the file.

## Run (offline)

```bash
audiocpp_cli --task asr --family sense_asr \
  --model models/SenseVoice-Small-GGUF/sensevoice-small-q8-audiocpp-v1.gguf \
  --backend cpu --audio samples/zh.wav
```

`audio_chunk_mode=auto` (default) segments long audio with the bundled silero
VAD; `audio_chunk_mode=fixed` splits on `audio_chunk_duration_sec`; `none` runs the
whole clip as one encoder pass. Chunk transcripts are joined at ASCII word
boundaries so CJK output stays space-free.

## Run (streaming)

```bash
audiocpp_cli --task asr --family sense_asr \
  --model models/SenseVoice-Small-GGUF/sensevoice-small-q8-audiocpp-v1.gguf \
  --backend cpu --mode streaming --audio - \
  --request-option audio_chunk_duration_sec=5 --request-option audio_chunk_mode=none \
  < 16k_s16.pcm
```

Buffered streaming holds accumulated PCM and transcribes one fixed window per
`audio_chunk_duration_sec`; each window yields a `partial_text` delta and
`finalize()` returns the full accumulated transcript. The SenseVoice service
also runs through `audiocpp_server` with
`POST /v1/audio/transcriptions/live` for chunked PCM ingest with SSE deltas.

## Run as a server (offline & streaming)

Start the WebUI-enabled server with the SenseVoice model:

```bash
audiocpp_server --ui --backend cpu \
  --config <(echo '{"models":[{"id":"sense_asr","family":"sense_asr","path":"models/SenseVoice-Small-GGUF/sensevoice-small-q8-audiocpp-v1.gguf","task":"asr","mode":"streaming"}]}')
```

Then open http://127.0.0.1:8080 and select "SenseVoice-Small (asr, 流式, 社区)" from the ASR tab.

Or load the model dynamically via the WebUI's model manager (click "📥 加载模型" after selecting the model).

For headless API-only mode without WebUI:

```bash
audiocpp_server --backend cpu \
  --config <(echo '{"models":[{"id":"sense_asr","family":"sense_asr","path":"models/SenseVoice-Small-GGUF/sensevoice-small-q8-audiocpp-v1.gguf","task":"asr","mode":"streaming"}],"ui":false}')
```

Then use the OpenAI-compatible endpoint:

```bash
# Offline transcription
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F 'model=sense_asr' -F 'file=@assets/resources/3.wav'

# Streaming transcription (SSE) — requires chunked transfer encoding
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions/live \
  -H 'Accept: text/event-stream' \
  -H 'Transfer-Encoding: chunked' \
  -F 'model=sense_asr' -F 'file=@assets/resources/3.wav'
```

## Request options

| Option | Values | Default | Meaning |
|---|---|---|---|
| `language` | auto\|zh\|en\|yue\|ja\|ko\|pt\|ru\|... | `auto` | Recognition language; `auto` lets the model infer it |
| `enable_itn` | true\|false | `true` | Inverse text normalization via the withitn query token |
| `keep_tags` | true\|false | `false` | Keep `<|...|>` meta tags inline |
| `audio_chunk_mode` | auto\|fixed\|none | `auto` | VAD segmentation, fixed split, or one pass |
| `audio_chunk_duration_sec` | seconds | `30` | Max chunk/window duration |

Session options: `sense_asr.weight_type` (native\|f32\|f16\|bf16\|q8_0),
`sense_asr.encoder_graph_arena_mb`, and `sense_asr.vad_model_path` (defaults to
the bundled `assets/framework/models/silero_vad`).

## Source and conversion

The port is based on the SenseVoice llama.cpp runtime at
`/workspace/SenseVoice/runtime/llama.cpp/sensevoice-server/`
(`sensevoice-server.cpp`): 80-mel filterbank + LFR, the SAN-M encoder, CTC
collapse, and SentencePiece detok. The reference FSMN-VAD and HTTP server layer
are replaced by the framework session plus the bundled silero_vad. The GGUF is
produced by that runtime's `export_sensevoice_gguf.py` script and carries the
`sv.*` metadata keys and `cmvn.shift`/`cmvn.scale` tensors (the offline fbank
path applies no CMVN, matching the reference offline engine).

## Verification notes

A 6-second Chinese sample (`sample.wav`) transcribes to `我想问我在滨海新区有房。`
in the offline path, matching the reference `llama-funasr-sensevoice` output
exactly. Streaming with 3-second windows emits per-window partials
(`我想问。` then `我在滨海新区有房。`) and the joined final transcript with a
~1.75s TTFT on CPU.
