# Higgs Audio v3 TTS

Higgs Audio v3 TTS is a voice-clone TTS model. The current integration uses
the framework chunker for long text and keeps the reference prompt state in the
model session.

## Install

The model manager installs the Q8_0 standalone GGUF package by default:

```bash
python3 tools/model_manager_v2.py install --models-root models higgs_audio_tts_4b_q8_0
```

## Quick Start

```bash
audiocpp_cli --task tts --family higgs_audio_tts --model models/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf --backend cuda --text "Hello from Higgs Audio." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

## Model

| Field | Value |
|---|---|
| Family | `higgs_audio_tts` |
| Model path | `models/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf` when installed through the model manager |
| Task | `tts` |
| Modes | `offline`, `streaming` |
| Languages | Model auto-handles supported languages |
| Voice input | Reference WAV through `--voice-ref`; transcript through `--reference-text` when known |
| Built-in voices | Not exposed |

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio. |
| `--reference-text` | text | empty string | Transcript for reference audio. |
| `--text-chunk-size` | integer chars | `1024` | Long-form chunk size. |
| `--max-tokens` | integer | `2048` | Maximum generated AR tokens per chunk. |
| `--temperature` | float | `0.8` | AR sampling temperature. |
| `--top-k` | integer | `30` | AR top-k sampling limit. The narrower default is less prone to premature EOC than the Python client's `50`. |
| `--top-p` | float | `0.8` | AR nucleus sampling limit. The Python client's unfiltered equivalent is `1.0`. |
| `--repetition-penalty` | float | `1.1` | Accepted for Python API compatibility; Higgs audio-code sampling does not consume it. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `max_tokens` | integer >= 0 | `2048` | Maximum generated AR tokens; zero uses the default. |
| `temperature` | float | `0.8` | AR sampling temperature. |
| `top_p` | float | `0.8` | AR nucleus sampling probability. |
| `top_k` | integer | `30` | AR top-k sampling limit. |
| `repetition_penalty` | float | `1.1` | Accepted for Python API compatibility; audio sampling does not consume this value. |
| `seed` | integer >= 0 | random | Torch sampling seed; omitted chooses a random seed. |
| `reference_text` | text | empty string | Transcript accompanying reference audio. |
| `text_chunk_size` | integer | `1024` | Framework long-form text chunk size. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunking mode. |
| `stream_frames_per_event` | integer >= 1 | `32` | Streaming audio frame cap; each codec frame contains 40 ms of audio. |
| `stream_chunk_policy` | `grow`, `fixed` | `grow` | Start with one frame and grow to the cap, or emit fixed-size chunks. Streaming only. |

## Streaming

Set the session mode to `streaming`. Audio is decoded and emitted during AR
generation, once all eight delayed codebooks for each frame are available.
The default chunk schedule is 1, 2, 4, 8, 16, then 32 frames per event;
the final event may be shorter. The non-causal codec needs ten frames of
future context, so a one-frame first event still waits for that context.
This is not a one-AR-step latency guarantee.

Streaming keeps bounded codec windows and reuses their graphs within the
session. Text chunking and reference conditioning remain the same as offline.
The final WAV contains exactly the concatenated audio deltas. Floating-point
differences from offline decoding are possible because codec graph shapes differ.

For the server, configure `"mode": "streaming"` on the model entry and request
`"stream_format": "sse"` from `/v1/audio/speech`. Streaming controls belong in
the request's `options` object. They are not applied to offline generation.

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `higgs_audio_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared AR and codec weight storage type. |
| `higgs_audio_tts.ar_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | shared setting | Autoregressive decoder weight storage override. |
| `higgs_audio_tts.codec_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | shared setting | Audio codec weight storage override. |
| `higgs_audio_tts.ar_weight_context_mb` | integer MiB >= 1 | not set | AR weight context size. |
| `higgs_audio_tts.codec_weight_context_mb` | integer MiB >= 1 | `1536` | Codec weight context size. |
| `higgs_audio_tts.ar_decode_graph_arena_mb` | integer MiB >= 1 | `512` | AR decode graph arena size. |
| `higgs_audio_tts.codec_decode_graph_arena_mb` | integer MiB >= 1 | `128` | Codec decode graph arena size. |
| `higgs_audio_tts.codec_encode_graph_arena_mb` | integer MiB >= 1 | `256` | Reference-audio codec encode graph arena size. |
| `higgs_audio_tts.reference_cache_slots` | integer >= 0 | `1` | Encoded reference-audio cache slots; `0` disables reuse. |
| `higgs_audio_tts.attention` | `auto`, `flash`, `eager` | `auto` | Attention kernel. `auto` uses flash except on Volta/Turing GPUs (e.g. V100), where it falls back to eager to avoid missing MMA kernels. |
