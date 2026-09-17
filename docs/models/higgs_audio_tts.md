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
| Modes | `offline` |
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

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `higgs_audio_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared AR and codec weight storage type. |
| `higgs_audio_tts.ar_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Autoregressive decoder weight storage override. |
| `higgs_audio_tts.codec_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Audio codec weight storage override. |
| `higgs_audio_tts.ar_weight_context_mb` | integer MiB >= 1 | `4096` (`1024` on 32-bit builds) | AR weight context size. |
| `higgs_audio_tts.codec_weight_context_mb` | integer MiB >= 1 | `1536` | Codec weight context size. |
| `higgs_audio_tts.ar_decode_graph_arena_mb` | integer MiB >= 1 | `512` | AR decode graph arena size. |
| `higgs_audio_tts.codec_decode_graph_arena_mb` | integer MiB >= 1 | `128` | Codec decode graph arena size. |
| `higgs_audio_tts.codec_encode_graph_arena_mb` | integer MiB >= 1 | `256` | Reference-audio codec encode graph arena size. |
| `higgs_audio_tts.reference_cache_slots` | integer >= 0 | `1` | Encoded reference-audio cache slots; `0` disables reuse. |
| `higgs_audio_tts.attention` | `auto`, `flash`, `eager` | `auto` | Attention kernel. `auto` uses flash except on Volta/Turing GPUs (e.g. V100), where it falls back to eager to avoid missing MMA kernels. |
