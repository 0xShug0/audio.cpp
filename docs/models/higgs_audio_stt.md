# Higgs Audio STT

Higgs Audio STT is an ASR model for Higgs Audio v3 STT assets. Offline mode
can split long audio before inference. Streaming mode consumes audio chunks and
emits partial text for each processed chunk.

## Model

| Field | Value |
|---|---|
| Family | `higgs_audio_stt` |
| Model directory | `models/higgs-audio-v3-stt` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Output | Transcription text |
| Streaming input | Audio chunks; preferred chunk duration is 4 seconds |
| Timestamps | Not exposed |

## Quick Start

Offline:

```bash
audiocpp_cli --task asr --family higgs_audio_stt --model models/higgs-audio-v3-stt --backend cuda --audio speech_16k.wav --text "Transcribe the speech." --text-out transcript.txt
```

Streaming:

```bash
audiocpp_cli --task asr --family higgs_audio_stt --model models/higgs-audio-v3-stt --backend cuda --mode streaming --audio speech_16k.wav --text "Transcribe the speech." --text-out transcript.txt
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. |
| `--text` | text | empty string | Prompt/context text for the ASR request. |
| `--language` | language code | model default (`en`) | Recognition language hint. |
| `--mode` | `offline`, `streaming` | `offline` | Full-context or streaming session. |
| `--max-tokens` | integer | model default | Maximum generated transcript tokens. |
| `--audio-chunk-mode` | `auto`, `fixed`, `none` | `auto` | Long-audio chunking mode. `auto` uses fixed chunks. |
| `--audio-chunk-seconds` | float seconds | `4` | Fixed audio chunk duration. Equivalent to request option `audio_chunk_duration_sec`. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `enable_thinking` | bool | `true` | Enable the model thinking prompt. |
| `audio_chunk_duration_sec` | float seconds | `4` | Fixed audio chunk duration. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `higgs_audio_stt.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared text decoder weight storage type. |
| `higgs_audio_stt.audio_encoder_weight_type` | `native`, `f32`, `f16` | `native` | Audio encoder convolution weight storage type. |
| `higgs_audio_stt.text_decoder_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `higgs_audio_stt.weight_type` or `native` | Text decoder matmul weight storage type. |
| `higgs_audio_stt.audio_encoder_graph_arena_mb` | integer MiB >= 0 | `512` | Audio encoder graph arena size. |
| `higgs_audio_stt.text_decoder_prefill_graph_arena_mb` | integer MiB >= 0 | `512` | Text decoder prefill graph arena size. |
| `higgs_audio_stt.text_decoder_decode_graph_arena_mb` | integer MiB >= 0 | `256` | Text decoder cached-step graph arena size. |
| `higgs_audio_stt.text_decoder_weight_context_mb` | integer MiB >= 0 | `4096` | Text decoder weight context arena size. |

## Compatibility

Compatibility aliases are applied before v1 option validation:

| Legacy request option | v1 request option |
|---|---|
| `audio_chunk_seconds` | `audio_chunk_duration_sec` |
| `audio_chunk_duration_seconds` | `audio_chunk_duration_sec` |
| `audio_chunk_duration` | `audio_chunk_duration_sec` |

| Legacy session option | v1 session option |
|---|---|
| `weight_type` | `higgs_audio_stt.weight_type` |
| `audio_encoder_weight_type` | `higgs_audio_stt.audio_encoder_weight_type` |
| `text_decoder_weight_type` | `higgs_audio_stt.text_decoder_weight_type` |
| `audio_encoder_graph_arena_mb` | `higgs_audio_stt.audio_encoder_graph_arena_mb` |
| `text_decoder_prefill_graph_arena_mb` | `higgs_audio_stt.text_decoder_prefill_graph_arena_mb` |
| `text_decoder_decode_graph_arena_mb` | `higgs_audio_stt.text_decoder_decode_graph_arena_mb` |
| `text_decoder_weight_context_mb` | `higgs_audio_stt.text_decoder_weight_context_mb` |

## GGUF Conversion

Standalone Q8_0 GGUF conversion uses the two-shard index. Map the shared
Whisper preprocessor configuration into the GGUF so the original directory
layout is not required:

```powershell
audiocpp_gguf.exe --input models\higgs-audio-v3-stt\model.safetensors.index.json --root models\higgs-audio-v3-stt --sidecar models\whisper-large-v3\preprocessor_config.json=preprocessor_config.json --output models\higgs-audio-v3-stt-Q8_0\model.gguf --type q8_0
```

The shared `whisper-large-v3/preprocessor_config.json` is required only while
creating the GGUF. Once embedded, the resulting GGUF can be moved, renamed, and
passed directly to `--model`; the external Whisper file and directory are no
longer required at runtime.
