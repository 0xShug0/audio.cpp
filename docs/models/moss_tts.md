# MOSS-TTS

MOSS-TTS in audio.cpp includes the larger Local Transformer model and the smaller Nano model. Both use a MOSS audio tokenizer under the model root and support text-only speech plus optional zero-shot voice cloning through the framework speaker-reference interface.

## MOSS-TTS-Local

MOSS-TTS-Local is the larger local-transformer path.

| Field | Value |
|---|---|
| Family | `moss_tts_local` |
| Model directory | `models/MOSS-TTS-Local-Transformer-v1.5` |
| Required codec layout | `audio_tokenizer/` directory inside the model root |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | Model auto-handles supported languages; `--language` can pass a language hint |
| Voice input | Optional reference WAV through `--voice-ref`; transcript through `--reference-text` when known |
| Built-in voices | Not exposed |

Text-only speech:

```bash
audiocpp_cli --task tts --family moss_tts_local --model /path/to/MOSS-TTS-Local-Transformer-v1.5 --backend cuda --text "Hello from MOSS-TTS-Local." --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task clon --family moss_tts_local --model /path/to/MOSS-TTS-Local-Transformer-v1.5 --backend cuda --text "Hello from MOSS-TTS-Local." --voice-ref /path/to/reference.wav --reference-text "Reference transcript when available." --out out.wav
```

### MOSS-TTS-Local Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for cloning. |
| `--reference-text` | text | empty string | Transcript for reference audio. |
| `--language` | language hint | auto | Optional language hint for the prompt template. |
| `--max-tokens` | integer | `4096` | Maximum generated audio frames. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic audio-token sampling. |
| `--temperature` | float | `1.7` | Audio-token sampling temperature. |
| `--top-p` | float | `0.8` | Audio-token nucleus sampling limit. |
| `--top-k` | integer | `25` | Audio-token top-k sampling limit. |
| `--repetition-penalty` | float | `1.0` | Audio-token repetition penalty. |
| `--text-chunk-size` | characters | `2048` | Framework long-form text chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |

### MOSS-TTS-Local Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `text_temperature` | float | `1.0` | Text-gate sampling temperature. |
| `text_top_p` | float | `1.0` | Text-gate nucleus sampling limit. |
| `text_top_k` | integer | `50` | Text-gate top-k sampling limit. |

### MOSS-TTS-Local Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `moss_tts_local.weight_type` | `auto`, `native`, `f32`, `f16`, `bf16`, `q8_0` | `auto` | Backbone weight storage type. |
| `moss_tts_local.reference_cache_slots` | integer slots | `1` | Prepared reference-voice cache slots; set `0` to disable reuse. |

## MOSS-TTS-Nano

MOSS-TTS-Nano is the smaller MOSS TTS path. It supports text-only continuation generation and voice cloning through the framework speaker-reference interface.

| Field | Value |
|---|---|
| Family | `moss_tts_nano` |
| Model directory | `models/MOSS-TTS-Nano-100M` |
| Required codec layout | `audio_tokenizer/` directory inside the model root |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | Model auto-handles supported languages |
| Voice input | Optional reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

Text-only continuation:

```bash
audiocpp_cli --task tts --family moss_tts_nano --model /path/to/MOSS-TTS-Nano-100M --backend cuda --text "Hello from MOSS-TTS-Nano." --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task clon --family moss_tts_nano --model /path/to/MOSS-TTS-Nano-100M --backend cuda --text "Hello from MOSS-TTS-Nano." --voice-ref /path/to/reference.wav --reference-text "Reference transcript when available." --out out.wav
```

### MOSS-TTS-Nano Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for cloning. When omitted, Nano uses text-only continuation mode. |
| `--reference-text` | text | empty string | Transcript for reference audio; valid only with `--voice-ref`. |
| `--max-tokens` | integer | `300` | Maximum generated audio frames per chunk. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic audio-token sampling. |
| `--temperature` | float | `1.7` | Audio-token sampling temperature. |
| `--top-p` | float | `0.8` | Audio-token nucleus sampling limit. |
| `--top-k` | integer | `25` | Audio-token top-k sampling limit. |
| `--repetition-penalty` | float | `1.0` | Audio-token repetition penalty. |
| `--text-chunk-size` | characters | `256` | Framework long-form text chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |

### MOSS-TTS-Nano Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `text_temperature` | float | `1.5` | Text-gate sampling temperature. |
| `text_top_p` | float | `1.0` | Text-gate nucleus sampling limit. |
| `text_top_k` | integer | `50` | Text-gate top-k sampling limit. |

### MOSS-TTS-Nano Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `moss_tts_nano.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Global and local-frame weight storage type. |
| `moss_tts_nano.global_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Global transformer weight storage type. |
| `moss_tts_nano.local_frame_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Local frame decoder weight storage type. |
| `moss_tts_nano.global_prefill_graph_arena_mb` | MB | `256` | Global prefill graph arena size. |
| `moss_tts_nano.global_decode_graph_arena_mb` | MB | `128` | Global decode graph arena size. |
| `moss_tts_nano.global_weight_context_mb` | MB | `512` | Global transformer weight context size. |
| `moss_tts_nano.local_frame_graph_arena_mb` | MB | `64` | Local frame decoder graph arena size. |
| `moss_tts_nano.local_frame_weight_context_mb` | MB | `128` | Local frame decoder weight context size. |
| `moss_tts_nano.audio_tokenizer_encoder_graph_arena_mb` | MB | `64` | Audio tokenizer encoder graph arena size. |
| `moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb` | MB | `64` | Audio tokenizer decoder graph arena size. |
| `moss_tts_nano.audio_tokenizer_weight_context_mb` | MB | `128` | Audio tokenizer weight context size. |
