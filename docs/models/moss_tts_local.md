# MOSS-TTS-Local

MOSS-TTS-Local is the larger local-transformer path.

| Field | Value |
|---|---|
| Family | `moss_tts_local` |
| Model directory | `models/MOSS-TTS-Local-Transformer-v1.5` |
| Required codec layout | `audio_tokenizer/` directory inside the model root |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | Model auto-handles supported languages; `--language` can pass a language hint |
| Voice input | Optional reference WAV through `--voice-ref`; Local does not consume a reference transcript |
| Built-in voices | Not exposed |

Text-only speech:

```bash
audiocpp_cli --task tts --family moss_tts_local --model /path/to/MOSS-TTS-Local-Transformer-v1.5 --backend cuda --text "Hello from MOSS-TTS-Local." --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task clon --family moss_tts_local --model /path/to/MOSS-TTS-Local-Transformer-v1.5 --backend cuda --text "Hello from MOSS-TTS-Local." --voice-ref /path/to/reference.wav --out out.wav
```

## Common Options (use directly)

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

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `max_tokens` | integer >= 1 | `4096` | Maximum generated audio frames. |
| `do_sample` | bool | `true` | Enable sampling. |
| `temperature` | float | `1.7` | Audio sampling temperature. |
| `top_p` | float | `0.8` | Audio nucleus sampling probability. |
| `top_k` | integer | `25` | Audio top-k sampling limit. |
| `repetition_penalty` | float | `1` | Audio repetition penalty. |
| `text_temperature` | float | `1` | Text-gate sampling temperature. |
| `text_top_p` | float | `1` | Text-gate nucleus sampling limit. |
| `text_top_k` | integer | `50` | Text-gate top-k sampling limit. |
| `seed` | integer >= 0 | random | Sampling seed; omitted chooses a random seed. |
| `text_chunk_size` | integer | `2048` | Framework long-form text chunk size. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunking mode. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `moss_tts_local.weight_type` | `auto`, `native`, `f32`, `f16`, `bf16`, `q8_0` | `auto` | Backbone weight storage type. |
| `moss_tts_local.reference_cache_slots` | integer slots | `1` | Prepared reference-voice cache slots; set `0` to disable reuse. |

