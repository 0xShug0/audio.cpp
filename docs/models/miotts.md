# MioTTS

MioTTS is a 1.7B voice-clone TTS path that uses MioCodec for acoustic decoding.
It requires a reference voice and a MioCodec model. Best-of-N candidate scoring
can optionally use Qwen3-ASR.

## Model

| Field | Value |
|---|---|
| Family | `miotts` |
| GGUF model | `models/MioTTS-1.7B-GGUF/miotts-1.7b-q8_0.gguf` |
| Required dependency | MioCodec through `--session-option miotts.codec_model_path=<dir>` |
| Task | `tts` |
| Modes | `offline` |
| Languages | Model auto-handles supported text languages; no explicit language selector is exposed |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

## Quick Start

```bash
audiocpp_cli --task tts --family miotts --model models/MioTTS-1.7B-GGUF/miotts-1.7b-q8_0.gguf --backend cuda --session-option miotts.codec_model_path=models/MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf --text "Hello from MioTTS." --voice-ref assets/resources/b.wav --out out.wav
```

With best-of-N scoring, also provide a Qwen3-ASR model:

```bash
audiocpp_cli --task tts --family miotts --model models/MioTTS-1.7B-GGUF/miotts-1.7b-q8_0.gguf --backend cuda --session-option miotts.codec_model_path=models/MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf --session-option miotts.best_of_n_asr_model_path=models/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf --request-option miotts.best_of_n_enabled=true --request-option miotts.best_of_n=2 --text "Hello from MioTTS." --voice-ref assets/resources/b.wav --out out.wav
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio. |
| `--text-chunk-size` | integer chars | `180` | Long-form chunk size. |
| `--max-tokens` | integer | `700` | Maximum generated LM tokens per chunk. |
| `--temperature` | float | `0.8` | LM sampling temperature. |
| `--top-k` | integer | `50` | LM top-k sampling limit. |
| `--top-p` | float | `1.0` | LM nucleus sampling limit. |
| `--repetition-penalty` | float | `1.0` | LM repetition penalty. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic LM sampling. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `max_tokens` | integer | model default | Generation control; defaults to the checkpoint config. |
| `top_k` | integer | model default | Generation control; defaults to the checkpoint config. |
| `top_p` | float | model default | Generation control; defaults to the checkpoint config. |
| `temperature` | float | model default | Generation control; defaults to the checkpoint config. |
| `repetition_penalty` | float | model default | Generation control; defaults to the checkpoint config. |
| `presence_penalty` | float | model default | Generation control; defaults to the checkpoint config. |
| `frequency_penalty` | float | model default | Generation control; defaults to the checkpoint config. |
| `do_sample` | bool | model default | Generation control; defaults to the checkpoint config. |
| `seed` | integer >= 0 | random | Sampling seed; omitted chooses a random seed. |
| `best_of_n` | integer | not set | Generate n candidates and select by ASR scoring. |
| `best_of_n_enabled` | bool | not set | Run best-of-N candidate selection. |
| `best_of_n_language` | `auto`, `en`, `ja` | not set | Language used to score candidates. |
| `text_chunk_size` | integer | shared setting | Framework long-form text chunk size; inherits the session setting. |

The legacy `miotts.best_of_n*` request names remain accepted. Session names
retain the `miotts.` prefix. Use only one name for each request control.

The legacy unprefixed session name `text_chunk_size` remains accepted for
`miotts.text_chunk_size`.

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `miotts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | not set | Weight storage type; defaults to native. |
| `miotts.codec_model_path` | directory | not set | MioCodec model used for acoustic decoding. |
| `miotts.best_of_n_asr_model_path` | directory | not set | Qwen3-ASR model used for best-of-N scoring. |
| `miotts.text_chunk_size` | integer | `180` | Default framework text chunk size. |
| `miotts.best_of_n_enabled` | bool | `false` | Enable candidate generation and ASR ranking. |
| `miotts.best_of_n_default` | integer | `1` | Default best-of-N candidate count. |
| `miotts.best_of_n_max` | integer | `8` | Maximum best-of-N candidate count. |
| `miotts.best_of_n_language` | `auto`, `en`, `ja` | `auto` | Default language used when scoring candidates. |
| `miotts.prefill_graph_arena_mb` | integer >= 1 | `1024` | Context or graph arena size in MiB. |
| `miotts.decode_graph_arena_mb` | integer >= 1 | `1024` | Context or graph arena size in MiB. |
| `miotts.weight_context_mb` | integer >= 1 | `512` | Context or graph arena size in MiB. |
| `miotts.codec_weight_context_mb` | integer >= 1 | `256` | Context or graph arena size in MiB. |
| `miotts.codec_constant_context_mb` | integer >= 1 | `256` | Context or graph arena size in MiB. |
| `miotts.global_graph_arena_mb` | integer >= 1 | `256` | Context or graph arena size in MiB. |
| `miotts.wave_graph_arena_mb` | integer >= 1 | `512` | Context or graph arena size in MiB. |
