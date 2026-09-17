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
| `miotts.best_of_n_enabled` | bool | `false` | Run best-of-N candidate selection. |
| `miotts.best_of_n` | integer | session default | Generate n candidates and select by ASR scoring. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `miotts.codec_model_path` | directory | sibling MioCodec directory | MioCodec model used for acoustic decoding. |
| `miotts.best_of_n_default` | integer | `1` | Default best-of-N candidate count. |
| `miotts.best_of_n_max` | integer | `8` | Maximum best-of-N candidate count. |
| `miotts.best_of_n_language` | `auto`, `en`, `ja` | `auto` | Default language used when scoring candidates. |
| `miotts.best_of_n_asr_model_path` | directory | sibling Qwen3-ASR directory | Qwen3-ASR model used for best-of-N scoring. |
