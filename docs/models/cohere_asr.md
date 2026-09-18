# Cohere Transcribe

[Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
is a multilingual speech recognition model released under Apache-2.0.

| Field | Value |
|---|---|
| Family | `cohere_asr` |
| Task | `asr` |
| Mode | `offline` |
| Output | Transcript text |
| Long-form audio | Quiet-energy chunks, up to 35 seconds each |
| Languages | English, French, German, Spanish, Italian, Portuguese, Dutch, Polish, Greek, Arabic, Japanese, Chinese, Vietnamese, Korean |

## Install

```bash
python3 tools/model_manager_v2.py install cohere_transcribe_bf16
```

For a smaller download, install `cohere_transcribe_q8_0` or
`cohere_transcribe_q4_0` instead and use the corresponding `-q8_0.gguf` or
`-q4_0.gguf` filename below. Each GGUF includes the configuration and tokenizer;
no conversion is needed.

## CLI

```bash
audiocpp_cli --task asr --family cohere_asr \
  --model models/Cohere-Transcribe-GGUF/cohere-transcribe-03-2026-bf16.gguf \
  --backend cuda --audio speech.wav \
  --request-option language=en --text-out transcript.txt --log
```

Long recordings are split automatically at quiet points.
Use `--backend cpu --threads 8` for CPU inference or `--backend vulkan` for Vulkan.

To transcribe French without punctuation and capitalization:

```bash
audiocpp_cli --task asr --family cohere_asr \
  --model models/Cohere-Transcribe-GGUF/cohere-transcribe-03-2026-bf16.gguf \
  --backend cuda --audio french.wav \
  --request-option language=fr --request-option pnc=false \
  --text-out transcript.txt --log
```

## Request Options (use with `--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `language` | `en` | `en`, `fr`, `de`, `es`, `it`, `pt`, `nl`, `pl`, `el`, `ar`, `ja`, `zh`, `vi`, or `ko`. |
| `pnc` | `true` | Request punctuation and capitalization. |
| `audio_chunk_mode` | `auto` | `auto` and `quiet_energy` use quiet boundaries; `fixed` uses equal-length chunks; `none` requires at most 35 seconds. |
| `audio_chunk_duration_sec` | `35` | Maximum audio chunk length, from 0.04 to 35 seconds. |
| `max_tokens` | `256` | Maximum generated tokens per chunk, up to 1014. Increase if generation reaches the limit before EOS. |

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `cohere_asr.weight_type` | `native` | Weight storage type override; `native` keeps the precision stored in the model file. |

Audio chunks must contain at least 20 ms.
Translation, diarization, timestamps, and live audio streaming are not supported.
