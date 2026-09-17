# Canary 180M Flash

[NVIDIA Canary 180M Flash](https://huggingface.co/nvidia/canary-180m-flash)
recognizes English, German, Spanish, and French speech. Source and target languages
can be selected independently. NVIDIA documents translation
quality for English-centric pairs. The weights use the CC-BY-4.0 license.

| Field | Value |
|---|---|
| Family | `canary_asr` |
| Task | `asr` |
| Mode | `offline` |
| Output | Transcript text |
| Long-form audio | Fixed chunks, up to 40 seconds each |
| Timestamps | Not exposed |

## Install

```bash
python3 tools/model_manager_v2.py install canary_180m_flash_f32
```

For a smaller download, install `canary_180m_flash_q8_0` instead and use
`canary-180m-flash-q8_0.gguf` in the commands below. Each GGUF includes the
configuration and tokenizers; no conversion is needed.

## CLI

```bash
audiocpp_cli --task asr --family canary_asr \
  --model models/Canary-180M-Flash-GGUF/canary-180m-flash-f32.gguf \
  --backend cuda --audio speech.wav \
  --text-out transcript.txt --log
```

Long recordings are chunked automatically. Inputs must contain at least 20 ms of
audio. To translate German speech to English:

```bash
audiocpp_cli --task asr --family canary_asr \
  --model models/Canary-180M-Flash-GGUF/canary-180m-flash-f32.gguf \
  --backend cuda --audio speech.wav \
  --request-option language=de --request-option target_language=en \
  --text-out translation.txt --log
```

## Request Options (use with `--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `language` | `en` | Input language: `en`, `de`, `es`, or `fr`. |
| `target_language` | Input language | Output language: `en`, `de`, `es`, or `fr`; identical source and target select transcription. |
| `pnc` | `true` | Punctuation and capitalization. |
| `audio_chunk_mode` | `auto` | `auto` or `fixed` splits long audio; `none` requires at most 40 seconds. |
| `audio_chunk_duration_sec` | `40` | Maximum chunk length, at most 40 seconds. |
| `max_tokens` | `0` | Generated tokens per chunk; zero derives a limit from encoder length. |

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `canary_asr.weight_type` | `native` | Weight storage type override; `native` keeps the precision stored in the model file. |

Use `--backend cpu --threads 8` for CPU inference or `--backend vulkan` for Vulkan.
