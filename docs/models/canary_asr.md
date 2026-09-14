# Canary 180M Flash

[NVIDIA Canary 180M Flash](https://huggingface.co/nvidia/canary-180m-flash)
recognizes English, German, Spanish, and French speech. Source and target languages
can be selected independently, matching the Python API. NVIDIA documents translation
quality for English-centric pairs. The weights use the CC-BY-4.0 license.

| Field | Value |
|---|---|
| Family | `canary_asr` |
| Task | `asr` |
| Mode | `offline` |
| Output | Transcript text |
| Long-form audio | Fixed chunks, up to 40 seconds each |
| Timestamps | Not exposed |

## Convert

Download the original checkpoint and convert it to safetensors:

```bash
hf download nvidia/canary-180m-flash --local-dir models/canary-180m-flash
python tests/canary_asr/convert_canary_to_safetensors.py \
  models/canary-180m-flash/canary-180m-flash.nemo \
  models/canary-180m-flash-safetensors
```

Create a standalone GGUF containing the configuration and all five tokenizers:

```bash
build/debug/bin/audiocpp_gguf \
  --input models/canary-180m-flash-safetensors/model.safetensors \
  --root models/canary-180m-flash-safetensors \
  --family canary_asr --model-spec model_specs/canary_asr.json \
  --type orig --output models/canary-180m-flash-f32.gguf
```

## CLI

```bash
audiocpp_cli --task asr --family canary_asr \
  --model models/canary-180m-flash-f32.gguf \
  --backend cuda --audio speech.wav \
  --text-out transcript.txt --log
```

Long recordings are chunked automatically. Inputs must contain at least 20 ms of
audio. A final fragment shorter than 20 ms borrows samples from the preceding chunk
without dropping audio. To translate German speech to English:

```bash
audiocpp_cli --task asr --family canary_asr \
  --model models/canary-180m-flash-f32.gguf \
  --backend cuda --audio speech.wav \
  --request-option language=de --request-option target_language=en \
  --text-out translation.txt --log
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `language` | `en` | Input language: `en`, `de`, `es`, or `fr`. |
| `target_language` | Input language | Output language: `en`, `de`, `es`, or `fr`; identical source and target select transcription. |
| `pnc` | `true` | Punctuation and capitalization. |
| `audio_chunk_mode` | `auto` | `auto` or `fixed` splits long audio; `none` requires at most 40 seconds. |
| `audio_chunk_duration_sec` | `40` | Maximum chunk length, at most 40 seconds. |
| `max_tokens` | `0` | Generated tokens per chunk; zero derives a limit from encoder length. |
| `canary_asr.weight_type` (session) | `native` | Matmul storage: `native`, `f32`, `f16`, `bf16`, or `q8_0`. |

Request options use `--request-option name=value`; session options use
`--session-option name=value`. The decoder uses a managed FP16 KV cache.
