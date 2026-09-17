# FireRedAudio

FireRedAudio is a multimodal speech and audio model. The native audio.cpp path
covers Chinese ASR, audio understanding, TTS cloning, voice design, semantic
editing, and acoustic editing from a single GGUF package.

## Quick Start

Chinese ASR:

```bash
audiocpp_cli \
  --task asr \
  --family firered_audio \
  --model models/FireRedAudio-GGUF/firered-audio-orig.gguf \
  --backend cuda \
  --audio input.wav \
  --text "Transcribe speech to text." \
  --request-option template_name=asr \
  --text-out transcript.txt
```

Voice cloning:

```bash
audiocpp_cli \
  --task clon \
  --family firered_audio \
  --model models/FireRedAudio-GGUF/firered-audio-orig.gguf \
  --backend cuda \
  --language zh \
  --text "安徽淮南秦师傅发现，停在小区的爱车右前驾驶窗玻璃被砸。" \
  --voice-ref reference/FireRedAudio/assets/examples/tts_zh_prompt.wav \
  --reference-text "同时，他强调微调要科学有序。" \
  --request-option template_name=tts_clone \
  --out firered_audio_clone.wav
```

Audio understanding:

```bash
audiocpp_cli \
  --task asr \
  --family firered_audio \
  --model models/FireRedAudio-GGUF/firered-audio-orig.gguf \
  --backend cuda \
  --audio input.wav \
  --text "What is happening in this audio?" \
  --request-option template_name=understand \
  --request-option enable_thinking=true \
  --request-option max_new_tokens=1024 \
  --text-out answer.txt
```

## Model

| Field | Value |
|---|---|
| Family | `firered_audio` |
| Tasks | `asr`, `tts`, `clon`, `vdes`, `gen` |
| Mode | `offline` |
| Default package | `models/FireRedAudio-GGUF/firered-audio-orig.gguf` |
| Generation paths | `tts_clone`, `voice_design`, `semantic_edit`, `acoustic_edit` |
| Text paths | `asr`, `understand` |

## Templates

| `template_name` | Task | Inputs |
|---|---|---|
| `asr` | `asr` | `--audio`, optional prompt through `--text` |
| `understand` | `asr` | `--audio`, question through `--text` |
| `tts_clone` | `clon` | `--text`, `--voice-ref`, `--reference-text` |
| `voice_design` | `vdes` | `--text`, `--request-option instruction=<text>` |
| `semantic_edit` | `gen` | `--audio`, `--request-option instruction=<text>` |
| `acoustic_edit` | `gen` | `--audio`, `--request-option instruction=<text>` |

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--language` | language tag | `zh` | Generation or text-path language tag. |
| `--voice-ref` | WAV path | required for clone | Prompt/reference voice. |
| `--reference-text` | text | empty | Transcript for the prompt audio. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `template_name` | `asr`, `understand`, `tts_clone`, `voice_design`, `semantic_edit`, `acoustic_edit` | path-dependent | Request template. |
| `language` | language tag | `zh` | Generation or text-path language tag. |
| `reference_text` | text | empty | Transcript for the prompt audio. |
| `instruction` | text | required for design/edit | Voice design or edit instruction. |
| `num_inference_steps` | integer > 0 | `10` | FireRedAudio DiT flow steps per latent patch. |
| `guidance_scale` | float >= 0 | `2.0` | FireRedAudio DiT CFG scale. |
| `max_new_audio_steps` | integer > 0 | `750` | Maximum generated RedAE latent patches. |
| `min_new_audio_steps` | integer >= 0 | `6` | Minimum audio patches before stop may be accepted. |
| `max_new_text_tokens` | integer > 0 | `512` | Text-mode token budget before audio mode. |
| `max_new_tokens` | integer > 0 | `300` | ASR/understanding text token budget. |
| `enable_thinking` | bool | `false` | Enable open thinking block for `understand`; invalid for plain `asr`. |
| `top_k` | integer >= 0 | `20` | Understanding top-k sampling. |
| `top_p` | `0..1` | `0.8` | Understanding nucleus sampling. |
| `temperature` | float >= 0 | `0.7` | Understanding sampling temperature. |
| `seed` | integer >= 0 | `1234` | Generation seed. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `firered_audio.reference_cache_slots` | integer >= 0 | `2` | Prepared reference-audio cache slots. |
| `firered_audio.mem_saver` | bool | `false` | Release runtime graphs after request phases. |
| `firered_audio.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Weight storage override for experiments. |
