# CosyVoice3

CosyVoice3 is a GGUF TTS family for zero-shot voice cloning, cross-lingual
speech, and instruction-conditioned speech. The default package is Q8_0.

## Quick Start

Download the default Q8_0 package:

```bash
python3 tools/model_manager_v2.py install cosyvoice3_q8_0 --models-root models
```

Zero-shot voice cloning:

```bash
audiocpp_cli \
  --task clon \
  --family cosyvoice3 \
  --model models/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf \
  --backend cuda \
  --text "This is a local CosyVoice3 voice cloning test." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years." \
  --request-option template_name=zero_shot \
  --out cosyvoice3_clone.wav
```

Instruction-conditioned speech:

```bash
audiocpp_cli \
  --task tts \
  --family cosyvoice3 \
  --model models/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf \
  --backend cuda \
  --text "Please read this with a calm and friendly tone." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years." \
  --request-option template_name=instruct \
  --request-option instruction="Speak warmly with clear articulation." \
  --out cosyvoice3_instruct.wav
```

## Model

| Field | Value |
|---|---|
| Family | `cosyvoice3` |
| Default GGUF | `models/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf` |
| Tasks | `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en`, `ja`, `ko`, `de`, `es`, `fr`, `it`, `ru`, `yue` |
| Voice input | Required reference WAV through `--voice-ref` |

## Templates

| `template_name` | Use |
|---|---|
| `zero_shot` | Voice cloning with prompt audio and transcript. |
| `cross_lingual` | Cross-lingual voice cloning from prompt audio. |
| `instruct` | Instruction-conditioned speech with prompt audio. |

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Prompt/reference speaker audio. |
| `--reference-text` / `--request-option reference_text=<text>` | text | empty | Transcript for prompt audio. |
| `--request-option template_name=<name>` | `zero_shot`, `cross_lingual`, `instruct` | `zero_shot` | Request template. |
| `--request-option instruction=<text>` | text | empty | Instruction text for `instruct` mode. |
| `--request-option text_chunk_size=<n>` | integer > 0 | `600` | Long-form text chunk size. |
| `--request-option text_chunk_mode=<mode>` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunk mode. |
| `--request-option max_tokens=<n>` | integer > 0 | `1600` | Maximum generated speech tokens. |
| `--request-option min_tokens=<n>` | integer >= 0 | `0` | Minimum generated tokens before stop is accepted. |
| `--request-option top_k=<n>` | integer > 0 | `25` | AR speech-token top-k sampling limit. |
| `--request-option num_inference_steps=<n>` | integer > 0 | `10` | Flow decoder Euler steps. |
| `--request-option seed=<n>` | integer >= 0 | `1986` | Generation seed. |
| `--session-option cosyvoice3.reference_cache_slots=<n>` | integer >= 0 | `4` | Prepared reference-audio cache slots. |
| `--session-option cosyvoice3.mem_saver=true\|false` | bool | `false` | Release cached runtime graphs after request phases. |
