# BreezeTTS 2

BreezeTTS 2 is a GGUF TTS family for instruction-conditioned speech and
prompt-audio voice cloning. The default package is Q8_0.

## Quick Start

Download the default Q8_0 package:

```bash
python3 tools/model_manager_v2.py install breeze_tts_2_q8_0 --models-root models
```

Voice cloning:

```bash
audiocpp_cli \
  --task clon \
  --family breeze_tts \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --backend cuda \
  --text "Please read this line in a clear and natural voice." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years." \
  --request-option instruction="Speak clearly and naturally." \
  --out breeze_tts_clone.wav
```

Voice design:

```bash
audiocpp_cli \
  --task tts \
  --family breeze_tts \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --backend cuda \
  --text "Welcome to the local voice demo." \
  --request-option instruction="A warm female narrator with calm pacing and studio clarity." \
  --out breeze_tts_design.wav
```

## Model

| Field | Value |
|---|---|
| Family | `breeze_tts` |
| Default GGUF | `models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf` |
| Tasks | `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en` |
| Voice input | Optional for `tts`; required for `clon` |

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required for `clon` | Prompt/reference speaker audio. |
| `--reference-text` / `--request-option reference_text=<text>` | text | empty | Transcript for prompt audio when cloning. |
| `--request-option instruction=<text>` | text | `Speak clearly and naturally.` | Voice or style instruction. |
| `--request-option text_chunk_size=<n>` | integer > 0 | `600` | Long-form text chunk size. |
| `--request-option text_chunk_mode=<mode>` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunk mode. |
| `--request-option max_tokens=<n>` | integer > 0 | `1500` | Maximum generated acoustic frames. |
| `--request-option guidance_scale=<f>` | float >= 0 | `1.0` | Classifier-free guidance scale. |
| `--request-option temperature=<f>` | float >= 0 | `0.9` | Backbone sampling temperature. |
| `--request-option depth_temperature=<f>` | float >= 0 | `0.9` | Depth decoder sampling temperature. |
| `--request-option top_k=<n>` | integer >= 0 | `50` | Top-k sampling limit; `0` disables top-k filtering. |
| `--request-option top_p=<f>` | `0..1` | `1.0` | Top-p sampling limit. |
| `--request-option seed=<n>` | integer >= 0 | `0` | Generation seed. |
| `--session-option breeze_tts.reference_cache_slots=<n>` | integer >= 0 | `1` | Prepared reference-audio cache slots. |
