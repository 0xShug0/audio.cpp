# ControlFoley

ControlFoley is a multimodal Foley generator. It can generate sound from text
alone, from video alone, from text plus video, or from reference audio plus
video.

## Quick Start

Text-to-audio:

```bash
audiocpp_cli \
  --task gen \
  --family controlfoley \
  --model models/ControlFoley-GGUF/controlfoley-large-44k-f32.gguf \
  --backend cuda \
  --text "A bird sings melodically in a forest." \
  --request-option duration_sec=10 \
  --request-option num_inference_steps=25 \
  --request-option guidance_scale=4.5 \
  --request-option seed=123 \
  --out foley.wav
```

Video-conditioned generation:

```bash
audiocpp_cli \
  --task gen \
  --family controlfoley \
  --model models/ControlFoley-GGUF/controlfoley-large-44k-f32.gguf \
  --backend cuda \
  --text "skateboarding" \
  --request-option video=/path/to/video.mp4 \
  --request-option duration_sec=8 \
  --request-option num_inference_steps=25 \
  --request-option guidance_scale=4.5 \
  --out foley_video.wav
```

## Model

| Field | Value |
|---|---|
| Family | `controlfoley` |
| Task | `gen` |
| Mode | `offline` |
| Packages | `controlfoley-large-44k-f32.gguf`, `controlfoley-large-44k-q8_0.gguf` |
| Text input | Optional prompt through `--text` |
| Video input | Optional video path through `--request-option video=<path>` |
| Audio input | Optional reference WAV through `--audio` |

## Paths

| Path | Required inputs |
|---|---|
| T2A | `--text` |
| TV2A | `--text`, `--request-option video=<path>` |
| TC-V2A | `--text`, `--request-option video=<path>`, `--request-option mask_away_clip=true` |
| AC-V2A | `--audio`, `--request-option video=<path>` |
| V2A | `--request-option video=<path>` |

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | prompt text | optional | Text conditioning prompt. |
| `--audio` | WAV path | optional | Reference audio for audio-conditioned video generation. |
| `--request-option video=<path>` | video path | optional | Video conditioning source. |
| `--request-option duration_sec=<seconds>` | float | `8` | Target temporal budget. |
| `--request-option num_inference_steps=<n>` | integer > 0 | `25` | Euler flow inference steps. |
| `--request-option guidance_scale=<f>` | float | `4.5` | Classifier-free guidance scale. |
| `--request-option negative_prompt=<text>` | text | empty | Negative text branch for CFG. |
| `--request-option mask_away_clip=true\|false` | bool | `false` | Disable OpenCLIP video conditioning while keeping visual and sync conditioning. |
| `--request-option seed=<n>` | integer >= 0 | `42` | Sampler seed. |
| `--session-option controlfoley.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Weight storage override for experiments. |
