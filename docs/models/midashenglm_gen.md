# MiDashengLM-Gen

MiDashengLM-Gen generates mixed audio from structured text prompts. Prompts can
describe speech, music, sound effects, and environment layers using the upstream
caption tags.

## Quick Start

```bash
audiocpp_cli \
  --task gen \
  --family midashenglm_gen \
  --model models/MiDashengLM-Gen-GGUF/midashenglm-gen-f32.gguf \
  --backend cuda \
  --text "<|caption|> A calm narrator speaks over soft rain and a quiet room tone. <|asr|> The lights are still on, and the rain is getting softer now. <|speech|> calm warm male narrator, close microphone, gentle emotion <|music|> sparse quiet piano chords <|sfx|> soft rain against a window <|env|> small quiet room at night" \
  --request-option duration_sec=10 \
  --request-option guidance_scale=2.0 \
  --request-option stop_threshold=0.5 \
  --request-option min_stop_step=5 \
  --request-option seed=20260817 \
  --out midasheng.wav
```

## Model

| Field | Value |
|---|---|
| Family | `midashenglm_gen` |
| Task | `gen` |
| Mode | `offline` |
| Packages | `midashenglm-gen-f32.gguf`, `midashenglm-gen-q8_0.gguf` |
| Input | Structured prompt through `--text` |
| Output | Generated waveform |

Useful prompt fields include `<|caption|>`, `<|asr|>`, `<|speech|>`,
`<|music|>`, `<|sfx|>`, and `<|env|>`. Use `<|unknown|>` for an empty branch
when you want the model to keep the tag structure but omit that content.

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | structured prompt | required | Prompt with MiDashengLM tags. |
| `--request-option duration_sec=<seconds>` | float | model budget | Target audio duration budget. |
| `--request-option guidance_scale=<f>` | float | `2.0` | Classifier-free guidance scale; `1.0` disables CFG. |
| `--request-option stop_threshold=<f>` | `0..1` | `0.5` | Stop probability threshold. |
| `--request-option min_stop_step=<n>` | integer >= 0 | `5` | Minimum generated steps before accepting stop. |
| `--request-option seed=<n>` | integer, `-1` for random | `0` | Generation seed. |
| `--session-option midashenglm_gen.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Weight storage override for experiments. |
