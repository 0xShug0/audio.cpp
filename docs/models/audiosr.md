# AudioSR

AudioSR performs audio super-resolution from an input waveform.

## Quick Start

```bash
audiocpp_cli \
  --task s2s \
  --family audiosr \
  --model models/AudioSR-GGUF/audiosr-basic-f32.gguf \
  --backend cuda \
  --audio input.wav \
  --request-option num_inference_steps=50 \
  --request-option guidance_scale=3.5 \
  --request-option ddim_eta=1.0 \
  --request-option seed=42 \
  --out output.wav
```

## Model

| Field | Value |
|---|---|
| Family | `audiosr` |
| Task | `s2s` |
| Mode | `offline` |
| Default package | `models/AudioSR-GGUF/audiosr-basic-f32.gguf` |
| Input | Source WAV through `--audio` |
| Output | Super-resolved waveform |

`ddim_eta` defaults to `1.0`, matching the official AudioSR inference default.
Use `--request-option ddim_eta=0` only for deterministic sampler debugging.

Long audio is processed with bounded overlapping chunks once the input exceeds
`audio_chunk_duration_sec`. The defaults match the official Python long-audio
path:

```bash
--request-option audio_chunk_duration_sec=15 \
--request-option audio_chunk_overlap_sec=2
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Source waveform to super-resolve. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `num_inference_steps` | integer > 0 | `50` | DDIM inference steps. The uniform DDIM schedule must produce exactly this many timesteps; not every count is valid (e.g. 51 fails with 1000 diffusion timesteps). |
| `guidance_scale` | float | `3.5` | Classifier-free guidance scale. Any finite value is accepted. |
| `ddim_eta` | float >= 0 | `1.0` | DDIM eta; 1.0 matches the official AudioSR default. Must be nonnegative; there is no runtime upper bound. |
| `audio_chunk_duration_sec` | seconds > 0 | `15` | Long-audio chunk size; must exceed the overlap. |
| `audio_chunk_overlap_sec` | seconds >= 0 | `2` | Overlap must be smaller than `audio_chunk_duration_sec`, including after rounding to samples. |
| `seed` | integer <= 2147483647 | `42` | Sampler seed. Omitted or negative values use fixed seed 42, not a random seed; zero is a fixed seed. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `audiosr.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Weight storage override for experiments. |
