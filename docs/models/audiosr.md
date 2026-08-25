# AudioSR

AudioSR performs audio super-resolution from an input waveform.

## Quick Start

```bash
audiocpp_cli \
  --task gen \
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
| Task | `gen` |
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

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Source waveform to super-resolve. |
| `--request-option num_inference_steps=<n>` | integer > 0 | `50` | DDIM sampler steps. |
| `--request-option guidance_scale=<f>` | float | `3.5` | Classifier-free guidance scale. |
| `--request-option ddim_eta=<f>` | float >= 0 | `1.0` | DDIM eta. |
| `--request-option audio_chunk_duration_sec=<seconds>` | seconds | `15` | Long-audio chunk size. |
| `--request-option audio_chunk_overlap_sec=<seconds>` | seconds | `2` | Overlap between long-audio chunks. |
| `--request-option seed=<n>` | integer, `-1` for random | `42` | Sampler seed. |
| `--session-option audiosr.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Weight storage override for experiments. |
