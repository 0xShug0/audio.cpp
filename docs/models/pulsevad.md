# PulseVAD

PulseVAD detects speech regions in 16 kHz mono audio.

## Quick Start

```bash
audiocpp_cli \
  --task vad \
  --family pulsevad \
  --model models/PulseVAD-GGUF/pulsevad-2.1k-f32.gguf \
  --backend cpu \
  --audio speech_16k_mono.wav \
  --segments-out segments.json \
  --log
```

## Model

| Field | Value |
|---|---|
| Family | `pulsevad` |
| Task | `vad` |
| Mode | `offline` |
| Model directory | `models/PulseVAD-GGUF` |
| Student weights | `pulsevad-2.1k-f32.gguf` (F32, default) |
| Teacher weights | `pulsevad-81k-f32.gguf` (F32) |
| Input | 16 kHz mono WAV through `--audio` |
| Output | Speech segments through `--segments-out` |

To use the larger teacher, replace the filename with
`pulsevad-81k-f32.gguf`. Both variants use the same family and options.
When both are installed, pass the concrete GGUF file to select the variant.

Download packages: `pulsevad_2_1k_f32` and `pulsevad_81k_f32`, from
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/PulseVAD-GGUF).

For local safetensors checkpoints, use a separate directory per variant with
the filename `pulsevad-f32.safetensors`. Pass that directory or file to `--model`.

Convert input to 16 kHz mono before running. PulseVAD uses 200 ms windows;
shorter recordings are zero-padded, and partial trailing windows are omitted.

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | 16 kHz mono WAV path | required | Audio to analyze. |
| `--segments-out` | JSON path | not set | Save detected speech segments. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `threshold` | float in `[0, 1]` | `0.5` | Speech probability threshold. |
| `hop_size_samples` | integer > 0 | `1600` | Window step in samples; the default is 100 ms. |
| `min_speech_duration_ms` | integer >= 0 | `100` | Minimum speech segment duration. |
| `min_silence_duration_ms` | integer >= 0 | `100` | Minimum silence before closing a speech segment. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `pulsevad.weight_type` | `native`, `f32`, `f16`, `bf16` | `native` | Weight storage override at session creation. |

Older `pulsevad.`-prefixed request option names remain accepted. Prefer the
unprefixed request names above; session options retain the family prefix.

## Convert Weights

From the repository root, prepare both checkpoints from a PulseVAD upstream
checkout, then package each variant with the standard GGUF tool:

```bash
python tests/pulsevad/convert_pulsevad.py \
  --upstream /path/to/PulseVAD \
  --output /path/to/PulseVAD-safetensors \
  --log pulsevad-conversion.log

audiocpp_gguf --family pulsevad --type orig \
  --input weights=/path/to/PulseVAD-safetensors/2.1k/pulsevad-f32.safetensors \
  --output models/PulseVAD-GGUF/pulsevad-2.1k-f32.gguf \
  --model-spec model_specs/pulsevad.json --no-sidecars
```

Repeat the GGUF command with `81k` instead of `2.1k` for the teacher.
