# UniverSR

UniverSR performs audio super-resolution to 48 kHz.

## Quick Start

```bash
audiocpp_cli \
  --task s2s \
  --family universr \
  --model models/UniverSR-GGUF/universr-audio-orig.gguf \
  --backend cuda \
  --audio input.wav \
  --request-option input_sample_rate=16000 \
  --out restored.wav \
  --log
```

## Model

| Field | Value |
|---|---|
| Family | `universr` |
| Task | `s2s` |
| Mode | `offline` |
| Model directory | `models/UniverSR-GGUF` |
| Audio weights | `universr-audio-orig.gguf` (F32, default) |
| Speech weights | `universr-speech-orig.gguf` (F32) |
| Input | Source WAV through `--audio` |
| Output | 48 kHz mono waveform |

For speech, replace the model filename with `universr-speech-orig.gguf`.
Both weights use the same family and options.

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Source audio to super-resolve. |
| `--out` | WAV path | not set | Save the restored waveform. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `input_sample_rate` | `8000`, `12000`, `16000`, `24000` | input file rate | Effective input bandwidth, expressed as a sample rate. |
| `sampler_mode` | `euler`, `midpoint`, `rk4` | `midpoint` | Integration method. |
| `num_inference_steps` | integer > 0 | `4` | Integration steps. |
| `guidance_scale` | float >= 0 | `1.5` | Guidance strength; `0` disables guidance. |
| `seed` | integer >= 0 | `42` | Initial noise seed. |
| `audio_chunk_duration_sec` | seconds >= 0 | `0` | Chunk duration; `0` processes the whole recording. |

Set `input_sample_rate` to the source's effective bandwidth rate, even if the
WAV was previously resampled to a higher rate. Omit it only when the file rate
is one of the supported values and describes the source bandwidth.

For long recordings, add `--request-option audio_chunk_duration_sec=6`.
Chunks are processed independently and concatenated without overlap; this can
change the output compared with whole-recording processing.

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `universr.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k` | `native` | Weight storage override at session creation. |

## Convert Weights

From the repository root, convert a checkpoint directory containing
`pytorch_model.bin` and `config.yaml`:

```bash
python tests/universr/convert_universr_gguf.py \
  --model-dir /path/to/universr-audio \
  --output models/UniverSR-GGUF/universr-audio-orig.gguf \
  --log universr-conversion.log
```

For speech, select the speech checkpoint directory and output filename.
The script uses `build/debug/bin/audiocpp_gguf` by default; use
`--audiocpp-gguf /path/to/audiocpp_gguf` for another executable.
