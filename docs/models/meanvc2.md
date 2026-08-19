# MeanVC2

MeanVC2 is wired as `--family meanvc2` for zero-shot voice conversion. It
converts source speech from `--audio` into the target speaker from `--voice-ref`.
The current audio.cpp package covers the recommended 120 ms / 40 ms quality
path.

## Install

```bash
python3 tools/model_manager_v2.py install meanvc2_120ms_40ms_f32
```

The default package installs a self-contained GGUF:

```text
models/MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf
```

## Quick Start

```bash
audiocpp_cli --task vc --family meanvc2 \
  --model models/MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf \
  --backend cuda \
  --audio assets/resources/a.wav \
  --voice-ref assets/resources/b.wav \
  --out converted.wav
```

Streaming mode accepts source-audio chunks and returns converted audio chunks
plus a final merged result:

```bash
audiocpp_cli --task vc --mode streaming --family meanvc2 \
  --model models/MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf \
  --backend cuda \
  --audio assets/resources/a.wav \
  --voice-ref assets/resources/b.wav \
  --out converted_stream.wav
```

## Model

| Field | Value |
|---|---|
| Family | `meanvc2` |
| Task | `vc` |
| Modes | `offline`, `streaming` |
| Model path | `models/MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf` |
| Source input | Source speech WAV through `--audio` |
| Target voice | Target speaker WAV through `--voice-ref` |
| Language | Language agnostic |

The runtime resamples input internally as needed. Streaming uses 160 ms preferred
input chunks at 16 kHz and keeps conversion state in the session until
`finish_stream`.

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Source speech to convert. |
| `--voice-ref` | WAV path | required | Target speaker reference. |
| `--seed` / `--request-option seed=<n>` | integer >= 0 | `42` | Random seed for MeanVC2 flow noise. |

MeanVC2 currently exposes no load or session tuning options through the model
spec. The GGUF package carries all required component weights.
