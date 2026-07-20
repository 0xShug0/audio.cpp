# Audio Tools

| Model | Family | Task(s) | Quick Start |
|---|---|---|---|
| MioCodec | `miocodec` | `vc`, `s2s` | [MioCodec](#miocodec) |
| Seed-VC | `seed_vc` | `vc`, `svc` | [Seed-VC](#seed-vc) |
| VeVo2 | `vevo2` | TTS, SVC, VC, editing | [VeVo2](#vevo2) |
| HTDemucs | `htdemucs` | `sep` | [HTDemucs](#htdemucs) |
| Mel-Band RoFormer | `mel_band_roformer` | `sep` | [Mel-Band RoFormer](#mel-band-roformer) |

This page covers voice conversion, codec, and source-separation families. These models do not share one interface: conversion models consume source speech plus a target voice, while the separation models consume a mixture and write named stems.

Common CLI shape:

```bash
audiocpp_cli --task <task> --family <family> --model <model-dir> --backend cuda ...
```

## MioCodec

MioCodec is a speech codec and voice-conversion path. In the CLI it is exposed as conversion tasks, not as a low-level token encode/decode tool.

| Field | Value |
|---|---|
| Family | `miocodec` |
| Model directory | `models/MioCodec-25Hz-44.1kHz-v2` |
| Tasks | `vc`, `s2s` |
| Modes | `offline` |
| Input | Source speech WAV through `--audio` |
| Conditioning | Target/reference voice WAV through `--voice-ref` |
| Output | Single converted WAV through `--out` |

Voice conversion:

```bash
audiocpp_cli --task vc --family miocodec --model models/MioCodec-25Hz-44.1kHz-v2 --backend cuda --audio source.wav --voice-ref target.wav --out converted.wav
```

Speech-to-speech:

```bash
audiocpp_cli --task s2s --family miocodec --model models/MioCodec-25Hz-44.1kHz-v2 --backend cuda --audio source.wav --voice-ref target.wav --out converted.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Source speech audio. |
| `--voice-ref` | WAV path | required | Target speaker/reference audio. |
| `--task` | `vc`, `s2s` | required | Conversion task. |
| `--out` | WAV path | required | Output audio path. |
| `--session-option miocodec.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Model weight type when supported by each component. |

## Seed-VC

Seed-VC provides voice conversion and singing voice conversion routes. See [Seed-VC](models/seed_vc.md) for the full route manual.

```bash
audiocpp_cli --task vc --family seed_vc --model models/Seed-VC --backend cuda --audio source.wav --voice-ref target.wav --out converted.wav
```

## VeVo2

VeVo2 covers speech, singing, voice conversion, singing conversion, and editing routes. See [VeVo2](models/vevo2.md) for the full route manual.

```bash
audiocpp_cli --task vc --family vevo2 --model models/VeVo2 --backend cuda --audio source.wav --voice-ref target.wav --out converted.wav
```

## HTDemucs

HTDemucs separates a music mixture into stems. The current integration writes the model stems as named output artifacts under `--out-dir`; it does not expose the upstream two-stems shortcut as a separate CLI task.

| Field | Value |
|---|---|
| Family | `htdemucs` |
| Model directory | `models/htdemucs` |
| Task | `sep` |
| Modes | `offline` |
| Input | 44.1 kHz music mixture WAV through `--audio` |
| Output | Stem files under `--out-dir` |
| Stems | Vocals, drums, bass, and other when produced by the model package |

```bash
audiocpp_cli --task sep --family htdemucs --model models/htdemucs --backend cuda --audio song_44k.wav --out-dir stems
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | 44.1 kHz WAV path | required | Input music mixture. |
| `--out-dir` | directory | required | Directory for separated stems. |
| `--backend` | `cpu`, `cuda`, `vulkan`, `metal`, `best` | `cpu` | Compute backend. |

## Mel-Band RoFormer

Mel-Band RoFormer is wired as a vocal/source-separation model. The CLI uses the framework separation task and writes named artifacts under `--out-dir`.

| Field | Value |
|---|---|
| Family | `mel_band_roformer` |
| Model directory | `models/mel-roformer-mlx` |
| Task | `sep` |
| Modes | `offline` |
| Input | 44.1 kHz music mixture WAV through `--audio` |
| Output | Named separated artifacts under `--out-dir` |
| Notes | Chunking/overlap behavior is internal to the integration; no user chunk option is exposed here |

```bash
audiocpp_cli --task sep --family mel_band_roformer --model models/mel-roformer-mlx --backend cuda --audio song_44k.wav --out-dir stems
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | 44.1 kHz WAV path | required | Input music mixture. |
| `--out-dir` | directory | required | Directory for separated outputs. |
| `--backend` | `cpu`, `cuda`, `vulkan`, `metal`, `best` | `cpu` | Compute backend. |

For backend weight-type controls, use `audiocpp_cli --inspect --model <model-dir> --family <family>`.
