# Audio Tools

| Model | Family | Task(s) | Quick Start |
|---|---|---|---|
| MioCodec | `miocodec` | `vc`, `s2s` | [MioCodec](#miocodec) |
| Seed-VC | `seed_vc` | `vc`, `svc` | [Seed-VC](#seed-vc) |
| VeVo2 | `vevo2` | TTS, SVC, VC, editing | [VeVo2](#vevo2) |
| HTDemucs | `htdemucs` | `sep` | [HTDemucs](#htdemucs) |
| BS-RoFormer | `bs_roformer` | `sep` | [BS-RoFormer](#bs-roformer) |
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
| GGUF model | `models/MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf` |
| Tasks | `vc`, `s2s` |
| Modes | `offline` |
| Input | Source speech WAV through `--audio` |
| Conditioning | Target/reference voice WAV through `--voice-ref` |
| Output | Single converted WAV through `--out` |

Voice conversion:

```bash
audiocpp_cli --task vc --family miocodec --model models/MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf --backend cuda --audio assets/resources/a.wav --voice-ref assets/resources/b.wav --out converted.wav
```

Speech-to-speech:

```bash
audiocpp_cli --task s2s --family miocodec --model models/MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf --backend cuda --audio assets/resources/a.wav --voice-ref assets/resources/b.wav --out converted.wav
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

## BS-RoFormer

BS-RoFormer separates vocals from a 44.1 kHz music mixture using explicit,
non-overlapping frequency bands. The native implementation accepts either the
converted SafeTensors package or a standalone GGUF with the package spec and
`config.json` embedded.

| Field | Value |
|---|---|
| Family | `bs_roformer` |
| Task | `sep` |
| Modes | `offline` |
| Input | 44.1 kHz mono or stereo WAV through `--audio` |
| Output | `vocals.wav` and derived `instrumental.wav` under `--out-dir` |
| Weight types | `native`, `f32`, `f16`, `bf16`, `q8_0` |

Standalone GGUF:

```bash
audiocpp_cli --task sep --model models/BS-RoFormer-ep368_Q8/BS-RoFormer-ep368_Q8.gguf --backend cuda --audio song_44k.wav --out-dir stems
```

Converted SafeTensors package:

```bash
audiocpp_cli --task sep --family bs_roformer --model models/BS-RoFormer-ep368 --backend cuda --audio song_44k.wav --out-dir stems
```

CUDA uses F32-accumulating Flash Attention while CPU and other backends keep
the explicit attention path. The packaged overlap count remains the
quality-oriented default. Lower overlap is an opt-in speed/quality tradeoff:

| Session option | Default | Notes |
|---|---:|---|
| `bs_roformer.num_overlap` | package `num_overlap` (`4` for ep368) | Set to `2` or `1` for fewer model passes. This is faster but changes boundary blending and can reduce separation quality. |
| `bs_roformer.weight_type` | `native` on device backends | Optional storage override such as `f16` or `f32`; measure it on the target backend because converting Q8 weights to F16 is not necessarily faster. |

Fast single-pass example:

```bash
audiocpp_cli --task sep --model models/BS-RoFormer-ep368_Q8/BS-RoFormer-ep368_Q8.gguf --backend cuda --audio song_44k.wav --out-dir stems-fast --session-option bs_roformer.num_overlap=1
```

The conversion helper preserves the checkpoint's fused QKV weights, explicit
`freqs_per_bands` layout, global final RMSNorm, and mask-estimator depth:

```bash
python tests/bs_roformer/convert_reference_ckpt.py \
  --ckpt model_bs_roformer.ckpt \
  --config-path model_bs_roformer.yaml \
  --output-dir models/BS-RoFormer
```

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
