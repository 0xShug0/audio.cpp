# Audio Tools

| Model | Family | Task(s) | Quick Start |
|---|---|---|---|
| MeanVC2 | `meanvc2` | `vc` | [MeanVC2](#meanvc2) |
| MioCodec | `miocodec` | `vc`, `s2s` | [MioCodec](#miocodec) |
| PersonaPlex | `personaplex` | `s2s` | [PersonaPlex](#personaplex) |
| RVC | `rvc` | `vc` | [RVC](#rvc) |
| Seed-VC | `seed_vc` | `vc`, `svc` | [Seed-VC](#seed-vc) |
| VeVo2 | `vevo2` | TTS, SVC, VC, editing | [VeVo2](#vevo2) |
| MuScriptor | `muscriptor` | audio to MIDI/events | [MuScriptor](#muscriptor) |
| HTDemucs | `htdemucs` | `sep` | [HTDemucs](#htdemucs) |
| BS-RoFormer | `bs_roformer` | `sep` | [BS-RoFormer](#bs-roformer) |
| Mel-Band RoFormer | `mel_band_roformer` | `sep` | [Mel-Band RoFormer](#mel-band-roformer) |

This page covers voice conversion, codec, audio-to-symbolic, and source-separation
families. These models do not share one interface: conversion models consume
source speech plus a target voice, audio-to-symbolic models consume audio and
write structured artifacts, and separation models consume a mixture and write
named stems.

Common CLI shape:

```bash
audiocpp_cli --task <task> --family <family> --model <model-dir> --backend cuda ...
```

## MeanVC2

MeanVC2 is a zero-shot voice conversion model. It takes source speech through
`--audio` and a target speaker reference through `--voice-ref`. See
[MeanVC2](models/meanvc2.md) for streaming behavior and options.

```bash
python3 tools/model_manager_v2.py install meanvc2_120ms_40ms_f32

audiocpp_cli --task vc --family meanvc2 \
  --model models/MeanVC2-GGUF/meanvc2-120ms-40ms-fp32.gguf \
  --backend cuda \
  --audio assets/resources/a.wav \
  --voice-ref assets/resources/b.wav \
  --out converted.wav
```

## PersonaPlex

PersonaPlex is a speech-to-speech conversational model. It consumes user audio,
a packaged or reference voice prompt, and an optional system/persona prompt.
See [PersonaPlex](models/personaplex.md) for offline and streaming examples.

```bash
python3 tools/model_manager_v2.py install personaplex_7b_v1_q4_k

audiocpp_cli --task s2s --family personaplex \
  --model models/PersonaPlex-GGUF \
  --backend cuda \
  --audio user.wav \
  --text "You are a concise assistant. Answer naturally and briefly." \
  --request-option voice_id=NATF2 \
  --out reply.wav
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

## RVC

RVC is an offline retrieval-based voice-conversion model. The audio.cpp package
uses native HuBERT content features, RMVPE pitch extraction, optional IVF
retrieval blending, and packaged v1/v2 voices. The F16 GGUF package is
self-contained and embeds the package spec plus packaged retrieval sidecars.

| Field | Value |
|---|---|
| Family | `rvc` |
| GGUF model | `models/RVC-GGUF/rvc-f16.gguf` |
| Task | `vc` |
| Modes | `offline` |
| Input | Source speech WAV through `--audio` |
| Packaged voices | `default`, `manthos`, `chocola`, `fraise` |
| Output | Single converted WAV through `--out` |

Packaged voice without retrieval:

```bash
audiocpp_cli --task vc --family rvc --model models/RVC-GGUF/rvc-f16.gguf --backend cuda --audio source.wav --out converted.wav --request-option voice_id=default
```

Packaged voice with retrieval blending:

```bash
audiocpp_cli --task vc --family rvc --model models/RVC-GGUF/rvc-f16.gguf --backend cuda --audio source.wav --out converted.wav --request-option voice_id=default --request-option retrieval_blend=0.5
```

User RVC checkpoint with a matching FAISS retrieval index:

```bash
audiocpp_cli --task vc --family rvc --model models/RVC-GGUF/rvc-f16.gguf --backend cuda --audio source.wav --out converted.wav --request-option voice_model_path=/path/to/voice.pth --request-option retrieval_index_path=/path/to/voice.index --request-option retrieval_blend=0.5
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--request-option voice_id=<id>` | `default`, `manthos`, `chocola`, `fraise` | `default` | Packaged RVC voice id. Ignored when `voice_model_path` is set. |
| `--request-option voice_model_path=<path>` | `.pth` or `.pt` path | unset | User RVC checkpoint path. |
| `--request-option pitch_extractor=rmvpe` | `rmvpe` | `rmvpe` | Pitch extractor for F0-enabled voices. |
| `--request-option pitch_path=<path>` | CSV path | unset | Optional F0 override file with `time,Hz` rows sorted by time. |
| `--request-option retrieval_index_path=<path>` | FAISS `.index` path | unset | User retrieval index used when `retrieval_blend` is greater than 0 for a user checkpoint. |
| `--request-option retrieval_blend=<rate>` | `0.0` to `1.0` | `0.0` | IVF retrieval feature blend rate. `0` disables retrieval. |
| `--request-option semitone_shift=<n>` | integer | `0` | Semitone pitch shift before synthesis. |
| `--request-option pitch_filter_radius=<n>` | integer >= 0 | `3` | Median filter radius for F0 smoothing; values greater than 2 enable filtering. |
| `--request-option output_sample_rate=<hz>` | integer >= 0 | `0` | Output sample rate; `0` keeps the selected voice model sample rate. |
| `--request-option rms_mix_rate=<rate>` | float | `0.25` | RMS envelope mix rate after conversion. |
| `--request-option unvoiced_protection=<rate>` | `0.0` to `1.0` | `0.33` | Unvoiced consonant protection strength. |
| `--request-option speaker_id=<n>` | integer >= 0 | `0` | Speaker embedding id for multi-speaker checkpoints. |
| `--request-option audio_pad_duration_sec=<sec>` | integer >= 1 | `1` | Long-audio chunk pad duration. |
| `--request-option split_query_sec=<sec>` | integer >= 1 | `5` | Quiet-point query window for long-audio splitting. |
| `--request-option split_center_sec=<sec>` | integer >= 1 | `30` | Long-audio split center stride. |
| `--request-option split_threshold_sec=<sec>` | integer >= 1 | `32` | Input duration before quiet-point splitting is used. |
| `--session-option rvc.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `f32` | Tensor storage type for native RVC, HuBERT, and RMVPE weights. |
| `--session-option rvc.voice_cache_slots=<n>` | integer >= 0 | `4` | User voice model cache slots. Set `0` to disable caching. |

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

## MuScriptor

MuScriptor is an audio-to-symbolic tool that converts music audio into
note-event JSON or a MIDI file. See [MuScriptor](models/muscriptor.md) for
streaming, sampling, and full option details.

```bash
python3 tools/model_manager_v2.py install muscriptor

audiocpp_cli --task midi --family muscriptor \
  --model models/MuScriptor-Small-GGUF/muscriptor-small-f32.gguf \
  --backend cuda \
  --audio song.wav \
  --request-option instruments=drums,electric_bass \
  --out result.mid
```

Use `--request-option output_format=json --out events.json` when you want the
generated note-event JSON instead of MIDI.

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
| `--session-option htdemucs.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | backend-dependent | Weight storage type. Defaults to `f32` for host graph planning, `f16` on CUDA, and `native` otherwise. |

Schema-v1 option compatibility:

| Legacy/session input | Schema-v1 option | Notes |
|---|---|---|
| `weight_type` | `htdemucs.weight_type` | Accepted as a compatibility alias for direct session-option callers. Prefer the family-prefixed form. |

## BS-RoFormer

BS-RoFormer separates vocals from a 44.1 kHz music mixture using explicit,
non-overlapping frequency bands. The native implementation accepts either the
converted SafeTensors package or a standalone GGUF with the package spec and
`config.json` embedded.

BS-RoFormer is v1-native. `model_specs/bs_roformer.json` is the single source
of truth for metadata, packages, session options, and GGUF/safetensors
resources. Its loader uses the generic spec-backed path; the shared Mel-Band
RoFormer loader remains unchanged.

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
| Notes | Uses the package overlap count by default; `mel_band_roformer.num_overlap` can lower the overlap for faster inference with a quality tradeoff |

```bash
audiocpp_cli --task sep --family mel_band_roformer --model models/mel-roformer-mlx --backend cuda --audio song_44k.wav --out-dir stems
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | 44.1 kHz WAV path | required | Input music mixture. |
| `--out-dir` | directory | required | Directory for separated outputs. |
| `--backend` | `cpu`, `cuda`, `vulkan`, `metal`, `best` | `cpu` | Compute backend. |
| `--session-option mel_band_roformer.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | backend-dependent | Weight storage type. Defaults to `f32` when the backend requires a host graph plan, otherwise `native`. |
| `--session-option mel_band_roformer.num_overlap=<n>` | integer `>= 1` | package config | Number of overlapping inference windows. Lower values improve throughput but can reduce boundary quality. |

Schema-v1 option compatibility:

| Legacy/session input | Schema-v1 option | Notes |
|---|---|---|
| `weight_type` | `mel_band_roformer.weight_type` | Accepted as a compatibility alias for direct session-option callers. Prefer the family-prefixed form. |
| `num_overlap` | `mel_band_roformer.num_overlap` | Accepted as a compatibility alias for direct session-option callers. Prefer the family-prefixed form. |

For backend weight-type controls, use `audiocpp_cli --inspect --model <model-dir> --family <family>`.
