# SoulX-Singer

SoulX-Singer supports singing synthesis (SVS) from aligned phonemes and notes or
an F0 contour, and singing voice conversion (SVC) from existing vocals. Both use
reference audio to select the voice. One family serves both tasks, with separate
upstream checkpoints; choose the matching package.

## Model

| Field | Value |
|---|---|
| Family | `soulx_singer` |
| Tasks | SVS: `music` (`--task gen`); SVC: `svc` |
| Mode | Offline |
| Input | SVS: reference singing and aligned metadata; SVC: source vocals and reference audio |
| Output | 24 kHz mono audio |
| Languages | Mandarin, English, Cantonese |
| Packages | F16, Q8_0 |

The [upstream project](https://github.com/Soul-AILab/SoulX-Singer/tree/81aeb3ae772c70093c3de74dc23c92d983801ae4) supplies
example metadata. The runtime consumes that format directly. It does not
automatically transcribe lyrics, align phonemes or convert MIDI into metadata.

## Install

Download `model.pt` from
[Soul-AILab/SoulX-Singer](https://huggingface.co/Soul-AILab/SoulX-Singer).
Keep an upstream source checkout for its phoneme vocabulary and example metadata:

```bash
git clone https://github.com/Soul-AILab/SoulX-Singer models/SoulX-Singer-source
python tests/soulx_singer/convert_gguf.py --variant svs \
  --source models/SoulX-Singer/model.pt \
  --phoneset models/SoulX-Singer-source/soulxsinger/utils/phoneme/phone_set.json \
  --staging models/SoulX-Singer/svs-staging --type q8_0 \
  --output models/SoulX-Singer-GGUF/soulx-singer-q8_0.gguf \
  > convert-soulx-singer.log 2>&1
```

Use `--type f16` and an `-f16.gguf` filename for the F16 package. The converter
uses `build/debug/bin/audiocpp_gguf`, enables its logging, and embeds configuration,
the phoneme vocabulary, model contract and license notices. Vocoder matrices
remain FP16; score conditioning and normalization weights remain FP32.

## Run

```bash
audiocpp_cli --task gen --family soulx_singer \
  --model models/SoulX-Singer-GGUF/soulx-singer-q8_0.gguf \
  --backend cuda --threads 8 --voice-ref reference.wav \
  --request-option reference_metadata_file=reference.json \
  --request-option target_metadata_file=target.json \
  --request-option control=score \
  --request-option seed=2026 --out singing.wav --log
```

Metadata is a JSON array of segments with upstream `phoneme`, `duration`,
`note_pitch`, `note_type`, `f0` and `time` fields. Durations are seconds;
segment times are milliseconds. Token and contour fields are space-separated
strings; F0 values are Hz at 50 frames per second. Melody control requires F0
contours in both files; score control can omit them. Reference metadata describes the supplied
reference audio. Target metadata defines the output segments and timeline.
Use upstream examples as the format reference, including its phoneme vocabulary.

## Singing voice conversion

Download `model-svc.pt` from
[Soul-AILab/SoulX-Singer](https://huggingface.co/Soul-AILab/SoulX-Singer),
`rmvpe/rmvpe.pt` from
[Soul-AILab/SoulX-Singer-Preprocess](https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess),
and `model.safetensors` from [openai/whisper-base](https://huggingface.co/openai/whisper-base).
Then convert with the built GGUF tool:

```bash
python tests/soulx_singer/convert_gguf.py --variant svc \
  --source models/SoulX-Singer/model-svc.pt \
  --whisper models/whisper-base \
  --rmvpe models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt \
  --staging models/SoulX-Singer/svc-staging --type q8_0 \
  --output models/SoulX-Singer-SVC-GGUF/soulx-singer-svc-q8_0.gguf \
  > convert-soulx-singer-svc.log 2>&1
```

Use `--type f16` and an `-f16.gguf` filename for the F16 package. The converter
uses `build/debug/bin/audiocpp_gguf`, enables its logging, and embeds the model
contract, configuration and license notices. Quantization applies to the DiT
matrices; vocoder matrices remain FP16 and the content/pitch encoders remain FP32.

## Run

```bash
audiocpp_cli --task svc --family soulx_singer \
  --model models/SoulX-Singer-SVC-GGUF/soulx-singer-svc-q8_0.gguf \
  --backend cuda --threads 8 \
  --audio singing.wav --voice-ref reference.wav \
  --request-option seed=2026 --out converted.wav --log
```

The package contains all inference weights and configuration. F0 preprocessing
requires a build with SOXR. Longer recordings are processed using pitch-aware
segments with context; this is offline chunking, not streaming playback.

## Options

| Request option | Default | Meaning |
|---|---|---|
| `reference_metadata_file` | Required for SVS | Reference metadata JSON file |
| `target_metadata_file` | Required for SVS | Target metadata JSON file |
| `control` | `melody` | `score` uses note pitches; `melody` uses the F0 contour |
| `num_steps` | `32` | Euler denoising steps |
| `guidance_scale` | `3` | Guidance strength; `0` disables CFG |
| `seed` | `2026` | Sampling seed; `-1` selects a random seed |
| `pitch_shift` | `0` | Semitone shift, from `-24` to `24` |
| `auto_pitch_shift` | `false` | Match median pitch when `pitch_shift=0` |

Session options are `weight_type=native` and `attention=auto`. The native
weight type preserves package storage. Both score and melody modes reuse
session-owned conditioning, denoising and vocoder graphs.

## License

SoulX-Singer code and checkpoints are Apache-2.0 licensed. SVC packages also
contain Whisper and RMVPE dependencies; retain their accompanying notices.
