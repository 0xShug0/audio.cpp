# KugelAudio-0-Open

KugelAudio generates 24 kHz mono speech from text, using a bundled preset voice
or an optional acoustic encoder for reference-audio voice cloning. This port
does not apply a watermark.

Reference cloning is an experimental audio.cpp extension. The original open
processor accepts presets or prepared voice features, but does not expose a
raw-WAV cloning API. This implementation uses the acoustic encoder included in
the original checkpoint and the existing voice-feature conditioning path.

Upstream: [KugelAudio-0-Open](https://huggingface.co/kugelaudio/kugelaudio-0-open).
The model and upstream implementation use the MIT license.

## Quick Start

```bash
audiocpp_cli --task tts --family kugelaudio \
  --model /path/to/kugelaudio-0-open-q8_0.gguf --backend cuda \
  --text "The next train leaves in ten minutes." \
  --request-option voice_id=english_female --seed 1234 \
  --out speech.wav --log
```

The GGUF includes the tokenizer and all four preset voices. No separate Python
environment or tokenizer download is required for inference. Published preset-only
packages omit the acoustic encoder; cloning requires a locally converted package
or a separate encoder file as described below.

Packages are available in BF16, Q8_0 (default), and Q4_K. Each package contains
the same four voices; select its GGUF file with `--model`.

## Common Options

Use these as ordinary CLI flags.

| Option | Description |
| --- | --- |
| `--text` | Text to speak. The model infers the language from the text. |
| `--seed` | Random seed; `-1` chooses a random seed. |
| `--mode` | `offline` or `streaming`. |
| `--out` | Output WAV path. |

## Request Options

Use with `--request-option name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `voice_id` | `default` | `default`, `clear`, `english_female`, or `english_male`. |
| `num_inference_steps` | `20` | Diffusion steps per acoustic frame, from 1 to 1000; omitted uses `diffusion_head_config.ddpm_num_inference_steps` from the checkpoint. |
| `guidance_scale` | `3` | Diffusion guidance; `1` disables the unconditional pass. |
| `max_tokens` | `2048` | Maximum generated tokens per text chunk. |
| `do_sample` | `false` | Sample AR control tokens rather than choose the highest logit. |
| `temperature` | `1` | AR temperature when sampling is enabled; `0` uses greedy decoding. |
| `seed` | `1234` | Random seed; `-1` chooses a random seed. |
| `text_chunk_size` | `300` | Maximum Unicode codepoints per long-form chunk. |
| `text_chunk_mode` | `default` | `default`, `tag_aware`, `japanese`, or `endline`. |

The `default` and `clear` presets are German female voices. `english_female`
and `english_male` are British English voices. Presets can speak other supported
languages, but upstream recommends their native languages for best reliability.

## Session Options

Use with `--session-option kugelaudio.name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `weight_type` | `native` | Generator weight storage type. The optional reference encoder retains source values in F32 during preparation. |
| `reference_encoder_path` | unset | Optional local acoustic-encoder GGUF or safetensors file, from the same checkpoint. Only needed if the main GGUF omits its encoder. |

## Long-Form and Streaming

Long text uses the framework text chunker. Each text chunk starts a new
generation with the same voice features and an incremented seed.

Streaming sends decoded audio frames as they become available. Offline output
is peak-normalized if it exceeds full scale. Streaming cannot retrospectively
normalize already-delivered audio, so it sends the decoder output directly.
The same behavior applies to cloned voices.

## Conversion

Build `audiocpp_gguf` first, then run:

```bash
python tests/kugelaudio/convert_gguf.py \
  --model-dir /path/to/original-checkpoint \
  --tokenizer-dir /path/to/Qwen2.5-tokenizer \
  --output-dir /path/to/KugelAudio-0-Open-GGUF
```

The original checkpoint must include its safetensors index and `voices/*.pt`.
Add `--type q8_0` or `--type q4_k` to create a quantized package.
The tokenizer directory must contain `tokenizer.json` and `tokenizer_config.json`
from `Qwen/Qwen2.5-1.5B`. Conversion excludes the semantic encoder and embeds
the preset acoustic features.
The acoustic encoder is also excluded unless `--include-reference-encoder` is
specified. That flag produces a `*-clone.gguf` containing the original encoder
values, without quantizing them. Preset-only conversions embed a preset-only
capability contract.

To augment an existing preset GGUF without changing its quantized tensors:

```bash
python tests/kugelaudio/add_reference_encoder.py \
  --model /path/to/kugelaudio-0-open-q8_0.gguf \
  --encoder /path/to/acoustic_encoder.safetensors \
  --converter /path/to/audiocpp_gguf \
  --output /path/to/kugelaudio-0-open-q8_0-clone.gguf
```

The encoder file must contain the original
`model.acoustic_tokenizer.encoder.*` tensors from the matching checkpoint. It is
not a separately published model-manager download. The augmented GGUF includes
the current clone-aware contract and preserves the original preset tensors.

## Reference-Audio Cloning

```bash
audiocpp_cli --task clon --family kugelaudio \
  --model /path/to/kugelaudio-0-open-q8_0-clone.gguf \
  --backend cuda --voice-ref reference.wav \
  --text "Hello, this is a test of the cloned voice." \
  --seed 1234 --out cloned.wav --log
```

Use a clean single-speaker recording, up to 30 seconds; the runtime rejects
longer recordings rather than silently truncating them. References are downmixed
and resampled to 24 kHz, normalized to -25 dBFS RMS, then encoded. Silence,
non-finite samples and incomplete frames are rejected. A mono 24 kHz reference
avoids resampler differences from the original Python loader. No transcript is
required. Speaker reference audio takes precedence over `voice_id`.

For a published preset-only GGUF, supply
`--session-option kugelaudio.reference_encoder_path=/path/to/acoustic_encoder.safetensors`.
Its old embedded contract does not advertise cloning, so also select the current
local contract with `--model-spec-override /path/to/model_specs/kugelaudio.json`.
Preset generation needs neither option and remains compatible with old packages.
Encoder configuration is parsed only when reference encoding is requested;
preset-only generation also works with decoder-only configuration files.

Only the latest normalized reference and its encoded features are cached per
session. Repeated requests with the same reference reuse them. Encoder weights
and graphs are released before synthesis; they do not remain resident beside
the generator between requests. Failed reference preparation does not replace
the previous prepared voice. `prepare({})` clears that prepared voice.

Cloning quality depends on the speaker, recording and text. Encoding parity is
not a claim of speaker-identity equivalence or a supported upstream cloning API.
CUDA uses GGML's existing TF32 math policy; small feature differences from a
strict F32 CPU reference are expected. No shared backend policy is changed.

The existing server speech endpoint accepts `voice_ref` as a local path or a
base64 WAV. Configure a TTS model with the encoder-enabled package, then send:

```json
{"model":"kugel-clone","input":"Hello from the cloned voice.","voice_ref":"/path/to/reference.wav","seed":1234}
```

See [the validation record](../reports/kugelaudio_reference_cloning.md) for
reference comparisons, session checks and reproducible commands.
