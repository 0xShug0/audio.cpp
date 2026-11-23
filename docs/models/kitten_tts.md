# KittenTTS

`kitten_tts` provides native GGML inference for KittenTTS Mini 0.8 and
KittenTTS 2 Mini, producing 24 kHz mono audio. Both use the same predictor
execution path and portable eSpeak-ng frontend.

| Variant | Languages | Voices | Package |
|---|---|---|---|
| Mini 0.8 | English | 8 | `kitten-tts-mini-0.8-orig.gguf` |
| TTS2 Mini | 16 checkpoint languages | 118 language-qualified voices | `kitten-tts-2-mini-f32.gguf` |

Checkpoint metadata selects the variant automatically; existing 0.8 packages
remain supported. This family is separate from the larger `kitten_tts2` model.

## Install

Install eSpeak-ng when it is not included in the audio.cpp build, then install
the model package:

```bash
sudo apt install espeak-ng libespeak-ng1
python tools/model_manager_v2.py install kitten_tts_mini_0_8_orig --models-root models
```

## Run

```bash
audiocpp_cli --task tts --family kitten_tts \
  --model models/KittenTTS-GGUF/kitten-tts-mini-0.8-orig.gguf \
  --backend cpu --text "This lightweight model runs without a GPU." \
  --voice-id Leo --out kitten.wav
```

Mini 0.8 voices are `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`,
`Kiki`, and `Leo`.

### TTS2 Mini

```bash
python tools/model_manager_v2.py install kitten_tts_2_mini_f32 --models-root models
audiocpp_cli --task tts --family kitten_tts \
  --model models/KittenTTS-GGUF/kitten-tts-2-mini-f32.gguf \
  --backend cuda --language en --voice-id Eleanor \
  --text "Hello there, this little kitten can speak clearly and naturally." \
  --out kitten-mini2.wav --log
```

Use a packaged voice name or its language-qualified ID to disambiguate names
shared across languages. The package contains its vocabulary, voice styles,
and language metadata; no model sidecars are required.

The portable English frontend can produce different phonemes from upstream
Misaki. Japanese text requires external G2P: supply upstream phonemes through
`--request-option 'phonemes=["PHONEMES"]'`. Each list entry is synthesized as
one chunk; text chunking is bypassed for supplied phonemes. Japanese text
without supplied phonemes reports an explicit error.

### Request options (use with `--request-option`)

`--voice-id` takes precedence over `--request-option voice_id=...`.

| Option | Range | Default | Description |
|---|---:|---:|---|
| `voice_id` | packaged voice ID or alias | `expr-voice-5-m` | Built-in voice, also selectable with `--voice-id`. |
| `language` | checkpoint language code | voice language | Selects the frontend language and language-qualified voice. |
| `phonemes` | JSON string list | absent | Externally prepared phoneme chunks; bypasses text G2P and chunking. |
| `speed` | positive float | `1.0` | Speech speed multiplier; larger values produce faster speech. Must be finite and positive. |
| `seed` | non-negative integer | random | Decoder noise seed for the current request; omitted requests choose a fresh random seed. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `text_chunk_size` | positive integer | `400` | Maximum Unicode codepoints per chunk. |

The common `--speaking-rate` option also sets speech speed.

### Session options (use with `--session-option`)

| Option | Description |
|---|---|
| `kitten_tts.weight_type` | Storage type for matrix multiplication weights. |
| `kitten_tts.conv_weight_type` | Storage type for convolution weights. |
| `kitten_tts.graph_capacity_mode` | Predictor graph capacity policy: `fixed`, `tiered`, `grow`, or `double`; defaults to `grow` for both variants. |
| `kitten_tts.max_input_tokens` | Maximum predictor input token capacity; defaults to `min(512, checkpoint context_length)`. |
| `kitten_tts.pre_tail_tokens` | Optional prebuilt predictor tail capacity. |
| `kitten_tts.espeak_library_path` | Explicit path to the eSpeak-ng shared library. |
| `kitten_tts.espeak_data_path` | Explicit path to `espeak-ng-data`. |

## Convert the official model

Download `kitten_tts_mini_v0_8.onnx`, `voices.npz`, and `config.json` from the
upstream model repository, then run:

```bash
python tests/kitten_tts/convert_onnx_to_safetensors.py \
  --onnx kitten_tts_mini_v0_8.onnx \
  --voices voices.npz \
  --output-dir kitten-converted/ggml

cp config.json kitten-converted/config.json
audiocpp_gguf \
  --input kitten=kitten-converted/ggml/kitten_tts.safetensors \
  --root kitten-converted \
  --output kitten-tts-mini-0.8-orig.gguf \
  --type orig --family kitten_tts --model-spec model_specs/kitten_tts.json
```

### TTS2 Mini conversion

Download the complete official TTS2 Mini checkpoint, including its configuration,
vocabulary and language voice packs, then run from the repository root:

```bash
python tests/kitten_tts/convert_tts2_mini.py \
  --model /path/to/kitten-tts-2-mini --output-dir kitten-mini2-converted
audiocpp_gguf \
  --input kitten=kitten-mini2-converted/ggml/kitten_tts.safetensors \
  --root kitten-mini2-converted --output kitten-tts-2-mini-f32.gguf \
  --type f32 --family kitten_tts --model-spec model_specs/kitten_tts.json
```

The converter preserves the official neural tensors and embeds the voice packs.

The upstream code and model are licensed under Apache-2.0.
