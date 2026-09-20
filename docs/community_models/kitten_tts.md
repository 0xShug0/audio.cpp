# KittenTTS

`kitten_tts` provides native GGML inference for the 80M-parameter
[KittenTTS Mini 0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8)
English text-to-speech model. It uses the shared eSpeak-ng frontend and
produces 24 kHz mono audio with eight built-in voices.

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

Available voices are `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`,
`Kiki`, and `Leo`.

### Request options (use with `--request-option`)

| Option | Range | Default | Description |
|---|---:|---:|---|
| `speed` | positive float | `1.0` | Speech speed multiplier; larger values are faster. |
| `seed` | non-negative integer | random | Decoder noise seed. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `text_chunk_size` | positive integer | `400` | Maximum Unicode codepoints per chunk. |

The common `--speaking-rate` option also sets speech speed.

### Session options (use with `--session-option`)

| Option | Description |
|---|---|
| `kitten_tts.weight_type` | Storage type for matrix multiplication weights. |
| `kitten_tts.conv_weight_type` | Storage type for convolution weights. |
| `kitten_tts.graph_capacity_mode` | Predictor graph capacity policy: `fixed`, `tiered`, `grow`, or `double`. |
| `kitten_tts.max_input_tokens` | Maximum predictor input token capacity. |
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

The upstream code and model are licensed under Apache-2.0.
