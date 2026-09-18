# Piper TTS

`piper_tts` provides native GGML inference for Piper VITS voices. The initial
package is the English `en_US-lessac-medium` voice and produces 22.05 kHz mono
audio. eSpeak-ng supplies the phonemes; ONNX Runtime is not used at inference.

## Install

Install eSpeak-ng when it is not included in the audio.cpp build:

```bash
sudo apt install espeak-ng libespeak-ng1
python tools/model_manager_v2.py install piper_lessac_medium_orig --models-root models
```

## Run

```bash
audiocpp_cli --task tts --family piper_tts \
  --model models/Piper-TTS-GGUF/piper-en-us-lessac-medium-orig.gguf \
  --backend cuda --text "The northern wind crossed the quiet harbor." \
  --out piper.wav
```

### Request options (use with `--request-option`)

| Option | Range | Default | Description |
|---|---:|---:|---|
| `speed` | `0.5`-`2.0` | `1.0` | Speech speed multiplier; larger values are faster. |
| `variation` | `0.0`-`1.0` | `0.667` | Acoustic latent noise scale. |
| `duration_variation` | `0.0`-`2.0` | `0.8` | Stochastic duration noise scale. |
| `seed` | non-negative integer | `1234` | Generation seed. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `text_chunk_size` | positive integer | `280` | Maximum Unicode codepoints per chunk. |

### Session options (use with `--session-option`)

| Option | Description |
|---|---|
| `piper_tts.espeak_library_path` | Explicit path to the eSpeak-ng shared library. |
| `piper_tts.espeak_data_path` | Explicit path to `espeak-ng-data`. |

## Convert an official voice

Download an ONNX voice and its matching `.onnx.json` configuration from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices), then run:

```bash
python tests/piper_tts/convert_piper_onnx.py \
  --onnx en_US-lessac-medium.onnx \
  --config en_US-lessac-medium.onnx.json \
  --output-dir piper-lessac-converted

audiocpp_gguf \
  --input weights=piper-lessac-converted/model.safetensors \
  --root piper-lessac-converted \
  --output piper-en-us-lessac-medium-orig.gguf \
  --type orig --family piper_tts --model-spec model_specs/piper_tts.json
```

The archived Piper runtime source is MIT licensed. Voice and dataset licenses
are voice-specific; check the model card beside each voice before redistribution.
