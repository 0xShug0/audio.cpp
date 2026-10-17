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
| `piper_tts.exec_plan` | JSON object of component execution plans. Default `{}` keeps existing execution. Select the RKNN decoder with `{"decoder":{"impl":"rknn","artifacts":{"main":"/models/decoder.rknn"}}}`, and pin the duration stages to the host with `{"duration_encoder":{"backend":"cpu"},"duration_flows":{"backend":"cpu"}}`. No params. RKNN requires `ENGINE_ENABLE_RKNN=ON`; see [Rockchip NPU](../build/rockchip-npu.md#component-execution-plans). |

On RK3588 the decoder is 75% of the runtime, so the NPU path is worth the extra file: measured
RTF 0.198 → 0.0696 and peak RSS 299 → 175 MB on `en_US-lessac-medium`.

The two duration stages are components of the same option, so each can run on the host while the
decoder stays on the device. `duration_encoder` already runs on the host on CUDA by default: host
duration alignment keeps small TF32 differences from being amplified by `expand_prior` while the
expensive decoder stays on the device. `duration_flows` runs on the device by default, and it is
the stage worth watching on a device backend, because the flows are the sampler: a rounding
difference there is a different syllable length, and `expand_prior` then spreads it over the whole
frame layout. Measured on an RK3588 Mali G610 with `--backend vulkan`, six runs of one fixed-seed
request produced six different renders, and the same draw lands outside `exp` outright as
`Piper TTS duration predictor produced a non-finite duration` — on v0.9.2 that abort is the normal
outcome of the plain `--backend vulkan` path (1 of 1 runs, first request, no sidecar involved),
not only the one-in-six draw the RKNN decoder sidecar used to add. Pinning
the stage to the host is the fix for that, and on a desktop Vulkan device it costs about 1.9x on
the duration stage. The placement is not math-neutral on CUDA either: host flows change the render
there too (`24298.6` → `24310.2 ms` of audio on a 443-character request), so treat both fields as
render-affecting, not as a free performance knob.

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
