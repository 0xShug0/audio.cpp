# CosyVoice3

CosyVoice3 is a GGUF TTS family for zero-shot voice cloning, cross-lingual
speech, and instruction-conditioned speech. The default package is Q8_0.

## Quick Start

Download the default Q8_0 package:

```bash
python3 tools/model_manager_v2.py install cosyvoice3_q8_0 --models-root models
```

Zero-shot voice cloning:

```bash
audiocpp_cli \
  --task clon \
  --family cosyvoice3 \
  --model models/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf \
  --backend cuda \
  --text "This is a local CosyVoice3 voice cloning test." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years." \
  --request-option template_name=zero_shot \
  --out cosyvoice3_clone.wav
```

Instruction-conditioned speech:

```bash
audiocpp_cli \
  --task tts \
  --family cosyvoice3 \
  --model models/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf \
  --backend cuda \
  --text "Please read this with a calm and friendly tone." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years." \
  --request-option template_name=instruct \
  --request-option instruct="Speak warmly with clear articulation." \
  --out cosyvoice3_instruct.wav
```

## Model

| Field | Value |
|---|---|
| Family | `cosyvoice3` |
| Default GGUF | `models/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf` |
| Tasks | `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en`, `ja`, `ko`, `de`, `es`, `fr`, `it`, `ru`, `yue` |
| Voice input | Required reference WAV through `--voice-ref` |

## Templates

| `template_name` | Use |
|---|---|
| `zero_shot` | Voice cloning with prompt audio and transcript. |
| `cross_lingual` | Cross-lingual voice cloning from prompt audio. |
| `instruct` | Instruction-conditioned speech with prompt audio. |

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Prompt/reference speaker audio. |
| `--reference-text` | text | empty | Transcript for prompt audio. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `reference_text` | text | empty | Transcript for prompt audio. |
| `template_name` | `zero_shot`, `cross_lingual`, `instruct` | `zero_shot` | Request template. |
| `instruct` | text | empty | Instruction text for `instruct` mode. Legacy `instruction` remains accepted. |
| `text_chunk_size` | integer > 0 | `600` | Long-form text chunk size. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunk mode. |
| `max_tokens` | `-1` or integer > 0 | `-1` | Automatic maximum: 20 times the target text token count. |
| `min_tokens` | `-1` or integer >= 0 | `-1` | Automatic minimum: twice the target text token count; must not exceed the resolved maximum. |
| `top_k` | integer > 0 | `25` | AR speech-token top-k sampling limit. |
| `num_inference_steps` | integer > 0 | `10` | Flow decoder Euler steps. |
| `seed` | integer >= 0 | `1986` | Generation seed. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `cosyvoice3.reference_cache_slots` | integer >= 0 | `4` | Prepared reference-audio cache slots. |
| `cosyvoice3.mem_saver` | bool | `false` | Release cached runtime graphs after request phases. |
| `cosyvoice3.exec_plan` | JSON object | `{}` | Component execution plans. The only component is `flow`; see the RKNN section. |

## Rockchip NPU (RKNN) estimator

The flow decoder is the expensive half of this family: the DiT estimator runs once per Euler step
(`num_inference_steps`, default 10) over the whole utterance, and on an RK3588 it is ~79% of
synthesis. `cosyvoice3.exec_plan` voices that one stage through `librknnrt` and leaves the AR LLM
and the HiFT vocoder on the host, which is the split the reference RKNN port of this model
measured best: a quantized NPU LLM degrades the speech tokens, and a quantized HiFT adds
high-frequency artifacts.

RKNN has no runtime graph builder, so each engine is compiled offline for one fixed sequence
length. Build them on an x86_64 host from the same checkpoint the GGUF was converted from:

```bash
python3 tools/rknn/prepare_cosyvoice3_rknn_flow.py \
    --estimator-onnx flow.decoder.estimator.fp32.onnx \
    --seq-len 200 --seq-len 500 --seq-len 900 --seq-len 1100 \
    --output-dir models/CosyVoice3-RKNN
```

Then point the session at the ladder (order does not matter; the smallest engine that fits is
used, and an utterance longer than the largest engine runs the CPU graph). The ladder travels as
the `flow` component's `main` artifact, comma-separated — see
[component execution plans](../build/rockchip-npu.md#component-execution-plans):

```bash
audiocpp_cli --task clon --family cosyvoice3 --model models/CosyVoice3-GGUF/cosyvoice3-q8_0.gguf \
    --backend cpu --session-option 'cosyvoice3.exec_plan={"flow":{"impl":"rknn","artifacts":{"main":"models/CosyVoice3-RKNN/flow_estimator_seq200.rknn,models/CosyVoice3-RKNN/flow_estimator_seq900.rknn"}}}' \
    --text "..." --voice-ref assets/resources/b.wav --reference-text "..." --out out.wav
```

Build with `-DENGINE_ENABLE_RKNN=ON` (plus `ENGINE_RKNN_LIBRARY` / `ENGINE_RKNN_INCLUDE_DIR`
pointing at the board's `librknnrt`); a plan selecting `impl: rknn` throws at load in a build
without it rather than falling back silently. Each engine is ~700 MB and an utterance is padded up
to the engine's
length, so the ladder is a time trade: the estimator costs about `S^2`, and the padding is exact
(measured bit-identical to an unpadded run), not a quality one.

### What it measures on an RK3588

Measured on an Orange Pi 5 Plus (RK3588, 32 GB, `librknnrt` 2.3.2, `--backend cpu --threads 4`,
10 Euler steps, same text/seed/reference on both paths), `dit_ms` for the whole ODE loop:

| mel frames | engine (pad) | ggml CPU | RKNN |
|---|---|---|---|
| 102 | seq200 (98) | 8.48 s | 11.49 s |
| 212 | seq500 (288) | 19.14 s | 57.10 s |
| 102 | seq110 (8) | 8.48 s | 6.14 s |
| 212 | seq250 (38) | 19.29 s | 18.51 s |
| 402 | seq400 (2) | 38.87 s | 38.84 s |

The estimator on the NPU is at parity with four A76 cores running the ggml graph, so the padding is
what decides: a coarse ladder loses by 3x, a dense one breaks even above 200 frames and wins 1.38x
only at the shortest lengths. Peak RSS is 3.8-6.8 GB against 2.3-2.6 GB.

Two ceilings bound the ladder. The flow's frame count includes the prompt mel, so a 4-second
reference puts every utterance above 700 frames. And the driver's IOVA arena is 32-bit on this
kernel: one engine asks 0.7 GB at seq110 and 4.71 GB at seq900, and above about 4 GB `rknn_init`
fails with `-4` (`RKNPU: failed to allocate IOVA: -12`), which is why the ladder is loaded lazily,
one engine resident. Engines above ~700 frames are therefore unusable on this board, and long
utterances — where the estimator is 79% of synthesis — run on the ggml graph.

The fp16 cost of the graph, measured on the PC simulator against the fp32 ONNX (seq200, 106 real
frames): snr 44.0 dB, tail snr 60.5 dB, `max_abs 0.5544`, mean `-3.83689` against `-3.83806`. There
is no systematic mel bias, so no offset knob is needed.

Legacy `cosyvoice3.conv_weight_type` remains accepted but has no consumer and
does not change convolution storage. The implementation uses the shared
`cosyvoice3.weight_type` setting with its existing component storage rules.
