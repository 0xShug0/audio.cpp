# MiniMax Music 3

MiniMax Music 3 is a text-to-music model with lyrics conditioning. The
audio.cpp port is packaged as a component GGUF directory containing the language
model, RVQ depth decoder, condition encoder, flow transformer, vocoder, tokenizer
files, and model configs.

## Install

Install the default GGUF package with the model manager:

```bash
python3 tools/model_manager_v2.py install minimax_music3_q4_0
```

The default package uses the current balanced component mix:

```text
MiniMax-Music3-GGUF/
  config.json
  config/
    language_model.json
    rvq_depth_decoder.json
    condition_encoder.json
    transformer.json
    vocoder.json
  tokenizer/
    tokenizer.json
    tokenizer_config.json
  language_model_q4_0.gguf
  rvq_depth_decoder_q8_0.gguf
  condition_encoder.gguf
  transformer_q4_0.gguf
  vocoder.gguf
```

## Run

Pass the package directory as `--model`. Lyrics are required through the
`lyrics` request option.

```bash
CAPTION='A bright pop rock song with clean drums, crisp rhythm guitars, a clear female vocal, an energetic chorus, and polished studio production.'
LYRICS='[verse] City lights are shining low. I keep moving with the glow. [chorus] Turn it up and let it fly. Sing the melody tonight.'

audiocpp_cli \
  --task gen \
  --family minimax_music3 \
  --model models/MiniMax-Music3-GGUF \
  --backend cuda \
  --text "$CAPTION" \
  --request-option "lyrics=$LYRICS" \
  --request-option duration_sec=30 \
  --request-option num_inference_steps=30 \
  --out music3.wav \
  --metrics
```

`duration_sec` is the autoregressive audio-frame budget, aligned with the
official Python model's `max_new_tokens` style control. It is not a hard output
duration cap: the final song can be shorter depending on prompt, lyrics, seed,
and sampling parameters. Larger budgets also increase memory use because the AR
and flow stages have to reserve enough room for longer generated sequences.

## Component Selection

The model directory may contain multiple component GGUF precisions. Select a
component by filename with session options:

```bash
audiocpp_cli \
  --task gen \
  --family minimax_music3 \
  --model models/MiniMax-Music3-GGUF \
  --backend cuda \
  --session-option minimax_music3.language_model_gguf=language_model_q8_0.gguf \
  --session-option minimax_music3.rvq_depth_decoder_gguf=rvq_depth_decoder_q8_0.gguf \
  --session-option minimax_music3.flow_transformer_gguf=transformer_q8_0.gguf \
  --text "$CAPTION" \
  --request-option "lyrics=$LYRICS" \
  --request-option duration_sec=30 \
  --out music3-q8.wav
```

Do not combine component GGUF selection with `weight_type` for the same
component. The GGUF file already determines the stored tensor precision.

## Options

Common request options:

| Option | Default | Notes |
|---|---:|---|
| `lyrics` | required | Lyrics to sing. Tags such as `[verse]`, `[chorus]`, `[bridge]`, and `[outro]` can be used in the text. |
| `duration_sec` | `20` | AR frame budget expressed in seconds. It affects maximum length, performance, and memory use, but is not a hard final audio duration. |
| `num_inference_steps` | `30` | Flow matching Euler steps per chunk. Lower values are faster. |
| `guidance_scale` | `1.7` | Flow transformer CFG scale. Set `0` for the non-CFG flow path. |
| `ar_guidance_scale` | `1.5` | Semantic/depth AR CFG scale. Set `0` for the non-CFG AR path. |
| `top_k` | `50` | Top-k sampling for semantic and residual code sampling. |
| `seed` | `0` | Generation seed. Try several seeds for diffusion-style music generation. |

Common session options:

| Option | Default | Notes |
|---|---:|---|
| `minimax_music3.language_model_gguf` | `language_model_q4_0.gguf` | Language model component file relative to the model root. |
| `minimax_music3.rvq_depth_decoder_gguf` | `rvq_depth_decoder_bf16.gguf` | RVQ depth decoder component file relative to the model root. |
| `minimax_music3.flow_transformer_gguf` | `transformer_q4_0.gguf` | Flow transformer component file relative to the model root. |
| `minimax_music3.graph_context_mb` | `32` | Runtime graph arena size in MiB. |
| `minimax_music3.weight_context_mb` | `32` | Weight context size in MiB. |
| `minimax_music3.mem_saver` | `true` | Load large generation stages only while needed to reduce peak VRAM. |

## Notes

The default package is intended to make the model practical on common GPUs,
while higher-precision component combinations are useful for quality and
performance experiments. Longer songs and higher precision settings may need
substantially more VRAM.
