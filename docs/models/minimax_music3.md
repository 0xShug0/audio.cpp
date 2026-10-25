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
| `guidance_scale` | `1.7` | Flow transformer CFG scale; must be > 0. Set `1` to use the conditional prediction without guidance amplification. |
| `ar_guidance_scale` | `1.5` | Semantic/depth AR CFG scale; must be > 0. Set `1` to use conditional logits without guidance amplification. |
| `top_k` | `50` | Top-k sampling for semantic and residual code sampling. |
| `seed` | `0` | Generation seed. Try several seeds for diffusion-style music generation. |

Both scales use `uncond + scale * (cond - uncond)`. Setting both to `1`
removes CFG's effect, but the runtime still computes the unconditional branches;
this does not select a faster conditional-only path. Zero is rejected.

Common session options:

| Option | Default | Notes |
|---|---:|---|
| `minimax_music3.language_model_gguf` | `language_model_q4_0.gguf` | Language model component file relative to the model root. |
| `minimax_music3.rvq_depth_decoder_gguf` | `rvq_depth_decoder_q8_0.gguf` | RVQ depth decoder component file relative to the model root. |
| `minimax_music3.flow_transformer_gguf` | `transformer_q4_0.gguf` | Flow transformer component file relative to the model root. |
| `minimax_music3.graph_arena_mb` | `32` | Host graph-metadata arena size in MiB, not device workspace; legacy `minimax_music3.graph_context_mb` remains accepted. |
| `minimax_music3.weight_context_mb` | `32` | Weight context size in MiB. |
| `minimax_music3.mem_saver` | `true` | Load large generation stages only while needed to reduce peak VRAM. |
| `minimax_music3.perf_mode` | `off` | `cuda_perf` selects 256-row attention-prefix buckets. Explicit controls override the preset. |
| `minimax_music3.kv_cache_alignment` | `1`; `256` with `cuda_perf` | Attention-prefix bucket size. `1` keeps full-capacity attention. |

## Performance options

Attention buckets retain all valid history, but numerical differences can change
sampled songs. The preset does not change weight precision, flow steps,
`mem_saver`, pipeline overlap, CFG reuse or chunk hops.

```bash
--session-option minimax_music3.perf_mode=cuda_perf
```

The loader always prepares only the 16,384 semantic-token head rows plus EOS,
preserving the selected weight dtype. Already-compact GGUFs are unchanged;
no checkpoint conversion is required.

Explicit `minimax_music3.kv_cache_alignment=1` restores normal attention even
when the preset is selected.

Everything below is opt-in; the defaults reproduce the reference trajectory
exactly. The two cheapest wins that need no extra GGUFs are
`flow_uncond_interval=3` and `flow_chunk_hop_frames=150`.

Request options:

| Option | Default | Notes |
|---|---:|---|
| `flow_uncond_interval` | `1` | Evaluate the flow unconditional CFG branch only every N-th step and reuse the cached guidance delta in between. `2`-`3` measures flow -25..-40% at ~0.3 dB mel-L1. Values <= 1 disable guidance reuse. |
| `flow_uncond_warmup` | `2` | Initial steps in each chunk that evaluate both CFG branches before reuse. Nonpositive values add no warmup. |
| `flow_chunk_hop_frames` | `0` | Flow chunk hop in AR frames (~25/sec). `0` keeps the model config (100, 50% chunk overlap). Positive values are capped at the configured chunk size; `150` measures flow ~-35% with consistently rederived crops and carry. |
| `ensemble_takes` | `1` | Integer in `[1, 16]`. Generate K takes that share one batched AR pass. Named outputs (`take_01`..) are emitted only when K > 1; K = 1 returns the primary audio. Amortizes the AR stage across takes. |
| `ensemble_prefix_frames` | `0` | With ensembles: share a master AR prefix of this many frames across takes (intro-lock), then let takes diverge. |

Session options:

| Option | Default | Notes |
|---|---:|---|
| `minimax_music3.pipeline_overlap` | `false` | Overlap AR decoding with per-chunk denoise/vocode on a second, lower-priority CUDA stream. Combining with `mem_saver=true` is an error. Long requests and nondefault chunk hops use sequential execution. |

Optional component GGUFs: `rvq_depth_decoder_q4_k.gguf` (the depth stage is
weight-bandwidth-bound; measures depth about -24% against `q8_0` with no
audible difference in our listening) can be selected with
`minimax_music3.rvq_depth_decoder_gguf` as shown above.

Environment variables (supported):

| Variable | Effect |
|---|---|
| `MM3_LEGACY_ROPE=1` | Use the original split-table rope path instead of the fused native NEOX rope. |
| `MM3_LEGACY_GLU=1` | Use the original slice+silu+mul GLU instead of the fused SwiGLU kernel. |
| `MM3_DEPTH_GPU_SAMPLE=1` | Device philox top-k sampler for the depth stage (byte-exact, perf-neutral option). |
| `MM3_DEPTH_GPU_FRAME=1` | Full GPU depth frame with device residual ids and hidden accumulation (byte-exact, perf-neutral option). |

## Notes

The default package is intended to make the model practical on common GPUs,
while higher-precision component combinations are useful for quality and
performance experiments. Longer songs and higher precision settings may need
substantially more VRAM.
