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

## Model

| Field | Value |
|---|---|
| Family | `minimax_music3` |
| Tasks | `music` |
| Modes | `offline` |
| Languages | `auto` |
| Default package | `minimax_music3_q4_0` |
| Input | Music style prompt and lyrics |
| Output | Music audio |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `lyrics` | `string` | required | Lyrics to sing. Structure tags such as [verse] and [chorus] are supported. |
| `duration_sec` | `float` | `20.0` | Positive AR frame budget expressed in seconds; not a hard final waveform duration limit. Declared range: minimum 0.1. |
| `num_inference_steps` | `int` | `30` | Flow matching Euler steps per chunk. Declared range: minimum 1. |
| `guidance_scale` | `float` | `1.7` | Flow transformer classifier-free guidance scale. Must be > 0. Declared range: minimum 0.0. |
| `ar_guidance_scale` | `float` | `1.5` | Autoregressive semantic and depth CFG scale. Must be > 0. Declared range: minimum 0.0. |
| `top_k` | `int` | `50` | Top-k sampling for semantic and residual code sampling. Declared range: minimum 1. |
| `seed` | `int` | `0` | Generation seed. Declared range: minimum 0. |
| `flow_uncond_interval` | `int` | `1` | Evaluate unconditional flow CFG every N-th step, reusing the guidance delta between evaluations. Values <= 1 disable reuse and evaluate both branches every step. |
| `flow_uncond_warmup` | `int` | `2` | Initial flow steps that evaluate both CFG branches before guidance-delta reuse; nonpositive values add no warmup. |
| `ensemble_takes` | `int` | `1` | Independent takes in a batched AR pass using consecutive seeds; flow/vocoder decode each take. Named outputs take_01 through take_NN are emitted only when ensemble_takes > 1; otherwise only primary audio is returned. Declared range: minimum 1, maximum 16. |
| `ensemble_prefix_frames` | `int` | `0` | Shared intro frames before ensemble takes fork, approximately 25 frames per second. Zero disables intro locking. Declared range: minimum 0. |
| `flow_chunk_hop_frames` | `int` | `0` | Flow chunk hop in AR frames, approximately 25 per second. Zero uses checkpoint configuration; positive values are capped at the configured chunk size. Declared range: minimum 0. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `minimax_music3.weight_type` | `native`, `bf16`, `f16`, `q8_0`, `q4_0`, `q4_k` | `native` | Shared weight storage type. |
| `minimax_music3.language_model_gguf` | `string` | `language_model_q4_0.gguf` | Language model component GGUF file relative to the model root. |
| `minimax_music3.rvq_depth_decoder_gguf` | `string` | `rvq_depth_decoder_q8_0.gguf` | RVQ depth decoder component GGUF file relative to the model root. |
| `minimax_music3.flow_transformer_gguf` | `string` | `transformer_q4_0.gguf` | Flow transformer component GGUF file relative to the model root. |
| `minimax_music3.graph_arena_mb` | `int` | `32` | Host graph-metadata arena size in MiB, not device workspace; legacy graph_context_mb remains accepted. Declared range: minimum 1. |
| `minimax_music3.weight_context_mb` | `int` | `32` | Weight context size in MiB. Declared range: minimum 1. |
| `minimax_music3.mem_saver` | `bool` | `true` | Load large generation stages only while they are needed to reduce peak VRAM. |
| `minimax_music3.perf_mode` | `off`, `cuda_perf` | `off` | Performance preset. cuda_perf selects attention-prefix buckets of 256 rows. Explicit controls override preset values. |
| `minimax_music3.kv_cache_alignment` | `int` | `1`; `256` with `cuda_perf` | Attention-prefix bucket size. 1 preserves full-capacity attention; cuda_perf selects 256. Retains all valid history, but graph shapes may change sampled output. Explicit values override the preset. Declared range: minimum 1. |
| `minimax_music3.pipeline_overlap` | `bool` | `false` | Overlap AR decoding with condition/flow/vocoder work on a second backend stream. Requires mem_saver=false; mem_saver=true is an error. Long requests and nondefault chunk hops use sequential execution. |

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

## Usage Notes

Both scales use `uncond + scale * (cond - uncond)`. Setting both to `1`
removes CFG's effect, but the runtime still computes the unconditional branches;
this does not select a faster conditional-only path. Zero is rejected.

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

Session options:

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
