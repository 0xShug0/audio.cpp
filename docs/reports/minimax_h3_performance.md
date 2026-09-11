# MiniMax-H3 Performance Notes

Date: 2026-08-11

Branch: `dev`

Hardware: NVIDIA GeForce RTX 5090, CUDA backend, server `threads=8`

This check compares the normal DiT GGUF against the INT8 ConvRot DiT GGUF in
server mode. Server detailed logging was disabled. Each model variant used a
separate server process:

| Variant | DiT file |
|---|---|
| DiT | `dit.gguf` |
| INT8 ConvRot DiT | `dit_int8.gguf` |

Each server received one short warmup request first, then three measured
requests in the same session: `none`, `spectrum`, and `first_block_cache`.

Prompt:

```text
A lively four-speaker podcast roundtable in a small studio. The host, Maya, is warm and curious. The comedian, Leo, is playful and quick. The scientist, Dr. Chen, is thoughtful and calm. The producer, Sam, is dry and amused. Maya says, "Welcome back to Tiny Big Questions, where today we ask whether coffee makes us productive or just louder." Leo says, "I vote louder. My notes are mostly exclamation marks and one drawing of a croissant." Dr. Chen says, "I think the ritual matters more than the caffeine. The cup gives your brain a starting line." Sam says, "My brain missed the starting line and is currently looking for parking." The speakers laugh naturally, overlap lightly, react to each other, and continue in a friendly conversational rhythm with clear distinct voices, no music, and no background noise.
```

Shared measured-request options:

| Option | Value |
|---|---:|
| `seed` | `20260808` |
| `height` | `32` |
| `width` | `32` |
| `num_frames` | `481` |
| `num_inference_steps` | `20` |
| `guidance_scale` | `1.0` |
| `return_video` | `false` |
| `minimax_h3.mem_saver` | default `true` |

Warmup used `num_inference_steps=2`, `num_frames=21`, and
`dit_acceleration=none`.

## Results

| DiT file | Acceleration | Wall ms | Audio sec | RTF | x realtime | Peak VRAM MiB | Speedup vs `dit.gguf` |
|---|---|---:|---:|---:|---:|---:|---:|
| `dit.gguf` | `none` | 12886.40 | 20.05 | 0.6427 | 1.56x | 16513 | baseline |
| `dit_int8.gguf` | `none` | 11692.00 | 20.05 | 0.5831 | 1.71x | 18406 | 9.3% |
| `dit.gguf` | `spectrum` | 7501.48 | 20.05 | 0.3741 | 2.67x | 16617 | baseline |
| `dit_int8.gguf` | `spectrum` | 7331.46 | 20.05 | 0.3657 | 2.73x | 18522 | 2.3% |
| `dit.gguf` | `first_block_cache` | 8270.58 | 20.05 | 0.4125 | 2.42x | 17389 | baseline |
| `dit_int8.gguf` | `first_block_cache` | 7887.51 | 20.05 | 0.3934 | 2.54x | 19286 | 4.6% |

Peak VRAM was sampled with `nvidia-smi` during each server session. The per-row
peak is estimated from the request timing interval. Whole-session peak was
17389 MiB for `dit.gguf` and 19286 MiB for `dit_int8.gguf`.

## ASR Check

The common transcript content matched the main spoken prompt: Tiny Big Questions,
coffee/caffeine/productivity, notes/exclamation marks/croissant, ritual/cup,
starting line, and parking. ASR made small recognition errors such as
`coffee -> cocky` and several variants of `croissant`, but none of the six
outputs collapsed into noise.

## 30s DiT Step Sweep

These sweeps used CUDA server mode, `threads=8`, `num_frames=721`, `height=32`,
`width=32`, `guidance_scale=1.0`, and `seed=20260808`. The prompt was a plain
quoted reading passage sized for the 30s target. Framework Nemotron ASR
(`nemotron_asr`) was used for the transcript check, and WER is computed against
the quoted passage after simple lowercase alphanumeric token normalization.

`dit.gguf`, `dit_acceleration=none`:

| DiT steps | Wall ms | Audio sec | RTF | x realtime | Nemotron ASR WER |
|---:|---:|---:|---:|---:|---:|
| 4 | 6937.46 | 30.675 | 0.2262 | 4.42x | 100.0% |
| 8 | 9410.08 | 30.675 | 0.3068 | 3.26x | 39.1% |
| 12 | 12090.60 | 30.675 | 0.3942 | 2.54x | 1.6% |
| 16 | 14629.50 | 30.675 | 0.4769 | 2.10x | 1.6% |
| 20 | 17213.50 | 30.675 | 0.5612 | 1.78x | 1.6% |
| 24 | 20091.70 | 30.675 | 0.6550 | 1.53x | 1.6% |

`dit.gguf`, `dit_acceleration=spectrum`:

| DiT steps | Wall ms | Audio sec | RTF | x realtime | Nemotron ASR WER |
|---:|---:|---:|---:|---:|---:|
| 4 | 6320.38 | 30.675 | 0.2060 | 4.85x | 100.0% |
| 8 | 7080.35 | 30.675 | 0.2308 | 4.33x | 100.0% |
| 12 | 7720.13 | 30.675 | 0.2517 | 3.97x | 1.6% |
| 16 | 8479.36 | 30.675 | 0.2764 | 3.62x | 4.7% |
| 20 | 9270.63 | 30.675 | 0.3022 | 3.31x | 6.2% |
| 24 | 9916.69 | 30.675 | 0.3233 | 3.09x | 3.1% |

`dit.gguf`, `dit_acceleration=first_block_cache`:

| DiT steps | Wall ms | Audio sec | RTF | x realtime | Nemotron ASR WER |
|---:|---:|---:|---:|---:|---:|
| 4 | 6801.17 | 30.675 | 0.2217 | 4.51x | 100.0% |
| 8 | 9685.66 | 30.675 | 0.3158 | 3.17x | 39.1% |
| 12 | 9546.81 | 30.675 | 0.3112 | 3.21x | 18.8% |
| 16 | 10860.90 | 30.675 | 0.3541 | 2.82x | 4.7% |
| 20 | 11016.10 | 30.675 | 0.3591 | 2.78x | 1.6% |
| 24 | 11726.50 | 30.675 | 0.3823 | 2.62x | 6.2% |

`dit_int8.gguf`, `dit_acceleration=none`:

| DiT steps | Wall ms | Audio sec | RTF | x realtime | Nemotron ASR WER |
|---:|---:|---:|---:|---:|---:|
| 4 | 6752.33 | 30.675 | 0.2201 | 4.54x | 56.2% |
| 8 | 8822.52 | 30.675 | 0.2876 | 3.48x | 4.7% |
| 12 | 11036.60 | 30.675 | 0.3598 | 2.78x | 3.1% |
| 16 | 13253.00 | 30.675 | 0.4320 | 2.31x | 3.1% |
| 20 | 15373.80 | 30.675 | 0.5012 | 2.00x | 3.1% |
| 24 | 17600.10 | 30.675 | 0.5738 | 1.74x | 3.1% |

`dit_int8.gguf`, `dit_acceleration=spectrum`:

| DiT steps | Wall ms | Audio sec | RTF | x realtime | Nemotron ASR WER |
|---:|---:|---:|---:|---:|---:|
| 4 | 6343.60 | 30.675 | 0.2068 | 4.84x | 100.0% |
| 8 | 6960.58 | 30.675 | 0.2269 | 4.41x | 51.6% |
| 12 | 7520.18 | 30.675 | 0.2452 | 4.08x | 12.5% |
| 16 | 8187.32 | 30.675 | 0.2669 | 3.75x | 10.9% |
| 20 | 8783.71 | 30.675 | 0.2863 | 3.49x | 3.1% |
| 24 | 9488.47 | 30.675 | 0.3093 | 3.23x | 3.1% |

`dit_int8.gguf`, `dit_acceleration=first_block_cache`:

| DiT steps | Wall ms | Audio sec | RTF | x realtime | Nemotron ASR WER |
|---:|---:|---:|---:|---:|---:|
| 4 | 6849.84 | 30.675 | 0.2233 | 4.48x | 56.2% |
| 8 | 8927.28 | 30.675 | 0.2910 | 3.44x | 4.7% |
| 12 | 8973.95 | 30.675 | 0.2925 | 3.42x | 21.9% |
| 16 | 10095.00 | 30.675 | 0.3291 | 3.04x | 12.5% |
| 20 | 10259.20 | 30.675 | 0.3345 | 2.99x | 45.3% |
| 24 | 10916.50 | 30.675 | 0.3559 | 2.81x | 1.6% |

In this 30s reading case, the fastest stable measured point was
`dit.gguf` + `spectrum` at 12 steps. The normal DiT path also reached stable ASR
at 12 steps, but with higher RTF. INT8 without acceleration was transcript-stable
by roughly 8-12 steps, while INT8 + `first_block_cache` needed 24 steps in this
run.

## CUDA 13 Server Sweep

This sweep reran the 30s reading case in CUDA 13 server mode with
`dit_acceleration=none`. It used the same `threads=8`, `num_frames=721`,
`height=32`, `width=32`, `guidance_scale=1.0`, seed, prompt, and short warmup
shape as the CUDA 12.9 server rows above.

`dit.gguf`, CUDA 13:

| DiT steps | Wall ms | Audio sec | RTF | x realtime | Gain vs CUDA 12.9 |
|---:|---:|---:|---:|---:|---:|
| 4 | 6901.74 | 30.675 | 0.2250 | 4.44x | +0.5% |
| 8 | 9434.47 | 30.675 | 0.3076 | 3.25x | -0.3% |
| 12 | 11973.10 | 30.675 | 0.3903 | 2.56x | +1.0% |
| 16 | 14506.40 | 30.675 | 0.4729 | 2.11x | +0.8% |
| 20 | 17244.20 | 30.675 | 0.5622 | 1.78x | -0.2% |
| 24 | 19856.10 | 30.675 | 0.6473 | 1.54x | +1.2% |

`dit_int8.gguf`, CUDA 13:

| DiT steps | Wall ms | Audio sec | RTF | x realtime | Gain vs CUDA 12.9 |
|---:|---:|---:|---:|---:|---:|
| 4 | 6871.01 | 30.675 | 0.2240 | 4.46x | -1.8% |
| 8 | 8815.82 | 30.675 | 0.2874 | 3.48x | +0.1% |
| 12 | 10860.00 | 30.675 | 0.3540 | 2.82x | +1.6% |
| 16 | 13167.40 | 30.675 | 0.4293 | 2.33x | +0.6% |
| 20 | 15081.60 | 30.675 | 0.4917 | 2.03x | +1.9% |
| 24 | 17179.80 | 30.675 | 0.5601 | 1.79x | +2.4% |

For this no-acceleration server sweep, CUDA 13 was a small improvement rather
than a large one: total wall time across the six measured step counts improved
by about 0.6% for `dit.gguf` and about 1.2% for `dit_int8.gguf`. The INT8 path
showed the clearest gain at larger step counts.

## Memory-Reduction Options

These runs used `dit.gguf` only, server mode, CUDA, `threads=8`, no detailed
server logging, the same short warmup shape, and the same measured prompt/options
as the 2x3 run above. Each option set used a fresh server process. The baseline
is the `dit.gguf` + `dit_acceleration=none` row above, where
`minimax_h3.mem_saver` is already at its default `true`.

| Option set | Wall ms | RTF | Peak VRAM MiB | Peak delta vs baseline | ASR status |
|---|---:|---:|---:|---:|---|
| baseline: `mem_saver=true` | 12886.40 | 0.6427 | 16513 | baseline | valid |
| `mem_saver=true`, `dit_layerwise=true`, `dit_layerwise_batch=4` | 56154.80 | 2.8007 | 15796 | -717 MiB | valid |
| `mem_saver=true`, `dit_layerwise=true`, `dit_layerwise_batch=4`, `dit_mlp_chunk_tokens=1024` | 56460.30 | 2.8160 | 15661 | -852 MiB | valid |

DiT layerwise staging reduced peak VRAM by roughly 0.7-0.9 GiB, but with a
large runtime cost. The MLP chunking option gave the lowest measured peak in
this run.

## Audio Duration Scaling

These runs used `dit.gguf`, default `minimax_h3.mem_saver=true`,
`dit_acceleration=none`, CUDA server mode, `threads=8`, no detailed server
logging, and one short warmup before the measured duration sweep. The prompt and
all generation options other than `num_frames` were unchanged.

| Target sec | `num_frames` | Output sec | Wall ms | RTF | Peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|
| 5 | 121 | 5.175 | 7076.73 | 1.3675 | 16246 |
| 10 | 241 | 10.125 | 8905.27 | 0.8795 | 16366 |
| 15 | 361 | 15.075 | 10754.00 | 0.7134 | 16530 |
| 20 | 481 | 20.050 | 13046.70 | 0.6507 | 16559 |
| 25 | 601 | 25.700 | 15272.80 | 0.5943 | 16702 |
| 30 | 721 | 30.675 | 17704.70 | 0.5772 | 16830 |
| 60 | 1441 | 60.425 | 34435.90 | 0.5699 | 17674 |

With `mem_saver=true`, peak VRAM increased slowly across this audio-only sweep:
about 1.4 GiB from the 5-second request to the 60-second request. Runtime scales
more visibly than memory, while RTF improves as fixed request overhead is
amortized over longer audio.
