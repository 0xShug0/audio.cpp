# Echo-TTS parity run 1: findings

Source: `echo_ref.npz`, 146 arrays, generated from `audio_prompts/musk1.wav`,
seed 0, 40 steps, sequence_length 640, model dtype bfloat16.

## Components now verified against real data

| Component | Result |
| --- | --- |
| Byte tokenizer + normalisation | **exact** — all 140 ids and the normalised string match byte-for-byte |
| PCA orientation | **confirmed** — `pca.components` is `(80, 1024)`, as assumed |
| `pca_unproject` (C++) | max abs 5.7e-06 against numpy on the real basis (z_q range 11.97) |
| PCA round trip (C++) | max abs 5.3e-06 against the real speaker latent (range 2.51) |
| Reconstructed z_q vs `ae.encode_zq` | min -10.2110 / max +11.9744 vs reference -10.2115 / +11.9740 |
| `find_flattening_point` (C++) | **exact** — 140 of 640 frames, matching the reference heuristic |
| `latent_scale` | float32(1/18) exactly |

The flattening-point match is worth calling out: it ran on the real generated
latent, not a synthetic one, and 140 frames is 6.502 s of audio from a 29.72 s
window. Getting this wrong changes the output duration silently, and it is
sensitive to the variance convention — I checked that `ddof=0` also lands on 140
here, so this particular case would not have caught a wrong choice. The
implementation uses the unbiased estimator to match `torch.std`, which is right
for the general case regardless.

## RNG: same stream, not bit-exact

`generate_torch_cuda_randn(51200, 0)` was compared against
`sampler.initial_noise`:

    cosine                1.000000000000   (1 - cos = 3.1e-14)
    median error          2 ULP
    p99 error             149 ULP
    correlation           1.000000000000 to 12 digits
    rank agreement        99.64%

Same Philox stream and same normal transform; the residual is CUDA-vs-host libm
precision in the transcendental calls. **Seeded parity will be near-identical but
not bit-exact**, so a 40-step trajectory will diverge slightly from the
reference. Against the PR's cosine >= 0.999 gate this is irrelevant (the noise
alone passes with ~3e10 margin), but any test written to expect bit-equality
would fail for reasons that are not bugs. Write the gates as cosine plus
max-abs-error, not equality.

Also confirmed: `dit.x_input` is bitwise identical to `sampler.initial_noise`,
so the fixed-timestep probe and the sampler share a starting draw.

## Massive activations in the late blocks

Activation magnitude grows monotonically through the stack:

| Block | std | max |
| ---: | ---: | ---: |
| 0 | 0.266 | 5.69 |
| 12 | 0.393 | 14.88 |
| 20 | 1.147 | 38.75 |
| 22 | 2.619 | 97.50 |
| 23 | 5.893 | 187.00 |

std grows 22x and max 33x from first block to last, with most of it in the final
four blocks. Separately, the layer-23 key caches for **both** text and speaker
peak at exactly 510.0 while layers 0 and 12 peak near 8-10. The two paths share
one `k_norm` weight per layer and carry unrelated inputs, so an identical maximum
points at a large element in that weight rather than at the data — the standard
massive-activation / attention-sink pattern.

Consequences:

1. **F16 activations are safe.** 510 and 187 are far below the 65504 F16 ceiling.
   No overflow risk in the planned conversion.
2. **The converter's decision to keep norm weights in F32 was right for a reason
   that was not known when it was made.** `KEEP_F32_SUBSTRINGS` already covers
   `q_norm` and `k_norm`. Quantising a weight with a ~510 outlier to a
   block-scaled int8 would destroy the small elements sharing its block.
3. **Q8_0 (milestone M3) needs care in the last four blocks.** A per-block int8
   scale resolves roughly 1/127 of the block maximum, which at block 23 is ~1.5
   absolute against a std of 5.89. Mixed precision — leaving blocks 20-23 at F16
   — is the obvious first thing to try if Q8_0 degrades quality.

## Still unverified

The DiT graph itself. Every check above exercises host-side code; nothing has run
the ggml graph, because that needs a build. The per-block dumps in this file are
exactly what the block-by-block comparison will consume once it can run, and the
growth table above doubles as a smoke test: a port that gets the residual stream
right should reproduce that monotone 22x growth, and one that gets adaLN gating
or the half-head RoPE wrong will not.

Useful next request, if another run is cheap: `--full-blocks`, which dumps every
block activation in full rather than stats plus a 64-value prefix. Not needed
until there is a build to compare against.
