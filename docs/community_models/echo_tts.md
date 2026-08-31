# Echo-TTS

Echo-TTS is an English zero-shot voice-cloning TTS model. A 2.8B diffusion transformer (EchoDiT)
generates 80-dimensional latents in PCA space, which the Fish S1-DAC autoencoder decodes to 44.1 kHz
audio. Cloning takes a reference wav with **no transcript required**.

Upstream: [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) ·
autoencoder: [jordand/fish-s1-dac-min](https://huggingface.co/jordand/fish-s1-dac-min)

| Family | `echo_tts` |
|---|---|
| Tasks | `clon` |
| Modes | offline |
| Languages | en |
| Sample rate | 44 100 Hz |
| Model directory | `models/echo-tts` |

## Status

**Work in progress.** Every stage is *implemented*; the open question is how much of it is
*verified*. Those are tracked separately on purpose, because a clean build and plausible audio
prove neither.

| Milestone | Scope | Implemented | Numerically verified |
|---|---|---|---|
| M0 | Family registration, model spec v1 | yes | n/a |
| M1 | GGUF conversion, DiT, PCA inverse, Fish decode | yes | **yes** — denoiser and 40-step trajectory both above gate |
| M2 | Native speaker encoding (Fish encoder + RVQ) | yes | folded into the denoiser probe below |
| M3 | Long-form via the framework text chunker | yes | **no** |
| M4 | Q8_0 conversion, RTF and memory evidence | partial | **no** |

Cloning is self-contained — `session.cpp` calls `codec_->encode_latents` directly, so no pre-computed
speaker latent is required.

### Numerical parity against PyTorch

Measured on an RTX 3090 (sm_86), CUDA, F16 GGUF, by `tests/echo_tts/echo_tts_dit_parity.cpp`
against a dump from `tools/community_models/echo_tts/echo_tts_reference.py`. Gates are cosine over the
flattened tensors **and** max-absolute-error, never equality: cosine alone cannot see a uniform
scale error, and the host Philox stream matches CUDA to ~2 ULP rather than bit-exactly.

**The reference's own weight dtype dominates every number here**, and getting it wrong is what made
the sampler look broken for most of this port's life. ggml accumulates in F32 regardless of the
stored weight type, so the like-for-like comparison against an F16 GGUF is a **float32** reference,
not an F16 one. Run that way, every check passes:

| Check | vs float32 reference | Gate | Verdict |
|---|---|---|---|
| Denoiser, one conditional forward at t = 0.7 | cosine 0.999999899, max-abs 0.0126 | ≥ 0.999, ≤ 0.25 | **PASS** |
| 40-step sampler, reference initial noise injected | cosine 0.999516, max-abs 0.318 | ≥ 0.999, ≤ 4.0 | **PASS** |
| 40-step sampler, our own seeded draw | cosine 0.999542, max-abs 0.319 | ≥ 0.999, ≤ 4.0 | **PASS** |

For contrast, the same GGUF against lower-precision references — this is the whole of the residual
that earlier revisions of this document reported as unexplained:

| Check | vs bfloat16 reference | vs float16 reference |
|---|---|---|
| Denoiser | cosine 0.999977 | cosine 0.999999188 |
| Sampler, injected noise | cosine 0.913082 | cosine 0.988459 |
| Sampler, seeded draw | cosine 0.905481 | cosine 0.976972 |

Generate the F32 dump with `--force-dtype float32`, which also disables TF32 — TF32 has a 10-bit
mantissa, no better than F16, and Ampere would otherwise use it for matmuls and defeat the point.

**The denoiser probe passes and is the number that carries the port.** It is what settles the four
details that fail *silently* rather than loudly — half-head RoPE, the interleaved (not NEOX) rotary
pairing, the speaker patchify reshape, and the adaLN `shift/scale/gate` order. A 0 % WER cannot do
that job: it scores the words, not the speaker identity, so a wrong patchify reshape in particular
could yield fluent, correctly-worded speech in the wrong voice and still read as 0 %.

Note what the probe does **not** isolate. The reference text ids and speaker latents are injected,
but `prepare_conditioning()` then runs this port's own text encoder, speaker encoder and KV
projections, so the number covers the combined conditioning-plus-denoiser path. That is enough to
catch a wrong block; it is not enough to localise one. Per-block activation dumps
(`echo_tts_pack_reference.py --blocks`) exist for that and have not been run.

**The 40-step trajectory now passes, and the earlier shortfall was the reference, not this port.**
The evidence that pointed there before it was confirmed:

- **It was not the RNG.** Injecting the reference's own initial noise scored no better than our
  seeded draw (0.988 vs 0.977), so the Philox difference was never the mechanism.
- **It tracked dtype.** Each step up in reference precision moved the seeded trajectory: 0.905 at
  bfloat16, 0.977 at float16, 0.9995 at float32.
- **It compounded with step count.** At 4 steps the bfloat16 comparison scored 0.9965; at 40 it
  scored 0.9055. Monotonic degradation with step count is the signature of accumulating rounding in
  one of the two implementations, amplified every step by dual CFG at 3.0 and 8.0.
- **A line-by-line review of the sampler found no defect** — schedule, inclusive CFG bounds, single
  application of `truncation_factor`, three-lane CFG combination, the Euler update and the
  speaker-KV boundary all agree with `inference.py`.

The rounding was PyTorch's, not ours. Credit to @dignome for running the float32 reference that
settled it; the numbers above were then reproduced independently on an RTX 3090 with the gates
enforced.

### Host-side checks

`tests/echo_tts/echo_tts_host_units.cpp` is registered with `add_test` and needs neither a GPU nor
the checkpoint. It covers WhisperD normalisation (including the asymmetric quote rewrite and the
bare-`S1` tag suppression), full byte-token id vectors, truncation and padding, PCA projection and
inversion pinned independently against hand-computed values on a rectangular non-symmetric basis,
and the flattening-point crop including its standard-deviation *and* mean thresholds.

Separately, and **by hand rather than in CI**: PCA project/unproject at 5.7e-06 against numpy on
the real (80, 1024) basis, the Euler dual-CFG update at 6.6e-07 against a numpy transcription of
`inference.py`, the timestep embedding at 0.0 diff, and the tokenizer at 140/140 ids on the parity
prompt. `combine_cfg_lanes` and `euler_timestep_schedule` have no registered coverage.

### End-to-end run, RTX 3090 (sm_86), CUDA, F16 GGUF

| Check | Result |
|---|---|
| Conversion | `manifest OK`; 1117 DiT tensors written, 219 blockwise tensors dropped, 495 codec tensors |
| GGUF verifier | pass — 1614 tensors (`dit_weights/` 1117, `ae/` 495, `pca/` 2), F16 1043 / F32 571 |
| `latent_scale` | 0.0555555559694767 (= 1/18), matching the reference |
| Generation | exit 0, 44 100 Hz mono, no NaNs, peak 0.80 (below the normalisation threshold) |
| ASR round-trip, 15 words | WER 0 % — the only diffs are Whisper writing spoken "dot" as punctuation |
| ASR round-trip, 32 words | WER 0.0 %, 0 edits |
| Throughput | 9.195 s of audio in 7.89 s wall — **RTF 0.86 cold**, including the 5.5 GB model load |

Transcription used `faster-whisper-large-v3-turbo`. Generation cost is essentially constant across
those two runs (7.75 s vs 7.89 s) because the window is fixed at 640 frames, so longer text inside
one chunk is close to free.

WER on 32 words is a small sample and scores intelligibility only. It shows the pipeline runs end to
end and produces the right words; the denoiser cosine above is what shows the graph is right.

### What is still missing

- No per-block DiT activation dump, so the passing denoiser cosine proves correctness without
  localising where any future regression lives.
- No A/B of the flash-attention path against `AUDIOCPP_ECHO_TTS_NO_FLASH=1` on a fixed seed.
- No listening comparison of F16 against Q8_0.
- Warm RTF and VRAM-stability-across-requests numbers.

None of these block the port from working; they bound how much the numbers above prove.

## Known limitations

### Fixed 29.72-second generation window

Echo is trained to generate at most **640 latents**, and 640 × 2048 ÷ 44100 = **29.7215 s**. This is
a property of the model, not of this port.

Behaviour outside that window:

- Text corresponding to more than ~30 s is **spoken faster** to fit, rather than truncated. This is
  learned behaviour arising from global attention over the text, not an explicit compression step.
- The upstream tokenizer hard-truncates text past **768 UTF-8 bytes**.
- Requesting a shorter window does *not* compress the whole utterance into it — upstream documents
  that the model generates a **prefix** of the utterance instead.

`long_form` is therefore **not** claimed in `capabilities` at this stage.

### Blockwise generation does not extend the window

Upstream ships a blockwise sampler that generates in connected blocks and supports continuing from
existing audio. It **subdivides** the ≤30 s window rather than extending it: upstream requires
`sum(block_sizes) + continuation_length < 640` "to be in-distribution with training data", and
documents prefix plus continuation as "up to 30 seconds combined". Upstream also notes blockwise
"hasn't been thoroughly tested".

## Licence — read before using output commercially

Echo-TTS is **CC-BY-NC-SA-4.0**, and the restriction covers **generated audio, not only the
weights**. The output constraint is inherited from the Fish S1-DAC autoencoder — the same mechanism
that makes Fish Speech's own outputs non-commercial.

Practically: **audio produced by this model may not be used commercially**, regardless of how the
rest of your stack is licensed. audio.cpp itself is Apache 2.0 and is unaffected; model weights are
a separate download.

There is existing precedent in-tree — `fish_audio` (Fish Audio S2 Pro) carries the identical
output restriction from the identical dependency.

## Why this model

Selected by comparing every model tracked in [tts-bench](https://github.com/5uck1ess/tts-bench) — a
public benchmark covering **62 local TTS models** across speed, objective scores, and blind human
preference — against audio.cpp's existing support table.

| Measure | Echo-TTS | Field |
|---|---|---|
| Blind cloning Elo | **1162** | #3 of 40 (35 games; 738 cloning votes total) |
| Speaker similarity (SIM) | **0.836** | 2nd of 41 scored models |
| UTMOS (naturalness) | 4.21 | — |
| WER (intelligibility) | 7.45 % | — |
| Frozen pairwise study | **21-1-6** | near-tied 1st of 28 |

Two honest caveats: the cloning arena averages ~30 games per model, so gaps under ~100 Elo are
noise, and the ranking uses a single reference clip. Echo's standing is robust to both — it is
top-3 on human votes *and* 2nd on objective SIM, which are independent measurements.

Compute profile suits a GGUF port: ~2.8 B parameters at 1.35× RTFx and 9.4 GB VRAM in PyTorch on an
RTX 3090, so there is real work to amortise.

## Architecture

| Component | Params | Role |
|---|---:|---|
| EchoDiT trunk, 24 blocks | 1.75 B | Joint attention + SwiGLU MLP, adaLN timestep modulation |
| Text encoder | 294 M | UTF-8 **byte** tokens (256 vocab) — no phonemizer or G2P |
| Speaker encoder | 294 M | Reference PCA latents → speaker states |
| Latent encoder | 294 M | Blockwise only; omitted in M1 |
| PCA state | 83 K | Fish 1024-D ↔ DiT 80-D, `latent_scale` = 1/18 |
| Fish S1-DAC | 391 M weights | Reference encoding and waveform decoding |

Sampling is 40 Euler steps with **two independent CFG scales** — text (default 3.0) and speaker
(default 8.0) — gated to `t ∈ [0.5, 1.0]`.

Note the Fish checkpoint stores an additional 303.6 M elements of `freqs_cis` and `causal_mask`
buffers. These are regenerated at runtime rather than shipped in the GGUF.

## Options

| Option | Type | Default | Description |
|---|---|---|---|
| `text_guidance_scale` | float | 3.0 | Guidance scale on the text condition. |
| `speaker_guidance_scale` | float | 8.0 | Guidance scale on the speaker condition. |
| `num_inference_steps` | int | 40 | Euler sampler steps. |
| `truncation_factor` | float | 0.8 | Initial-noise truncation. |
| `speaker_kv_scale` | float | 1.0 | Force-speaker KV scaling; 1.5 is upstream's default when enabled. Raise only if the model drifts to a different speaker on out-of-distribution text. |
| `seed` | int | 0 | RNG seed for the initial latent. |
| `reference_duration_sec` | float | 15.0 | Trim the speaker reference before encoding. Also available as a session default. |
| `max_duration_sec` | float | — | Cap the generation window (up to 29.7215 s). Quantised down to a 46.44 ms latent frame; larger values clamp. Unset, the window is estimated per chunk. |
| `guidance_interval` | int | 1 | Refresh the two unconditional CFG lanes only every Nth guided step. 1 reproduces upstream exactly. See [performance notes](echo_tts_performance.md). |

Session options are `echo_tts.reference_duration_sec` and
`echo_tts.reference_cache_slots`.

## Text format

Prompts follow the [WhisperD](https://huggingface.co/jordand/whisper-d-v1a) transcription style:

- `[S1] ` is prepended automatically when neither `[S1]` nor `[S2]` is present.
- Colons, semicolons, and em dashes are normalised to commas.
- Commas generally function as pauses.
- Exclamation points and other emphatic punctuation increase expressiveness but can reduce quality.

Multi-speaker dialogue is expressed with `[S1]` / `[S2]` tags.

## Reference audio

Up to 5 minutes is accepted; 10 seconds or less works well. Audio is mixed to mono, resampled to
44.1 kHz, and peak-limited before encoding.

## Running it

```
audiocpp_cli \
    --family echo_tts \
    --model /path/to/Echo-TTS-GGUF \
    --task clon \
    --voice-ref reference.wav \
    --text "[S1] Alright, I'm going to demo this new model." \
    --out out.wav
```

The speaker reference is `--voice-ref`, not `--target-voice`; the latter is for
path-based voice conversion. No transcript of the reference is needed. Useful
request options: `num_inference_steps` (default 40), `text_guidance_scale`
(3.0), `speaker_guidance_scale` (8.0), `truncation_factor` (0.8), and `seed`.

## Quantisation

`--precision q8_0` produces a roughly half-size GGUF:

| | F16 | Q8_0 |
| --- | ---: | ---: |
| DiT | 4.76 GB | 2.53 GB |
| codec | 0.78 GB | 0.50 GB |

Q8_0 packs 32 weights per block behind one shared scale, so a tensor qualifies
only when its last logical dimension is a multiple of 32. The converter routes
each tensor accordingly rather than quantising blindly:

* **Q8_0** -- every 2-D matmul weight with a conforming row length. That is all
  but one DiT tensor, and 78% of codec weights.
* **F16** -- convolution kernels (`ggml_conv_1d` has no quantised path, which is
  why `codec.cpp` takes matmul and conv storage types separately) and the one
  non-conforming matmul, `in_proj.weight` at (2048, 80).
* **F32** -- norm weights, biases, snake alphas, LayerScale/ConvNeXt gammas and
  the codebooks, exactly as at other precisions.

Round-trip error is around 6e-05 RMSE with cosine similarity above 0.9999 on
weight-like distributions. Because the scale is per 32-weight block, the large
outliers this model carries in its late blocks and in `k_norm` degrade only
their own block rather than a whole row -- and `k_norm` is F32 regardless.

Quality has not been compared against F16 on real audio. Start with `orig` and
treat Q8_0 as an experiment until someone listens to both.

## Limiting the reference length

`reference_duration_sec` trims the speaker reference before encoding. Shorter
references cost less and often clone better -- upstream's guidance favours
around 10 s, and a long clip averages timbre over more prosodic variation.

Per request (bare name):

```
--request-option reference_duration_sec=30
```

As a default for a CLI run or a server, in the session scope (family-prefixed,
which is how the framework namespaces session and load options):

```
--session-option echo_tts.reference_duration_sec=30
```

In a server config file the same key goes under `session_options`, with a string
value. A request value overrides the session default. Values above the trained
maximum of 297.1 s are clamped rather than rejected. Trimming happens before
chunked encoding, so a cap also bounds encode time and VRAM.

## Reference encoding cost

Encoding the speaker reference is linear in its length: one Fish encode pass per
~29.7 s chunk, so a 4m29s clip is ten passes against one for a 28 s clip. At the
trained maximum that is roughly 22% of a request's arithmetic, before per-graph
launch overhead.

The result depends only on the audio and the trim length, so it is cached across
requests. A server rotating a few voices pays the cost once per voice instead of
once per request:

```
--session-option echo_tts.reference_cache_slots=8
```

Default 4; `0` disables it. Each slot holds only the projected latent, at most
2 MB. The cache lives with the session, so it helps a running server and not a
one-shot CLI invocation. Beyond caching, the levers are `reference_duration_sec`
and shorter references generally -- around 10 s is one chunk, the floor.

## The Fish S1-DAC autoencoder

Echo decodes its 80-D PCA latents through the Fish S1 DAC and encodes speaker
references with the same model. audio.cpp ships that codec as a framework
runtime, `engine::codecs::FishDacCodecRuntime`, so Echo and `fish_audio` share
one implementation -- but **not** the weights. `fish_audio` ships Fish Audio S2
Pro; Echo is trained against the S1 DAC (`jordand/fish-s1-dac-min`), and
`pca_state.safetensors` is fitted to that
codec's latent space specifically. Pointing Echo at S2 Pro would produce
plausible-looking latents and wrong audio, with no error anywhere.

The S1 weights are therefore packaged inside Echo's own GGUF, in the `ae`
namespace:

```
python3 tools/community_models/echo_tts/convert_echo_tts.py \
    --model-dir /path/to/echo-tts-base \
    --fish-dir  /path/to/fish-s1-dac-min \
    --outfile   Echo-TTS-GGUF/model.gguf
```

No companion model and no extra options are needed at run time. Two details the
converter handles:

* **Weight normalisation is folded.** The checkpoint stores it in two forms --
  `conv.parametrizations.weight.original0/original1` on the convolutions and
  legacy `weight_g`/`weight_v` on the quantiser projections -- and `codec.cpp`
  expects plain `conv.weight`. Both reduce to `w = g * v / ||v||` with the norm
  taken over every axis but the first. Note that for `ConvTranspose1d` axis 0 is
  the *input* channel count, so `g` is sized by input channels there; the
  decoder's four transposed convolutions are the only place this bites.
* **Fused qkv projections are split.** `autoencoder.py` keeps one `wqkv`
  linear and splits its output into three equal blocks; `codec.cpp` loads
  `attention.q_proj` / `k_proj` / `v_proj` separately, so the converter
  partitions the weight rows in the same order.
* **Exact tensor shapes are carried in metadata.** `ggml_n_dims()` ignores
  trailing dimensions of size 1, so a `(1, C, 1)` snake alpha would read back as
  `(C, 1)` and fail `codec.cpp`'s `{1, C, 1}` shape check. The converter emits
  `audiocpp.tensor_ranks` (INT32) and `audiocpp.tensor_shapes` (INT64) in tensor
  order, which audio.cpp uses in preference to the lossy inference.
* **The GGUF embeds its own model spec.** `package.cpp` refuses to load a
  published GGUF that does not, so the converter copies
  `model_specs/echo_tts.json` into the `audiocpp.model_spec.*` metadata keys.
  A distributed file is therefore self-describing and does not depend on the
  reader having a matching `model_specs/` checkout. Use `--model-spec` to embed
  a spec from elsewhere.
* **Namespaces are separated by `/`, not `.`** -- `dit_weights/...`, `pca/...`,
  `ae/...`. `PrefixedTensorSourceView` matches on `prefix + "/"`, so a
  dot-separated name is never routed and the loader reports the namespace as
  non-existent rather than the tensor as missing.
* **The codec namespace is `ae`, not `codec_weights`.** ggml caps tensor
  names at 64 characters (`GGML_MAX_NAME`) and rejects the whole file at load
  time if any name reaches it. The longest name `codec.cpp` loads is already 60
  characters, so only a three-character prefix fits; `codec_weights.` would push
  157 of the 455 codec tensors over. The converter refuses to write a GGUF that
  would trip this, and `verify_echo_gguf.py` re-checks it.
* **Registered buffers are dropped.** Two causal masks and three RoPE tables
  account for 305 MB of the 1.87 GB checkpoint and are rebuilt at graph
  construction, so they are not stored.

That leaves roughly 1.57 GB of codec weights on top of the 4.76 GB DiT.
`docs/community_models/echo_tts_autoencoder_reuse.md` covers how the two
families share the codec implementation and where the seam sits in
`src/framework/codecs/fish_dac_codec_runtime.cpp`.
