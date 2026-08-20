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
| M1 | GGUF conversion, DiT, PCA inverse, Fish decode | yes | **no** — see below |
| M2 | Native speaker encoding (Fish encoder + RVQ) | yes | **no** |
| M3 | Long-form via the framework text chunker | yes | **no** |
| M4 | Q8_0 conversion, RTF and memory evidence | partial | **no** |

Cloning is self-contained — `session.cpp` calls `codec_->encode_zq` directly, so no pre-computed
speaker latent is required.

What *is* verified today is host-side only: the byte tokenizer and WhisperD normalisation (exact,
140/140 ids), PCA project/unproject (5.7e-06 against numpy on the real basis), the flattening-point
crop (exact), the Euler dual-CFG update (6.6e-07 against a numpy transcription of `inference.py`),
and the timestep embedding (0.0 diff). The seeded noise matches the reference Philox stream to
cosine 1.000000000000, with a median error of 2 ULP — near-identical, not bit-exact, so gates must
be written as cosine plus max-absolute-error rather than equality.

### End-to-end run, RTX 3090 (sm_86), CUDA, F16 GGUF

The full pipeline has been executed and the output checked objectively:

| Check | Result |
|---|---|
| Conversion | `manifest OK`; 1117 DiT tensors written, 219 blockwise tensors dropped, 495 codec tensors |
| GGUF verifier | pass — 1614 tensors (`dit_weights/` 1117, `ae/` 495, `pca/` 2), F16 1043 / F32 571 |
| `latent_scale` | 0.0555555559694767 (= 1/18), matching the reference |
| Generation | exit 0, 44 100 Hz mono, no NaNs, peak 0.80 (below the normalisation threshold) |
| ASR round-trip, 15 words | WER 0 % — the only diffs are Whisper writing spoken "dot" as punctuation |
| ASR round-trip, 32 words | **WER 0.0 %, 0 edits** |
| Throughput | 9.195 s of audio in 7.89 s wall — **RTF 0.86 cold**, including the 5.5 GB model load |

Transcription used `faster-whisper-large-v3-turbo`. Generation cost is essentially constant across
those two runs (7.75 s vs 7.89 s) because the window is fixed at 640 frames, so longer text inside
one chunk is close to free.

That rules out the failure modes which produce plausible audio rather than an error: half-head RoPE,
the rotary pairing convention (interleaved, not NEOX), the speaker patchify reshape, and the adaLN
`shift/scale/gate` order would each yield fluent-sounding but wrong speech, not a 0 % WER. Tensor
names are settled by `manifest OK` against the real checkpoint.

### What is still missing

A 0 % WER proves the pipeline is right end to end. It is **not** per-tensor parity, and this port
does not yet have any:

- No cosine ≥ 0.999 comparison of the DiT graph against PyTorch at a fixed timestep, and no
  per-block activation dump compared against `echo_ref.npz`.
- No A/B of the flash-attention path against `AUDIOCPP_ECHO_TTS_NO_FLASH=1` on a fixed seed.
- No regression test for `fish_audio` itself. `build_decode_quantizer` was **restructured**, not
  merely extended, so that core family's decode path changed and needs its own coverage.
- No listening comparison of F16 against Q8_0.
- No C++ unit tests are registered; the host-side checks above were run by hand and never committed.

This PR stays in draft until that evidence exists.

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
| `target_voice` | string | — | Reference wav for cloning. No transcript needed. |
| `cfg_scale_text` | float | 3.0 | Guidance scale on the text condition. |
| `cfg_scale_speaker` | float | 8.0 | Guidance scale on the speaker condition. |
| `num_steps` | int | 40 | Euler sampler steps. |
| `truncation_factor` | float | 0.8 | Initial-noise truncation. |
| `speaker_kv_scale` | float | 1.0 | Force-speaker KV scaling; 1.5 is upstream's default when enabled. Raise only if the model drifts to a different speaker on out-of-distribution text. |
| `seed` | int | 0 | RNG seed for the initial latent. |

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
request options: `num_steps` (default 40), `cfg_scale_text` (3.0),
`cfg_scale_speaker` (8.0), `truncation_factor` (0.8), and `seed`.

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

`reference_max_seconds` trims the speaker reference before encoding. Shorter
references cost less and often clone better -- upstream's guidance favours
around 10 s, and a long clip averages timbre over more prosodic variation.

Per request (bare name):

```
--request-option reference_max_seconds=30
```

As a default for a CLI run or a server, in the session scope (family-prefixed,
which is how the framework namespaces session and load options):

```
--session-option echo_tts.reference_max_seconds=30
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
one-shot CLI invocation. Beyond caching, the levers are `reference_max_seconds`
and shorter references generally -- around 10 s is one chunk, the floor.

## The Fish S1-DAC autoencoder

Echo decodes its 80-D PCA latents through the Fish S1 DAC and encodes speaker
references with the same model. audio.cpp already implements that codec for the
`fish_audio` family, so Echo reuses the implementation -- but **not** the
weights. `fish_audio` ships Fish Audio S2 Pro; Echo is trained against the S1
DAC (`jordand/fish-s1-dac-min`), and `pca_state.safetensors` is fitted to that
codec's latent space specifically. Pointing Echo at S2 Pro would produce
plausible-looking latents and wrong audio, with no error anywhere.

The S1 weights are therefore packaged inside Echo's own GGUF, in the `ae`
namespace:

```
python3 tools/community_models/convert_echo_tts.py \
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
`src/models/fish_audio/codec.cpp`.
