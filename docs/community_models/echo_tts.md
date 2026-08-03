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

**Work in progress.** Landing in stages, each gated on numerical parity against the reference
implementation:

| Milestone | Scope | State |
|---|---|---|
| M0 | Family registration, model spec v1 | done |
| M1 | GGUF conversion, DiT, PCA inverse, Fish decode | in progress |
| M2 | Native speaker encoding (Fish encoder + RVQ) | not started |
| M3 | Long-form generation | not started |
| M4 | Quantisation, RTF and memory evidence | not started |

Until M2 lands, cloning requires a pre-computed speaker latent, so the model is not yet
self-contained. This PR stays in draft until the full evidence pack exists.

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
