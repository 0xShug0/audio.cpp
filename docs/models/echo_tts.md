# Echo-TTS

Echo-TTS is an English zero-shot voice-cloning TTS model. A 2.8B diffusion transformer (EchoDiT)
generates 80-dimensional latents in PCA space, which the Fish S1-DAC autoencoder decodes to 44.1 kHz
audio. Cloning takes a reference wav with **no transcript required**.

Upstream: [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) ·
autoencoder: [jordand/fish-s1-dac-min](https://huggingface.co/jordand/fish-s1-dac-min)

## Quick Start

```
audiocpp_cli \
    --family echo_tts \
    --model /path/to/Echo-TTS-GGUF \
    --task clon \
    --voice-ref reference.wav \
    --text "[S1] Alright, I'm going to demo this new model." \
    --out out.wav
```

`tts` is an alias to the same synthesis path. Route selection does not change
the model's existing input requirements.

The speaker reference is `--voice-ref`, not `--target-voice`; the latter is for
path-based voice conversion. No transcript of the reference is needed. Useful
request options: `num_inference_steps` (default 40), `text_guidance_scale`
(3.0), `speaker_guidance_scale` (8.0), `truncation_factor` (0.8), and `seed`.

## Model

| Field | Value |
|---|---|
| Family | `echo_tts` |
| Tasks | `tts`, `clon` |
| Modes | `offline` |
| Languages | `en` |
| Default package | `echo_tts_q8_0` |
| Input | Text and reference voice audio; no reference transcript required |
| Output | Generated speech audio |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--voice-ref` | Reference audio for voice cloning. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `text_chunk_size` | `int` | `300` | Maximum Unicode codepoints per text chunk; must be positive, default 300. Declared range: minimum 1. |
| `text_guidance_scale` | `float` | `3` | Classifier-free guidance scale on the text condition (independent-cfg only). Advertised range is recommended, not enforced by the parser. Declared range: minimum 0. |
| `speaker_guidance_scale` | `float` | `8` | Classifier-free guidance scale on the speaker condition (independent-cfg only). Advertised range is recommended, not enforced by the parser. Declared range: minimum 0. |
| `guidance_scale` | `float` | `8` | Classifier-free guidance scale on the text and speaker condition (join-cfg only). Advertised range is recommended, not enforced by the parser. Declared range: minimum 0. |
| `t_schedule_mode` | `linear`, `sway` | `linear` | RF Euler timestep schedule: linear or sway; default linear. |
| `sway_coeff` | `float` | `-1.0` | Sway Sampling coefficient; must be finite and produce strictly decreasing timesteps. Default -1.0. |
| `num_inference_steps` | `int` | `40` | Euler sampler steps. Default 40 for linear timestep schedule, 12 for sway. Declared range: minimum 1. |
| `truncation_factor` | `float` | `0.8` | Initial-noise truncation factor. Advertised range is recommended, not enforced by the parser. Declared range: minimum 0, maximum 1. |
| `speaker_kv_scale` | `float` | `1` | Force-speaker KV scaling. 1.0 disables; 1.5 is the upstream default when enabled. Advertised range is recommended, not parser-enforced; zero is invalid for reciprocal scaling. Declared range: minimum 1. |
| `seed` | `int` | `0` | RNG seed for the initial latent. Negative signed-integer values are treated as zero. |
| `reference_duration_sec` | `float` | `15` | Trim the speaker reference to at most this many seconds before encoding. Shorter references are cheaper and often clone better; upstream's guidance favours around 10 s. Defaults to 15 s. Values above the trained maximum of 297.1 s are clamped. Set per request, or as a session default from CLI or server config. Must be positive. |
| `max_duration_sec` | `float` | — | Cap the generation window, in seconds (up to 29.7215 s, the trained maximum; larger values are clamped to it). Quantised down to a whole latent frame of 46.44 ms. Left unset, the window is estimated per chunk from text length and widened automatically if the utterance does not finish inside it, which is substantially cheaper for short text. Must be positive; rounded to at least one latent frame. Declared range: maximum 29.7215. |
| `guidance_interval` | `int` | `1` | Refresh the two unconditional CFG lanes only every Nth step inside the guidance window, reusing the previous guidance correction in between. 1 reproduces upstream exactly. Worth raising to 2 at 30+ steps, where the correction is still measured 8-10 times across the guided phase (roughly 1.3x fewer denoiser evaluations); not recommended below ~20 steps, where it is already coarsely sampled and staleness shows up as degraded timbre and text adherence. Declared range: minimum 1. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `echo_tts.reference_duration_sec` | `float` | `15` | Trim the speaker reference to at most this many seconds before encoding. Shorter references are cheaper and often clone better; upstream's guidance favours around 10 s. Defaults to 15 s. Values above the trained maximum of 297.1 s are clamped. Set per request, or as a session default from CLI or server config. Must be positive. |
| `echo_tts.reference_cache_slots` | `int` | Model/session default | How many encoded speaker references to keep. Encoding is linear in reference length, so reusing a voice across requests avoids repeating it. 0 disables caching; default 4. |
| `echo_tts.mem_saver` | `bool` | `false` | Release Fish S1-DAC decode graph space before encoding reference speaker audio and encode graph space after. Avoids having both allocations resident simulaneously at the cost of rebuilding the graphs when a speaker cache-miss occurs. |
| `echo_tts.cfg_mode` | `independent`, `joint` | `independent` | Type of classifier free guidance used. |

## Text Chunk Size

Use `--text-chunk-size 300` or `--request-option text_chunk_size=300` to set the
maximum Unicode codepoints per text chunk. The default is 300; explicit values
must be positive. Each chunk is generated separately within the model's fixed
audio window.

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
