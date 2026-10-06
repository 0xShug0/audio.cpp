# Sopro V2 Turbo (`sopro_tts`)

[samuel-vitorino/sopro-v2-turbo](https://huggingface.co/samuel-vitorino/sopro-v2-turbo) is a
120M-parameter zero-shot voice-cloning TTS covering English, European Portuguese, French and
German, released under Apache-2.0. It clones from 5–20 s of reference audio and outputs
24 kHz mono.

> **Not the same model as `soprano_tts`.** audio.cpp's existing `soprano_tts` family is
> [ekwek/Soprano-1.1-80M](https://huggingface.co/WalkingCat/Soprano-1.1-80M-GGUF), an unrelated
> project with a Qwen3 backbone and a different decoder. Sopro V2 Turbo shares nothing with it
> beyond a similar name, so it ships as its own family. The `--family` hints `sopro`,
> `sopro_v2` and `sopro_v2_turbo` all resolve to `sopro_tts`.

## Installation

Both packages come from the model repo. The f16 GGUF is a single self-contained file and the
one the WebUI installs; the upstream safetensors checkpoint runs directly in full precision:

```bash
python3 tools/model_manager_v2.py install sopro_v2_turbo_f16
# -> models/sopro-v2-turbo-GGUF/sopro-v2-turbo-f16.gguf
python3 tools/model_manager_v2.py install sopro_v2_turbo_safetensors
# -> models/sopro-v2-turbo/{config.json,tokenizer.model,*.safetensors}
```

The GGUF was packed from the safetensors with the command below. `audiocpp_gguf` takes one
namespaced input per stage, and `--root` makes it embed `config.json` and `tokenizer.model` as
sidecars. Keep the mel front ends' filterbanks and windows in f32 when packing your own:
`--type f16` would otherwise round them too, and the reference mel the acoustic head copies from
drifts by several dB.

```bash
build/bin/audiocpp_gguf \
    --input model=models/sopro-v2-turbo/model.safetensors \
    --input semantic_encoder=models/sopro-v2-turbo/semantic_encoder.safetensors \
    --input speaker_encoder=models/sopro-v2-turbo/speaker_encoder.safetensors \
    --input vocoder=models/sopro-v2-turbo/vocoder.safetensors \
    --family sopro_tts --root models/sopro-v2-turbo \
    --keep-type 'vocoder/feature_extractor.*=f32' --keep-type 'vocoder/head.istft.window*=f32' \
    --keep-type 'speaker_encoder/frontend.*=f32' --keep-type 'semantic_encoder/frontend.*=f32' \
    --output models/sopro-v2-turbo-GGUF/sopro-v2-turbo-f16.gguf --type f16
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=sopro_tts
cmake --build build --target audiocpp_cli -j"$(nproc)"
```

## Run

Zero-shot cloning always needs a reference clip:

```bash
build/bin/audiocpp_cli \
    --task tts --family sopro_tts \
    --model models/sopro-v2-turbo \
    --backend cpu --threads 8 \
    --text "Sopro is a lightweight text-to-speech model that runs on device." \
    --voice-ref ref.wav \
    --request-option language=en \
    --out out.wav --metrics
```

`--task clon` works the same way. The reference is resampled to 24 kHz, cropped at a pause near
`ref_seconds`, and level-normalised before the speaker and semantic encoders see it. That
normalisation is boost-only and peak-guarded (sopro 2.2): a reference already at or above the
−19.8 dB prompt level is passed through untouched, and a boost is never large enough to push
the peak past 0.95. The output gain is fixed by the level the reference ends up at: it maps that
level to −23 dB (`output_gain`), the same for every segment and in both run modes.

## Streaming

`--mode streaming` emits audio as the semantic LM runs, like `SoproTTS.stream` upstream:

```bash
build/bin/audiocpp_cli \
    --task tts --family sopro_tts \
    --model models/sopro-v2-turbo \
    --backend cpu --threads 8 --mode streaming \
    --text "$(cat article.txt)" \
    --voice-ref ref.wav --language en \
    --out stream.wav --out-dir chunks/
```

Each event carries a `chunk_<n>` named audio buffer that is already levelled, gated and faded,
so a consumer can play events back to back; `--out` writes the whole utterance, which is exactly
the concatenation of the events.

How it works (`streaming.cpp`, a port of `sopro/streaming.py`):

- The reference prompt is trimmed to a whole number of semantic tokens and its complete
  `stream_chunk_frames` (64-frame) chunks are solved once, before the first text segment.
- Every 16 semantic tokens, the acoustic head solves the frames that can no longer change
  (a 3-token lookahead margin behind the last token, rounded down to whole chunks). Each frame
  attends to everything solved before it within its chunk boundary, and the solver keeps every
  DiT layer's keys and values per Euler step, so a chunk costs only its own frames.
- The vocoder decodes each chunk once its receptive field (27 frames each side) is complete and
  overlap-adds it into a streaming ISTFT.
- The opening audio of each text segment is held until speech starts, then the lead-in is cut
  the same way the offline path trims it.

Streaming is not a re-cut of the offline output: it solves with chunk-limited attention and its
own noise draws, as the Python package does, so the two modes differ for the same `seed`. On an
8-core Apple M-series CPU at 6 threads the first chunk arrives about 1.6–1.8 s after the request
starts (reference encoding ~0.3–0.45 s, prompt solve ~0.8–0.95 s, first chunk ~0.4–0.5 s for a
10 s reference), and each later 0.68 s chunk takes about 0.2 s. `ref_seconds` scales the prompt
solve.

### Voice cache

`sopro_tts.voice_cache_slots=<n>` (off by default) keeps the last `n` prepared voices, keyed by
the clip and `ref_seconds`, together with each streamed voice's solved prompt, like reusing a
`prepare_reference` result upstream. A repeated voice skips the encoders and the prompt solve,
which brought the first chunk down from ~1.8 s to ~0.4–0.5 s in the setup above. A streamed
10 s voice holds about 60 MB.

To prepare a voice without generating anything, pass `audiocpp_session_prepare` a request that
carries only the voice audio (plus `ref_seconds` or `num_inference_steps` if the requests that
follow set them).

The cache never changes the audio: the draws a voice needs for itself (room tone for a clip
without a usable pause, the prompt noise) are seeded from the clip rather than the request.

## Options

| Request option | Type | Default | Meaning |
|---|---|---|---|
| `language` | string | *(empty)* | Prepends `<\|lang_xx\|>`; one of `en`, `pt`, `fr`, `de`. Optional, helps on ambiguous text |
| `temperature` | float | 0.8 | Semantic LM sampling temperature; `0` selects arg-max |
| `top_p` | float | 0.9 | Nucleus threshold, applied after top-k renormalisation |
| `top_k` | int | 25 | Top-k truncation; `0` disables |
| `num_inference_steps` | int | 2 | Acoustic rectified-flow Euler steps |
| `max_seconds` | float | 30.0 | Audio cap per segment; long text is split, so total length is unbounded |
| `min_seconds` | float | 0.4 | Minimum audio before the semantic LM may emit EOS |
| `ref_seconds` | float | 10.0 | Reference window used for cloning |
| `text_chunk_size` | int | 300 | Max codepoints per synthesis segment |
| `seed` | int | *(random)* | Seeds semantic sampling and the acoustic noise prior |

| Session option | Default | Meaning |
|---|---|---|
| `sopro_tts.language` | *(empty)* | Default language tag for requests that do not set one |
| `sopro_tts.voice_cache_slots` | `0` | Prepared voices kept between requests; see [Voice cache](#voice-cache) |
| `sopro_tts.matmul_weight_type` | `f32` | Storage type for matmul weights (`native`, `f32`, `f16`, `bf16`, `q8_0`) |
| `sopro_tts.conv_weight_type` | `f32` | Storage type for convolution weights (`native`, `f32`, `f16`) |

Every request-option default comes from the checkpoint's `config.json` `generation` block, so a
retrained variant picks up its own values without a code change.

## Architecture notes

Five stages run per request, mirroring `sopro/` upstream:

1. **Text** (`text_tokenizer.cpp`) — SentencePiece unigram, 8192 pieces, plus the reference's
   punctuation clean-up and sentence/clause/word segmentation. No phonemiser.
2. **Speaker encoder** (`speaker_encoder.cpp`, ~11M) — 16 kHz log-mel into a three-stage gated
   depthwise ResNet with squeeze-excite, then attentive-statistics pooling for identity and
   multi-scale mean/std pooling for style. The convolution trunk runs on the backend; the two
   pooling heads and their MLPs run on the host, where they cost a few hundred kFLOP.
3. **Semantic encoder** (`semantic_encoder.cpp`, ~82M) — a Whisper-style front end and six
   non-causal transformer layers, resampled to one frame per 1024 output samples and quantised
   by an FSQ head with levels `[7,5,5,5,5]` (4375 codes).
4. **Semantic LM** (`semantic_lm.cpp`) — 12 pre-norm blocks, dim 512, QK RMS-norm, SwiGLU,
   half-rotation RoPE. The prompt is `[style prefix | text | prompt tokens | BOS]`, where the
   prompt tokens are the first `prompt_tokens` reference tokens for the first segment and the
   tail of the previous segment's tokens after that. Because the only structural difference
   from a Qwen3 decoder is a LayerScale vector on each residual branch, and those branches end
   in a bias-free projection, the scale is folded into that projection's rows at load time and
   the shared `QwenCausalDecodeRuntime` runs the stack unmodified. The eight-query style prefix
   cross-attention runs on the host.
5. **Acoustic head + vocoder** (`acoustic.cpp`, `vocoder.cpp`) — an 8-block adaptive-layer-norm
   DiT solving a rectified flow in two Euler steps on a sway-sampled time grid, with the prompt
   mel re-pinned after every step; then an 8-block Vocos ConvNeXt backbone and one centred
   ISTFT over the full band. The vocoder config is causal with a lookahead of 3 frames on every
   7-tap conv, which is the stock centred Vocos layout; other lookaheads are padded to match.
   `mu` (the upsampled semantic conditioning) is built in its own graph because it is constant
   across solver steps.

Implementation details worth knowing:

- **Front-end buffers come from the checkpoint.** torchaudio stores its analysis window and mel
  filterbank as persistent buffers, and all three front ends load those rather than rebuilding
  the filterbank, which removes the usual mel-parity risk. A checkpoint exported without them
  fails at load with a message naming the missing tensor.
- **Grouped convolutions are split.** The DiT's causal positional embedding uses
  `Conv1d(512, 512, k=31, groups=16)`; ggml has no grouped conv1d, so the weight is split into
  16 independent convolutions at load time.
- **The velocity graph re-uploads every leaf per Euler step.** `ggml_gallocr` exempts only
  `GGML_TENSOR_FLAG_OUTPUT` tensors from being freed and reused
  (`ggml_gallocr_free_node` in `ggml-alloc.c`); an *input* leaf's arena space is handed to a
  later intermediate once its last consumer has run. That is correct for a one-shot graph, but
  the solver replays the velocity graph once per step, so staging `mu`, `cond_mel`, `cond_mask`,
  `spk` and the RoPE positions once would leave the second and later steps reading whatever
  overwrote them. `SoproVelocityGraph::run` uploads all of them before every compute; it costs
  a few hundred kB per step against a multi-GFLOP DiT pass.

## Known limitations

- **Without the voice cache, every request prepares its voice again.** The reference arrives
  with each request, so a stream pays the encoders and the prompt solve unless
  [`sopro_tts.voice_cache_slots`](#voice-cache) is set.
- **Sampling RNG is not torch-bit-exact.** `sample_next_token` reproduces the reference's
  masking, temperature, top-k and top-p arithmetic exactly, but draws from a seeded
  `std::mt19937_64` rather than torch's generator, so a given `seed` will not reproduce the
  Python output sample-for-sample. The same `seed` is reproducible within audio.cpp.
- **No `int8` AR path.** The upstream `--int8` CPU option quantises only the semantic LM; the
  closest here, `sopro_tts.matmul_weight_type=q8_0`, quantises the matmuls of every stage.
- The text front end is deliberately minimal upstream: prefer words to symbols (`one plus two`,
  not `1 + 2`), and avoid mixing languages inside one sentence.

## Validation status

Verified against the real checkpoint on a 16-core x86-64 CPU build, 8 threads. Every stage was
reimplemented independently in numpy, driven from the checkpoint's own weights, and diffed
against the C++.

| Stage | Check | Result |
|---|---|---|
| Tensor inventory | 762 names + shapes vs. the four real files | exact match |
| Vocoder mel front end | vs. numpy STFT + checkpoint filterbank | max diff 1.9e-3 |
| Vocos backbone + ISTFT head | vs. numpy, all 14 blocks of the sopro 2.1 vocoder | max diff 1.0e-5 |
| Semantic encoder mel | vs. numpy | max diff 1.9e-5 |
| Semantic encoder transformer | vs. numpy, all 6 layers | max diff 6.0e-5 |
| FSQ token ids | vs. numpy | 188/188 identical |
| Speaker encoder mel / trunk / heads | vs. numpy | max diff 3.7e-5 |
| Acoustic `mu`, `spk`, time embedding | vs. numpy | max diff 1.3e-5 |
| Acoustic velocity field, every Euler step | vs. numpy | max diff 5.8e-3 |
| Acoustic self-reconstruction | NMSE vs. the reference's own mel | 0.38 |
| Fixed `seed` reproducibility | byte-identical WAV across runs | pass |
| Streaming chunk sum | chunks vs. `--out` | exact |
| Voice cache on vs. off | server, 7 requests per mode (repeats, a new seed, an eviction) | byte-identical |
| Voice prepared ahead vs. not | C API, `audiocpp_session_prepare` with only the voice, both modes | byte-identical |
| Long-form, 6026 chars, `text_chunk_size=200` | ~371 s of audio, Apple M3, 6 threads | offline 118 s, streaming 136 s; peak RSS 1.57 / 1.95 GB |
| `matmul_weight_type` f16 / bf16 / q8_0 | vs. f32, same `seed`, both modes | f16 0.12–0.16 dB, bf16 0.5–2.1 dB spectrum; q8_0 samples different tokens |
| Published f16 GGUF | vs. safetensors, 12 cases, same `seed`, both modes | 0.03–0.08 dB spectrum on 8 of 12, both greedy cases included; the f16 LM samples different tokens on the 4 long sampled cases |
| `orig` GGUF | vs. safetensors, same `seed`, both modes | byte-identical |

The numpy references above are independent reimplementations from the same source, which
catches implementation bugs but not a shared misreading of the architecture.

### Parity with the Python package (sopro 2.2.0)

Measured end to end against `SoproTTS.synthesize` and `SoproTTS.stream` from the published 2.2.0
wheel and the published checkpoint, CPU backend: EN and PT, single- and multi-segment text, 16-
and 32-bit references, one reference short enough to take the room-tone crop branch. For these
checks the C++ run was given the Python run's random draws (AR tokens, solver noise, room tone),
so every other stage is computed independently on each side.

| Check | Result |
|---|---|
| Reference crop, normalisation, 16 kHz resample | max sample diff 5e-7 |
| Reference semantic tokens | identical, all cases |
| Reference analysis mel | max diff 6e-5 to 1.3e-3 on 16-bit references |
| Acoustic solve (2 Euler steps), PT case | max diff 6e-4, mean 6.5e-5 (normalised mel units) |
| Vocoder + gain + trims + fades, PT case | 77.5 dB waveform SNR when fed the same solved mel |
| Greedy AR (temperature 0, C++ samples its own tokens) | identical tokens and length, 2 cases |
| Output length | identical, all cases |
| Output level | within 0.02 dB |
| Output spectrum (STFT magnitude, cells within 60 dB of peak) | mean diff 0.03–0.05 dB on 16-bit references |
| Streaming vs `SoproTTS.stream`, 5 cases (up to 47 chunks, 2 segments) | identical length and chunk boundaries |
| Streaming output spectrum | mean diff 0.03 dB on 16-bit references |

Two things to read the numbers right:

- **Compare spectra, not samples.** The vocoder predicts phase, so the waveform is very
  sensitive to its input: adding 1e-5 of noise to the normalised mel moves the Python output
  itself to ~52 dB waveform SNR, and the 6e-4 float32 difference of the solve gives ~31 dB.
  The magnitude spectra stay within a few hundredths of a dB.
- **Near-silent reference bands are float32 noise.** On two 32-bit studio references the analysis
  mel differs (up to 0.3) in cells ~30 dB below the frame, mostly near 11–12 kHz. Those cells
  sit below float32 resolution: adding 1e-8 of noise to the Python reference moves them by up
  to 0.45. Those outputs differ by 0.16–0.26 dB mean; with the C++ reference mel fed to Python
  the gap closes to 0.01–0.02 dB.

### Debugging

`tests/sopro_tts/sopro_probe.cpp` (built with `-DENGINE_BUILD_WARMBENCH=ON`) exercises each
stage in isolation against a reference clip:

```bash
build/bin/sopro_probe models/sopro-v2-turbo reference.wav /tmp/soproprobe
```

It reports the `crop_on_pause` decision, a mel round trip through the vocoder (which is
phase-invariant and so the meaningful vocoder check), the FSQ token histogram, the speaker
embedding statistics, and an acoustic self-reconstruction NMSE. It also writes
`probe_vocoder_roundtrip.wav` — the reference passed through mel then the vocoder; if that
sounds like the speaker, the whole back half of the pipeline is fine.

Setting `SOPRO_DUMP_DIR=<dir>` additionally dumps the encoder and solver intermediates as raw
f32 for diffing against a reference implementation.

## References

- Model card: <https://huggingface.co/samuel-vitorino/sopro-v2-turbo>
- Reference implementation: <https://github.com/samuel-vitorino/sopro>
- Blog post: <https://research.haloneuro.ai/posts/sopro-v2>
