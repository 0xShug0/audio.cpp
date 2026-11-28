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
`reference_duration_sec`, and level-normalised before the speaker and semantic encoders see it. That
normalisation is boost-only and peak-guarded (sopro 2.2): a reference already at or above the
−19.8 dB prompt level is passed through untouched, and a boost is never large enough to push
the peak past 0.95. The output gain is fixed by the level the reference ends up at: it maps that
level to −23 dB (`output_gain`), the same for every segment and in both run modes.

## Model

| Field | Value |
|---|---|
| Family | `sopro_tts` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | `en`, `pt`, `fr`, `de` |
| Default package | `sopro_v2_turbo_safetensors` |
| Input | Text; voice conditioning as shown in the examples |
| Output | Generated speech audio |

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
| `language` | `string` | see description | Language tag (en, pt, fr, de); omitted inherits transcript language, then the session default. |
| `temperature` | `float` | see description | Semantic sampling temperature; nonpositive selects greedy decoding. Omitted uses checkpoint generation.temperature. |
| `top_p` | `float` | see description | Semantic nucleus cutoff; values outside [0,1] are disabled/clamped by the sampler. Omitted uses checkpoint generation.top_p. |
| `top_k` | `int` | see description | Semantic top-k; nonpositive disables filtering. Omitted uses checkpoint generation.top_k. |
| `num_inference_steps` | `int` | see description | Positive acoustic sampling steps; omitted uses checkpoint generation.steps. Declared range: minimum 1. |
| `max_duration_sec` | `float` | see description | Positive maximum segment duration; must be >= min_duration_sec. Omitted uses checkpoint generation.max_seconds. |
| `min_duration_sec` | `float` | see description | Nonnegative minimum duration; must be <= max_duration_sec. Omitted uses checkpoint generation.min_seconds. Declared range: minimum 0. |
| `reference_duration_sec` | `float` | see description | Positive reference window in seconds; omitted uses checkpoint generation.ref_seconds. |
| `text_chunk_size` | `int` | see description | Positive codepoint budget per segment; omitted uses checkpoint generation.max_segment_chars. Declared range: minimum 1. |
| `seed` | `int` | *(random)* | Non-negative seed for semantic sampling and the acoustic noise prior; omit for a random seed. Declared range: minimum 0. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `sopro_tts.language` | `string` | `` | Default language tag for requests that do not set one. |
| `sopro_tts.voice_cache_slots` | `int` | `0` | Reference voices kept prepared between requests, so a repeated voice skips the encoders and, when streaming, the prompt solve. Output is identical with or without it. A streamed 10 s voice holds about 60 MB. Default 0 (off). Declared range: minimum 0. |
| `sopro_tts.matmul_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `f32` | Storage type for the matmul weights of every stage (semantic LM, acoustic DiT, encoders, vocoder head). |
| `sopro_tts.conv_weight_type` | `native`, `f32`, `f16` | `f32` | Storage type for convolution weights (speaker/semantic encoders and the Vocos backbone). |

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
10 s reference), and each later 0.68 s chunk takes about 0.2 s. `reference_duration_sec` scales the prompt
solve.

### Voice cache

`sopro_tts.voice_cache_slots=<n>` (off by default) keeps the last `n` prepared voices, keyed by
the clip and `reference_duration_sec`, together with each streamed voice's solved prompt, like reusing a
`prepare_reference` result upstream. A repeated voice skips the encoders and the prompt solve,
which brought the first chunk down from ~1.8 s to ~0.4–0.5 s in the setup above. A streamed
10 s voice holds about 60 MB.

To prepare a voice without generating anything, pass `audiocpp_session_prepare` a request that
carries only the voice audio (plus `reference_duration_sec` or `num_inference_steps` if the requests that
follow set them).

The cache never changes the audio: the draws a voice needs for itself (room tone for a clip
without a usable pause, the prompt noise) are seeded from the clip rather than the request.

## Usage Notes

Numeric generation defaults come from the checkpoint's `config.json` `generation` block, so a
retrained variant picks up its own values without a code change. Language inherits the transcript
language, then the session default; an omitted seed is random.

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

## References

- Model card: <https://huggingface.co/samuel-vitorino/sopro-v2-turbo>
- Reference implementation: <https://github.com/samuel-vitorino/sopro>
- Blog post: <https://research.haloneuro.ai/posts/sopro-v2>
