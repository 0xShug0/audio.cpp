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

The upstream safetensors checkpoint runs directly — no conversion step:

```bash
python3 tools/model_manager_v2.py install sopro_v2_turbo_safetensors
# -> models/sopro-v2-turbo/{config.json,tokenizer.model,*.safetensors}
```

To run from GGUF instead, pack the four stages into one file. `audiocpp_gguf` takes one
namespaced input per stage, and `--root` makes it embed `config.json` and `tokenizer.model` as
sidecars, so the resulting `.gguf` is self-contained:

```bash
build/bin/audiocpp_gguf \
    --input model=models/sopro-v2-turbo/model.safetensors \
    --input semantic_encoder=models/sopro-v2-turbo/semantic_encoder.safetensors \
    --input speaker_encoder=models/sopro-v2-turbo/speaker_encoder.safetensors \
    --input vocoder=models/sopro-v2-turbo/vocoder.safetensors \
    --family sopro_tts --root models/sopro-v2-turbo \
    --output models/sopro-v2-turbo-GGUF/sopro-v2-turbo-f16.gguf --type f16
```

No public audio.cpp GGUF build of this family is published yet, so the spec's default
`sopro_v2_turbo_f16` package has `download.kind = "unsupported"` and expects the file above to
be produced locally.

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
`ref_seconds`, and level-normalised before the speaker and semantic encoders see it.

## Streaming

`--mode streaming` emits one pull event per text segment instead of one buffer at the end:

```bash
build/bin/audiocpp_cli \
    --task tts --family sopro_tts \
    --model models/sopro-v2-turbo \
    --backend cpu --threads 8 --mode streaming \
    --text "$(cat article.txt)" \
    --voice-ref ref.wav --language en \
    --text-chunk-size 120 \
    --out stream.wav --out-dir segments/
```

Each event carries a `segment_<n>` named audio buffer that is already levelled, trimmed and
faded, so a consumer can play events back to back; `--out` still writes the whole utterance,
and it is exactly the concatenation of the events. The reference voice is encoded once in
`start_stream`, so every event after the first costs only its own LM, solver and vocoder pass.

**Granularity is one text segment, not one frame.** The acoustic DiT and the Vocos vocoder both
see a whole span at once, and this checkpoint ships no causal vocoder, so a segment is the
smallest unit that can leave without boundary artefacts. `text_chunk_size` is the latency dial:
on a 16-core CPU build at 8 threads, 8 solver steps and a 14 s reference, `text_chunk_size=120`
put the first audio out at ~3.1 s for a 6.7 s segment, against ~9.9 s for the same text offline.

Two things to know before turning it down further:

- Every segment re-solves the *whole* reference mel prompt alongside its own frames, so the
  per-segment cost has a floor of roughly `ref_seconds` worth of DiT work. Segments shorter
  than about 1 s take longer to generate than to play, even though the stream as a whole stays
  ahead of real time (`text_chunk_size=40` measured 0.75 RTF overall, with the shortest
  segment at 2.1). Lowering `ref_seconds` shrinks that floor at some cost to cloning fidelity.
- Streaming reproduces the offline waveform for the same `seed`, sample count included, with
  one deliberate exception: offline level-matches over the finished utterance, which a stream
  cannot see, so the first segment fixes the gain for the rest. The measured difference is a
  constant scale factor (1.15x, +1.2 dB, on the clip above) with a −47 dB residual.

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

| Load option | Default | Meaning |
|---|---|---|
| `sopro_tts.matmul_weight_type` | `f32` | Storage type for matmul weights (`native`, `f32`, `f16`, `bf16`, `q8_0`) |
| `sopro_tts.conv_weight_type` | `f32` | Storage type for convolution weights (`native`, `f32`, `f16`) |

Every default comes from the checkpoint's `config.json` `generation` block, so a retrained
variant picks up its own values without a code change.

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
   half-rotation RoPE. The prompt is `[style prefix | text | carried tokens | BOS]`. Because the
   only structural difference from a Qwen3 decoder is a LayerScale vector on each residual
   branch, and those branches end in a bias-free projection, the scale is folded into that
   projection's rows at load time and the shared `QwenCausalDecodeRuntime` runs the stack
   unmodified. The eight-query style prefix cross-attention runs on the host.
5. **Acoustic head + vocoder** (`acoustic.cpp`, `vocoder.cpp`) — an 8-block adaptive-layer-norm
   DiT solving a rectified flow in two Euler steps on a sway-sampled time grid, with the prompt
   mel re-pinned after every step; then a 14-layer Vocos ConvNeXt backbone and one centred
   ISTFT. `mu` (the upsampled semantic conditioning) is built in its own graph because it is
   constant across solver steps.

Two implementation details worth knowing:

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
  overwrote them. `SoproAcousticGraphs::upload_constants` re-uploads all of them before every
  compute; it costs a few hundred kB per step against a multi-GFLOP DiT pass.

## Known limitations

- **Streaming is segment-level, not frame-level.** The upstream frame-level path (chunked DiT
  attention plus the causal vocoder, `vocoder_streaming.safetensors`) is not implemented, and
  that vocoder is not part of the published checkpoint this family loads. What ships is one
  pull event per text segment; see [Streaming](#streaming) for the latency it actually buys.
- **Sampling RNG is not torch-bit-exact.** `sample_next_token` reproduces the reference's
  masking, temperature, top-k and top-p arithmetic exactly, but draws from a seeded
  `std::mt19937_64` rather than torch's generator, so a given `seed` will not reproduce the
  Python output sample-for-sample. The same `seed` is reproducible within audio.cpp.
- **No `int8` AR path.** The upstream `--int8` CPU option has no equivalent; use
  `sopro_tts.matmul_weight_type=q8_0` instead.
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
| Vocos backbone + ISTFT head | vs. numpy, all 14 blocks | max diff 1.0e-5 |
| Semantic encoder mel | vs. numpy | max diff 1.9e-5 |
| Semantic encoder transformer | vs. numpy, all 6 layers | max diff 6.0e-5 |
| FSQ token ids | vs. numpy | 188/188 identical |
| Speaker encoder mel / trunk / heads | vs. numpy | max diff 3.7e-5 |
| Acoustic `mu`, `spk`, time embedding | vs. numpy | max diff 1.3e-5 |
| Acoustic velocity field, every Euler step | vs. numpy | max diff 5.8e-3 |
| Acoustic self-reconstruction | NMSE vs. the reference's own mel | 0.38 |
| Fixed `seed` reproducibility | byte-identical WAV across runs | pass |
| Streaming vs. offline, same `seed` | 22.5 s clip, 4 segments | identical sample count; a constant 1.15x gain, −47 dB residual |
| Streaming segment sum | segments vs. `--out` | exact |
| Long-form, 6026 chars | 371.6 s of audio, 48 segments | offline and streaming both complete; peak RSS 1.083 vs 1.086 GB |
| `matmul_weight_type` f16 / bf16 / q8_0 | runs clean | pass |
| Single-file GGUF | end to end | pass |

Numeric parity against the upstream PyTorch implementation has still not been measured
directly; the numpy references above are independent reimplementations from the same source,
which catches implementation bugs but not a shared misreading of the architecture.

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
