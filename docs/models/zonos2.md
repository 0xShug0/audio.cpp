# ZONOS2

`zonos2` is a community port of Zyphra's
[ZONOS2](https://huggingface.co/Zyphra/ZONOS2) text-to-speech model, built on
audio.cpp framework modules and validated against Zyphra's ggml runtime
[Zyphra/zonos2.cpp](https://github.com/Zyphra/zonos2.cpp).

The model has three parts:

- **Backbone.** 28 pre-norm layers at width 2048 with gated grouped-query
  attention (16 query heads, 4 KV heads, QK-norm with a learned per-head
  temperature, interleaved RoPE). Layers 0-2 and 27 have a dense SwiGLU FFN;
  layers 3-26 are 16-expert MoE layers (top-1, top-2 on layer 26) whose router
  is an MLP over a state carried from one MoE layer to the next (EDA). The
  output is 9 delayed codebooks of 1024 codes at 86.13 frames per second.
- **Decoder.** The Descript Audio Codec 44.1 kHz decoder, which turns the
  codes into 44.1 kHz mono audio (512 samples per frame).
- **Speaker encoder.** An ECAPA-TDNN that turns a 24 kHz reference clip into a
  2048-dimensional speaker embedding.

MoE routing, the expert matmuls and the optional expert cache all use the
framework MoE components: `MoeModule`, `DiskExpertWeightStore` and
`StagedMoeRuntime`.

## Build

The family is standalone; no other model needs to be built with it:

```bash
cmake -S . -B build/zonos2 -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=zonos2
cmake --build build/zonos2 -j 8 --target audiocpp_cli audiocpp_server
```

On macOS with Apple clang, add `-DENGINE_ENABLE_OPENMP=OFF -DGGML_OPENMP=OFF`
(Apple clang ships no OpenMP), or use `scripts/build_metal.sh`.

## Download

```bash
python3 tools/model_manager_v2.py install zonos2_q8_0 --models-root models
```

| Package | Files | Size | Use |
|---|---|---:|---|
| `zonos2_q8_0` (default) | `zonos2-q8_0.gguf`, `dac.gguf`, `spk-encoder.gguf` | 8.82 GB | TTS and voice cloning |

The files come from
[Zyphra/ZONOS2-GGUF](https://huggingface.co/Zyphra/ZONOS2-GGUF). The backbone
keeps its attention, router, embedding and head weights in F16 and its routed
experts in Q8_0. The repository also has F16, Q6_K, Q5_K and Q4_K backbones;
they load the same way when `--model` points at them, but only Q8_0 has been
validated here.

The directory holds several GGUFs, so pass the backbone file itself as
`--model`. The other files are found next to it. Run from the repository root,
or pass `--model-spec-override model_specs`, so the runtime finds
`model_specs/zonos2.json`.

## Run

```bash
M=models/ZONOS2-GGUF/zonos2-q8_0.gguf

# Text to speech
build/zonos2/bin/audiocpp_cli --task tts --family zonos2 --model $M --backend metal \
  --text "Hello from audio.cpp." --seed 7 --out out.wav

# Voice cloning
build/zonos2/bin/audiocpp_cli --task clon --family zonos2 --model $M --backend metal \
  --text "Hello from audio.cpp." --voice-ref voice.wav --out clone.wav

# Streaming
build/zonos2/bin/audiocpp_cli --task tts --family zonos2 --model $M --backend metal --mode streaming \
  --text "Hello from audio.cpp." --out stream.wav
```

Output is 44.1 kHz mono. Reference audio must be a WAV file; it is mixed to
mono and resampled to 24 kHz for the speaker encoder.

The model reads raw UTF-8 bytes. There is no text normalization: write
numbers, dates, units and symbols as words ("forty-two dollars", not "$42").

### Voices

ZONOS2 has no built-in voices. Each request takes at most one voice source:

- `--voice-ref <wav>`, encoded by the speaker encoder;
- `--request-option speaker_embedding_file=<npy>`: a float32 `[2048]` or
  `[1,2048]` embedding, for example from Zyphra's `spk-encoder-cli`.

To encode a voice once and reuse it, use `encode_reference_only`. It writes an
`.npy` file and does not generate speech:

```bash
build/zonos2/bin/audiocpp_cli --task clon --family zonos2 --model $M \
  --voice-ref voice.wav --request-option encode_reference_only=true --out-dir voice/
# voice/zonos2_speaker_embedding.npy
build/zonos2/bin/audiocpp_cli --task clon --family zonos2 --model $M --text "..." \
  --request-option speaker_embedding_file=voice/zonos2_speaker_embedding.npy --out out.wav
```

A session also keeps its most recent reference encodings in memory
(`zonos2.voice_cache_slots`), so repeating the same `--voice-ref` on a server
skips the encoder.

### Conditioning

The prompt can carry conditioning rows before the text:

- With a voice, a background row says whether the reference is clean
  (`clean_speaker_background=true`) or noisy (the default), and an accurate-mode
  row is added unless `accurate_mode=false`, which gives a more expressive,
  less literal read.
- `speaking_rate_bucket` (0-7) adds a speaking-rate row.
- `quality_buckets` adds one row per listed recording-quality feature. The
  default is `trailing_silence_s:3`; `none` removes every quality row. Features
  and bucket ranges: `lufs` 0-11, `estimated_snr` 0-11, `max_pause` 0-11,
  `estimated_bandlimit_hz` 0-7, `leading_silence_s` 0-7,
  `trailing_silence_s` 0-7.

### Long form

Text longer than `text_chunk_size` code points (300 by default) is split with
the framework chunker. Each chunk is generated with the same voice and
conditioning and seed `seed + i`, and the chunks are concatenated.

## Model

| Field | Value |
|---|---|
| Family | `zonos2` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | `en`, `fr`, `de`, `es`, `it`, `pt`, `ja`, `zh`, `ko` (the training languages; the model takes no language input) |
| Default package | `zonos2_q8_0` |
| Output | 44.1 kHz mono |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `seed` | `int` 0..4294967295 | random | Sampler seed. Long-form chunk `i` uses `seed + i`. |
| `max_tokens` | `int` >= 1 | `2000` | Generated frames per chunk (86.13 frames/s), including the 10-frame end-of-speech drain. |
| `temperature` | `float` >= 0 | `1.15` | Sampling temperature; `0` decodes greedily. |
| `top_k` | `int` >= 1 | `106` | Top-k per codebook; ties at the threshold are kept. |
| `top_p` | `float` 0..1 | `0.0` | Nucleus probability per codebook; `0` disables it. |
| `min_p` | `float` 0..1 | `0.18` | Drops candidates below `min_p` times the most likely one. |
| `repetition_penalty` | `float` >= 1 | `1.2` | Penalty on codes seen in the repetition window, once per distinct code. |
| `repetition_window` | `int` >= 0 | `50` | Generated frames of per-codebook history; `0` disables the penalty. |
| `repetition_codebooks` | `int` -1..9 | `8` | Leading codebooks the penalty applies to; `-1` applies it to all nine. |
| `speaker_embedding_file` | path | - | float32 `.npy` speaker embedding, shape `[2048]` or `[1,2048]`. |
| `encode_reference_only` | `bool` | `false` | Return the `--voice-ref` embedding as an `.npy` artifact without generating speech. |
| `clean_speaker_background` | `bool` | `false` | Mark the reference as clean instead of noisy (cloning only). |
| `accurate_mode` | `bool` | `true` | Accurate (on) or expressive (off) marker row (cloning only). |
| `speaking_rate_bucket` | `int` 0..7 | none | Adds a speaking-rate row. |
| `quality_buckets` | `feature:bucket,...` or `none` | `trailing_silence_s:3` | Recording-quality rows; features and ranges under Conditioning. |
| `fade_out_ms` | `float` 0..2000 | `0` | Cosine fade over the last N ms of the output. |
| `text_chunk_size` | `int` >= 1 | `300` | Maximum code points per long-form chunk. |
| `text_chunk_mode` | `word_budget`, `tag_aware`, `japanese`, `endline` | `word_budget` | Framework chunking policy. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `zonos2.encoder_backend` | `cpu`, `same` | `cpu` | Backend for the speaker encoder. |
| `zonos2.voice_cache_slots` | `int` >= 0 | `4` | Reference encodings kept per session; `0` disables the cache. |
| `zonos2.stream_chunk_frames` | `int` 1..256 | `40` | Maximum frames per streamed decoder chunk. Chunks start at one frame and grow to this size. |
| `zonos2.stream_context_frames` | `int` 16..64 | `24` | Decoder context frames on each side of a streamed chunk; 16 or more is seam-free. |
| `zonos2.disk_experts` | `bool` | `false` | Use the expert cache (below). |
| `zonos2.pin_experts` | `bool` | `false` | With `disk_experts`, hold the expert banks in pinned host memory. |
| `zonos2.expert_cache_size` | `int` 4..16 | `8` | Expert slots per MoE layer with `disk_experts`. |
| `zonos2.disk_expert_read_threads` | `int` 1..64 | `1` | Host threads copying selected experts when `pin_experts` is off. |

## Expert cache

With `zonos2.disk_experts=true`, the routed expert banks (about 7.7 GB of the
8.5 GB Q8_0 file) stay in the memory-mapped GGUF. For each MoE layer, the
experts a step selects are copied into a fixed number of slots
(`expert_cache_size`) on the compute backend. This uses the framework's
`DiskExpertWeightStore`, and prefill uses `StagedMoeRuntime`. Routing is
unchanged. Every weight, resident or cached, uses the backend's plain buffers,
so the cache's logits are bit-identical to resident experts: the parity test
checks this, and on the CPU a seeded take with the cache equals the resident
take sample for sample.

On an Apple M3 with 16 GB of RAM, the peak memory footprint of a Metal request
was 8.6 GB resident and 5.2 GB with the cache (CPU resident: 9.3 GB). The
cache is slower: on Metal, generation took 4.4 times as long per frame.

## Performance

Measured on an Apple M3 (16 GB, 4 performance cores) with the Q8_0 package and
a 14-word sentence, while the machine was already using its swap. Wall time
covers one request after the model is loaded; RTF is wall time divided by
audio duration. "Frames/s" is the backbone generation rate (real time is 86.1).

| Backend | Mode | Frames/s | Wall | Audio | RTF | Peak footprint |
|---|---|---:|---:|---:|---:|---:|
| CPU, 4 threads | TTS | 26.6 | 27.8 s | 5.42 s | 5.13 | 9.3 GB |
| Metal | TTS | 42.0 | 16.4 s | 6.84 s | 2.39 | 8.6 GB |
| Metal | Clone (`--voice-ref`) | 46.3 | 13.9 s | 6.05 s | 2.30 | 8.8 GB |
| Metal | TTS, `disk_experts` | 9.6 | 54.1 s | 5.77 s | 9.38 | 5.2 GB |

For comparison, Zyphra's `zonos2-cli` generated 25.7 frames/s on the CPU (4
threads) and 47.6 frames/s on Metal on the same machine. Encoding a 6-second
reference took 0.18 s. The DAC decode of a 600-frame take took 2.1 s on Metal
and about 10 s on the CPU.

On the CPU, 8 threads were slower than 4 (19.7 frames/s): the M3 has four
performance cores.

## Validation

`tests/community_models/zonos2/make_oracle.sh` runs Zyphra's zonos2.cpp tools
on a reference voice, and compiles `oracle_decode.cpp` against its build tree
to record its own teacher-forced KV-cache decode. `zonos2_parity` compares each
component against that output. The tests are registered with CTest when the
build is configured with
`-DAUDIOCPP_ZONOS2_MODEL_DIR=<gguf dir> -DAUDIOCPP_ZONOS2_PARITY_DIR=<oracle dir>`:

| Test | Result (CPU) |
|---|---|
| `zonos2_prompt_test` | Prompt rows equal `zonos2-cli --build-prompt`, with and without a speaker |
| `zonos2_parity_backbone` | Prefill + 100 teacher-forced steps: logits bit-identical to zonos2.cpp (max difference 0) |
| `zonos2_parity_backbone_speaker` | The same with a cloned speaker: bit-identical |
| `zonos2_parity_backbone_disk` | Expert cache bit-identical to resident experts |
| `zonos2_parity_decoder` | SNR 59.1 dB against `dac-cli` after 16-bit quantization, same length; 40-frame windows with 24 context frames identical to one pass |
| `zonos2_parity_speaker` | Embedding cosine 1.000000 against `spk-encoder-cli` |

CUDA and Vulkan smoke runs on an NVIDIA RTX 4090 complete TTS and voice cloning
with the Q8_0 package and a five-second reference WAV. The speaker encoder runs
on the selected backend with `--session-option zonos2.encoder_backend=same`.
Vulkan cloning was verified with a Vulkan-only build (see Known limitations).

With greedy decoding and no repetition penalty, the generated codes equal
`zonos2-cli --greedy` frame for frame over 2,000 frames, and the audio differs
by at most 3 LSB of 16-bit PCM.

Quality against zonos2.cpp (Qwen3-ASR-1.7B WER over 4 English prompts; speaker
cosine from the ECAPA encoder against the reference clip; CPU):

| | zonos2.cpp | audio.cpp |
|---|---:|---:|
| WER, TTS (4 seeds) | 0.37% | 4.10% |
| WER, clone (2 seeds) | 0.00% | 0.00% |
| Speaker cosine, clone | 0.964 | 0.968 |
| WER, TTS, Metal (2 seeds) | - | 2.99% |

The audio.cpp TTS figure is one take out of 16: seed 2 of the shortest prompt
repeated the sentence and ran to 20 s. The other 15 takes were word-perfect.
Both runtimes use the same backbone (bit-identical logits) but different
samplers (see Known limitations), so their seeded takes differ.

A 1,284-character paragraph (234 words) split into 7 chunks on Metal produced
70.5 s of audio. The transcript was complete except for one sentence of 29
words that one chunk dropped, which gives 14.5% WER; numbers written as words
came back as digits, which the scorer also counts.

On the CPU, streaming output equals offline output exactly, cloning from an
`.npy` embedding equals cloning from the clip, and the expert cache equals the
resident path.

A long-lived CPU server answered five requests (three plain, two with
`voice_ref`, one streamed as SSE with 15 audio deltas) at a peak memory
footprint of 10.1 GB.

## Known limitations

- Not implemented: emotion directions, speaker blending, the upstream
  long-form continuation (prefix-conditioned chunks), classifier-free guidance
  and text normalization.
- Sampling follows the upstream Python sampler (repetition penalty applied
  once per distinct code over generated frames only, ties kept at the top-k
  threshold), not zonos2.cpp's. Fixed seeds therefore do not reproduce
  zonos2.cpp's sampled audio; greedy decoding matches it.
- Long-form text is generated as independent chunks, so the voice and pacing
  can drift between chunks, and a chunk can occasionally drop or repeat a
  sentence.
- The Q8_0 backbone needs about 9 GB. On a 16 GB machine that is already under
  memory pressure, generation slows down several times; `zonos2.disk_experts`
  lowers the footprint to about 5 GB at a large speed cost.
- On the tested NVIDIA 550.144.03 driver stack, a build with both CUDA and
  Vulkan enabled can segfault in a `[vkps] Update` thread during CLI shutdown
  after Vulkan cloning. A Vulkan-only build exits cleanly; configure it with
  `-DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=ON`.

## License

Weights: Apache-2.0 ([Zyphra/ZONOS2-GGUF](https://huggingface.co/Zyphra/ZONOS2-GGUF)).
The runtime was written for audio.cpp and numerically validated against
[Zyphra/zonos2.cpp](https://github.com/Zyphra/zonos2.cpp).
