# ZONOS3-Mini

`zonos3` is a community port of Zyphra's
[ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) text-to-speech model,
built on audio.cpp framework modules from the native ggml runtime in
[Zyphra/ZONOS3](https://github.com/Zyphra/ZONOS3) (`servers/ggml`).

The model has four parts:

- **Backbone.** 11 layers at width 1024. Layers 3 and 7 use global
  grouped-query attention; the other nine use Gated DeltaNet. Every layer is
  joined by Full AttnRes (learned attention over all earlier sublayer outputs
  instead of a residual stream). Layer 0 has a dense SwiGLU FFN; layers 1-10
  use a latent top-2-of-64 MoE. The output is 10 delayed codebooks of 4096
  codes at 15 frames per second.
- **Decoder.** The KEEP-2 MR-iSTFT decoder, which turns the codes into 44.1 kHz
  mono audio.
- **Speaker encoder.** MECT-B2, which turns a reference clip into a
  192-dimensional speaker embedding.
- **Raw-reference encoder (optional).** The ZAC encoder with a w2v-BERT
  teacher, which turns a reference clip into codes that are placed before the
  generated speech.

MoE routing, the expert matmuls and the optional expert cache all use the
framework MoE components: `MoeModule`, `DiskExpertWeightStore` and
`StagedMoeRuntime`.

## Build

The family is standalone; no other model needs to be built with it:

```bash
cmake -S . -B build/zonos3 -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=zonos3
cmake --build build/zonos3 -j 8 --target audiocpp_cli audiocpp_server
```

On macOS with Apple clang, add `-DENGINE_ENABLE_OPENMP=OFF -DGGML_OPENMP=OFF`
(Apple clang ships no OpenMP), or use `scripts/build_metal.sh`.

### Text normalization

Numbers, symbols and abbreviations are rewritten into spoken form by
[zyphra-textnorm](https://github.com/Zyphra/zyphra-textnorm), a Rust library.
It is only linked when you ask for it, because it needs `cargo`:

```bash
git clone --branch v0.1.0 https://github.com/Zyphra/zyphra-textnorm ../zyphra-textnorm
cmake -S . -B build/zonos3 ... \
  -DAUDIOCPP_ZONOS3_TEXTNORM=ON -DAUDIOCPP_ZONOS3_TEXTNORM_DIR=$PWD/../zyphra-textnorm
```

Without it, the runtime only collapses line breaks, and the first request
prints a one-line warning. Write numbers as words in that case ("forty-two
dollars", not "$42"); the model reads digits and symbols less reliably.

## Download

```bash
python3 tools/model_manager_v2.py install zonos3_mini_ptq --models-root models
```

| Package | Files | Size | Use |
|---|---|---:|---|
| `zonos3_mini_ptq` (default) | `backbone-ptq.gguf`, `zac-keep2.gguf`, `spk-encoder.gguf` | 1.16 GB | TTS and speaker-embedding cloning |
| `zonos3_mini_ptq_raw_reference` | the above + `zac-encoder.gguf`, `ssl-teacher.gguf` | 6.74 GB | adds `speaker_mode=reference` |

The files come from the `gguf/` folder of
[Zyphra/ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini). The backbone
is a post-training quantization of the release checkpoint: routed experts
Q4_0, GDN input and attention projections Q8_0, everything else F16/F32. The
decoder is F16, and the encoders are F32.

The directory holds several GGUFs, so pass the backbone file itself as
`--model`. The other files are found next to it. Run from the repository root,
or pass `--model-spec-override model_specs`, so the runtime finds
`model_specs/zonos3.json`.

## Run

```bash
M=models/ZONOS3-Mini-GGUF/backbone-ptq.gguf

# Text to speech
build/zonos3/bin/audiocpp_cli --task tts --family zonos3 --model $M --backend cpu \
  --text "Hello from audio.cpp." --seed 7 --out out.wav

# Voice cloning from a speaker embedding (default speaker_mode)
build/zonos3/bin/audiocpp_cli --task clon --family zonos3 --model $M --backend metal \
  --text "Hello from audio.cpp." --voice-ref voice.wav --out clone.wav

# Voice cloning from raw-reference codes (needs the raw-reference package)
build/zonos3/bin/audiocpp_cli --task clon --family zonos3 --model $M \
  --text "Hello from audio.cpp." --voice-ref voice.wav \
  --request-option speaker_mode=reference --out clone_ref.wav

# Streaming
build/zonos3/bin/audiocpp_cli --task tts --family zonos3 --model $M --mode streaming \
  --text "Hello from audio.cpp." --out stream.wav
```

Output is 44.1 kHz mono. Reference audio must be a WAV file. It is mixed to
mono, resampled to 44.1 kHz, and cut to its first 60 seconds.

### Voices

ZONOS3 has no built-in voices. Each request takes at most one voice source:

- `--voice-ref <wav>`, encoded by the speaker encoder (`speaker_mode=embedding`)
  or by the raw-reference encoder (`speaker_mode=reference`);
- `--request-option speaker_embedding_file=<npy>`: a float32 `[192]` or
  `[1,192]` embedding;
- `--request-option reference_codes_file=<npy>`: `[T,10]` integer or float
  codes, 0..4095. Only the first 900 frames (60 s) are used.

To encode a voice once and reuse it, use `encode_reference_only`. It writes an
`.npy` file and does not generate speech:

```bash
build/zonos3/bin/audiocpp_cli --task clon --family zonos3 --model $M \
  --voice-ref voice.wav --request-option encode_reference_only=true --out-dir voice/
# voice/zonos3_speaker_embedding.npy
# (with speaker_mode=reference: voice/zonos3_reference_codes.npy)
build/zonos3/bin/audiocpp_cli --task clon --family zonos3 --model $M --text "..." \
  --request-option speaker_embedding_file=voice/zonos3_speaker_embedding.npy --out out.wav
```

The `gguf/voices/` folder of the Hugging Face repository has preset voices.
Each comes as a WAV, a `.embedding.npy` and a `.codes.npy` file; the `.npy`
files work directly with the two file options.

Encoding a 10-second reference with the speaker encoder takes about 3 seconds
on the CPU of an Apple M3, as with ZONOS3's own encoder. To skip it, encode a
voice once with `encode_reference_only` and pass the `.npy` file afterwards. A
session also keeps its most recent reference encodings in memory
(`zonos3.voice_cache_slots`), so repeating the same `--voice-ref` on a server
skips the encoder.

### Conditioning

The prompt carries three condition rows: loudness, bandwidth (fmax) and
speaking rate. By default each is "unknown". `loudness` (LUFS), `fmax` (Hz) and
`speaking_rate_bps` (UTF-8 bytes per second) pick a bucket from the
checkpoint's bucket edges. `condition_buckets=a,b,c` sets the bucket ids
directly; an id equal to the table's bucket count means unknown.

`guidance_scale` above 1 applies classifier-free guidance on the speaker. Each step
then also runs a copy of the prompt without the speaker, which doubles the
backbone work.

### Long form

Text longer than `text_chunk_size` code points (300 by default) is split with
the framework chunker. Each chunk is generated with the same voice and
conditions and seed `seed + i`, and the chunks are concatenated.
When no voice reference is supplied, later chunks use acoustic codes from the
first chunk as a speaker reference to keep the generated voice consistent.
Explicit speaker embeddings and reference codes remain the reference for every chunk.

## Model

| Field | Value |
|---|---|
| Family | `zonos3` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | `en`, `fr`, `de`, `es`, `it`, `pt`, `ja`, `zh`, `ko`. `language` only selects the text-normalization grammar; the model takes no language input. |
| Default package | `zonos3_mini_ptq` |
| Output | 44.1 kHz mono |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `language` | `en_us`, `en_gb`, `fr_fr`, `de`, `es`, `it`, `pt_br`, `ja`, `cmn`, `ko` (aliases `en`, `fr`, `pt`, `zh`) | `en_us` | Text-normalization grammar. |
| `text_normalization` | `bool` | `true` | Rewrite numbers and symbols into spoken form (needs `AUDIOCPP_ZONOS3_TEXTNORM`). |
| `seed` | `int` | random | Sampler seed. Long-form chunk `i` uses `seed + i`. |
| `max_tokens` | `int` | `2000` | Generated 15 Hz rows per chunk, including the 11-row end-of-speech drain. |
| `temperature` | `float` | `0.75` | Sampling temperature; `0` decodes greedily. |
| `top_k` | `int` | `80` | Top-k per codebook; ties at the threshold are kept. |
| `top_p` | `float` | `0.69` | Nucleus probability per codebook. |
| `min_p` | `float` | `0.0` | Drops candidates below `min_p` times the most likely one. |
| `repetition_penalty` | `float` | `2.4` | Penalty on codes seen within the repetition window. |
| `repetition_window` | `int` | `96` | Frames of per-codebook history the penalty looks at. |
| `repetition_codebooks` | `int` | `6` | Leading codebooks the penalty applies to; `-1` applies it to all ten. |
| `guidance_scale` | `float` | `1.0` | Speaker classifier-free guidance; `1` disables it. Needs a voice. |
| `speaker_mode` | `embedding`, `reference` | `embedding` | How `--voice-ref` audio conditions the prompt. |
| `speaker_embedding_file` | path | - | float32 `.npy` speaker embedding, shape `[192]` or `[1,192]`. |
| `reference_codes_file` | path | - | `.npy` reference codes `[T,10]`, values 0..4095. |
| `encode_reference_only` | `bool` | `false` | Return the `--voice-ref` embedding (or codes) as an `.npy` artifact without generating speech. |
| `loudness` | `float` | unknown | Target loudness in LUFS. |
| `fmax` | `float` | unknown | Target bandwidth in Hz. |
| `speaking_rate_bps` | `float` | unknown | Target speaking rate in UTF-8 bytes per second. |
| `condition_buckets` | `a,b,c` | - | Explicit bucket ids; each overrides the matching value option. |
| `silence_prefill_frames` | `int` 0..30 | `3` | Silence frames placed before generation. They are not part of the output. |
| `headroom_db` | `float` 0..24 | `3.0` | Output attenuation in dB. |
| `soft_limit` | `bool` | `true` | -1 dBFS tanh soft limiter after the gain. |
| `text_chunk_size` | `int` | `300` | Maximum code points per long-form chunk. |
| `text_chunk_mode` | `word_budget`, `tag_aware`, `japanese`, `endline` | `word_budget` | Framework chunking policy. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `zonos3.encoder_backend` | `cpu`, `same` | `cpu` | Backend for the speaker and raw-reference encoders. |
| `zonos3.keep_reference_encoder` | `bool` | `false` | Keep the raw-reference encoder (about 3.4 GiB peak RSS while encoding) loaded between requests. |
| `zonos3.voice_cache_slots` | `int` | `4` | Reference encodings kept per session; `0` disables the cache. |
| `zonos3.stream_chunk_frames` | `int` 1..64 | `16` | Maximum frames per streamed decoder chunk. Chunks start at one frame and grow to this size. |
| `moe_disk_experts` | `bool` | `false` | Use the expert cache (below). |
| `moe_pin_experts` | `bool` | `false` | With `disk_experts`, hold the expert banks in pinned host memory. |
| `moe_expert_cache_size` | `int` 4..64 | `16` | Expert slots per MoE layer with `disk_experts`. |
| `moe_disk_expert_read_threads` | `int` 1..64 | `1` | Host threads copying selected experts when `pin_experts` is off. |
| `moe_cpu_expert_layers` | `int` 0..routed layers | `0` | Place the first N routed expert layers on CPU. Dense backbone operations remain on the selected backend. |
| `moe_cpu_share` | `float` 0..1 | `0` | With staged experts, share remaining cache misses with CPU after cache hits and vacant GPU slots are handled. |

The four older `zonos3.disk_experts`, `zonos3.pin_experts`,
`zonos3.expert_cache_size` and `zonos3.disk_expert_read_threads` spellings remain
accepted aliases. Do not supply both spellings of the same setting. CPU layer
placement uses routed-layer ordinals and takes precedence over fractional sharing.
With staged experts, CPU-selected layers also retain bounded expert caches.

## Expert cache

With `zonos3.disk_experts=true`, the routed expert banks stay in the
memory-mapped GGUF. For each MoE layer, the experts a step selects are copied
into a fixed number of slots (`expert_cache_size`) on the compute backend. This
uses the framework's `DiskExpertWeightStore`, and prefill uses
`StagedMoeRuntime`. Routing is unchanged. The parity test checks that the
cache's logits are bit-identical to resident experts using the same plain
expert kernels. Normal CPU-resident execution still uses packed kernels when
available; see Known limitations for their rounding differences.

The resident expert banks total only about 425 MB, so the cache is off by
default. It is for setups where that memory matters more than speed.

## Performance

Measured on an Apple M3 (8 CPU threads) with a 27-word English sentence. Wall
time covers one generation request after the model is loaded. RTF is wall time
divided by audio duration. Clone runs include encoding a 10-second `--voice-ref`
clip; the encoders run on the CPU by default.

| Backend | Mode | Wall | Audio | RTF | Peak RSS |
|---|---|---:|---:|---:|---:|
| CPU | TTS | 2.67 s | 8.87 s | 0.30 | 2.06 GiB |
| CPU | Clone (`--voice-ref`) | 5.87 s | 7.00 s | 0.84 | 2.05 GiB |
| CPU | TTS, `disk_experts` | 3.38 s | 8.80 s | 0.38 | 1.81 GiB |
| Metal | TTS | 0.97 s | 9.07 s | 0.11 | 1.71 GiB |
| Metal | Clone (`--voice-ref`) | 4.43 s | 6.93 s | 0.64 | 1.71 GiB |
| Metal | TTS, `disk_experts` | 1.58 s | 8.67 s | 0.18 | 1.42 GiB |

A 1,235-character paragraph split into 6 chunks took 36.4 s on the CPU for
73.9 s of audio (RTF 0.49). A long-lived CPU server answered five requests at
a peak RSS of 1.29 GiB. Its second request with the same `voice_ref` took 6.6 s
instead of 23.4 s because the voice cache skipped the encoder.

## Validation

`tests/community_models/zonos3/make_oracle.sh` runs the ZONOS3 native tools on
a reference voice. `zonos3_parity` compares each component against that output.
The tests are registered with CTest when the build is configured with
`-DAUDIOCPP_ZONOS3_MODEL_DIR=<gguf dir> -DAUDIOCPP_ZONOS3_PARITY_DIR=<oracle dir>`:

CUDA and Vulkan smoke runs on an NVIDIA RTX 4090 complete TTS and raw-reference
cloning from a five-second WAV, including speaker CFG and explicit condition
buckets. To run the reference encoders on the selected backend instead of the
default CPU, pass `--session-option zonos3.encoder_backend=same`.

| Test | Result (CPU) |
|---|---|
| `zonos3_parity_backbone` | 100 teacher-forced steps: minimum logit cosine 1.000000, argmax agreement 100% |
| `zonos3_parity_backbone_disk` | Logits bit-identical to the resident path with plain expert kernels |
| `zonos3_parity_speaker` | Embedding cosine 1.000000 |
| `zonos3_parity_decoder` | SNR 113.8 dB, same length, 7-frame chunks bit-identical to one pass |
| `zonos3_parity_reference` | All 145 × 10 reference codes identical |

Quality against the ZONOS3 runtime (Qwen3-ASR-1.7B WER, speaker cosine from
the MECT-B2 encoder; 4 prompts × 2 seeds, CPU):

| | ZONOS3 runtime | audio.cpp |
|---|---:|---:|
| WER, TTS | 3.08% | 3.08% |
| WER, clone | 3.08% | 3.08% |
| Speaker cosine to the reference, clone | 0.815 | 0.835 |
| WER, TTS, Metal | - | 1.54% |
| WER, TTS, Metal, `disk_experts` | - | 1.54% |

Every error is on the prompt containing "7" and "3": the speech was correct,
but the ASR transcript wrote "seven" and "three", which the scorer counts as
different words. The long-form paragraph above scored 0.45% WER.

On the CPU, given the same inputs and seed, these all produce the same audio as
ZONOS3's `zonos3-cli`: TTS, speaker-embedding cloning, raw-reference cloning
from `--voice-ref`, reference codes, speaker CFG and condition buckets. The
output has the same length and differs by at most 1 LSB of 16-bit PCM.
Streaming output equals offline output exactly.

## Known limitations

- Not implemented: phoneme input, voice-design steering, accents, emotion
  directions, LACI and speech editing.
- The CPU backbone matches ZONOS3 only when the expert banks use the CPU
  backend's repacked kernels, as ZONOS3 does. The expert cache cannot use
  repacked buffers: they only accept whole-tensor uploads. Its logits are
  bit-identical to the resident path with plain expert kernels, and they
  differ from ZONOS3's by kernel rounding (minimum per-step logit cosine
  0.99777, argmax agreement 93.2% over 100 teacher-forced steps). Sampled
  takes therefore differ from the resident path's.
- Raw-reference encoding (`speaker_mode=reference`) is slow on the CPU. For
  a 10-second clip it used about 1,740 CPU-seconds, the same as ZONOS3's own
  encoder; one run took 226 s of wall time on an 8-thread Apple M3. Peak
  RSS was about 3.4 GiB. The encoder is released after each encode unless
  `zonos3.keep_reference_encoder` is set.

## License

The weights are Apache-2.0 according to the
[model card](https://huggingface.co/Zyphra/ZONOS3-Mini). The runtime code is
ported from [Zyphra/ZONOS3](https://github.com/Zyphra/ZONOS3) (MIT, Copyright
2026 Zyphra, Inc.). zyphra-textnorm (MIT) is linked only when you build with
`AUDIOCPP_ZONOS3_TEXTNORM=ON`.
