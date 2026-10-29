# Underdog Bark 3.0 Pro validation

The later [complete-recording comparison](underdog_bark3_original_comparison.md)
adds original PyTorch CUDA and five-minute full-context evidence. In particular,
opt-in windowed long-form output differs from the original whole-context
transcript; exact native F32/BF16 whole-recording parity was established on
CUDA/Vulkan by explicitly selecting `offline_mode=full_context`.

Validated on 2026-10-29 against audio.cpp `2597b300` plus the model-local port.
Windows 11, Ryzen 9 7950X3D, 8 inference threads, RTX 3090 24 GiB;
MSVC 14.43, CUDA 12.4, Vulkan SDK 1.4.350.0. Both a CPU-only build and a combined
CUDA/Vulkan build were compiled in Release mode with Ninja (`-j 4`, CUDA SM 86).
Build, conversion and ordinary CLI/server commands are in
[the model guide](../models/underdog_bark3.md). No shared backend/kernel changes
or experimental parallel runtime are used.

## Checkpoint and packages

Original checkpoint: `ConwayResearch/Underdog-Bark-3.0-Pro`, revision
`4b5c51e04aa8d16dda9ecdff71bc29f7360149e4`; SHA-256
`6f84b33e63ebd1851e910e8b2fe299d29f5e0efc6f57e84d92d2f79af464e0e5`.
The comparison uses the unmodified BF16 source expanded to F32 by the original
Transformers runner, including its 4000-sample silence prefix.

| Standalone GGUF | Bytes | Decimal GB | SHA-256 |
|---|---:|---:|---|
| F32 | 2,509,679,520 | 2.510 | `887dea62a61d4878dd8af5c4d32fb3e4147520b2bf1b93b2711bf48ef43c5ddf` |
| BF16 | 1,255,564,960 | 1.256 | `cefa279f1d3a75fcc29da9e8eba39c3400a5b2fd5377777c14d0e2a8ed52c0b5` |
| Mixed Q8_0 | 915,914,384 | 0.916 | `6b8237f74da8e1410d33a1eae4d339992cba68baf7667484d7f9c92c583b604a` |

The package includes configs, tokenizer, embedded Bark model spec v1,
conversion provenance and upstream NOTICE/license. No Python runtime is needed
for native inference. The local package directory is
`models/Underdog-Bark-3.0-Pro-GGUF`; these are not published download entries.

## Reference parity and lifecycle

Six unique mono 16 kHz inputs were compared against Transformers 5.19.0,
PyTorch 2.8.0 CPU/F32: checked-in LibriSpeech `2086-149220-0033.wav` (7.435 s),
four English video excerpts (7.9925, 12, 12 and 30 s), and 1 s silence.
The excerpts came from the local `We need to talk about 5min.mp4` at 0, 20,
240 and 90 seconds respectively; they are not shipped with the source.
Each native session runs the six clips, then repeats short/long/short without
reloading. Comparisons explicitly select `offline_mode=full_context` and the
embedded Bark contract.

| Backend | F32 exact transcripts | BF16 exact transcripts | Q8 exact transcripts |
|---|---:|---:|---:|
| CPU | 9/9 | 9/9 | 9/9 |
| CUDA | 9/9 | 9/9 | 9/9 |
| Vulkan | 9/9 | 9/9 | 9/9 |

Every returned word span was within the original recording's sample bounds.
Silence returned empty text. These are fixture checks, not a large-corpus WER
or forced-alignment accuracy claim. For F32 CPU LibriSpeech intermediate parity:

| Boundary | Compared values | Maximum absolute error | Mean absolute error |
|---|---|---:|---:|
| Mel features | 769 x 128 | 1.6892e-4 | 8.9690e-7 |
| Encoder | 96 valid frames x 1024 | 1.6838e-6 | 2.5118e-8 |

The padded 97th encoder row is outside the valid-frame mask and is excluded.
The production encoder entry point was dumped with:

```bash
build/bark3/bin/parakeet_parity_dump \
  --model models/underdog-bark-3.0-pro-f32.gguf \
  --audio librispeech-with-4000-leading-zeros.wav --backend cpu \
  --encoder-only 1 --output-dir build/bark3/encoder-parity
```

`parakeet_parity_dump` exercises the raw encoder, bypassing the session adapter,
so its fixture alone must be pre-padded. Ordinary CLI/server callers must not
add the prefix themselves.

The Q8 legacy HTTP server passed on all three backends:

- Mixed short requests, silence, repeat requests and a subsequent short request
  after a five-minute long-form recording, using the same loaded offline model.
- Rejection of `audio_chunk_mode=vad` through `/v1/tasks/run`, then a successful
  transcription on the same session.
- Three raw-PCM chunked HTTP requests to `/v1/audio/transcriptions/live`, with
  one final SSE transcript each, exact short-clip reference text, and reuse
  after end-of-stream. Server config uses `model_spec_override: "embedded"`.
- Five-minute bounded-window transcription with identical text on CPU, CUDA
  and Vulkan; word spans remain on the unpadded recording timeline.

Weight-free C++ tests cover input/capacity prefix handling, original input
ownership, downmix/resampling, timestamp clamping, out-of-order chunk rejection,
callback/return timestamps, active-stream prepare rejection, failure/restart,
empty input and unsupported VAD. Python conversion tests cover reversible
projection scaling, BatchNorm gain compensation, architecture rejection and
variant-specific metadata. The existing stock Parakeet CPU golden offline,
long-form, stereo/fixed-window and streaming/reset checks pass. Phonon Q8's
LibriSpeech transcript and the existing 13 converter tests also pass.
Relevant variant, hotword, GGUF resource-source and model-spec CTests passed
in CPU and GPU builds; loader/catalog sync passed.

## Observed time and memory

Warm native whole-context transcription, one reused session, eight threads;
representative final requests from the mixed-length matrix, excluding load:

| Backend | Storage | 7.435 s clip | 30 s clip |
|---|---|---:|---:|
| CPU | F32 | 0.511 s | 1.745 s |
| CPU | BF16 | 0.487 s | 1.613 s |
| CPU | Q8_0 | 0.415 s | 1.390 s |
| CUDA | F32 | 0.038 s | 0.093 s |
| CUDA | BF16 | 0.037 s | 0.083 s |
| CUDA | Q8_0 | 0.039 s | 0.083 s |
| Vulkan | F32 | 0.073 s | 0.111 s |
| Vulkan | BF16 | 0.062 s | 0.106 s |
| Vulkan | Q8_0 | 0.084 s | 0.158 s |

The five-minute (299.994 s) Q8 server pass used Bark auto scheduling (now opt-in),
including its prefix. Times below are HTTP wall time, after prior requests;
memory covers the whole repeated-request/streaming pass, including model load.

| Backend | Five-minute wall time | RTF | Peak extra GPU memory, including model | Peak process RSS |
|---|---:|---:|---:|---:|
| CPU | 70.29 s | 0.2343 | 0 GiB | 2.50 GiB |
| CUDA | 3.35 s | 0.0112 | 2.03 GiB | 2.43 GiB |
| Vulkan | 4.50 s | 0.0150 | 1.77 GiB | 2.20 GiB |

GPU memory is NVML device-total usage sampled every 50 ms minus the 567.1 MiB
pre-server baseline; this includes weights, workspace, allocator caches and
backend context, and is not a weights-excluded figure. RSS is sampled process
working set. Short CUDA requests stayed around 2.0 GiB extra; after the long
request memory returned to that range. These finite-run observations do not
prove absence of every possible leak. Measurements are single-machine samples,
not a statistical performance comparison with the original Python runner.

Local artifacts: `outputs/ub3p/reference-results.json`, `native-encoder/`,
`matrix-full-context/`, `final-validation/`, `server/` and `server-embedded/`
under `outputs/ub3p`, plus `models.json` (package/payload checksums). The final
metadata-only repack verified identical tensor-payload hashes. Source and test
helpers are in the working source tree; generated weights/audio/logs are not
part of the patch.

Known limits: English only; no corpus-level accuracy certification; Metal/HIP
untested; windowed and buffered-streaming outputs need not match whole-context
decoding. Full-context attention scales with input length. VAD preprocessing
and Bark-specific hotword biasing are not admitted. See the model guide for
the exact original-sample timing and buffered-streaming contract.
