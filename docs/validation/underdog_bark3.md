# Underdog Bark 3.0 Pro validation

Validated follow-up for PR #1130, tested on Windows 11, Ryzen 9 7950X3D
(eight inference threads), RTX 3090 24 GiB, MSVC 14.43, CUDA 12.4 and Vulkan SDK
1.4.350.0. CPU-only and combined CUDA/Vulkan Release builds pass. Build,
conversion and CLI/server commands are in [the model guide](../models/underdog_bark3.md).
The [initial original-runner comparison](underdog_bark3_original_comparison.md)
records the earlier CPU failure and Q8 profile; the fixes below supersede them.
No shared framework, optimizer or backend/kernel behavior is changed.

## CPU memory fix

CPU schedule pruning removed tensor-view nodes before ggml allocation. The
allocator then lost last-use boundaries for large temporary backing buffers.
The Bark-only CPU policy retains those nodes and releases the one-shot position
projection producer after copying its outputs to the persistent graph inputs.

| Five-minute full-context CPU graph | Before | After |
|---|---:|---:|
| Graph buffer | 37,815,713,312 bytes (35.2 GiB) | 2,812,205,088 bytes (2.62 GiB) |
| Allocation reduction | | **92.56%** |
| Full original F32-reference transcript | Could not complete safely on this machine | Exact match, without chunking |

This is the graph buffer, not total process RAM. Measured total peaks are
5.17 GiB for F32 and 4.45 GiB for protected Q8. A weight-free regression builds
32 dependent cropped views, measures allocation without allocating the old
oversized layout, and executes the corrected graph to verify exact values.

## Packages and Q8 precision

Source: `ConwayResearch/Underdog-Bark-3.0-Pro`, pinned revision
`4b5c51e04aa8d16dda9ecdff71bc29f7360149e4`, checkpoint SHA-256
`6f84b33e63ebd1851e910e8b2fe299d29f5e0efc6f57e84d92d2f79af464e0e5`.
699 inference tensors are prepared; 24 training-only counters are omitted.
Tokenizer/configs, spec v1, conversion provenance, NOTICE and CC-BY-4.0 license
are embedded. No external files or Python are needed for native ASR inference.
VAD additionally loads the existing optional framework VAD model.

| Standalone GGUF | Bytes | SHA-256 |
|---|---:|---|
| f32 | 2,509,681,888 | `68b4eef71c528ab4ebbf99478bbc7198bb84d3b7bee0ac0cfe171d7f242fb339` |
| bf16 | 1,255,567,328 | `d59e8e8fa2e26d942c8a912846e4fc4f0ecf3ed8de2dc44e6e1ef44cfc51aae5` |
| q8_0 | 908,687,328 | `d3fedf4d4fea36405bfce7ec0e8c132cb2c89b393ea2f4a61f97ba19352a5c46` |

Q8 now retains `decoder.*`, `encoder_projector.*` and `joint.head.*` in their
exact source BF16 values. Conversion rejects non-BF16-representable staging
values for those tensors. It still quantizes the eligible encoder matrices;
BatchNorm compensation remains F32. The protected package is **7,227,056 bytes
smaller** than the initial 915,914,384-byte Q8 because the generic conversion
previously left some LSTM tensors F32. This is mixed Q8 storage, not a lossless
whole-model claim. Reconvert old Q8 packages to obtain the new policy/spec.

## Reference and whole-recording results

Original references use the unmodified checkpoint through Transformers 5.19.0,
PyTorch 2.8.0 CPU/F32, including its 4000-sample prefix. The publisher's
unmodified CUDA/BF16 runner was also executed; see the initial comparison.
Six unique mono 16 kHz fixtures were checked: LibriSpeech
`2086-149220-0033.wav` (7.435 s), four local English video excerpts (7.9925,
12, 12 and 30 s), and 1 s silence. The video excerpts start at 0, 20, 240 and
90 seconds in `We need to talk about 5min.mp4` and are not shipped with source.
Each native session also repeats short/long/short without reloading.

| Backend | F32 exact text / timestamp bounds | BF16 | Protected Q8 |
|---|---:|---:|---:|
| CPU | 9/9 | 9/9 | 9/9 |
| CUDA | 9/9 | 9/9 | 9/9 |
| Vulkan | 9/9 | 9/9 | 9/9 |

All **81/81** checks pass with the final full-context default. Silence returns
empty text. The initial F32 CPU intermediate comparison had maximum absolute
error 1.684e-6 over 96 valid encoder frames; mel error was 1.689e-4. The padded
97th encoder row is not valid output. These fixtures do not establish corpus WER
or forced-alignment accuracy.

The complete 299.99375 s recording now matches the original F32 transcript
**exactly, including punctuation/case, in every native result below**:

| Backend | GGUF | Processing time | Peak extra GPU memory incl. model | Peak process RSS |
|---|---|---:|---:|---:|
| CPU | f32 | 48.986 s | 0.00 GiB | 5.17 GiB |
| CPU | q8_0 | 62.621 s | 0.00 GiB | 4.45 GiB |
| CUDA | f32 | 0.859 s | 5.19 GiB | 2.97 GiB |
| CUDA | bf16 | 0.820 s | 5.03 GiB | 2.17 GiB |
| CUDA | q8_0 | 0.801 s | 4.51 GiB | 2.21 GiB |
| VULKAN | f32 | 0.899 s | 5.33 GiB | 2.91 GiB |
| VULKAN | bf16 | 0.908 s | 5.04 GiB | 2.12 GiB |
| VULKAN | q8_0 | 0.892 s | 4.61 GiB | 2.11 GiB |

GPU timing is the median of the last two of three full requests after a short
warmup; CPU timing is one full request and includes its full-graph build. Both
exclude model load/file I/O and include preprocessing/decoding. CPU times varied
between runs; this is a memory/quality fix, not a CPU speedup claim. Original
PyTorch CUDA timings were 2.436 s F32 and 2.049 s BF16 on the same recording.
NVML device-total memory was sampled every 100 ms minus a pre-process baseline;
RSS is sampled process working set. Memory includes model and temporary buffers.

## Buffered streaming frontend reuse

Bark streams now cache one window of **unnormalized** log-mel features. Only
frames with their full FFT/preemphasis support inside both windows are reused.
Window edges are recomputed, and F64 normalization is repeated over the entire
current window in the original reduction order. An unaligned hop grid or absent
overlap falls back to fresh extraction; starting/resetting a stream clears the
cache. Default raw-cache data is about 0.68 MiB of host RAM (1401 x 128 floats),
plus frontend metadata, with no additional GPU tensors. This is Bark-only;
stock Parakeet/Phonon, offline inference, attention and shared kernels retain
their existing paths. No GGUF reconversion is required.

The model-local unit test compares every normalized float bit-for-bit against
fresh extraction for growing/sliding windows, boundary impulses, silence,
unaligned windows/tails, absent overlap and zero/one-valid-frame cases. Actual
streaming before/after comparisons pass **81/81** checks: the six fixtures above
plus repeated short/long/short, for CPU/CUDA/Vulkan x F32/BF16/Q8. Text and word
timestamps match exactly within each backend/precision. The complete five-minute
Q8 stream also retains exactly the same text/timestamps on all three backends.
These comparisons use the prior buffered implementation at `889e3161`, not the
original model's different whole-recording context. Both warm-bench binaries
include the `start_stream()` lifecycle correction; the previous benchmark only
called `reset()`, which cannot activate the Bark stream wrapper.

The 66 HTTP/SSE checks below were repeated successfully with this path, including
hotwords and stream reuse. The actual stock Parakeet streaming/reset regression
also passes. Default 2 s centers, 10 s left context and 2 s right context are
unchanged. Processing-time improvements do not reduce the amount of future
audio required for an output, and this remains buffered bidirectional inference.

| Five-minute Q8 buffered stream | Before | After | Less processing time |
|---|---:|---:|---:|
| CUDA | 3.154 s | 2.735 s | **13.3%** |
| Vulkan | 4.451 s | 3.911 s | **12.1%** |

These are medians of five streams after one warmup, with identical default
contexts, 3200-sample input chunks, eight inference threads and
`OMP_NUM_THREADS=8`, `OMP_WAIT_POLICY=PASSIVE` for both binaries. Input is fed
without real-time delays. Timing includes preprocessing/decoding and excludes
model load/file I/O. Logged frontend time falls by about 73-76%. Use the
`controlled-before/` and `controlled-after/` artifacts for these measurements;
unrestricted-thread exploratory trials were sensitive to background VM load.
A separate single-stream CPU trial without warmup, using the original automatic
frontend thread settings, measured 71.377 -> 69.696 s. That small CPU difference
is indicative only; encoder computation dominates its runtime.

Reproduce a before/after sequence with the same model and audio on each binary:

```bash
python tests/parakeet_tdt/compare_bark3_streaming.py \
  --before-bin build-before/bin/parakeet_warm_bench \
  --after-bin build/bark3/bin/parakeet_warm_bench \
  --model models/underdog-bark-3.0-pro-q8_0.gguf --backend cuda \
  --audio short-16k-mono.wav long-16k-mono.wav short-16k-mono.wav \
  --output-dir build/bark3/streaming-comparison
```

## VAD, hotwords and lifecycle

There are **66 passing server checks**, 22 per backend on CPU/CUDA/Vulkan:

- The six reference recordings in one long-lived Q8 session.
- Zero-score and explicit-empty hotwords preserve the original transcript;
  enabled word lists and word/spoken variants run. Raw-PCM HTTP/SSE streams
  accept a vocabulary prompt and then an unbiased request in the same server.
- Two speech clips separated by silence return both reference transcripts
  through offline Silero VAD, with word offsets inside the original recording.
  Each independent span receives its own model prefix. Silence-only VAD returns
  empty text, and max-token limits are respected across spans.
- Malformed hotwords, excessive scores, zero/non-finite VAD duration and unknown
  chunk mode return HTTP 400; a following valid request succeeds.

A separate default-path Silero CLI check reproduces LibriSpeech exactly. A
bounded high-strength positive control forces a requested new token sequence,
verifying hotword selection is active; it is not a recommended quality setting.
VAD/context changes and intentional biasing need not match the unbiased whole
recording. These are native features, not features of the publisher's runner.

Relevant CTests pass in CPU/GPU builds: model-local prefix/streaming/resampling,
allocation-lifetime/value checks, variant config, hotword policy, GGUF resource
source and model spec. Five Bark converter checks pass. Stock Parakeet golden
offline/long-form/stereo/fixed-window and streaming/reset regressions pass;
Phonon's unchanged converter checks and loader/catalog sync are also checked.

## Limits and artifacts

English only is the checkpoint's declared training/support scope. Its unlimited
bidirectional attention and whole-utterance normalization prevent standard
native KV caching from preserving old encoder activations as audio grows;
streaming remains buffered. Full-context attention still grows with duration
and retains the existing encoder frame limit. VAD is offline-only. Quantization
can differ on other recordings despite exact matches here. Metal/HIP and
large-corpus accuracy are untested.

Current local evidence is under `outputs/ub3p/`: `memory-fix/final/` (whole-file
runs and graph-buffer traces), `memory-fix/matrix/` (81 reference checks),
`features-test/` (66 HTTP/SSE checks), `models-features.json`, and `q8-profiles/`
(profile comparisons and old Q8 backup). `original-comparison/` retains the
original reference outputs and baseline timing. Changes are model-local and
were kept separate from shared backend/kernel work.
`streaming-optimization/` contains the streaming baseline/updated benchmarks,
81 comparison checks in `matrix/`, and the repeated HTTP/SSE checks in
`features-test/`.
