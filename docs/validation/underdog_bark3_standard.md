# Underdog Bark 3.0 validation

PR #1132 adds standard Bark 3.0 to the existing `parakeet_tdt` community family,
using shared Bark support from #1130 and #1131. Tested on Windows 11, Ryzen 9
7950X3D, RTX 3090 24 GiB, MSVC 14.43, CUDA 12.4 and Vulkan SDK 1.4.350.0.
CPU-only and combined CUDA/Vulkan Release builds pass. No shared framework,
backend/kernel or multi-slot behavior is changed. Exact build/conversion/run
commands are in [the model guide](../models/underdog_bark3.md).

## Source, isolation and storage

Source: `ConwayResearch/Underdog-Bark-3.0`, pinned revision
`85db2ad18ab41e534a222ee253c29191d77ed919`. The 163,398,645-byte `model.bark`
has SHA-256 `c18de9418d73cbeb80b8c1287b392d480a56e191ef4a81d1baaa1b02e66d92ed`.
Base processor/tokenizer sidecars use NVIDIA revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b` and are checked by hash.

All **699 inference tensors** independently decoded by the native converter's
packed reader exactly match the unmodified official `load_state_dict(...,
as_numpy=True)` before compatibility transforms. The container has 723 records;
24 training-only counters are excluded. Projection/BatchNorm compensation
matches the existing native loader. Every package embeds tokenizer/configs,
English spec v1, tensor hashes/provenance, NOTICE and the full CC-BY-4.0 license.
The reader does not execute downloaded Python.

The standard profile is immutable and selects its own checkpoint, provenance,
docs and `underdog_bark3_standard` variant. Plus defaults remain unchanged;
cross-model staging is rejected. No module-global model settings are mutated.

| GGUF storage | Bytes | SHA-256 |
|---|---:|---|
| f32 | 2,509,744,800 | `b75407e7aedf5db9b784f0d6da57d6f152b5a9e27a6cbcaa51d3742d45fbffe7` |
| mixed-f16 | 1,301,648,160 | `f9062eb55154ba2e4d7bcfdf7e524db4eff448a3752963ff7490da364e3f5406` |
| q8_0 | 1,183,845,280 | `b15480349003b18fd96e4bcf3bb3747363a541ade00c354c6efe6afc015b384b` |
| bf16 | 1,300,278,432 | `6b391938a3275ceb76ac6d3a5c7dcc934a2ac69b2fb81505542868a3ecbaea10` |

The compressed download is not a native 163 MB GPU codebook implementation.
F32/mixed-F16 preserve staged weight values; mixed-F16 retains non-representable
integer/scale products, normalization and compensated projections in F32.
Protected Q8 uses Q8_0 for eligible attention matrices, exact F16 for sensitive
feed-forward/pointwise weights, and F32 for non-representable products.

The initial fully rounded BF16 profile changed capitalization/token splitting
on one clip for CPU/CUDA. Standard BF16 now retains the predictor,
encoder-to-joint projection and joint output head in F32. This corrects all
tested comparisons while leaving Plus's existing BF16 policy unchanged.
Q8/BF16 remain rounded derivatives; these results do not promise universal parity.

## Original-reference matrix

Reference environment: Python 3.11.15, PyTorch 2.8.0, Transformers 5.19.0,
NumPy 2.4.6 and constriction 0.5.0. Short references use the independently
unpacked official FP32 checkpoint on CPU. The standard reference harness also
runs the pinned, unmodified official loader on CUDA and cross-checks its local
pinned processor against official `transcribe()`. It does not load native
staging weights. All inputs use identical mono 16 kHz samples and 4000 prefix
samples, with the prefix removed from returned native timestamps.

Six fixtures: checked-in LibriSpeech `2086-149220-0033.wav` (7.435 s), four
local video excerpts (7.9925, 12, 12 and 30 s), and 1 s silence. Excerpts start
at 0, 20, 240 and 90 s in `We need to talk about 5min.mp4`; that local video is
not shipped. Each session also repeats short/long/short without reloading.

| Backend | F32 | Lossless mixed-F16 | Protected Q8 | Protected BF16 |
|---|---:|---:|---:|---:|
| CPU | 9/9 | 9/9 | 9/9 | 9/9 |
| CUDA | 9/9 | 9/9 | 9/9 | 9/9 |
| Vulkan | 9/9 | 9/9 | 9/9 | 9/9 |

All **108/108** checks match original text including punctuation/case, with word
timestamps bounded by the original recording. The original checkpoint emits
`Yeah.` on the one-second silence input; unfiltered native inference preserves
it on all tested backends. Opt-in VAD skips the non-speech input and returns
empty text. Matching the reference does not validate hallucination-free ASR,
corpus WER or independent forced-alignment timestamp accuracy.

## Five-minute offline full-context time and memory

The complete recording is 299.99375 s and its original CUDA/FP32 transcript
contains 823 whitespace-separated words. Original PyTorch CUDA takes
**2.572 s**, the median of the last two of three requests after warmup.

| Backend | GGUF | Processing time | Sampled peak extra VRAM incl. model (GiB) | Peak RSS (GiB) | Text vs original FP32 |
|---|---|---:|---:|---:|---|
| CPU | f32 | 33.646 s | 0.00 | 5.17 | Exact |
| CPU | mixed-f16 | 34.637 s | 0.00 | 4.56 | Exact |
| CPU | q8_0 | 33.664 s | 0.00 | 4.52 | Exact |
| CPU | bf16 | 34.305 s | 0.00 | 4.56 | Exact |
| CUDA | f32 | 0.801 s | 5.90 | 2.95 | Exact |
| CUDA | mixed-f16 | 0.769 s | 5.06 | 2.15 | Exact |
| CUDA | q8_0 | 0.780 s | 4.64 | 2.11 | Exact |
| CUDA | bf16 | 0.776 s | 5.07 | 2.18 | Exact |
| VULKAN | f32 | 0.885 s | 5.70 | 2.91 | Exact |
| VULKAN | mixed-f16 | 0.888 s | 4.70 | 2.10 | Exact |
| VULKAN | q8_0 | 0.900 s | 4.68 | 2.03 | Exact |
| VULKAN | bf16 | 0.934 s | 4.60 | 2.12 | Exact |

All **12/12** full-recording outputs match the original transcript exactly,
with bounded timestamps. GPU timing is the median of the last two of three
full requests after short warmup; CPU is one full request including graph build.
Times exclude model load/file I/O and include preprocessing/decoding. All runs
use eight inference threads, `OMP_NUM_THREADS=8`, `OMP_WAIT_POLICY=PASSIVE`.
RTF is below 1 for every result. No CPU thread-count tuning was performed.

NVML device total minus a pre-process baseline and process working set were
sampled every 100 ms. Memory includes the model, temporary allocations and pools;
these are sampled peaks, not weights-excluded cache sizes. Exact matches on these
recordings do not make rounded Q8/BF16 generally lossless. Mixed-F16 is the
recommended package when preserving the source weight values.

## HTTP/SSE, reuse and regressions

**66/66 server checks** pass with mixed-F16: six original references in one
server, zero/empty/enabled/spoken hotword controls, two VAD speech spans with
original-sample timestamps, VAD silence/token budget, malformed options returning
HTTP 400 followed by successful recovery, and prompted/unbiased raw-PCM streams.

Ten additional identical requests per backend after two warmups retain the
reference text and show these post-request memory ranges:

| Backend | Requests | RSS range (MiB) | Device VRAM range (MiB) |
|---|---:|---:|---:|
| CPU | 10 | 0.03 | 0.00 |
| CUDA | 10 | 1.11 | 0.00 |
| VULKAN | 10 | 1.28 | 0.00 |

These are idle post-request ranges, separate from the full-recording peak table.
Ten focused CTests pass across CPU/GPU builds. Three standard-profile, five Plus,
five Pro and thirteen Phonon converter tests pass. Loader/catalog sync and its
self-tests pass; actual stock Parakeet streaming/finalize/reset remains stable.

## Limits and artifacts

English only. Full-context attention grows with duration and retains the encoder
frame limit. Buffered streaming, bounded windows and VAD change context and may
change recognition; this is not native cache-aware streaming. VAD is offline-only.
The source checkpoint's silence hallucination is retained. Metal/HIP and corpus
WER are unvalidated. No packed ANS GPU kernels, GGUF upload or unpublished
model-manager downloads are claimed.

Local evidence: `outputs/ub3/unpack-parity.json`, `models.json`,
`reference-results.json`, `official-gpu-reference.json`, `matrix/`,
`matrix-results.json`, `full/`, `full-results.json`, `features-test/`,
`memory-reuse/`, `memory-reuse-results.json`. Initial BF16 is retained under
`precision-profiles/`; the hashes above identify the final packages.

## Main merge and shared guide/config regression

Updated against main `aeb7bcf1` after Plus (#1131) merged. Bark C++ inference
sources are unchanged from the previous Standard validation. One model guide
(`docs/models/underdog_bark3.md`) and one lazy-loaded server example
(`examples/server/underdog-bark3-asr.json`) now cover Standard, Plus and Pro;
per-variant measurement records are retained. Conversion metadata, family UI
references and validation links point to the shared guide.

CPU-only and combined CUDA/Vulkan builds pass. Ten focused CTest executions,
26 converter tests and catalog self-tests/synchronization pass. All 54
Standard/Plus mixed-F16 repeated-request comparisons pass on CPU/CUDA/Vulkan,
with exact text and word timestamps against the earlier runs. All 18 routes in
the shared config pass (three variants x offline/streaming x three backends),
including Pro's original `bark3` / `bark3-stream` IDs and exact original text for
offline requests. These are compatibility checks, not new timing measurements.
Local commands/artifacts are under `outputs/bark-native/conflict-validation-*`.
