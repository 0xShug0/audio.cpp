# Underdog Bark 3.0 Plus validation

PR #1131 extends the existing `parakeet_tdt` community family using the shared
Bark support in #1130. Tested on Windows 11, Ryzen 9 7950X3D, RTX 3090 24 GiB,
MSVC 14.43, CUDA 12.4 and Vulkan SDK 1.4.350.0. CPU-only and combined CUDA/Vulkan
Release builds pass. No shared backend/kernel or multi-slot changes are included.
Build/run commands are in [the model guide](../models/underdog_bark3.md).

## Source and reconstruction

Source: `ConwayResearch/Underdog-Bark-3.0-Plus`, revision
`d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb`. The 179,524,845-byte `model.bark`
has SHA-256 `fdd9f1052e4fdead07d71e8cd2138813eb0c209fd6ff8b1b6f47737e1e501857`.
Processor/tokenizer sidecars are pinned to NVIDIA revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b` and checked by hash.

All **699 inference tensors** independently decoded by the new converter match
the unmodified official `bark_loader.load_state_dict(..., as_numpy=True)`
exactly, before the documented projection/BatchNorm compatibility transforms.
The 723-record container also has 24 training-only counters, which are dropped.
The converter does not execute downloaded Python. Every GGUF embeds spec v1,
tokenizer/configs, tensor hashes, provenance, NOTICE and full CC-BY-4.0 license.

| GGUF storage | Bytes | SHA-256 |
|---|---:|---|
| f32 | 2,509,744,608 | `e50decf0ce8215c101d4714fc0581579662cd6d3599f54f603dfab7e1e776426` |
| mixed-f16 | 1,301,647,968 | `088473600a8855d5488dc5b9efbe46bb3bdc55ee6116bd7e2b5009a07e10127a` |
| q8_0 | 1,183,845,088 | `7d0602eafdfde51efb696d15c29781a2d68435ccf2f41000094cd2421b588983` |
| bf16 | 1,264,020,704 | `17524c9eb96f523341bd2c9c44c91efb2f17f7744cbe5c083422a596c51c8748` |

The compressed download expands to ordinary native weights; these are not
180 MB native ANS/codebook kernels. `mixed-f16` retains every staged F32 value,
keeping non-F16-representable scale products, normalization and compensated
projection tensors F32. Q8 applies the existing Phonon mixed policy: eligible
attention matrices use Q8_0; feed-forward/pointwise matrices retain exact F16;
non-representable products/normalization/projection retain F32. The initial
broad-Q8 profile changed Vulkan punctuation on a 30-second clip and is superseded
by this protected profile. BF16 and Q8 still add weight rounding.

## Original-runner and repeated-session parity

Reference environment: Python 3.11.15, PyTorch 2.8.0, Transformers 5.19.0,
NumPy 2.4.6, constriction 0.5.0. Short references use the official independently
unpacked FP32 checkpoint on CPU. The new reference harness additionally executes
the unmodified official loader and compares its own local pinned processor path
to official `transcribe()` on CUDA. It never loads audio.cpp's staged weights.
All inputs use mono 16 kHz samples and the publisher's 4000-sample prefix.

Six fixtures: checked-in LibriSpeech `2086-149220-0033.wav` (7.435 s), four local
video excerpts (7.9925, 12, 12 and 30 s), and 1 s silence. Excerpts start at
0, 20, 240 and 90 seconds in `We need to talk about 5min.mp4`; the local video
is not shipped. Each session also repeats short/long/short without reloading.

| Backend | F32 | Lossless mixed-F16 | Protected Q8 | BF16 |
|---|---:|---:|---:|---:|
| CPU | 9/9 | 9/9 | 9/9 | 9/9 |
| CUDA | 9/9 | 9/9 | 9/9 | 9/9 |
| Vulkan | 9/9 | 9/9 | 9/9 | 9/9 |

All **108/108** checks match original text exactly, including punctuation/case,
and keep word timestamps inside the original audio bounds. Timestamp accuracy
is not independently validated as forced alignment. These clips do not
establish corpus WER or universal quantized parity.

## Full five-minute recording: time and memory

All times below use offline full context. The complete recording is 299.99375 s.
Its original CUDA/FP32 transcript contains
816 whitespace-separated words. Original PyTorch CUDA processing takes
**2.563 s**, the median of the last two of three requests after warmup.

| Backend | GGUF | Processing time | Sampled peak extra VRAM incl. model (GiB) | Peak RSS (GiB) | Text vs original FP32 |
|---|---|---:|---:|---:|---|
| CPU | f32 | 34.291 s | 0.00 | 5.17 | Exact |
| CPU | mixed-f16 | 34.551 s | 0.00 | 4.56 | Exact |
| CPU | q8_0 | 33.512 s | 0.00 | 4.52 | One comma differs |
| CPU | bf16 | 34.214 s | 0.00 | 4.52 | Exact |
| CUDA | f32 | 0.816 s | 5.19 | 2.98 | Exact |
| CUDA | mixed-f16 | 0.786 s | 5.29 | 2.17 | Exact |
| CUDA | q8_0 | 0.805 s | 4.64 | 2.11 | Exact |
| CUDA | bf16 | 0.783 s | 5.03 | 2.18 | Exact |
| VULKAN | f32 | 0.924 s | 5.33 | 2.91 | Exact |
| VULKAN | mixed-f16 | 0.898 s | 5.08 | 2.11 | Exact |
| VULKAN | q8_0 | 0.909 s | 5.06 | 2.05 | Exact |
| VULKAN | bf16 | 0.886 s | 4.57 | 2.10 | Exact |

GPU timing is the median of the last two of three full requests after a short
warmup. CPU timing is one full request and includes the full graph build. Model
load/file I/O are excluded; preprocessing/decoding are included. All runs use
eight inference threads, `OMP_NUM_THREADS=8` and `OMP_WAIT_POLICY=PASSIVE`.
Memory includes the loaded model and temporary/pool allocations. NVML device
total was sampled every 100 ms minus a pre-process baseline; RSS is sampled
process working set. These are sampled peaks, not model-excluded cache sizes.

F32 and lossless mixed-F16 match the full original transcript on all backends.
CPU Q8 omits one comma (`models,` -> `models`), with the same normalized word
sequence; its timestamps stay bounded. All tested CUDA/Vulkan packages and CPU
BF16 match exactly. Mixed-F16 is recommended when preserving source values.
Rounded formats can change other recordings. All measured RTFs are below 1.

## Server, memory reuse and regressions

**66/66 HTTP/SSE checks** pass using mixed-F16: six references in one server,
zero/empty/enabled/spoken hotwords, two VAD speech spans with absolute timestamp
checks, VAD silence/token budgets, malformed options returning HTTP 400 and
valid-request recovery, plus unbiased/prompted raw-PCM streaming. The two-span
input joins two copies of LibriSpeech with four-second silence gaps.

Ten additional identical requests per backend after two warmups retain the
reference transcript and show the following post-request memory ranges:

| Backend | Requests | RSS range (MiB) | Device VRAM range (MiB) |
|---|---:|---:|---:|
| CPU | 10 | 0.02 | 0.00 |
| CUDA | 10 | 0.41 | 0.00 |
| VULKAN | 10 | 1.26 | 0.00 |

This is idle post-request reuse, separate from the full-recording peak table.
Ten focused CTests pass across the CPU/GPU builds (Bark prefix/frontends/variant,
hotwords, GGUF sources and model specs). Five Plus, five Pro and thirteen Phonon
converter tests pass; loader/catalog sync and its self-tests pass. Actual stock
Parakeet buffered streaming/finalize/reset still passes with its Q8 model.

## Limits and artifacts

English only. Full-context attention grows with length and retains the encoder
frame limit. Buffered streaming, explicit bounded windows and VAD change context
and can differ from full-recording reference decoding; this is not native
cache-aware streaming. VAD is offline-only. Metal/HIP and large-corpus WER are
unvalidated. Weights are manually converted; no unpublished model-manager
downloads or Hugging Face GGUF upload are claimed.

Local evidence is under `outputs/ub3plus/`: `unpack-parity.json`, `models.json`,
`reference-results.json`, `official-gpu-reference.json`, `matrix/`,
`matrix-results.json`, `full/`, `full-results.json`, `features-test/`,
`memory-reuse/` and `memory-reuse-results.json`. Initial Q8 profiles are retained
separately in `q8-profiles/`; use the final model hashes above.
