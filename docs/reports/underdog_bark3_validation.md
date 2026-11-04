# Underdog Bark 3.0 validation

This report consolidates the Standard (#1132), Plus (#1131), and Pro (#1130)
validation records. They share the `underdog_bark3` family; current installation,
conversion, CLI, and server instructions are in the
[model guide](../models/underdog_bark3.md).

These are recorded results from those integrations, not fresh measurements of
current main. Package hashes identify the tested snapshots; later embedded-spec
updates can change a GGUF's file hash without changing its tensors. Superseded
results are separated in [Historical Pro comparison](#historical-pro-comparison).

## Setup and Methodology

- Windows 11, Ryzen 9 7950X3D, eight inference threads, RTX 3090 24 GiB.
- MSVC 14.43, CUDA 12.4, Vulkan SDK 1.4.350.0; CPU-only and combined CUDA/Vulkan
  Release builds passed.
- Reference: Python 3.11.15, PyTorch 2.8.0, Transformers 5.19.0, NumPy 2.4.6,
  constriction 0.5.0. The original checkpoint/loader was used rather than
  audio.cpp's staged weights. Pro also ran the publisher's unmodified CUDA/BF16
  transcription runner; Standard/Plus cross-checked the pinned processor against
  the publisher's CUDA transcription path.
- Identical mono 16 kHz audio and the publisher's 4,000-sample silence prefix.
  Native word positions exclude that prefix and are checked against the original
  recording bounds.
- Six unique short fixtures: checked-in LibriSpeech
  [`2086-149220-0033.wav`](../../tests/parakeet_tdt/assets/2086-149220-0033.wav)
  (7.435 s), four non-public recording excerpts (7.9925, 12, 12, and 30 s), and
  one second of silence. Sessions also repeat short/long/short without reloading.
- The full recording is 299.99375 s. It is not distributed with the repository;
  exact reproduction of its measured numbers requires that original input.
- GPU full-recording timing is the median of the last two of three requests
  after a short warmup. CPU timing is one full request including graph build.
  Timings exclude model loading/file I/O and include preprocessing and decoding.
- `OMP_NUM_THREADS=8`, `OMP_WAIT_POLICY=PASSIVE`. Device-total GPU usage minus a
  pre-process baseline and process RSS/working set were sampled every 100 ms.
  Reported peaks include model weights, workspaces, temporary allocations, and
  allocator pools. Idle reuse ranges are reported separately.

Exact transcripts on these fixtures are not corpus WER, independent forced
alignment, or a universal quantization-parity guarantee. Native BF16/Q8 labels
refer to stored weights, not identical activation arithmetic to PyTorch BF16.

## Sources and Conversion

| Variant | Upstream source | Revision | Source SHA-256 |
|---|---|---|---|
| Standard | [ConwayResearch/Underdog-Bark-3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) | `85db2ad18ab41e534a222ee253c29191d77ed919` | `c18de9418d73cbeb80b8c1287b392d480a56e191ef4a81d1baaa1b02e66d92ed` |
| Plus | [ConwayResearch/Underdog-Bark-3.0-Plus](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Plus) | `d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb` | `fdd9f1052e4fdead07d71e8cd2138813eb0c209fd6ff8b1b6f47737e1e501857` |
| Pro | [ConwayResearch/Underdog-Bark-3.0-Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro) | `4b5c51e04aa8d16dda9ecdff71bc29f7360149e4` | `6f84b33e63ebd1851e910e8b2fe299d29f5e0efc6f57e84d92d2f79af464e0e5` |

Standard and Plus source containers are 163,398,645 and 179,524,845 bytes.
Their independent unpacking matched all 699 inference tensors from the official
loader before compatibility transforms; 24 training-only counters were omitted.
Processor/tokenizer sidecars use NVIDIA revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. Conversion verifies source hashes
and does not execute downloaded Python.

The GGUFs embed tokenizer/configuration, spec v1, provenance, tensor hashes,
NOTICE, and CC-BY-4.0 license. Projection and BatchNorm compensation account for
the native loader's scaling. Compressed source size is not expanded GGUF, RAM,
or VRAM size; native inference uses ordinary expanded weights.

Standard/Plus mixed-F16 preserves staged F32 values, keeping non-representable
products, normalization, and compensated projections in F32. Protected Q8 uses
eligible attention matrices in Q8_0 and sensitive feed-forward/pointwise matrices
in exact F16. Standard BF16 additionally retains the predictor, encoder-to-joint
projection, and output head in F32; Plus retains its own BF16 policy.

Pro originates in BF16. Its protected Q8 keeps `decoder.*`,
`encoder_projector.*`, and `joint.head.*` in exact source BF16, with BatchNorm
compensation in F32. Quantized formats remain rounded derivatives.

## Standard

### Tested Packages

| GGUF storage | Bytes | SHA-256 |
|---|---:|---|
| f32 | 2,509,744,800 | `b75407e7aedf5db9b784f0d6da57d6f152b5a9e27a6cbcaa51d3742d45fbffe7` |
| mixed-f16 | 1,301,648,160 | `f9062eb55154ba2e4d7bcfdf7e524db4eff448a3752963ff7490da364e3f5406` |
| q8_0 | 1,183,845,280 | `b15480349003b18fd96e4bcf3bb3747363a541ade00c354c6efe6afc015b384b` |
| bf16 | 1,300,278,432 | `6b391938a3275ceb76ac6d3a5c7dcc934a2ac69b2fb81505542868a3ecbaea10` |

### Reference Comparisons

| Backend | F32 | Lossless mixed-F16 | Protected Q8 | Protected BF16 |
|---|---:|---:|---:|---:|
| CPU | 9/9 | 9/9 | 9/9 | 9/9 |
| CUDA | 9/9 | 9/9 | 9/9 | 9/9 |
| Vulkan | 9/9 | 9/9 | 9/9 | 9/9 |

All **108/108** short/repeated checks match original-reference text, including
punctuation/case, with timestamps inside the input bounds. The source emits
`Yeah.` for one-second silence; native inference preserves it. Opt-in VAD
returns empty text for that non-speech input.

### Full-recording Performance

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

All 12 outputs match the original CUDA/F32 transcript (823 words). Original
PyTorch CUDA/F32 processing time was 2.572 s.

## Plus

### Tested Packages

| GGUF storage | Bytes | SHA-256 |
|---|---:|---|
| f32 | 2,509,744,608 | `e50decf0ce8215c101d4714fc0581579662cd6d3599f54f603dfab7e1e776426` |
| mixed-f16 | 1,301,647,968 | `088473600a8855d5488dc5b9efbe46bb3bdc55ee6116bd7e2b5009a07e10127a` |
| q8_0 | 1,183,845,088 | `7d0602eafdfde51efb696d15c29781a2d68435ccf2f41000094cd2421b588983` |
| bf16 | 1,264,020,704 | `17524c9eb96f523341bd2c9c44c91efb2f17f7744cbe5c083422a596c51c8748` |

### Reference Comparisons

| Backend | F32 | Lossless mixed-F16 | Protected Q8 | BF16 |
|---|---:|---:|---:|---:|
| CPU | 9/9 | 9/9 | 9/9 | 9/9 |
| CUDA | 9/9 | 9/9 | 9/9 | 9/9 |
| Vulkan | 9/9 | 9/9 | 9/9 | 9/9 |

All **108/108** short/repeated checks match original-reference text, including
punctuation/case, with timestamps inside the input bounds.

### Full-recording Performance

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

The original CUDA/F32 transcript contains 816 words and takes 2.563 s.
CPU Q8 omits one comma (`models,` → `models`), with the same normalized word
sequence; all other listed outputs match exactly. Mixed-F16 is preferred when
preserving the staged source values.

## Pro

### Tested Packages

| Standalone GGUF | Bytes | SHA-256 |
|---|---:|---|
| f32 | 2,509,681,888 | `68b4eef71c528ab4ebbf99478bbc7198bb84d3b7bee0ac0cfe171d7f242fb339` |
| bf16 | 1,255,567,328 | `d59e8e8fa2e26d942c8a912846e4fc4f0ecf3ed8de2dc44e6e1ef44cfc51aae5` |
| q8_0 | 908,687,328 | `d3fedf4d4fea36405bfce7ec0e8c132cb2c89b393ea2f4a61f97ba19352a5c46` |

### Reference Comparisons

| Backend | F32 exact text / timestamp bounds | BF16 | Protected Q8 |
|---|---:|---:|---:|
| CPU | 9/9 | 9/9 | 9/9 |
| CUDA | 9/9 | 9/9 | 9/9 |
| Vulkan | 9/9 | 9/9 | 9/9 |

All **81/81** short/repeated checks match original-reference text, including
punctuation/case, with timestamps inside the input bounds. Silence returns
empty text. The initial CPU/F32 encoder comparison had maximum absolute error
1.684e-6 over 96 valid frames; mel error was 1.689e-4. The padded 97th encoder
row is not valid output.

### Full-recording Performance

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

All listed native outputs match the original F32 transcript exactly. Original
PyTorch CUDA timings were 2.436 s F32 and 2.049 s BF16. CPU timings varied across
runs; the allocation correction below is a memory fix, not a CPU-speed claim.

## Pro CPU Allocation Correction

CPU schedule pruning removed view nodes needed for allocation lifetime tracking.
The model-local CPU policy retains them and releases the one-shot position
projection producer after copying to persistent graph inputs.

| Five-minute full-context CPU graph | Before | After |
|---|---:|---:|
| Graph buffer | 37,815,713,312 bytes (35.2 GiB) | 2,812,205,088 bytes (2.62 GiB) |
| Allocation reduction | | **92.56%** |
| Full original F32-reference transcript | Could not complete safely on this machine | Exact match, without chunking |

The 92.56% reduction is for the graph buffer, not total process RAM. Total
sampled RSS is 5.17 GiB for F32 and 4.45 GiB for protected Q8. A weight-free
regression checks dependent cropped-view lifetimes and exact output values
without allocating the old oversized layout.

## Buffered Streaming

Buffered streaming re-encodes bidirectional context; it is not native
cache-aware inference. The Pro frontend-reuse comparison caches one window of
unnormalized log-mel features, recomputes edges, and repeats full-window F64
normalization in the same reduction order. Reset clears the cache. The default
raw cache occupies about 0.68 MiB of host RAM and adds no GPU tensors.

Normalized frontend floats matched fresh extraction bit-for-bit for growing and
sliding windows, impulses, silence, unaligned tails, absent overlap, and
zero/one-valid-frame cases. The real streaming comparison passed **81/81**
CPU/CUDA/Vulkan × F32/BF16/Q8 short/repeated checks. Full-recording Q8 streams
also retained exact before/after text and timestamps on all three backends.
The baseline was the prior buffered implementation at `889e3161`, not the
publisher's different full-recording context. Both benchmark binaries called
`start_stream()`; older reset-only measurements did not activate the wrapper.

| Five-minute Q8 buffered stream | Before | After | Less processing time |
|---|---:|---:|---:|
| CUDA | 3.154 s | 2.735 s | **13.3%** |
| Vulkan | 4.451 s | 3.911 s | **12.1%** |

These are medians of five streams after one warmup, with two-second centers,
ten-second left context, two-second right context, 3,200-sample input chunks,
and the common eight-thread settings. Input was fed without real-time delays.
Timing includes preprocessing/decoding and excludes model loading/file I/O.
Frontend time decreased by approximately 73–76%; required lookahead is unchanged.
The separate uncontrolled CPU trial is excluded from performance conclusions.

## Server, Lifecycle, and Compatibility

Each variant passed **66 HTTP/SSE checks**, 22 per CPU/CUDA/Vulkan backend:

- Six original-reference recordings in one long-lived session.
- Zero-score and explicit-empty hotwords preserve the transcript; enabled
  lists and word/spoken alternatives execute. Prompted and unbiased raw-PCM
  streams work consecutively.
- Two speech spans separated by silence return both reference transcripts with
  original-sample offsets through offline VAD. Each span gets its own prefix;
  silence-only VAD returns empty text and token budgets remain effective.
- Malformed hotwords, excessive scores, invalid VAD durations, and unknown chunk
  modes return HTTP 400, followed by successful valid-request recovery.

Standard/Plus each measured ten further identical requests after two warmups:

| Variant | Backend | Requests | Idle RSS range (MiB) | Idle device VRAM range (MiB) |
|---|---|---:|---:|---:|
| Standard | CPU | 10 | 0.03 | 0.00 |
| Standard | CUDA | 10 | 1.11 | 0.00 |
| Standard | Vulkan | 10 | 1.28 | 0.00 |
| Plus | CPU | 10 | 0.02 | 0.00 |
| Plus | CUDA | 10 | 0.41 | 0.00 |
| Plus | Vulkan | 10 | 1.26 | 0.00 |

These are idle post-request ranges, not the peak-memory figures above.
Model-local prefix/frontend/allocation/hotword tests, converter checks,
GGUF/spec checks, loader/catalog synchronization, and stock Parakeet
streaming/reset regressions passed in the recorded builds.

After the shared guide/config integration on main `aeb7bcf1`, 54 Standard/Plus
mixed-F16 repeated comparisons retained exact text and timestamps. All 18
shared-config routes passed (three variants × offline/streaming × three
backends), retaining Pro's `bark3` and `bark3-stream` IDs. These were compatibility
checks, not new timing measurements. Family selection has since moved to the
dedicated `underdog_bark3` registration; use the current model guide for commands.

## Limits

English only. Full-context attention and encoder frame limits still constrain
long input. Bounded windows, streaming, VAD, and intentional hotword bias can
change recognition relative to unfiltered full-context decoding. VAD is
offline-only. The source Standard model's silence hallucination is preserved.
Metal/HIP, corpus-level WER, and independent forced-alignment accuracy were not
validated by these records. Quantized storage can differ on unseen recordings.

The full recording and four excerpts are non-public; their measurements are
recorded evidence, not a portable dataset. Public reproduction can use the
checked-in LibriSpeech fixture, the current model guide, and the existing
`tests/parakeet_tdt/` converter, reference, server, and streaming checks.

## Historical Pro Comparison

This is the initial implementation/profile, retained to distinguish superseded
findings from the later Pro results above. Its broad-Q8 punctuation differences
and 35.2 GiB CPU allocation failure were corrected later. Original PyTorch
BF16 versus F32 differences are not a native-port regression.

| Implementation | Backend | Storage / execution dtype | Warm time | Extra peak VRAM, including model | Exact F32-reference text |
|---|---|---|---:|---:|---|
| Original PyTorch | CUDA | F32 | 2.436 s | 5.39 GiB | Yes |
| Original PyTorch | CUDA | BF16 | 2.049 s | 2.85 GiB | One punctuation/case change |
| audio.cpp | CUDA | F32 GGUF | 0.842 s | 5.25 GiB | Yes |
| audio.cpp | CUDA | BF16 GGUF | 0.788 s | 5.26 GiB | Yes |
| audio.cpp | CUDA | Mixed Q8_0 GGUF | 0.781 s | 4.52 GiB | One punctuation/case change |
| audio.cpp | Vulkan | F32 GGUF | 0.917 s | 5.33 GiB | Yes |
| audio.cpp | Vulkan | BF16 GGUF | 0.892 s | 5.05 GiB | Yes |
| audio.cpp | Vulkan | Mixed Q8_0 GGUF | 0.875 s | 4.98 GiB | Two punctuation/case changes |

All entries had the same normalized word sequence on this one recording.
Original PyTorch BF16 and initial native CUDA Q8 used `change, and` where the
F32 reference used `change. And`. Initial Vulkan Q8 additionally used
`quickly, and` instead of `quickly. And`. The later protected-Q8 profile
supersedes these native Q8 results.

The earlier windowed `auto` policy took 3.35 s CUDA / 4.50 s Vulkan and changed
eight word spans as well as punctuation/case. Those numbers cannot establish
full-context parity. Full context was restored as the default; `auto` and
`long_form` remain explicit choices. The initial CPU full-context run failed on
a 37,815,713,312-byte graph allocation; the later model-local lifetime correction
resolved it. GPU full-context inference was not affected by that CPU failure.
