# Bark 3.0 Pro: complete-recording PyTorch comparison

This records the initial PR implementation and Q8 profile. The later
[current validation](underdog_bark3.md) supersedes its CPU allocation failure
and Q8 punctuation differences: allocator lifetimes are corrected, Q8 protects
the decoder's source precision, and opt-in native VAD/hotwords are validated.

Follow-up to [the initial validation](underdog_bark3.md), 2026-10-29. The complete
`We need to talk about 5min.mp4` recording was extracted once to mono 16 kHz WAV
(299.99375 seconds) and supplied to both implementations. Both add the
publisher's 4000-sample silence prefix exactly once. Source revision, package
hashes and hardware are unchanged from the initial validation.

The original checkpoint was executed without conversion using Transformers
5.19.0 and PyTorch 2.8.0: CPU/F32, CUDA/F32 and CUDA/BF16. CUDA uses the official
`2.8.0+cu128` wheel in an isolated Python environment. The benchmark follows the
publisher's processor and `model.generate` calls without compiling or replacing
model operations. The publisher's **unmodified** `runtime/transcribe_bark_pro.py`
was also executed directly on the full recording; its CUDA/BF16 transcript
exactly matches the benchmark's CUDA/BF16 result.

## Full-context results

Original CPU/F32 is the text reference. Original CUDA/F32 matches it exactly.
Native F32 and BF16 GGUFs match that full transcript exactly on CUDA and Vulkan.
All native storage/backend combinations have the same words after removing
punctuation and case. Q8 has only the specific punctuation/capitalization
differences below. This is a one-recording result, not corpus-level accuracy.

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

Timing excludes model loading and audio-file I/O, includes audio preprocessing,
generation and text decoding, and uses the median of the last two of three full
runs after a short-clip warmup. Both use eight host threads and the same RTX
3090. Native CLI benchmark result serialization is outside its timed session
call. GPU memory uses NVML device-total usage sampled every 100 ms minus each
pre-process baseline; peaks include loading, weights, workspace and allocator
caches. It is not an estimate of weights alone. Native BF16/Q8 labels describe
GGUF storage; they do not imply the same activation arithmetic as PyTorch BF16.

Compared with original PyTorch F32, native CUDA F32 takes about **65.4% less
processing time (2.89x throughput)** on this recording. Native BF16 has a larger
GPU footprint than the original PyTorch BF16 execution despite being faster.
These are repeated runs on one machine, not a broad performance guarantee.

Compared with the original F32 transcript:

- Original PyTorch BF16 and native CUDA Q8 use `change, and` where F32 emits
  `change. And`. Those two complete transcripts match each other exactly.
- Native Vulkan Q8 makes that same change and also emits `quickly, and`
  instead of `quickly. And`.

## Context policy matters

The original publisher runner processes the entire recording in one context.
The final port uses the same full-context default; the comparison selected it
explicitly for clarity:

```text
--session-option parakeet_tdt.offline_mode=full_context
```

The previously reported native Q8 server times of 3.35 s CUDA and 4.50 s Vulkan
used the development `auto` policy, which switches to overlapping bounded windows
beyond 30 seconds including the prefix. Those windowed outputs agree across
the three native backends but **do not exactly match the original full-context
transcript**: punctuation/capitalization is lost and there are eight differing
word spans, including spelling/number formatting, `want to`/`wanna`, and the
final clipped phrase. This difference cannot be presented as equal-quality
whole-context parity. Full context is both faster and more faithful for this
five-minute input on the tested GPU, at the cost of more VRAM. Publication audit
therefore restored `full_context` as the default; `auto` and `long_form` remain
explicit choices rather than silently changing the context policy.

Native CPU full context on this same five-minute sample failed to allocate a
**37,815,713,312-byte (35.2 GiB) CPU graph buffer** on the test machine. Original
PyTorch CPU full context succeeds. The native CUDA/Vulkan full-context paths
both succeed with the GPU usage above; the CPU allocation failure is not a
GPU failure. Native CPU bounded-window execution remains functional as recorded
in the initial report. The final publication change only restores the default
context policy; the explicitly selected full-context benchmark is unchanged.

The final publication build was rechecked without any offline-mode override:
all six CUDA/Vulkan storage combinations reproduce their explicit full-context
results. The legacy server also returned the exact F32-reference transcript on
two consecutive full-recording BF16 requests per GPU backend, with word spans
bounded to the unpadded recording. The six-clip CPU/CUDA/Vulkan matrix passes
all 81 checks using the final default and regenerated embedded contracts.

## Reproduction and artifacts

Build and convert using [the model guide](../models/underdog_bark3.md). Extract:

```bash
ffmpeg -i 'We need to talk about 5min.mp4' -ar 16000 -ac 1 movie-full.wav

# Publisher's original runner, unmodified, using its original source checkpoint.
python models/Underdog-Bark-3.0-Pro-source/runtime/transcribe_bark_pro.py \
  movie-full.wav --model models/Underdog-Bark-3.0-Pro-source \
  --device cuda --dtype bfloat16

# Reusable original-runner timing helper; discard the first full run for the table.
python tests/parakeet_tdt/bark3_reference.py \
  --source models/Underdog-Bark-3.0-Pro-source --audio movie-full.wav \
  --device cuda --dtype float32 --iterations 3 \
  --warmup-audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
  --output build/bark3/pytorch-full.json

# Native matched context, original-reference exact text with F32/BF16 storage.
build/bark3/bin/audiocpp_cli --task asr --family parakeet_tdt \
  --model models/underdog-bark-3.0-pro-bf16.gguf --model-spec-override embedded \
  --backend cuda --threads 8 --audio movie-full.wav \
  --session-option parakeet_tdt.offline_mode=full_context --text-out native.txt

# Timing: same loaded session, one short warmup and three full requests.
build/bark3/bin/parakeet_warm_bench \
  --model models/underdog-bark-3.0-pro-f32.gguf --model-spec-override embedded \
  --audio movie-full.wav --warmup-audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
  --backend cuda --threads 8 --warmup 1 --iterations 3 --offline-mode full_context \
  --timing-file build/bark3/full-timing.log
```

For Vulkan, choose the RTX 3090's index from `--list-devices` and use
`--backend vulkan --device <index>`. The benchmark used
`GGML_VK_VISIBLE_DEVICES=1`, then selected device 0.

Local evidence lives under `outputs/ub3p/original-comparison/`: `comparison.json`,
`comparison.md`, original CPU/CUDA JSON/transcripts, six native CUDA/Vulkan
JSON/transcripts, the native CPU allocation-failure log, and the original
publisher-runner command, transcript and exact-match check. The timed Python
driver is `outputs/ub3p/full_reference.py`; the native orchestrator is
`outputs/ub3p/full_native.py`. Ready-to-run local server configs using BF16 full
context are `outputs/ub3p/server-cuda-full-context.json` and
`outputs/ub3p/server-vulkan-full-context.json`.
