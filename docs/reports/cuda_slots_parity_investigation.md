# CUDA two-slot output parity investigation

Follow-up to PR #706, based on `92e09c80`, on Windows and an RTX 3090
(24 GiB), 2026-09-27. Changes are organized into shared infrastructure and
separate model commits for PR #706.

## Results on the regular server

All seven tested family/task pairs now pass at two CUDA slots. The production
allowlist increases from 82 to 89 CUDA entries, with these additions capped at
two. Three/four slots, other checkpoints, longer inputs and other tasks are not
certified. Vulkan eligibility is unchanged: these seven remain capped at one.

| Model | Task | Retained change | Exact paired outputs vs updated single-slot | One-slot peak (MiB) | Two-slot peak (MiB) |
|---|---|---|---:|---:|---:|
| AudioSR | s2s | Package-scoped CUDA inference guard | 8/8 | 6026 | 8692 |
| MOSS TTS Local | tts | Share one CUDA runtime and guard prepare/run | 8/8 | 17218 | 17356 |
| MOSS TTSD | tts | Share immutable CUDA backbone weights | 8/8 | 13176 | 17872 |
| MioCodec | vc | Deterministic CUDA ISTFT overlap-add | 8/8 | 3752 | 7284 |
| MioTTS | tts | Same opt-in through its MioCodec pipeline | 8/8 | 4628 | 9102 |
| NeuTTS | tts | Refresh codec inputs and deterministic ISTFT | 8/8 | 1782 | 3414 |
| VieNeu v3 Turbo | tts | Fixed seed in validation; no inference fix | 8/8 | 976 | 1776 |

GPU memory is sampled device-wide with `nvidia-smi` during isolated tests. It
includes model weights, buffers and desktop/driver usage; it is not a per-slot
cache allocation measurement. Sampling can miss brief peaks. Original MOSS
Local and TTSD two-slot attempts reached the 24063 MiB guard; the harness stopped
its own server, rather than observing a natural out-of-memory crash.

AudioSR and MOSS Local accept two active HTTP requests but **serialize GPU
inference** within the loaded package. No inference throughput speedup is
claimed for them. MOSS TTSD keeps separate graphs, KV caches and execution
contexts; only immutable weights are shared, without a model inference lock.

## What the investigation established

- AudioSR was stable across serial requests, but overlapping CUDA requests
  produced waveform differences (roughly 51-52 dB SNR in the original fixture).
  The guard preserves the original serial output. The underlying CUDA/library
  cause is still unconfirmed; this is a correctness workaround.
- MioCodec and MioTTS differed slightly even in serial runs. Their CUDA ISTFT
  accumulated overlapping frames using floating-point atomic adds, whose order
  varies with scheduling. The new opt-in kernel assigns one thread per output
  sample and accumulates contributing frames in a fixed order. The existing
  atomic kernel remains the default for other callers.
- NeuTTS tracing found identical generated speech codes but varying codec head
  values, including serial variation. Marking the FSQ tensors as graph inputs
  alone was insufficient. Re-uploading positions before every graph execution,
  together with deterministic ISTFT, made the real-model tests repeatable.
  The allocator/capture mechanism behind the stale input is not established.
- Sharing only MOSS Local backbone/depth weights, even with guarded inference
  and temporary codec graph release, still exceeded the memory guard. Those
  experiments were discarded. The retained package cache shares the entire
  CUDA runtime, keyed by device, thread count and session options. Weak cache
  ownership avoids an assets/runtime cycle; per-session text processing and
  reference voice caches remain separate. Prepare/run are guarded together.
- MOSS TTSD fits by sharing its read-only backbone tensor storage. A package
  cache distinguishes device and storage configuration; other delay-decoder
  families keep the original loading path unless explicitly opted in.
- VieNeu's original comparison omitted its seed. With an explicit seed, the
  original implementation already passes repeated and overlapping requests.
  This investigation does not establish an inference bug in VieNeu.

No ggml backend or ggml CUDA kernels were changed. There **is** a new opt-in
CUDA reconstruction kernel in the audio framework, enabled only by the
MioCodec/MioTTS and NeuTTS paths. CPU/Vulkan retain their ISTFT path. The FSQ
input refresh applies to its shared runtime (currently used by NeuTTS); MOSS
TTSD weight sharing and MOSS Local runtime sharing are CUDA-only.

Tested packages were `audiosr-basic-f32.gguf`,
`moss-tts-local-v1.5-q8_0.gguf`, `moss_ttsd_q8_0_codec_f16.gguf`,
`miocodec-25hz-44khz-v2-q8_0.gguf`, `miotts-1.7b-q8_0.gguf`,
`neutts-2e-orig.gguf` and `vieneu-v3-turbo-q8_0.gguf`, with their catalogued
sidecars and default storage options. MOSS Local retains its original CUDA
storage precision; no precision reduction or shorter generation limit was
used to make the two-slot test fit.

## Quality against the original implementation

The original single-slot WAV from the diagnostic binary was compared with the
updated regular-server single-slot WAV, using the same checkpoint, input and
seed. Sample rate, channel count and frame count were unchanged in all seven.

| Model | Original WAV comparison | Maximum PCM difference | Different PCM scalars | Error SNR (dB) |
|---|---|---:|---:|---:|
| AudioSR | Byte-identical | 0 | 0 | Exact |
| MOSS TTS Local | Byte-identical | 0 | 0 | Exact |
| MOSS TTSD | Byte-identical | 0 | 0 | Exact |
| VieNeu v3 Turbo | Byte-identical at fixed seed | 0 | 0 | Exact |
| MioCodec | Quantization-level change | 1 PCM16 step | 27 | 109.88 |
| MioTTS | Quantization-level change | 1 PCM16 step | 11 | 110.94 |
| NeuTTS | Quantization-level change | 1 PCM16 step | 8 | 108.89 |

One PCM16 step is `1/32768` in normalized audio. The three reconstruction
outputs are **not byte-identical to the original**, which itself varied with
accumulation order. These measurements establish numerical closeness on the
tested fixtures, not a perceptual quality certificate for every input. Updated
single-slot and two-slot output bytes are identical in every retained test.

## Validation and evidence

The normal `audiocpp_server.exe`, with the production slot policy, was tested
for each family using a fresh one-slot server and a fresh two-slot server:

1. Two different single-slot reference requests, then both repeated.
2. Simultaneous cold requests, repeated warm requests, then two mixed waves
   with different inputs/prompts and seeds, in both orders: 56/56 exact paired
   outputs overall, with two active slots observed in every wave.
3. Three serial requests after concurrency, checked against their references.
4. Model unload, verification of unloaded state, lazy reload and exact output.

Seeds were explicit for stochastic targets; MioCodec does not sample. AudioSR
and MioCodec used a cropped second audio input; TTS used a second prompt/seed.
Reference voices were held fixed. All seven unload/reload checks passed, and
no retained run triggered the VRAM guard. Sampled GPU usage after unload was
330-442 MiB. This does not cover every reference voice or simultaneous unload
while an inference request is active.

New unit coverage tests deterministic ISTFT against host and old CUDA output
with repeated concurrent runtimes, shared-weight cache ownership/loading,
and repeated FSQ codec graphs with independent execution contexts. The small
synthetic FSQ fixture also passes without the position refresh; the actual
NeuTTS checkpoint tests, rather than that fixture, reproduce the defect.

CUDA and Vulkan CLI/server builds passed. Nine focused CUDA CTests and seven
Vulkan CTests passed (16/16), covering session/slot policy, single-slot
compatibility, busy handling, configuration, mapped weight sources and the
new tests above. CUDA ISTFT cases include FFT sizes 256, 258, 392 and 1920,
single/multiple frames and non-dividing hop lengths; comparisons with the
host/old CUDA path require maximum error <= 1e-5 and RMS error <= 1e-6.

CUDA controls BS-RoFormer, Kokoro and MOSS VoiceGenerator passed the same
mixed/cold/warm/reload sequence (24/24 paired outputs) and matched their
original catalog cold output. Higgs v3 did not meet this sequence's stricter
cold-to-warm equality criterion: its cold and warm WAVs differ even at a fixed
seed. This also occurs in the original catalog and the pre-change diagnostic
binary. Comparing the same request sequence before/after these changes gave
14/14 identical recorded response hashes for Higgs, including mixed requests
and reload; the two serial-repeat boolean outcomes also matched. This is an
existing state-dependent output difference, not a newly demonstrated
regression. Higgs has not been patched or newly certified by this task.

Vulkan controls BS-RoFormer, Kokoro and MOSS VoiceGenerator also passed the
same sequence: 24/24 exact paired outputs, successful unload/reload and
original Vulkan catalog cold-output matches. These are controls for existing
Vulkan support; the seven target models were not enabled or certified for
multiple Vulkan slots by this CUDA investigation.

Requests, logs, WAVs and JSON measurements are under the workspace's
`outputs/model-two-slot-audit/cuda-parity-production/`. Original diagnostics
are under `cuda-parity-investigation/`; discarded experiments are retained
separately for investigation. These large artifacts are not repository files.
