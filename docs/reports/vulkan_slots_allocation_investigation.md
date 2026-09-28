# Vulkan two-slot allocation investigation

Follow-up to PR #706 commit `c427c70d`, on Windows and an RTX 3090
(24 GiB), Vulkan device 1, 2026-09-27. The changes are organized into shared
infrastructure and separate model commits for PR #706. They address the four allocation failures recorded in the earlier
Vulkan audit without changing model precision, request limits or Vulkan kernels.

## Retained fixes

Each loaded package now owns a cache for immutable Vulkan weights, keyed by
component, device, storage type and weight-context size. Slot graphs, KV caches,
samplers, reference caches and execution contexts remain independent. No model
inference mutex or shared generation runtime was introduced.

- FireRed Audio shares its Qwen3.5 backbone weights. Patch encoder, RedAE and
  flow components remain separate; sharing the backbone alone was sufficient.
- HeartMuLa shares both generator and codec weights. The audited package keeps
  its existing `heartmula.mem_saver=true` setting; this investigation did not
  introduce that setting or shorten the generation request.
- MOSS TTS v1.5 opts into the delay-backbone weight cache on Vulkan. Heads and
  codec runtimes remain separate. MOSS TTSD's existing cache remains CUDA-only;
  its call site now explicitly supplies no cache on other backends. Other delay
  families still use the original loading path unless explicitly opted in.
- PersonaPlex shares its LM and depformer weights. Mimi codec components and
  all conversation state remain separate. Streaming admission is unchanged;
  only the offline speech-to-speech task was validated here.

The reusable weight cache can now retain a loader's existing shared allocation
directly. It rejects null loader results and permits a subsequent retry rather
than caching failure. Existing value-returning loaders use the same synchronized
cache path. CUDA/CPU model loading retains its previous behavior; the four new
sharing branches are selected only for Vulkan.

## Regular-server results

All four tested family/task pairs now pass at two Vulkan slots. The local
Vulkan allowlist increases from 63 to 67 entries; these four additions are
capped at two. The CUDA table remains at 89 entries with unchanged capacities.

| Model | Task | Original two-slot run | Updated one-slot peak (MiB) | Updated two-slot peak (MiB) | Exact paired outputs |
|---|---|---|---:|---:|---:|
| FireRed Audio | clon | Model-buffer allocation failure | 13027 | 15745 | 8/8 |
| HeartMuLa | gen | Inference-buffer allocation failure | 14097 | 20803 | 8/8 |
| MOSS TTS v1.5 | tts | Model-buffer allocation failure | 13580 | 19185 | 8/8 |
| PersonaPlex | s2s | Inference-buffer allocation failure | 14143 | 17516 | 8/8 |

GPU memory was sampled device-wide with `nvidia-smi`, including model weights,
runtime allocations, retained pools and driver/desktop usage. It is not an
isolated per-slot cache measurement, and sampling can miss brief peaks. The
original runs did not finish, so no percentage savings against a completed
original two-slot peak is claimed. No updated run triggered the VRAM guard
(total device memory minus 512 MiB).

The tested checkpoints are `firered-audio-q8_0.gguf`, `heartmula-q8_0.gguf`,
`moss_tts_v15_q8_0_codec_f16.gguf` and `personaplex-7b-v1-q8_0.gguf`, with the
catalogued sidecars and session options. Results do not certify other variants,
three/four slots, all reference voices, longer inputs or other tasks.

## Output quality and request isolation

Before changing the implementation, the regular Vulkan executable was saved.
For each model, two different requests were run on that original one-slot binary
and compared with the updated regular one-slot server: **8/8 original response
comparisons were exact, including WAV bytes**, and each model's original catalog
cold-output hash also matched. Precision and default generation limits were
unchanged. Seed 1234 was used for the catalog request and 5678 for the alternate
request; HeartMuLa's alternate lyrics were also changed.

Updated one-slot and two-slot servers then exercised:

1. Single-slot reference requests and serial repetitions.
2. Simultaneous cold starts, a warm identical pair, then mixed request/seed waves
   in both orders: **32/32 exact paired outputs** overall, with two active slots
   observed in every wave.
3. Three serial requests after concurrency, each matching its reference.
4. Explicit unload, verification of unloaded state, lazy reload and exact output.

All four models returned HTTP 200 throughout the retained sequence, released
active leases and passed unload/reload. Device usage after unload was sampled
at 593-2196 MiB. The remaining usage was not attributed to individual
allocations, and no claim is made that every byte was immediately returned
to the operating system.

Reference audio was fixed in this primary sequence. It does not establish
cross-reference isolation for every voice or every request route.

## Preliminary timing observations

| Model | Warm single request (s) | Warm concurrent pair wall time (s) |
|---|---:|---:|
| FireRed Audio | 2.729 | 5.075 |
| HeartMuLa | 4.842 | 8.418 |
| MOSS TTS v1.5 | 1.200 | 2.181 |
| PersonaPlex | 7.357 | 12.309 |

These are individual observations from the parity checks, not repeated
statistical throughput benchmarks. The pair column is elapsed time for both
requests, not latency per request. The primary result is successful inference
within 24 GiB while preserving output, not a universal performance percentage.

## Verification and evidence

The new Vulkan unit test uploads a small immutable matrix, destroys the backend
that uploaded it, then reads it through two independent Vulkan execution
contexts with repeated concurrent matrix multiplies. It checks results against
the host calculation and verifies that dropping the package cache releases
weight ownership. This exercises buffer lifetime as well as concurrent reads.
The portable cache test also covers retaining an existing shared allocation,
cache hits without re-upload, null-load failure and retry.

Vulkan's eight focused CTests passed, including the new GPU lifetime test,
session pool/policy, single-slot compatibility, busy/configuration handling and
mapped source concurrency. CUDA and Vulkan CLI/server builds passed.

CUDA's nine focused CTests also passed (17/17 across the two backend suites).
All four target models' cold/warm single-slot CUDA response hashes matched
the original CUDA catalog (8/8), with no VRAM guard or active-lease failure.
CUDA/Vulkan model timing differences are not used as a quality comparison;
each backend is compared with its own original output.

CUDA two-slot controls MOSS TTSD and MOSS VoiceGenerator also passed the full
cold/warm/mixed/reload sequence: 16/16 exact paired outputs, original catalog
cold-output matches and no VRAM guard. These controls exercise the shared
delay-backbone constructor's existing CUDA opt-in and default uncached path.

Vulkan controls BS-RoFormer, Kokoro and MOSS VoiceGenerator passed the same
sequence: 24/24 exact paired outputs, original Vulkan catalog cold-output
matches, successful unload/reload and no VRAM guard. MOSS VoiceGenerator
exercises the unchanged default delay-backbone loading path on Vulkan.

Raw requests, original/updated WAVs, server logs and measurements are under the
workspace's `outputs/model-two-slot-audit/vulkan-allocation-candidate-v1/`.
Earlier allocation errors remain in `catalog-vulkan/`. These large artifacts
are not repository files.
