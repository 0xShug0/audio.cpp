# CUDA two-slot failure investigation

Local investigation on RTX 3090 (24 GiB), Windows, 2026-09-27. Based on PR #706 commit `c6b2be98`. The fixes are split into shared-infrastructure and per-model commits for PR #706.

## Revalidated regular server

Fixed seeds were used for stochastic models. A fresh one-slot server was compared with a fresh two-slot server; each ran cold and warm waves. Two-slot waves started both requests together. Parity compares decoded response content, including exact WAV bytes, after stripping timing fields. All waves returned HTTP 200; status polling observed two active slots. The CUDA policy enables these six additional family/task pairs at capacity two; higher counts are not certified.

| Model | Task | Cold/warm exact outputs | Peak process-test GPU usage (MiB) | Warm single request (s) | Warm concurrent pair (s) |
|---|---|---:|---:|---:|---:|
| Chatterbox | clon | 4/4 | 4584 | 0.778 | 1.179 |
| Chatterbox Turbo | tts | 4/4 | 3548 | 0.683 | 0.898 |
| F5-TTS | tts | 4/4 | 3250 | 1.788 | 3.525 |
| MMS Forced Aligner | align | 4/4 | 1072 | 0.041 | 0.046 |
| Seed-VC | vc | 4/4 | 5950 | 1.165 | 2.128 |
| Vevo2 | vc | 4/4 | 6960 | 1.143 | 1.870 |
| BS-RoFormer | sep | 4/4 | 2776 | 0.766 | 1.421 |
| Kokoro | tts | 4/4 | 2202 | 0.194 | 0.239 |

GPU usage is sampled device-wide during an otherwise isolated test, not an allocator-exact per-slot measurement. Timings are individual measurements, not a statistical performance benchmark. BS-RoFormer and Kokoro are previously supported control models, not new entries.

Single-slot output content for all six revalidated models is unchanged against the unfixed seeded baseline. Vevo2 already passed the fresh unfixed run, so this investigation does not establish that a specific patch fixed its historical failure.

## Fixes

- GGUF and Safetensors sources now protect mapped storage across lazy remapping, copying, synchronous backend upload and release. Multiple sessions can no longer unmap a source during another reader/upload.
- Chatterbox keeps the requested host thread count per caller, retains worker pools for each count and serializes submissions into each pool. This prevents callback replacement and pool destruction during another request. GPU inference is not globally serialized.
- F5-TTS protects its legacy process-wide DiT graph/model cache through upload, compute and readback; the Vocos graph has a separate lock. These GPU stages are serialized. Requests can overlap outside them; this is correctness support, not independent concurrent DiT execution or a claimed speedup.
- ACE-Step uses a mutex shared by the loaded package's slots around preparation, planner, conditioning, diffusion and VAE CUDA stages. It is released between generation stages and before CPU output processing. CPU and Vulkan do not acquire this lock; other model families are unaffected.
- No CUDA kernels or ggml backend code were changed. No new Vulkan model eligibility was added.

## ACE-Step guarded two-slot support

Serial generation did not reproduce the mismatch: four consecutive requests in an unguarded, two-slot-configured server matched the one-slot reference exactly. The shared-source fix removed the observed cold-start access violation, but overlapping CUDA inference still produced seeded waveform differences. Traces showed differences in both text encoding and diffusion in different runs; the underlying backend/library cause remains unconfirmed.

The retained workaround serializes ACE-Step's CUDA stages across sessions sharing one loaded package. The CUDA allowlist now permits its AudioGeneration task at capacity two (82 total audited family/task entries, including the six additions above). This supports overlapping HTTP requests, **not simultaneous ACE-Step GPU inference or a claimed throughput improvement**. Vulkan ACE-Step remains capped at one.

| Guarded diagnostic | Exact concurrent outputs | Warm single request (s) | Warm pair (s) | Sampled two-slot GPU peak (MiB) |
|---|---:|---:|---:|---:|
| Five-second request, six cold/warm waves | 12/12 | 2.001–2.016 | 3.912–3.966 | 13,746 |
| Default duration, cold/warm waves | 4/4 | 7.354 | 14.608 | 14,282 |
| Different prompts, seeds and durations (5/10 seconds), four waves | 8/8 | — | 4.882–4.939 | 13,666 |

Every diagnostic response returned HTTP 200, and status polling observed two active slots in every paired wave. The six five-second serial references were identical to the original seeded output. Results establish parity for the tested turbo Q8 model and requests; they do not certify every ACE-Step checkpoint or planner/edit route. Timings above use whole-wave wall time and are individual observations.

Locking diffusion alone was insufficient. cuBLAS workspace configuration and experimental CUDA graph-cache eviction also failed repeated parity checks; those experiments were discarded. A compute-sanitizer run exceeded the request timeout and was stopped; its partial log reported no error, but it is **incomplete and not a passing sanitizer test**.

Evidence: `outputs/model-two-slot-audit/ace-stage-guard-validation/`, `ace-stage-guard-default/`, `ace-stage-guard-mixed/`, and `ace-memcheck/`. The fixes are included as separate commits in PR #706.

The rebuilt regular CUDA server also passed default-duration cold/warm requests: 4/4 concurrent outputs matched its single-slot references, with HTTP 200 and two active slots in both waves. Sampled GPU peak was 13,606 MiB. Evidence: `outputs/model-two-slot-audit/ace-production-validation/`. Together with the diagnostic runs, this is 28/28 exact concurrent outputs on the retained workaround.

## Still excluded

| Model | Finding | Current CUDA cap |
|---|---|---:|
| DramaBox | One-slot sampled peak 22,212 MiB. Two slots reached the 24,063 MiB safety threshold; the test terminated its own server. This was not an independently observed natural crash. | 1 |
| LiveAvatar | One-slot sampled peak 14,396 MiB. Two slots reached the safety threshold; one prewarmed pair matched exactly but the next pair hit the memory guard. Not certified on this device. | 1 |

The original audit omitted seeds for ACE-Step and both Chatterbox variants; its unseeded parity comparisons cannot establish regressions. Follow-up ACE-Step tests include both bounded diagnostic durations and the original default duration.

## Verification

- CUDA and Vulkan server builds passed.
- Six focused CTests passed on each backend (12/12), including concurrent GGUF/Safetensors copy/scalar-read versus storage release and server slot policy/compatibility tests.
- Regular CUDA server: eight models x four parallel cold/warm responses = 32/32 exact reference matches, including the two control models.
- Failed experimental thread-local worker ownership caused a Windows teardown hang; it was discarded. The retained persistent-pool implementation passed fresh cold/warm request tests.

Raw logs, requests, WAVs and measurements for the six additions and controls are under `outputs/model-two-slot-audit/cuda-failure-production-validation/`. Original diagnostic evidence is under `cuda-failure-investigation/`; ACE-Step serial control is under `cuda-failure-fixed-v3/ace_step/`. Diagnostic executables temporarily bypassed the production allowlist; the regular CUDA binary now contains the six validated additions and guarded ACE-Step support.
