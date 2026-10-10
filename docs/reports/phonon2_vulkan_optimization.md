# Phonon-2 Vulkan optimization

Windows/RTX 3090, 8 threads, 2026-10-10. Baseline: the Phonon PR integration
at `37c8a2b7` (including CUDA dependency #861), whose model-only equivalent is
`eade1260`. This report compares the Vulkan follow-up against that initial
integration build. [Portable evidence](phonon2_vulkan_optimization.json)
contains the timing samples, tracked allocation peaks and regression results.

The Phonon Vulkan decoder now keeps hidden/cell state and predictor cache in
a separate device allocation. Graph copy nodes update them after all consumers
of the old state have run. Only logits return to the host each step. Request
reset clears the device state; blank steps reuse the resident predictor cache.
Encoder graph replacement releases the unused graph before allocating another,
and one-shot positional-projection buffers are freed after their results have
been copied to persistent main-graph inputs.

All changes are model-local, gated by Vulkan plus Phonon's explicit F32 and
token-duration policies. No shared backend/kernel changes or arithmetic
precision changes. CPU/CUDA and stock Parakeet retain their previous paths.

## Warm HTTP latency

Old/new/old/new server order, two process trials per implementation, two
warm-up requests then nine timed requests per clip/trial: median of 18 samples.
Model loading and graph initialization excluded; every response's text and
word timings equals the baseline response exactly.

| Storage | Audio | Before | After | Less time |
|---|---:|---:|---:|---:|
| F32 | 11.9925 s | 66.0 ms | 52.9 ms | 19.9% |
| F32 | 30 s | 140.3 ms | 102.1 ms | 27.2% |
| BF16 | 11.9925 s | 60.9 ms | 51.2 ms | 15.9% |
| BF16 | 30 s | 132.2 ms | 101.9 ms | 22.9% |
| Q8_0 | 11.9925 s | 65.6 ms | 50.5 ms | 22.9% |
| Q8_0 | 30 s | 132.9 ms | 98.4 ms | 26.0% |

The unchanged CUDA Q8 path measured 48.2 ms / 108.6 ms on the same two inputs
with 18 warm HTTP samples. Vulkan Q8 measured 50.5 ms / 98.4 ms in the
alternating comparison. This is sample latency, not a general backend ranking.

## Memory

Twenty clips with two requests per clip plus upload endpoint, per-build
Vulkan allocation logging. These figures include weights and all tracked
device buffers; they exclude driver allocations and other applications.

| Storage | Tracked peak before | Tracked peak after | Saved |
|---|---:|---:|---:|
| F32 | 2.88 GiB | 2.68 GiB | 205 MiB (6.9%) |
| Q8_0 | 2.17 GiB | 1.98 GiB | 195 MiB (8.8%) |

Whole-GPU NVML readings varied substantially during these passes; the earlier
Q8 3.34 GiB spike was not a consistent property of Q8 weight storage. Backend
logging is used for the allocation comparison, not a claim of an equal
reduction in Windows driver-inclusive VRAM. The raw NVML samples are retained.

## Validation

- Final F32/Q8 Vulkan matrix: 20/20 reference text and word timings, exact
  repeats and upload text. BF16: 20/20 text, 19/20 reference timings, identical
  to baseline; the existing `video_30s` 80 ms difference remains.
- Five-minute Q8 native long-form: all 817 words and timestamps identical
  before/after. Cold process time including loading: 7.25 s / 6.88 s; one
  trial, not the controlled warm benchmark above. Original packed-runtime
  long-form parity is still not established.
- Stock Parakeet Q8 Vulkan full-context and long-form text/timings identical
  to baseline. Variant/config and duration CTests: 2/2.
- Twelve requests per Vulkan storage variant: exact repeat results, zero
  sampled post-warm VRAM growth; RSS varies by at most 1.25 MiB.
- CPU/CUDA reference code paths are unchanged. AMD Vulkan was not tested;
  hardware coverage is the RTX 3090.

Build commands and reference setup are in [the original report](phonon2_validation.md).
An instrumented native timing probe can use:

```bash
build/<preset>/bin/parakeet_warm_bench --model models/Phonon-2-GGUF/phonon-2-q8_0.gguf \
  --backend vulkan --device 1 --threads 8 --audio speech.wav \
  --offline-mode full_context --warmup 2 --iterations 9 --timing-file warm.log
```

Select the Vulkan device index for your machine. Instrumented native timings
are separate from the HTTP table above. Raw scripts, logs and samples:
`outputs/phonon2-vulkan-opt-20261010/` in the local workspace.
