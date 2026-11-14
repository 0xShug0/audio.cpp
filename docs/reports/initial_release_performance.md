# Initial-release performance baseline

This report preserves the performance section removed from the README during the
v0.9.5 cleanup. These are historical measurements, not current-release results;
no benchmarks were rerun for this move. The charts and numerical results below
are retained from the original README.

> [!WARNING]
> These Python-relative numbers were measured for the initial release. Several model paths have improved substantially since then, so the figures below should be read as the original release baseline rather than the latest peak performance.

All performance metrics in this section were measured on Ubuntu with the CUDA backend on an NVIDIA GeForce RTX 5090. The Python-relative one-shot and long-lived-session comparisons come from direct framework/runtime API benchmark calls, not from `audiocpp_cli`; CLI path tests are separate and include app-layer request parsing, output writing, and other user-facing overhead.

**Absolute RTF depends on the GPU and system setup, but the Python-relative speedups are real because audio.cpp and the matching Python reference paths were measured on the same CUDA setup.**

audio.cpp already shows some genuinely exciting wins against the matching Python reference paths, especially on the TTS side, even when using the original model weights without quantization. The headline win is wall time: several TTS paths run **1.8x to up to 10x faster** than Python while cutting end-to-end latency by **45%-90%**.

- In one-shot runs, several TTS-family models already land far ahead of Python:
  - `vevo2`: **5.03x faster** with **80.11% less wall time**
  - `pocket tts`: **3.68x faster** with **72.80% less wall time**
  - `miotts`: **2.73x faster** with **63.39% less wall time**
  - `moss_tts_local`: **2.33x faster** with **57.07% less wall time**
  - `qwen3 tts`: **1.83x faster** with **45.34% less wall time**
  - `vibevoice`: **1.40x faster** with **28.75% less wall time**
- In long-lived-session runs, where the same loaded session serves multiple requests in sequence, the gains stay strong:
  - `pocket tts`: **3.22x faster** with **68.91% less wall time**
  - `qwen3 tts`: **2.74x faster** with **63.47% less wall time**
  - `moss_tts_local`: **2.66x faster** with **62.35% less wall time**
  - `miotts`: **2.28x faster** with **56.22% less wall time**
  - `vibevoice`: **1.77x faster** with **43.55% less wall time**
  - `vevo2`: **1.75x faster** with **42.72% less wall time**
- In long-form runs on the shared 6,026-character, 1,028-word passage, the strongest Python-relative wins still show up clearly:
  - `pocket tts`: **3.15x faster** with **68.23% less wall time**
  - `qwen3 tts`: **3.06x faster** with **67.33% less wall time**
  - `vibevoice`: **2.86x faster** with **65.07% less wall time**
  - `vevo2`: **1.77x faster** with **43.51% less wall time**
  - `chatterbox`: **1.58x faster** with **36.83% less wall time**
- These long-lived-session numbers are especially important for real applications, because they reflect the common case where model load, cached state, and reusable runtime setup are amortized across many requests.
- Bars below the 1.0x line are useful too: they spotlight exactly where more optimization work is still worth doing.

<p align="center">
  <img src="../../assets/figure/perf_one_shot_20260630.svg" alt="One-shot" width="720" />
</p>

<p align="center">
  <img src="../../assets/figure/perf_long_lived_session_20260630.svg" alt="Long-lived session" width="720" />
</p>

The figures report `Python wall time / audio.cpp wall time`. The 1.0x line means equal wall time; bars above 1.0x mean audio.cpp is faster than Python, and bars below 1.0x mean it is slower.

For TTS-family models, the measured one-shot RTF is:

| model | audio len (s) | wall time (s) | RTF | x faster than real time |
|---|---:|---:|---:|---:|
| chatterbox | 9.72 | 2.45 | 0.252 | 3.97x |
| miotts | 20.40 | 3.30 | 0.162 | 6.18x |
| moss_tts_local | 9.60 | 0.97 | 0.101 | 9.91x |
| omnivoice | 9.00 | 1.32 | 0.146 | 6.84x |
| pocket tts | 8.08 | 0.26 | 0.032 | 31.09x |
| qwen3 tts | 11.44 | 4.46 | 0.390 | 2.56x |
| vevo2 | 8.66 | 2.47 | 0.285 | 3.51x |
| vibevoice | 11.07 | 5.02 | 0.454 | 2.20x |
| voxcpm2 | 5.60 | 3.09 | 0.551 | 1.81x |

For long-form TTS tests, each run uses the same 6,026-character, 1,028-word input text (vibevoice uses 106,310 chars, 18,052 words, 4 speakers). Rows are CUDA unless marked CPU. The measured RTF is:

| model | audio len (s) | wall time (s) | RTF | x faster than real time |
|---|---:|---:|---:|---:|
| chatterbox | 391.24 | 58.57 | 0.150 | 6.68x |
| index tts2 | 422.12 | 139.95 | 0.332 | 3.02x |
| miotts | 399.16 | 66.59 | 0.167 | 5.99x |
| moss_tts_nano | 391.20 | 43.16 | 0.110 | 9.06x |
| moss_tts_local | 375.44 | 73.84 | 0.197 | 5.08x |
| omnivoice | 357.00 | 17.77 | 0.050 | 20.09x |
| pocket tts | 353.12 | 7.30 | 0.021 | 48.40x |
| qwen3 tts | 327.60 | 72.65 | 0.222 | 4.51x |
| supertonic | 379.32 | 2.02 | 0.005 | 187.62x |
| supertonic (CPU) | 379.40 | 61.40 | 0.162 | 6.18x |
| vevo2 | 457.68 | 52.47 | 0.115 | 8.72x |
| voxcpm2 | 315.84 | 72.70 | 0.230 | 4.34x |
| vibevoice | 5615.73 | 1376.84 | 0.245 | 4.08x |
