# F5-TTS slot validation

Tested on Windows/MSVC and an RTX 3090 (24 GiB), 2026-09-27.

## Fix

Use the loaded CPU backend public graph-compute API instead of a private weak/dlsym symbol unavailable on MSVC. The previous CPU-backend-not-loaded error was misleading. Carry the selected backend through synthesis so Vulkan runs DiT and Vocos on the requested GPU. Partition weight/backend/graph caches by backend and device, retaining existing DiT/vocoder locks and CUDA calculations.

Admit the validated Vulkan offline TTS task at two slots. CUDA already admits it at two. Higher capacities, other packages/tasks and CPU parallel slots are not certified.

## Validation

Habibi Unified original GGUF passed two serial references and repeats, four paired waves (cold/warm same input and two mixed-input orders), post-parallel serial requests, and unload/reload. Each wave reported two active slots, with both outputs byte-exact against its backend-specific single-slot reference. All requests returned HTTP 200; no VRAM guard triggered.

| Backend | Exact paired outputs | One-slot peak | Two-slot peak |
| --- | ---: | ---: | ---: |
| CUDA | 8/8 | 3.17 GiB | 3.17 GiB |
| Vulkan | 8/8 | 4.64 GiB | 4.64 GiB |

VRAM is sampled total NVIDIA device usage, including model weights, inference buffers and caches. It is not incremental slot memory. F5 shares locked graph caches; two active HTTP slots do not imply simultaneous GPU graph execution or a throughput increase.

Corrected CUDA output matches the saved pre-fix executable first-request waveform byte-for-byte. Original Vulkan failed before producing audio, so no successful Vulkan baseline exists. Corrected Vulkan output is repeatable and slot-exact, but differs from CUDA: 24 kHz / 93,953 samples, RMSE 0.111336, aligned waveform SNR 3.11 dB. These checks establish within-backend repeatability and CUDA preservation, not perceptual equivalence across backends.

Full F5 CPU server generation succeeded at the default 32 steps: 74.190 s, 24 kHz mono, 93,953 samples. The new F5 public CPU-backend regression verifies graph computation and repeated input updates on Windows. Both CUDA/Vulkan server and CLI builds and selected backend/slot CTests passed. No ggml kernels were changed.

Artifacts: `outputs/model-two-slot-audit/{cuda,vulkan}-f5-seed-final/f5_tts/`, `cuda-f5-seed-original/f5_tts/` and `cpu-f5-seed-final/f5_tts/`.
