# F5-TTS and Seed-VC slot validation

Tested on Windows/MSVC and an RTX 3090 (24 GiB), 2026-09-27.

## Fixes

- F5-TTS: use the loaded CPU backend public graph-compute API rather than a private weak/dlsym symbol unavailable on MSVC. The previous CPU-backend-not-loaded error was misleading. Carry the selected backend through synthesis so Vulkan runs DiT and Vocos on its requested GPU. Partition weight/backend/graph caches by backend and device. Retain existing DiT/vocoder locks and CUDA calculations.
- Seed-VC: restore CAMPPlus deferred constant tensors before each cached graph execution. Their arena storage can be reused after the last consumer; a one-time upload is insufficient. Input/output allocator flags and disabling Vulkan fusion did not repair this. Identical audio features/content codes/noise were stable, but style embeddings diverged (all 192 values, maximum difference 3.2806406), causing different audio. The same issue reproduced on CUDA.
- Admit only these two new Vulkan model/task entries, with a validated capacity of two. CUDA already admitted both at two. Higher capacities and other packages/tasks are not certified.

## Real-model results

Each row passed two serial references and their repeats, four paired waves (cold same input, warm same input, two mixed-input orders), two post-parallel serial requests, and unload/reload. Every wave reported two active slots and both outputs matched its backend-specific single-slot reference exactly. All requests returned HTTP 200; no VRAM guard triggered.

| Backend | Package / task | Exact paired outputs | One-slot peak | Two-slot peak |
| --- | --- | ---: | ---: | ---: |
| CUDA | Habibi Unified original GGUF / TTS | 8/8 | 3.17 GiB | 3.17 GiB |
| CUDA | SeedVC MLX Q8_0 / v2 voice conversion | 8/8 | 3.05 GiB | 5.83 GiB |
| VULKAN | Habibi Unified original GGUF / TTS | 8/8 | 4.64 GiB | 4.64 GiB |
| VULKAN | SeedVC MLX Q8_0 / v2 voice conversion | 8/8 | 4.38 GiB | 6.23 GiB |

VRAM is sampled total NVIDIA device usage, including model weights, inference buffers and caches. It is not the incremental cost of a slot. F5 shares locked graph caches, so two active HTTP slots do not imply simultaneous GPU graph execution or a throughput increase.

## Output and compatibility checks

- Both corrected CUDA models match the saved pre-fix executable first-request waveform byte-for-byte. Seed-VC warm pre-fix output was unstable and cannot serve as a valid reference. Corrected Seed-VC Vulkan matches its original cold first-request waveform exactly.
- Original F5 Vulkan failed before generating any audio, so it has no successful Vulkan output baseline. Corrected Vulkan output is repeatable and slot-exact, but differs from CUDA: same 24 kHz / 93,953 samples, RMSE 0.111336, aligned waveform SNR 3.11 dB. These checks establish within-backend repeatability and CUDA preservation, not perceptual equivalence across backends.
- Chatterbox and GLM TTS original/updated first and repeat waveform hashes match exactly on both CUDA and Vulkan. These exercise other callers of the shared CAMPPlus encoder. Other shared-component callers were not retested in this follow-up.
- Full F5 CPU server generation succeeded with the default 32 steps on Windows: 74.190 s, 24 kHz mono, 93,953 samples. CPU parallel-slot admission remains unchanged.

## Regression checks

- CUDA server + CLI and Vulkan server + CLI build successfully.
- 11 selected CTests passed in each build (22 total), including the new F5 public CPU-backend graph/reuse regression.
- New checkpoint-backed `campplus_graph_reuse_probe` passed on CPU, CUDA device 0, and Vulkan device 1. It repeats a 257-frame input, alternates different features at the same shape, alternates a 143-frame shape, and returns to the cached reference graph. All reference embeddings must be finite and byte-exact.
- Temporarily removing only the CAMPPlus constant refresh makes that probe fail on Vulkan with `cached CAMPPlus embedding changed`. The fixed source and executable were restored afterward.
- `git diff --check` passed. No ggml CUDA/Vulkan kernels were edited.

Artifacts are under `outputs/model-two-slot-audit/{cuda,vulkan}-f5-seed-final/`, `cpu-f5-seed-final/`, `{cuda,vulkan}-f5-seed-controls/`, `cuda-f5-seed-original/`, `vulkan-f5-seed-investigation/`, and `campplus-reuse-*.log`.
