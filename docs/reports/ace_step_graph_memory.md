# ACE-Step encoder memory and repeated timbre reuse

## Implementation

Text encoding, lyric encoding, and standard planner prefill use ggml graph-lifetime allocation instead of allocating storage for every intermediate tensor. Explicit inputs and retained outputs preserve uploaded positions/masks and the planner KV tensors read after execution. Backend graph resources are released before graph storage.

The conditioning encoder retains one exact CPU timbre result per session. A hit requires identical reference bytes, frame count, and compatible graph capacity. Changed or multiple references compute normally. Virtual frame capacity preserves the original padded graph history across cache hits; memory-saver graph release resets that history. The cache survives release of GPU graphs.

These are shared ACE-Step model changes. There are no CUDA/Vulkan kernels, precision changes, diffusion/VAE changes, or server/parallel-slot framework changes. CLI and server use the same implementation. The timbre graph itself retains its original allocation path.

## CUDA measurements

Windows, RTX 3090 (24 GiB), Release, eight CPU threads, one session, seed 1234, five-second generation, eight diffusion steps, direct generation with all CoT switches disabled, `ace_step.mem_saver=true`. Base/Turbo components are Q8; XL packages have Q8 DiT and bf16 text/planner/VAE. Formatted text/empty lyric/conditioning lengths: 114/11/126.

Caption:

> Tired of juggling a dozen Conda environments, hundreds of Python packages, and dependency conflicts just to try a few audio models? audio.cpp gives those paths a shared native runtime instead. Runs on Windows, Linux, and macOS, with support for NVIDIA, AMD, Apple Silicon, and CPU-only machines.

| Model | Original extra warm VRAM (GiB) | Optimized extra warm VRAM (GiB) | Less extra VRAM | Original warm time (s) | Optimized warm time (s) | Less time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Q8 | 2.367 | 0.846 | 64.3% | 0.990 | 0.918 | 7.3% |
| Turbo Q8 | 2.367 | 0.846 | 64.3% | 0.985 | 0.915 | 7.1% |
| XL Turbo Q8 DiT | 2.375 | 0.869 | 63.4% | 1.152 | 1.086 | 5.7% |
| XL SFT Q8 DiT | 2.375 | 0.869 | 63.4% | 1.157 | 1.089 | 5.9% |

Extra VRAM is the sampled request peak minus the immediately preceding loaded-idle device reading. Idle includes weights, backend state, and retained caches: these are incremental generation peaks, not all non-weight memory or total GPU savings. NVML sampling interval: 10 ms. Times are HTTP wall-time medians of requests 2?8, excluding lazy model loading. All 32 response artifacts/WAVs matched exactly.

This recorded table compares complete builds: upstream `bbaa20dc` without slots versus `fdcb297a` plus these model changes and pre-existing local server work, with one slot. It is not an isolated measurement of the slot framework or a fresh measurement on the standalone PR branch. The ACE-Step files were unchanged upstream between that baseline and the PR base `ed96b730`; the PR contains only the optimizations described above and a model probe. Fresh standalone-branch validation is listed separately below.

## Targeted quality coverage

- CUDA final timbre-cache addition versus the earlier graph-allocation version: 112/112 exact comparisons across four packages, memory saver on/off, repeats, changed prompts/lyrics/seeds/durations, planner generation, and Base/SFT guidance.
- Direct conditioning probe: ten reference histories each for Turbo and XL Turbo. Covers cache hits, changed bytes, frame growth/shrinkage, graph release, multiple references, and invalid reference ordering.
- CUDA CLI comparisons: Turbo and XL Turbo WAVs identical before/after the cache addition.
- Final Vulkan Base/Turbo: 28/28 exact before/after response comparisons, memory saver enabled, eight repeats and six mixed-history requests per package. This includes the new cache and the allocation changes. Timing/memory from this additional run are not reported because a CPU build ran concurrently.
- Earlier allocation-only Vulkan tests: ten exact Base/Turbo comparisons with memory saver disabled; sampled full-sequence device peak fell by 0.885/1.045 GiB respectively.

Warm cache savings apply to identical references, especially memory saver mode. Cold requests compute the original timbre graph; the cache does not remove first-request timbre allocation. Without memory saver, retained graphs dominate and warm speed differences were negligible. Longer lyrics or other graphs can dominate a mixed-workload peak. CPU, Metal, editing routes, longer songs and parallel sessions have not been validated by this focused PR.

## Standalone PR branch validation

Ported only the three model changes onto upstream `ed96b730`, without the development branch's slot/server edits. CUDA CLI/server and all enabled test targets built. The standalone one-session server matched the development candidate for all 56 mixed-history responses across Base/Turbo/XL Turbo/XL SFT (14 each). Both fresh Turbo and XL Turbo conditioning probes passed ten histories each. Compact hashes/counts are in `ace_step_graph_memory_standalone.json`.

On Windows MSVC, configure `CMAKE_CXX_FLAGS="/utf-8 /EHsc"`: without `/utf-8`, two unrelated Unicode-literal unit tests fail; both pass with the setting. The remaining `http_live_body_test` process crash (0xc0000409) also reproduces on the unchanged upstream executable. It is not an ACE-Step inference test. Final CTest: 97 passed, four skipped (missing model assets), one failed out of 102. These targeted results do not claim a fully passing repository gate.

The portable graph/cache source was also built and exercised on the existing Vulkan development branch (28 exact Base/Turbo comparisons above); the standalone PR branch has now also been built on Vulkan and validated with both XL variants (see below).

## Reproduction

Run commands from the repository root so model specifications can be located. Build the CLI/server and model probe using the normal project instructions, enabling `ENGINE_BUILD_MODEL_TESTS=ON`. The standalone probe requires a real GGUF; it is deliberately not an unconditional CTest requiring multi-gigabyte model downloads.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENGINE_ENABLE_CUDA=ON -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
build/bin/ace_step_timbre_cache_probe /path/ace-step-1.5-turbo-q8_0.gguf acestep-v15-turbo cuda 0
build/bin/ace_step_timbre_cache_probe /path/ace-step-1.5-xl-turbo-q8dit.gguf acestep-v15-xl-turbo cuda 0
```

For Vulkan, build with CUDA off and Vulkan on, then pass `vulkan` and the NVIDIA Vulkan device index (1 on this test system). Model packages were downloaded from `audio-cpp/audio.cpp-gguf`, under `ACE-Step1.5-GGUF`.

Server request configuration: `backend=cuda`, `device=0`, `threads=8`, `lazy_load=true`; a single `ace_step` generation model with `session_options.ace_step.mem_saver=true`. Select XL explicitly with `load_options.ace_step.dit_model_path=acestep-v15-xl-turbo` or `acestep-v15-xl-sft`, and bf16 text/planner session weight types. Submit the caption eight times to `/v1/tasks/run` with `duration_seconds=5`, `seed=1234`, `thinking=false`, and all `use_cot_*` switches false. Compare decoded audio SHA-256 and meaningful response fields against an identical baseline sequence; exclude profiling/timing metadata. Measure warm medians separately from cold load and subtract each request's prior idle memory reading, not the GGUF file size.

Raw local WAVs/traces/executables are retained under `outputs/ace-graph-memory-20260929/` and `outputs/ace-next-20260929/` in the test workspace; those large artifacts are not part of the repository. The compact recorded CUDA summary is adjacent to this document.

## Follow-up: standalone XL Vulkan validation

Both XL Turbo and XL SFT Q8-DiT passed **56/56 exact server comparisons** across memory saver on/off, plus **20/20 direct conditioning histories**. Both CLI/server and the conditioning probe built with Vulkan on the standalone PR branch. Controls used the same server/framework/build settings, with only the three ACE-Step model files restored to upstream `ed96b730`; candidate sources were restored byte-for-byte and rebuilt afterward.

| Model | Original warm time (s) | Optimized warm time (s) | Less time | Original extra warm VRAM (GiB) | Optimized extra warm VRAM (GiB) | Less extra VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XL Turbo Q8 DiT | 0.744 | 0.618 | 16.9% | 2.373 | 0.820 | 65.4% |
| XL SFT Q8 DiT | 0.721 | 0.620 | 14.1% | 2.373 | 0.820 | 65.4% |

Memory saver enabled, one session, RTX 3090, same supplied caption, seed 1234, five seconds, eight diffusion steps. Warm medians: requests 2-8. Extra VRAM is sampled request peak minus preceding loaded idle, not all non-weight bytes. No compilation/other GPU workload overlapped measurements. With memory saver disabled, measured warm latency fell by 3.1%/5.1%; incremental warm memory stayed unchanged, while full mixed-sequence device peaks (including weights) fell from 15.814/15.706 to 14.425 GiB. These measurements do not certify arbitrary requests or a particular smaller GPU capacity; optimized mixed-sequence peaks with memory saver on were 12.731/12.624 GiB.

See [the XL Vulkan report](ace_step_xl_vulkan.md) and [compact results](ace_step_xl_vulkan.json) for the full matrix, executable hashes, probes and reproduction. Multi-slot Vulkan XL, editing routes and longer songs remain outside this validation.
