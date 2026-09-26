# Higgs v3 CUDA prefill memory reuse

The Higgs AR prefill graph previously reserved storage for every intermediate
tensor across all decoder layers. On a cold cloned-voice request, that graph
includes the reference prefix as well as the requested text. Cached requests
process a smaller suffix, explaining why their memory peak was lower.

For CUDA prefill writing into an external KV cache, use `ggml_gallocr` to reuse
intermediate storage after its last consumer. Mark all six uploaded inputs as
inputs and retain the existing logits output flag. The KV cache remains
externally owned. RAII frees the allocator after backend graph resources.

No CUDA kernels, weights, arithmetic precision, sampling options, or graph
operations change. CLI and server share this implementation. Non-CUDA,
layerwise long-prompt prefill, and prefill exporting KV state without an
external cache keep their previous allocation behavior.

## Initial measurement

Windows 11, RTX 3090 (24 GiB), CUDA 12.4, Release MSVC, eight CPU threads,
bundled `assets/resources/b.wav`, seed 1234, `max_tokens=1024`, remaining
generation settings at defaults. Model: `higgs-audio-v3-tts.gguf`, 5,095,340,672
bytes, SHA-256 `6140c39b4d3baa24082fb0614069ad87644c8449525396865940dd8cbefe2db6`.
This local legacy GGUF uses `model_specs/higgs_audio_tts.json` as a spec override.

Text:

> Ptaszki ćwierkają, że Apple chce wrócić na rynek serwerów dostarczając chłonnemu rynkowi AI swoje własne rozwiązania bazujące na zmodyfikowanych układach z serii M.

All peaks below **exclude the 4.931 GiB loaded-model baseline**. They include
KV, graphs, codec work, and backend allocations, rather than only KV memory.
NVML sampling interval was approximately 10 ms. These are sampled process
peaks on an otherwise idle GPU, rather than allocator high-water marks.

| Additional peak VRAM | Before | After | Saved |
|---|---:|---:|---:|
| One session, first request | 5.570 GiB | 0.510 GiB | 5.061 GiB (90.8%) |
| One session, warm repeat | 1.172 GiB | 0.469 GiB | 0.703 GiB (60.0%) |
| Two concurrent sessions, first requests | 11.105 GiB | 1.014 GiB | 10.092 GiB (90.9%) |
| Two concurrent sessions, warm repeats | 2.309 GiB | 0.902 GiB | 1.406 GiB (60.9%) |

The concurrent-session rows are additional evidence from a separate local slots
experiment. **Slots are not part of this change.** That experiment holds all
other code constant between its before/after builds. Standalone validation
against the current upstream base is described below.

The actual KV allocation is unchanged: 144 MiB per session. Direct buffer-size
instrumentation identified the old cold prefill allocation as 5,734,983,680
bytes (5.341 GiB), the old warm prefill allocation as 992,608,000 bytes
(0.924 GiB), and the codec encoder graph as 482,499,200 bytes (460.15 MiB).
The temporary instrumentation is not included in the change.

One-session generation took 2.452 -> 2.286 s cold and 2.154 -> 2.126 s warm in
this measurement. Individual timing results are indicative, not a broad
throughput claim. Six cold/warm WAV comparisons across one and two sessions
were byte-identical. Eight additional WAV comparisons covered different seeds,
English/Polish text, output limits, reference reuse, and cache transitions.
Two unconditioned requests reached the existing token-limit error in both
builds; their errors matched and later referenced generation recovered.

## Standalone PR validation

Repeated against upstream `c7dbd4a4`, with no slots changes in either binary:

| One session | Upstream | Patched |
|---|---:|---:|
| First-request additional peak VRAM | 5.570 GiB | 0.510 GiB |
| Warm-repeat additional peak VRAM | 1.172 GiB | 0.469 GiB |
| First-request generation time | 2.498 s | 2.295 s |
| Warm-repeat generation time | 2.163 s | 2.135 s |

- Fresh Release CUDA build of the full model set, including CLI/server/GGUF.
- All 49 registered default CTest tests passed (extended/model tests disabled).
- Loader catalog sync self-tests and catalog/spec/package sync check passed.
- The checked-in benchmark passed eight byte-identical WAV comparisons and
  two matching token-limit errors, including successful generation after errors.
- CLI before/after generation of the Polish sentence was also byte-identical,
  and matched the server cold output.
- Cold WAV SHA-256 (both builds and CLI):
  `b34983738b8300a364ce4321a6bfd153b5da54d8238ca9693efea19fa1a8d5f3`.
- Warm WAV SHA-256 (both builds):
  `a11fb5604f732a56f2b9a40d4d9e517b8b5ec5049120addbd00873dfaa7a7e03`.
- `git diff --check` passed. Test processes were stopped and GPU returned to idle.

Local standalone artifacts are under `outputs/higgs-slots/memory-pr/runtime/`
and `outputs/higgs-slots/memory-pr/cli/` in the parent workspace. The benchmark
writes the same request, output, comparison, and memory metadata into whichever
output directory is supplied. Cold/warm outputs differ in the original
reference-cache behavior; parity comparisons match request order/cache state.

## Reproduction

Build upstream and patched server executables separately. On Windows, the
project build script supports:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_server -ModelSet full -CudaArchitectures 86-real -Jobs 8 -RunTests
```

Use a CUDA architecture matching your GPU. This machine has multiple CUDA
versions installed, so its clean configure additionally needed an explicit
`CMAKE_CUDA_COMPILER` pointing to CUDA 12.4 `nvcc.exe`; otherwise CMake found an
older toolkit first. Standard local build and CTest commands were then used.

The benchmark uses Python's standard library and NVML from the NVIDIA driver.
It runs the two executables sequentially on an otherwise idle GPU, measures
loaded-model memory, samples each request's peak, writes WAVs and JSON, checks
SHA-256 parity and matching error responses, and stops its servers.

```powershell
python tests/higgs_audio_tts/prefill_memory_bench.py `
  --before-server C:/bench/before/audiocpp_server.exe `
  --after-server C:/bench/after/audiocpp_server.exe `
  --model C:/models/higgs-audio-v3-tts.gguf `
  --model-spec model_specs/higgs_audio_tts.json `
  --reference assets/resources/b.wav `
  --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." `
  --output-dir outputs/higgs-prefill-memory
```

On Linux use the same arguments with local paths and normal shell continuations.
Other models, GPUs, and unchanged non-CUDA/layerwise paths have no new runtime
validation claimed by these measurements.
