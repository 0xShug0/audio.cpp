# CPU and Model Loading Optimizations

Date: October 27, 2026

Comparison baseline: `origin/main` at `e09bf3749f276a0138b944e080cabac5f82022c9`

## Changes

- On Linux, the automatic thread default uses the physical CPU cores allowed to the process.
  CPU affinity and sysfs topology are respected; other platforms use `std::thread::hardware_concurrency()`. The explicit `--threads N` option still overrides the default.
- OpenMP stages in audio DSP and Mel frontend projection use the same requested thread count.
  Small tasks avoid starting an unnecessarily large OpenMP team. Static scheduling distributes work evenly.
- HIP model weights are still copied in chunks from the mapped source file into the backend buffer.
  The host staging buffer is now at most 32 MiB and is reused within a weight store instead of allocating up to 64 MiB for every tensor. Its memory is released after the store.
- With timing logs enabled, each backend weight store reports separate
  `*.allocate_ms` and `*.upload_ms` values, separating VRAM buffer allocation from data upload.

The regular CPU file source already uses memory mapping and copies same-format tensors directly into the backend buffer. No additional intermediate copy was introduced for this path. The shared thread default improves CPU execution and also affects CPU work performed during GPU tasks.

## Hardware and Builds

- CPU: Intel Core Ultra 7 265K, Linux, 20 physical cores.
- GPU: AMD Radeon RX 9070 XT (`gfx1201`), ROCm/HIP; ROCm reports 16,304 MiB of available VRAM.
- Release builds with GCC 16.2, OpenMP 5.2, and HIP compiler Clang 22.
- The music model comparison used one build from unchanged `origin/main` and one build with all changes from this branch. Both used the same GPU, model directory, prompt, seed, and generation options. `--threads` was omitted in both runs so each build used its default.

## Before and After: MiniMax Music 3 on HIP

The run used package `minimax_music3_q4_0` (8,446,098,960 bytes), containing a Q4_0 language model, Q8_0 depth model, and the other included components. The short reproducible run used `duration_sec=1`, `num_inference_steps=1`, and seed `42`. The generated audio was 998.46 ms long and byte-identical across all six runs.

| Runtime (`metrics.wall_ms`) | `origin/main` | Branch with all patches |
|---|---:|---:|
| Individual runs | 2915.35 / 2923.69 / 2931.23 ms | 2890.04 / 2742.51 / 2747.99 ms |
| Median | 2923.69 ms | 2747.99 ms |
| Median change | — | **5.99% faster** |
| Reported host RSS peak | 1950 MB in all three runs | 1918 MB in all three runs |

SHA-256 of the generated WAV in all runs:
`656addfb04fc3031916a1734138570d856d2770f6926ec496801c047fed4ac98`.

Patched upload timings were 980.1 / 883.4 / 890.2 ms across the five weight stores. In the direct staging A/B test, with the same explicit thread count and timing instrumentation, the median was 1061.4 ms with the previous 64 MiB buffer per tensor and 904.6 ms with the reused 32 MiB buffer (**14.8% faster**). The measured host RSS peak fell from 1950 to 1918 MB.

The five reported weight buffers occupy 7640.95 MiB in the backend. The GPU total observed through ROCm system counters was approximately 8.0–8.6 GiB during model runs; this change does not reduce weight memory in VRAM. VRAM usage is therefore unchanged while staging RAM and transfer time decrease.

## Before and After: CPU TTS

With PocketTTS Q8_0, voice `alba`, seed `42`, and the same short text, the unchanged build (default 4 threads) and the new build (20 detected physical cores) each produced 6.4 seconds of audio. The three `metrics.wall_ms` measurements were 1412.81 / 1446.21 / 1373.47 ms versus 1321.39 / 1299.39 / 1270.15 ms. The medians were 1412.81 ms and 1299.39 ms, making the new build **8.0% faster**. All six WAV files had the same SHA-256:
`8b795e81595d0a8cc0785ff523ca5b4f0ba44b792d44454fc8f630d28d288a2a`.

An additional 60.88-second run was generated with the same CPU TTS build against the baseline. In this single comparison, runtime fell from 14,152.8 to 12,132.1 ms (**14.3% faster**); both WAV files were byte-identical, SHA-256:
`08f1580c2248ffa0c51b50ef27450eaab0bd9d2ac5e0de37a7b7fe2e77944ba0`.

## Tests

- HIP/OpenMP Release: `audiocpp_cli`, `mel_spectrogram_frontend_test`,
  `audio_dsp_test`, `weight_adapter_source_test --log`, and
  `cpu_extra_buffers_test` were built and passed.
- CPU/OpenMP Release:  `audiocpp_cli`, `mel_spectrogram_frontend_test`, `audio_dsp_test`, and
  `audiocpp_server` were built; DSP tests passed.
- CPU build without audio.cpp OpenMP: Mel frontend and DSP tests were built and passed.
- The Mel frontend was checked in CPU and HIP builds against a 60-second WAV; test cases compare results at 1, 2, and 4 threads bit-for-bit against the reference.
- Music and TTS runs used identical seeds and content before and after the changes; WAV checksums match.

## Rejected Experiments and Limitations

A 16 MiB staging buffer reduced peak host RAM further but made music model upload about 9% slower at the median, so it is not part of the patch. A reused 64 MiB buffer was faster than the old per-tensor allocation, but the 32 MiB buffer was slightly faster in the A/B test and reduced measured RSS peak.

The tested AVX2 Mel projection showed no reliable speedup in the 60-second frontend benchmark. `-march=native` for all runtime code helped only slightly at 20 threads and slowed a 12-thread run. Neither experiment was included.

These measurements show gains for the tested CPU TTS configuration and HIP music run. They do not establish the same percentage for every model family, CPU generation, or GPU. Model-specific memory and execution strategies require separate patches.

## Installer Configuration

The Arch Linux installer `scripts/install_arch_amd.sh` explicitly passes `--native-cpu ON` by default, enabling ggml CPU kernels for the build host instruction set. Set `AUDIOCPP_NATIVE_CPU=OFF` for portable CPU kernels. The Linux build script accepts the same environment variable; an explicit `--native-cpu ON|OFF` argument takes precedence. This option controls ggml CPU kernel compilation and does not add `-march=native` to all audio.cpp runtime code.

On Arch Linux, run `bash scripts/install_arch_amd.sh` and select the desired GPU target mode. The installer forwards its default `AUDIOCPP_NATIVE_CPU=ON` setting to the Linux build script. For portable CPU kernels, run `AUDIOCPP_NATIVE_CPU=OFF bash scripts/install_arch_amd.sh`.
