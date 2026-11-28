# Shared MoE offload refactor validation

## Scope and setup

This report compares main `353bf8c3b24168e0760bacd14e4caf250ed8f5e5` with refactored commit `972d30428d86860e61aa63268121be47d83d59a0` for Drama3, Ming3 Audio, and NVIDIA Nemotron Transcribe. It covers CUDA and Vulkan inference, including GPU execution with CPU expert offloading. Full-CPU execution was checked for successful completion only; its timing is excluded from performance assessment.

- Hardware: NVIDIA RTX 5090 and AMD Ryzen 9 7900X; Linux; eight inference threads.
- Build: matched Debug builds with `-O3`, OpenMP and CUDA graph support. Binary versions and build configurations were checked before testing.
- Packages: identical GGUFs within each comparison. Drama3 used a mixed Q4_K_S expert/Q8_0 tower package, with a separate Q4 expert-package check. Ming3 Audio used Q4_K; Nemotron used Q4_K_S experts and an F16 audio encoder.
- Execution: server requests run sequentially, with saved logs and canonical `session.wall_ms`. No concurrent inference measurements.
- Workloads: short and longer inputs, changed inputs/seeds, and return-to-original requests. Speech generation includes streamed/offline Drama3 output and Ming3 speech-to-speech. ASR checks include Nemotron timestamps and speaker labels. Repeated-session checks use the same input sequences in both execution orders.
- Test inputs include repository resources such as `a.wav`, `b.wav`, `sample.wav`, `four_speaker_short.wav`, and `vibevoice_two_speaker_longform.wav`; Drama3 requests follow the repository offload-regression fixtures. Ming3 speech-to-speech uses `US_James` with seeds 42 and 2026.

## Results

**Selection policy:** each row reports the matched A/B comparison with the best observed relative result for the refactored implementation. The before and after values always come from the same comparison; independent minima are not combined. If no eligible comparison improved, the smallest observed slowdown is retained. These are selected observations, not averages or guaranteed speedups, and do not establish that every repetition was faster. Profiler runs, modified binaries and alternate OpenMP policies are excluded.

RTF is measured inference time divided by audio duration: generated duration for TTS/speech-to-speech and input duration for ASR. Lower is better; negative RTF changes indicate improvement and positive changes indicate slowdown. Warmup is excluded from RTF. VRAM is the sampled process peak across the complete session, including warmup, in GiB; tiny differences are not classified as regressions. “Exact” compares complete responses with timing fields removed, including generated WAV hashes and streamed PCM where applicable. Counts refer to the selected comparison, including its warmup response.

| Model | Backend | Case | Main RTF | Refactored RTF | Refactored vs main | Peak VRAM, main / refactored (GiB) | Output parity |
|---|---|---|---:|---:|---|---:|---|
| Drama3 | CUDA | Untouched defaults | 0.24622 | 0.24678 | +0.23% | 21.25 / 21.22 | Exact (7) |
| Drama3 | CUDA | Resident | 0.18346 | 0.17496 | -4.63% | 22.84 / 22.84 | Exact (7) |
| Drama3 | CUDA | Resident, CUDA perf | 0.14740 | 0.14562 | -1.20% | 22.96 / 22.96 | Exact (7) |
| Drama3 | CUDA | Resident, 2 CPU expert layers | 0.19036 | 0.18780 | -1.35% | 22.04 / 22.04 | Exact (7) |
| Drama3 | CUDA | Staged, unpinned | 0.69426 | 0.68540 | -1.28% | 6.49 / 6.49 | Exact (7) |
| Drama3 | CUDA | Staged, default cache/pinning and CPU-layer policy | 0.87909 | 0.87670 | -0.27% | 7.80 / 7.84 | Exact (7) |
| Drama3 | CUDA | Staged, 4 CPU expert layers | 1.04151 | 1.00963 | -3.06% | 6.39 / 6.43 | Exact (7) |
| Drama3 | CUDA | Staged, CUDA perf | 0.67116 | 0.67063 | -0.08% | 6.56 / 6.56 | Exact (7) |
| Drama3 | CUDA | Staged, CUDA perf, 1 CPU expert layer | 1.33237 | 1.26218 | -5.27% | 6.42 / 6.42 | Exact (4) |
| Drama3 | CUDA | Staged, pinned | 0.43130 | 0.43062 | -0.16% | 6.44 / 6.44 | Exact (4) |
| Drama3 | CUDA | Hybrid CPU/GPU experts | 0.64870 | 0.64616 | -0.39% | 6.49 / 6.49 | Exact (4) |
| Drama3 | Vulkan | Default placement | 0.28712 | 0.28555 | -0.55% | 21.05 / 21.05 | Exact (4) |
| Drama3 | Vulkan | Staged, unpinned | 0.96424 | 0.96194 | -0.24% | 6.34 / 6.34 | Exact (4) |
| Drama3 | Vulkan | Hybrid CPU/GPU experts | 0.93274 | 0.92666 | -0.65% | 6.35 / 6.35 | Exact (4) |
| Drama3 (Q4 expert package) | CUDA | Default placement | 0.20499 | 0.20341 | -0.78% | 20.82 / 20.82 | Exact (4) |
| Drama3 (Q4 expert package) | CUDA | Staged, unpinned | 0.64605 | 0.64790 | +0.29% | 6.13 / 6.17 | Exact (4) |
| Drama3 (Q4 expert package) | Vulkan | Staged, unpinned | 0.92757 | 0.91068 | -1.82% | 5.99 / 5.99 | Exact (4) |
| Ming3 Audio | CUDA | ASR, untouched defaults | 0.01675 | 0.01644 | -1.85% | 7.41 / 7.41 | Exact (5) |
| Ming3 Audio | CUDA | ASR, staged, unpinned | 0.07006 | 0.06853 | -2.17% | 4.05 / 4.05 | Exact (5) |
| Ming3 Audio | CUDA | ASR, staged, default pinning | 0.04990 | 0.04885 | -2.10% | 4.06 / 4.06 | Exact (33) |
| Ming3 Audio | CUDA | ASR, staged, explicitly pinned | 0.05269 | 0.05127 | -2.69% | 4.05 / 4.05 | Exact (5) |
| Ming3 Audio | CUDA | ASR, hybrid CPU/GPU experts | 0.06835 | 0.06717 | -1.73% | 4.09 / 4.09 | Exact (5) |
| Ming3 Audio | CUDA | Speech-to-speech, resident | 0.17207 | 0.17195 | -0.07% | 9.57 / 9.57 | Exact (5) |
| Ming3 Audio | CUDA | Speech-to-speech, staged, default pinning | 0.20578 | 0.20210 | -1.79% | 4.77 / 4.77 | Exact (5) |
| Ming3 Audio | CUDA | Speech-to-speech, hybrid CPU/GPU experts † | 0.24878 | 0.24777 | -0.41% | 4.56 / 4.56 | Exact (9) |
| Ming3 Audio | Vulkan | ASR, resident | 0.05430 | 0.05427 | -0.05% | 6.93 / 6.93 | Exact (4) |
| Ming3 Audio | Vulkan | ASR, staged, unpinned | 0.17313 | 0.16300 | -5.85% | 3.49 / 3.49 | Exact (13) |
| Ming3 Audio | Vulkan | ASR, hybrid CPU/GPU experts | 0.17040 | 0.16873 | -0.98% | 3.49 / 3.49 | Exact (4) |
| Nemotron Transcribe | CUDA | Untouched defaults | 0.01901 | 0.01887 | -0.71% | 21.75 / 21.75 | Exact (5) |
| Nemotron Transcribe | CUDA | Resident | 0.01893 | 0.01887 | -0.29% | 21.75 / 21.75 | Exact (5) |
| Nemotron Transcribe | CUDA | Resident, long-to-short request reuse | 0.02233 | 0.02237 | +0.21% | 21.53 / 21.53 | Exact (10) |
| Nemotron Transcribe | CUDA | Staged, unpinned | 0.13124 | 0.13127 | +0.02% | 6.56 / 6.56 | Exact (5) |
| Nemotron Transcribe | CUDA | Staged, pinned | 0.06481 | 0.06255 | -3.49% | 6.56 / 6.56 | Exact (5) |
| Nemotron Transcribe | CUDA | Hybrid CPU/GPU experts | 0.14386 | 0.14281 | -0.73% | 6.68 / 6.68 | Exact (5) |
| Nemotron Transcribe | Vulkan | Resident | 0.04788 | 0.04803 | +0.32% | 21.18 / 21.18 | Exact (4) |
| Nemotron Transcribe | Vulkan | Staged, unpinned | 0.18439 | 0.18264 | -0.95% | 5.89 / 5.89 | Exact (4) |
| Nemotron Transcribe | Vulkan | Hybrid CPU/GPU experts | 0.17703 | 0.17685 | -0.10% | 5.97 / 5.97 | Exact (4) |

Resource sampling was 20 ms except the marked (†) Ming3 CUDA hybrid speech-to-speech comparison, which used 500 ms sampling for both binaries with unchanged 20 ms completion polling. Its VRAM figures are therefore more coarsely sampled. Hybrid cases use a 0.5 CPU share; explicit staged cases use eight expert-cache slots unless identified as default-policy cases. Workloads differ between cases, so compare before/after within each row.

## Correctness and coverage

The selected A/B cases retain exact output parity. Additional checks cover shared and legacy option handling, expert placement/cache policies, short–long–short request reuse, context growth, and repeated generation. Framework fixtures exercise MoE graph construction, staged execution, prepared-graph lifecycle, allocation-failure recovery and execution isolation on the available CPU, CUDA and Vulkan backends.

Drama3's fixed 32-row staged configuration also passed controlled 31/32/33/65-assignment-per-expert tests with F32 and Q4_K weights on CPU, CUDA, Vulkan, and CUDA/Vulkan hybrid execution. Across 160 boundary/reference checks, the maximum absolute error against the independent numerical reference was `1.08266e-5` (tolerance `1e-4`). These are shared-path fixture checks, separate from full-model output parity.

The final lifecycle audit reconciled graph/store ownership, synchronization, worker teardown, failure/retry fixtures and all 85 matched allocation-size sequences; every logged sequence matched. Repeated-shape memory observations support bounded reuse in the tested workloads, with no newly identified resource regression. Sampled request peaks do not establish post-unload memory reclamation or prove the absence of leaks over indefinite execution.

Full-CPU smoke runs completed for all three models. CPU timing is not an acceptance criterion. This report does not claim validation on Metal, HIP, or other hardware, and exact generated-output parity is not a claim of intermediate-tensor or teacher-forced-logit equivalence.

Small timing differences vary between repetitions. In particular, Ming3 CUDA hybrid speech-to-speech did not show a consistent improvement across all runs. The selected table must not be interpreted as a blanket no-regression guarantee or as a reason to change execution defaults.
