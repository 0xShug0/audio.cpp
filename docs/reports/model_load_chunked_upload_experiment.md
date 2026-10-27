# Chunked model loading and conversion experiment

Date: 2026-10-27
Branch: `ZeroDot1/model-load-optimizations`
Base: `upstream/main` at `1b8c8122`

## Scope

This branch combines four code changes for evaluation:

1. **Asynchronous chunked backend uploads:** enqueue transfers of mapped GGUF tensor data in bounded chunks, synchronizing before a staging slot is reused and after the store finishes.
2. **Lazy rotating staging buffers:** allocate two 64 MiB HIP/CUDA staging buffers only when a compatible upload is queued, then reuse them during the store upload.
3. **Reusable HIP fallback staging:** reuse a thread-local staging vector, capped at 64 MiB, for the synchronous HIP fallback path.
4. **Parallel format conversion:** decode large F16, BF16, and quantized GGUF tensors into F32 in OpenMP-parallel ranges/rows. Small tensors keep the serial path.

The file remains one mmap-backed GGUF. The GPU-upload change chunks transfers from that mapping; it does not split the model into separate files or add parallel disk readers.

## Test machine and model

- CPU: Intel Core Ultra 7 265K, 20 cores / 20 threads.
- Memory: 62 GiB system RAM.
- GPU: AMD Radeon RX 9070 XT, gfx1201, 16,304 MiB reported VRAM.
- Software: Linux x86_64, ROCm 7.2.53211, driver 7.3.0-4-cachyos.
- HIP build: Release, Clang 22.0.0, OpenMP enabled, full model catalog.
- CPU build: Release, GCC 16.2.1, OpenMP enabled, ACE-Step custom catalog.
- Model: ACE-Step 1.5 Turbo Q8 GGUF, `6,185,460,032` bytes (about 5.76 GiB).

Before each run, the test issued Linux `POSIX_FADV_DONTNEED` for the GGUF. This is advisory; it does not guarantee that every relevant page was evicted. Runs were sequential. Each CLI process exited before the next run. No claim is made that all system RAM or GPU memory was forcibly cleared.

## Direct file-read chunk test

The temporary benchmark used `O_DIRECT`, 4 MiB aligned reads, and five sequential repetitions per configuration. Every configuration read the same byte count and returned the same checksum. Median results:

| Reader strategy | Median time | Throughput | Change vs. one reader |
|---|---:|---:|---:|
| 1 reader, sequential | 1.27951 s | 4,610 MiB/s | baseline |
| 2 readers, sequential | 1.28064 s | 4,606 MiB/s | −0.09% |
| 4 readers, sequential | 1.28309 s | 4,597 MiB/s | −0.28% |
| 8 readers, sequential | 1.27846 s | 4,614 MiB/s | +0.08% |
| 20 readers, sequential | 1.36522 s | 4,321 MiB/s | −6.28% |
| 20 readers, round-robin | 1.34975 s | 4,370 MiB/s | −5.20% |
| 20 readers, random assignment | 1.36962 s | 4,307 MiB/s | −6.58% |

This separate exploratory test did not support parallelizing raw file reads on this NVMe device. The branch's implementation chunks mapped-file-to-GPU transfers; it does not parallelize disk reads.

## ACE-Step Q8 end-to-end test

Baseline (`upstream/main`) and candidate (this branch) used the same 5.761 GiB Q8 model, prompt/lyrics, seed (`1234`), HIP backend, and 60-second output request. Three sequential pairs were performed, with each process exiting before the next. The final candidate was built after lazy staging allocation. The same output was generated in all six runs: 60.000 seconds, 48 kHz, stereo. `POSIX_FADV_DONTNEED` was issued for the model before each run; Linux treats this as advisory, so this does not prove a fully cold disk cache. No system-wide RAM or VRAM clearing is claimed.

Per-run results:

| Pair | Variant | CLI wall (ms) | Outer wall (s) | Peak RSS (MiB) | Peak Linux virtual footprint (MiB) | Weight upload total (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Baseline | 15,105.02 | 15.917 | 1,340 | 18,139 | 5,115.26 |
| 1 | Candidate | 16,516.35 | 17.415 | 1,340 | 18,379 | 6,464.41 |
| 2 | Baseline | 16,659.15 | 17.715 | 1,341 | 18,139 | 6,660.99 |
| 2 | Candidate | 16,675.46 | 17.716 | 1,340 | 18,379 | 6,582.85 |
| 3 | Baseline | 16,647.45 | 17.715 | 1,340 | 18,139 | 6,626.56 |
| 3 | Candidate | 16,540.21 | 17.420 | 1,341 | 18,379 | 6,459.37 |

The weight upload total is the sum of the instrumented planner, VAE decoder, DiT, and text encoder uploads. Their individual timings in pairs 1/2/3 (ms) were:

| Component | Baseline | Candidate |
|---|---|---|
| Planner | 2,009.46 / 2,636.67 / 2,677.42 | 2,695.66 / 2,732.99 / 2,661.81 |
| VAE decoder | 17.79 / 17.82 / 17.77 | 46.58 / 45.57 / 43.44 |
| DiT | 2,389.87 / 3,035.80 / 3,061.23 | 2,890.57 / 2,919.80 / 2,927.96 |
| Text encoder | 698.14 / 970.70 / 870.13 | 831.61 / 884.49 / 826.17 |

The medians are:

| Measurement | Baseline median | Candidate median | Difference |
|---|---:|---:|---:|
| CLI-reported wall time | 16,647.45 ms | 16,540.21 ms | 0.64% faster |
| Outer process wall time | 17.715 s | 17.420 s | 1.67% faster |
| Weight-store upload time | 6,626.56 ms | 6,464.41 ms | 2.45% faster |
| Peak RSS | 1,340 MiB | 1,340 MiB | effectively unchanged |
| Peak Linux virtual footprint | 18,139 MiB | 18,379 MiB | +240 MiB |

These small median differences are not evidence of a large or reliable user-visible speedup. The individual runs varied: the first pair was much faster than the later pairs, and the candidate was slower in that first pair. The branch is retained as an experiment at the user's request, not presented as a proven general performance win.

All six WAV files were exactly 60.000 seconds at 48 kHz stereo. Baseline and candidate SHA-256 hashes were identical across all three pairs:

```text
6ee36704575fa3f17446360dc1993f3985ea59d37d4cf63fda5d9b0d74f73e5d
```

## Conversion-path test

The final branch candidate was run once with `ace_step.text_encoder_weight_type=bf16` on the Q8 GGUF to exercise quantized decoding and conversion for the text encoder. It completed successfully, generated a 60.000-second WAV, and reported 17,404.18 ms CLI wall time, 2,064 MiB peak RSS, and 19,956 MiB peak Linux virtual footprint. Its output SHA-256 was `5283064b789b650d5471f207880eb8595bdd0355c77001d6671e01d90f185f37`. This is a correctness run only: there was no final-branch baseline run for this conversion path, so no speed or memory comparison is available.

## Build and test results

- Full HIP Release build with all model targets: `audiocpp_cli` and `gguf_tensor_source_test` built successfully.
- HIP `gguf_tensor_source_test`: passed.
- CPU-only Release build with the custom ACE-Step catalog: `audiocpp_cli` and `gguf_tensor_source_test` built successfully.
- CPU-only `gguf_tensor_source_test`: passed.
- The full 60-second model-generation benchmark was run on HIP. CPU-only model-generation speed was not measured.

## Limitations

- Only one AMD discrete GPU and one Intel CPU were measured. There is no CUDA hardware result, no iGPU result, and no AMD CPU result.
- The raw parallel file-reader test showed no material gain and is not part of the production code changes.
- The Q8 run exercises chunked backend upload and staging reuse. The BF16 override exercises quantized-to-F32 decode, but only for one ACE-Step component.
- Memory readings are process RSS and Linux virtual footprint. They are not a full system-wide RAM or VRAM audit; sampled VRAM readings are not included as exact peaks.
- The current end-to-end measurements do not demonstrate a repeatable 5% speedup. Keep this branch experimental until broader, cooler, randomized-order runs show a clear benefit without memory or correctness regressions.

## Scope of this report

This report contains results from the current `ZeroDot1/model-load-optimizations` work only: its file-read chunk experiment, final branch ACE-Step baseline comparison, final candidate BF16 correctness run, and CPU/HIP builds and tensor-source tests. It intentionally excludes measurements from earlier, separate PR comparisons. Raw logs, JSON, WAV files, temporary benchmark harnesses, and build outputs are outside the repository at `/tmp/audio-cpp-pr1060-bench.5npAs6`; they were not committed.
