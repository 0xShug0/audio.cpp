# WeightNormFoldedTensorSource: backend upload performance

## Summary

`WeightNormFoldedTensorSource` inherited `TensorSource::set_backend_tensor`. Ordinary tensors passing through the wrapper were therefore materialized as host-side `TensorData` bytes and copied again to the backend, bypassing optimized upload implementations in the wrapped source.

The wrapper now delegates ordinary tensors to the wrapped source. Synthesized weight-normalized tensors are folded to F32 and uploaded through `set_backend_tensor_from_f32_parallel`. The requested output storage type and expected-shape checks are preserved. The F32 convenience override follows the same implementation.

## Benchmark method

A temporary synthetic `TensorSource` exercised `BackendWeightStore::load_tensor` with two tensors per iteration:

- Ordinary tensor: F32, 2048 × 4096, 32 MiB.
- Folded tensor: F32, 512 × 2048, 4 MiB.

The harness records calls to the wrapped source's `set_backend_tensor`. Each process performs one warm-up and five timed iterations; each iteration includes both loads, backend upload, and backend synchronization. The reported process value is the median of those five iterations. CPU results aggregate five independent process medians; HIP results include one baseline process and two post-change processes. The table uses the final post-change HIP process, and the additional HIP result is reported below.

Baseline and fixed implementations were compiled in matching temporary Release configurations. CPU settings were native CPU plus OpenMP, with eight runtime threads. The GPU backend was HIP on the RX 9070 XT. Harness, binaries, build directories, raw results, and logs are under `/tmp/audio-cpp-bug09` and `/tmp/audio-cpp-weight-norm-cpu-after`.

## Results

| Backend | Before median | After median | Change | Wrapped-source upload calls per process |
|---|---:|---:|---:|---:|
| AMD Radeon RX 9070 XT (HIP) | 11.834 ms | 4.0656 ms | 65.6% less time; 2.91× faster | 0 → 6 |
| Intel Core Ultra 7 265K (CPU, 8 threads) | 14.9958 ms | 6.95678 ms | 53.6% less time; 2.16× faster | 0 → 6 |

CPU process medians, in milliseconds:

- Before: 15.0830, 15.7379, 14.8985, 14.9958, 14.5151. Median: 14.9958 ms.
- After: 6.95678, 6.97671, 6.89995, 7.09618, 6.94903. Median: 6.95678 ms.

HIP process medians:

- Before: 11.834 ms.
- After: 4.74472 ms and 4.0656 ms. The table reports the final run; the spread is retained here for transparency.

The direct-upload count is six after the fix because each process performs one warm-up and five timed iterations, with one ordinary tensor delegated per iteration. The folded tensor still requires weight-norm calculation and F32 conversion. These measurements characterize this synthetic load/upload path; they are not end-to-end model-generation speedups.

Raw outputs are in `/tmp/audio-cpp-bug09/results/hip-before.txt`, `hip-after.txt`, `hip-after-final.txt`, `cpu-before-controlled-repeats.txt`, and `cpu-after-controlled-repeats.txt`.

## Regression test and builds

`gguf_tensor_source_test` adds a counting source and checks that an ordinary tensor delegates once, folded values arrive correctly, and a mismatched folded shape is rejected. The test passed in both CPU-only and HIP-enabled Release builds.

HIP build and test:

```sh
cmake --build /tmp/audio-cpp-bug09-hip-before --target audiocpp_cli gguf_tensor_source_test -j 8
/tmp/audio-cpp-bug09-hip-before/bin/gguf_tensor_source_test
```

CPU-only configuration and test:

```sh
cmake -S . -B /tmp/audio-cpp-weight-norm-cpu-after -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_CUDA=OFF \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_EXAMPLES=OFF -DENGINE_BUILD_MODEL_TESTS=OFF \
  -DENGINE_BUILD_EXTENDED_TESTS=OFF -DENGINE_BUILD_WARMBENCH=OFF \
  -DENGINE_ENABLE_NATIVE_CPU=ON -DENGINE_ENABLE_OPENMP=ON
cmake --build /tmp/audio-cpp-weight-norm-cpu-after --target gguf_tensor_source_test -j 8
/tmp/audio-cpp-weight-norm-cpu-after/bin/gguf_tensor_source_test
```

## 60-second music-model smoke test

ACE-Step Turbo Q8 (`ace-step-1.5-turbo-q8_0.gguf`, 5.8 GiB) loaded on HIP and generated `/tmp/audio-cpp-bug09/results/ace-step-60s.wav`. Its WAV header reports PCM 16-bit stereo at 48 kHz and 2,880,000 frames, exactly 60.000 seconds.

- Wall time: 17,742.60 ms.
- Peak RSS: 2,867 MiB.
- Peak virtual address space (Linux VSZ): 18,095 MiB.
- RTF: 0.30 (3.38× real time).

This verifies model loading and 60-second output on the AMD backend. ACE-Step does not use `WeightNormFoldedTensorSource`, so this is a separate smoke test and not evidence for the upload benchmark result.

## Environment and scope

- AMD Radeon RX 9070 XT, gfx1201; ROCm 7.2.53211; HIP reports 16,304 MiB VRAM.
- Intel Core Ultra 7 265K; GCC 16.2.1; Linux x86_64.
- The Intel integrated GPU was not benchmarked. The results cover CPU and AMD HIP only.
- Generated audio, model files, build trees, benchmark harness, and test outputs remain outside the repository.

## Revision

Local branch: `ZeroDot1/weight-norm-upload-perf`

Commit subject: `fix(assets): preserve direct uploads in weight norm tensor views`
