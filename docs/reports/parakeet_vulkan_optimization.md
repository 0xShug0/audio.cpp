# Parakeet TDT: Vulkan state and allocation optimization

On Vulkan, each decoder step previously downloaded and re-uploaded LSTM state
and the predictor cache. The new path keeps that state in independently owned
device storage and appends state copies after all consumers of the old state.
The joint graph borrows the resident predictor cache; request reset clears it.
The joint graph is destroyed before its owning step graph. Graph-allocation
failure leaves local resources owned by RAII objects.

The encoder releases a mismatched cached graph before allocating its replacement
and frees the one-shot positional producer allocation after copying its outputs.
CPU/CUDA keep their host-state path. Only two Parakeet implementation files change;
no GGML, shared backend, precision-policy or framework changes are included.
The branch builds directly on upstream `86a51cab` and needs neither #861 nor #863.
This follows the [request to separate optimization](https://github.com/0xShug0/audio.cpp/pull/863#issuecomment-6100453087).

## Timing

Windows, Ryzen 7950X3D, 8 threads, RTX 3090, MSVC 14.43, CUDA 12.4 and Vulkan SDK
1.4.350. Warm HTTP wall time excludes loading. Trials alternate before/after/
before/after; each clip has two warmups and nine measurements per process,
18 measured samples per implementation. Percentages mean less processing time.
The benchmark composes the minimal Phonon variant with/without these two files;
normal backend math is used throughout. Independent branch execution is also
checked below. The speech clips are 11.9925 s and approximately 30 s.

| Vulkan package | 12 s before (ms) | After (ms) | Less time | 30 s before (ms) | After (ms) | Less time |
|---|---:|---:|---:|---:|---:|---:|
| Phonon-2 F32 | 66.02 | 51.16 | 22.5% | 137.47 | 105.70 | 23.1% |
| Phonon-2 BF16 | 64.99 | 49.67 | 23.6% | 132.89 | 99.63 | 25.0% |
| Phonon-2 Q8_0 | 63.75 | 49.13 | 22.9% | 135.30 | 98.73 | 27.0% |
| Parakeet v3 Q8_0 | 59.44 | 47.96 | 19.3% | 123.95 | 91.02 | 26.6% |

CPU/CUDA controls retain their existing path. The observed 0.8-1.7% differences
do not establish an improvement or a regression; no speedup is claimed for them.

| Control | 12 s before (ms) | After (ms) | 30 s before (ms) | After (ms) |
|---|---:|---:|---:|---:|
| cpu Q8_0 | 479.24 | 487.59 | 1271.75 | 1282.04 |
| cuda Q8_0 | 40.44 | 41.14 | 89.23 | 90.69 |

## Memory

Each cell below gives the range of the two process peak measurements. GPU figures
are device-wide NVML samples every 20 ms, minus the desktop baseline before each
process; CPU figures are maximum process RSS. Both include loaded weights/work
buffers. Short peaks can be missed and driver/desktop activity can contribute.
The variable F32/BF16 peaks are retained; they are not a precise cache-savings
claim or a comparison to older tracked-allocation measurements.

| Vulkan package | Before VRAM range (GiB) | After VRAM range (GiB) | Before peak RSS (GiB) | After peak RSS (GiB) |
|---|---:|---:|---:|---:|
| Phonon-2 F32 | 2.80-4.46 | 2.63-2.70 | 2.88 | 2.89 |
| Phonon-2 BF16 | 2.18-3.90 | 1.98-3.25 | 2.09 | 2.12 |
| Phonon-2 Q8_0 | 2.13-2.14 | 1.99-2.00 | 2.14 | 2.13 |
| Parakeet v3 Q8_0 | 2.08-2.54 | 1.98-1.99 | 2.11 | 2.11 |

## Correctness and scope

- 120 distinct backend/package/input keys: exact before/after text and word
  timestamps on 20 speech, silence and music fixtures, plus exact warm repeats.
  Includes Phonon F32/BF16/Q8 and stock Parakeet v3 Q8 on Vulkan and Q8 CPU/CUDA controls.
- Independent upstream-based CPU/Vulkan build: 8/8 stock Q8 offline (3/12/30 s)
  and buffered-streaming text/timestamp comparisons and repeats pass.
- Combined Phonon Vulkan server: JSON/multipart hotwords, explicit-empty override,
  request isolation, three repeated streamed uploads and live 1003-byte PCM chunks pass.
  Snapshots remain revisable; committed deltas equal final text.
- Standalone targeted CTest 2/2; composed targeted CTest 4/4. Builds do not contain
  the CUDA F32 precision API. Sampled parity is not corpus-level WER evidence.
- No local Metal/HIP/MUSA or RTX 5090 runtime coverage. Other Parakeet package
  sizes and other-backend applicability can be tested independently of Phonon.

## Reproduce

In an MSVC 14.43 developer environment with the Vulkan SDK on PATH:

```bash
cmake -S . -B ../build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /utf-8' \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=parakeet_tdt \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON -DENGINE_BUILD_WARMBENCH=ON \
  -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF -DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF \
  -DGGML_CCACHE=OFF -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build ../build-vk -j 6 --target audiocpp_cli audiocpp_server \
  tdt_decoder_duration_loop_test partial_text_render_test
ctest --test-dir ../build-vk --output-on-failure \
  -R 'tdt_decoder_duration_loop_test|partial_text_render_test'
../build-vk/bin/audiocpp_cli.exe --task asr --family parakeet_tdt \
  --model ../../../models/Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf \
  --backend vulkan --device 1 --threads 8 --mode offline --audio recording.wav
../build-vk/bin/audiocpp_server.exe --config ../bench-stock-q8_0-vulkan-1-after.json --no-ui
```

The CLI input path stands for one of the fixtures listed in the original
`reference.json`/`holdout-reference.json` artifacts. Repeat with `--mode streaming`.
The server configuration records the same backend, device, threads and model
path and is preserved beside the benchmark. Device indices depend on the machine.
The benchmark composition enables CUDA as well as Vulkan and uses the Phonon
conversion documented in #863; only these two source files differ between trials.

The [portable JSON](parakeet_vulkan_optimization.json) records exact timing samples,
both process peaks, model/source hashes and the eight standalone cases. Commands,
server configs, logs and preserved before/after binaries remain under
`outputs/phonon2-review-20261010/`. No user audio is uploaded.
