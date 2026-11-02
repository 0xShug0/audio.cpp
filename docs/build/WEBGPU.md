# Native WebGPU backend

The native backend uses Dawn and the GGML WebGPU kernels described in [WEBGPU_KERNELS.md](WEBGPU_KERNELS.md). Browser/WASM integration is separate.

## Build and selection

WebGPU is disabled by default. Install Dawn and configure its CMake package location:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug \
  -DENGINE_ENABLE_WEBGPU=ON -DDawn_DIR=/path/to/lib/cmake/Dawn
cmake --build build/debug --target audiocpp_cli -j 8
```

Use `--backend webgpu` with the Linux build helper; expose Dawn through `CMAKE_PREFIX_PATH`. If using the individual switches, pass `--cuda off --webgpu on` to avoid automatic CUDA detection enabling both backends. Only the WebGPU backend requires C++20; the framework retains its existing language standard.

Select the backend with `--backend webgpu`. Device discovery, server configuration, the C API backend string and the existing `exec_plan` parser also recognize `webgpu`. Existing backend defaults are unchanged. `best` still delegates to GGML's existing automatic selection; enabling WebGPU makes it another available GPU candidate. WebGPU memory capacity is unknown and is reported as such, rather than as a GPU allocation limit.

## Model adaptations

- Irodori uses its existing padding-and-cropping codec construction on WebGPU, as on CPU/Vulkan/Metal.
- Nemotron protects the backing allocations of output views on WebGPU. The vendored `ggml_set_output()` only protects the view itself; otherwise graph workspace reuse can overwrite recurrent state. Other backend paths retain their existing output markings.
- VibeVoice permits WebGPU in its session backend validation.

No model defaults or shared operator math are changed. Emscripten build settings, browser examples, browser file I/O, eSpeak cross-compilation and unrelated timing additions are excluded.

## Validation

The backend-device test covers backend selection, device listings, implementation identity and live WebGPU graph-buffer lifetime across repeated registry queries. The generic LSTM test retains its default CPU cases and adds an explicit WebGPU graph-reuse comparison. The model-specific fixed-input RNNT comparison is in `tests/nemotron_asr/webgpu_rnnt_test.cpp`. Its target is built only with WebGPU and Nemotron enabled; it is not registered as an automatic CTest because it requires an external GGUF. It performs no autoregressive generation.

```bash
cmake --build build/debug --target nemotron_asr_webgpu_rnnt_test -j 8
build/debug/bin/nemotron_asr_webgpu_rnnt_test --model /path/to/nemotron.gguf --log
```

Earlier native validation logs remain under `logs/webgpu/`. They were produced on the complete development branch and are historical evidence, not a new validation of this extracted branch. Fresh checks for this split are saved under `logs/webgpu/native-pr/`; full waveform parity and cross-hardware performance are not established.

Fresh Debug checks on the native-only branch passed backend-device selection/lifetime (with CUDA, Vulkan and WebGPU registered), all 12 default CPU LSTM cases, the WebGPU 640-wide graph-reuse case, and the fixed-input RNNT predictor/cached-joint comparison (logit NMSE approximately `3.5e-15` versus CPU). A streaming Nemotron CLI smoke produced the expected transcript, “Concord returned to its place amidst the tents,” using `--backend webgpu`, without `exec_plan`. The `--version` output confirms Debug and CPU/CUDA/Vulkan/WebGPU support. These checks are not a complete model regression or performance sweep.
