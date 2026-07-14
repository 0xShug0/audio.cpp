# HIP/ROCm Backend Support

## Overview

This document describes the HIP/ROCm backend support added to audio.cpp for running models on AMD GPUs.

The GGML library vendored at `external/ggml/` already contains a HIP backend that compiles CUDA kernels (`ggml-cuda/*.cu`) as HIP code via a vendor header mapping (`ggml-cuda/vendors/hip.h`). The changes in this PR expose that functionality through audio.cpp's build system, backend abstraction layer, and CLI.

### How It Works

```
audio.cpp ENGINE_ENABLE_HIP=ON
  -> CMakeLists.txt sets GGML_HIP=ON
    -> external/ggml/src/ggml-hip/CMakeLists.txt
      -> compiles ../ggml-cuda/*.cu with GGML_USE_HIP
      -> vendor header maps cudaMalloc->hipMalloc, cublasCreate->hipblasCreate, etc.
      -> sets GGML_USE_CUDA on ggml target (same register path as native CUDA)
        -> audio.cpp backend.cpp sees #ifdef GGML_USE_CUDA == true
        -> ggml_backend_cuda_init() / ggml_backend_cuda_reg() work for HIP
        -> device name "ROCm0" distinguishes HIP from "CUDA0"
```

Key insight: HIP and CUDA share **the same GGML backend code**. The `ggml-hip/` directory contains only a `CMakeLists.txt` -- zero lines of unique kernel code. The runtime distinction is made via device name prefix (`"ROCm"` vs `"CUDA"`).

---

## File-by-File Changes

### 1. `CMakeLists.txt` (root)

**L24** -- New option default:
```cmake
set(ENGINE_DEFAULT_ENABLE_HIP OFF)
```

**L36-L38** -- Forward pre-existing GGML_HIP variable (allows `-DGGML_HIP=ON` to auto-enable):
```cmake
if (DEFINED GGML_HIP)
    set(ENGINE_DEFAULT_ENABLE_HIP ${GGML_HIP})
endif()
```

**L53** -- New CMake option:
```cmake
option(ENGINE_ENABLE_HIP "Build ggml with HIP/ROCm backend support" ${ENGINE_DEFAULT_ENABLE_HIP})
```

**L64** -- Guard CUDA language/Toolkit to CUDA-only builds:
```cmake
if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
```

**L81** -- Forward to vendored GGML:
```cmake
set(GGML_HIP ${ENGINE_ENABLE_HIP} CACHE BOOL "Build ggml with HIP backend support" FORCE)
```

**L497** -- Guard audio.cpp's own `.cu` files to CUDA-only builds:
```cmake
if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    target_sources(engine_runtime PRIVATE
        src/framework/audio/istft_cuda_runtime.cu
        src/framework/sampling/torch_random_cuda_runtime.cu
    )
```

> **Rationale:** `istft_cuda_runtime.cu` and `torch_random_cuda_runtime.cu` call CUDA APIs directly (`cuda_runtime.h`, `cufft.h`). They are optional GPU-accelerated paths with CPU fallbacks (gated by `ENGINE_HAS_CUDA_ISTFT` and `ENGINE_HAS_CUDA_TORCH_RANDOM`). Skipping them on HIP builds is safe and avoids adding a HIP port of these audio.cpp-specific routines.

---

### 2. `include/engine/framework/core/module.h`

**L16** -- New enum value:
```cpp
enum class BackendType {
    Cpu,
    Cuda,
    Hip,      // <-- added
    Vulkan,
    Metal,
    BestAvailable,
};
```

---

### 3. `src/framework/core/backend.cpp`

All HIP code paths reuse `#ifdef GGML_USE_CUDA` because `ggml-hip/CMakeLists.txt` defines `GGML_USE_CUDA` on the `ggml` target for static builds (line 92 of `external/ggml/src/ggml-hip/CMakeLists.txt`).

**L107-L117** -- `init_backend()` Hip case:
```cpp
case BackendType::Hip: {
#ifdef GGML_USE_CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(config.device);
    // ...
#endif
}
```

**L183-L190** -- `backend_type()` CUDA/HIP distinction:
```cpp
if (is_cuda_backend_handle(backend)) {
#ifdef GGML_USE_CUDA
    if (backend_name_has_prefix(backend, "ROCm")) {
        return BackendType::Hip;
    }
#endif
    return BackendType::Cuda;
}
```

`is_cuda_backend_handle()` compares device registration against `ggml_backend_cuda_reg()`, which returns the same registry for both CUDA and HIP. We disambiguate by checking the backend name prefix: HIP devices are named `"ROCm0"`, `"ROCm1"`, etc.

**L217** -- `release_backend_graph_resources()` by backend handle:
```cpp
if (backend_name_has_prefix(backend, "CUDA") || backend_name_has_prefix(backend, "ROCm")) {
```

**L228** -- `release_backend_graph_resources()` by BackendType:
```cpp
if (backend_type == BackendType::Cuda || backend_type == BackendType::Hip) {
```

**L318** -- `query_backend_memory()` by BackendConfig:
```cpp
case BackendType::Cuda:
case BackendType::Hip:       // <-- fall-through to CUDA path
```

---

### 4. `app/cli/args.cpp`

**L99-L101** -- CLI string to BackendType mapping:
```cpp
if (value == "hip") {
    return engine::core::BackendType::Hip;
}
```

### 5. `app/cli/main.cpp`

**L48** -- Help text:
```
--backend cpu|cuda|hip|vulkan|metal|best
```

---

### 6. `external/sentencepiece/src/CMakeLists.txt` (build compatibility fix)

**L266** -- Disable `-fPIC` on Windows (clang Windows target does not support it):
```cmake
# Original:
if (NOT MSVC)
# Fixed:
if (NOT MSVC AND NOT WIN32)
```

> This is not HIP-specific, but is required when compiling with clang (the HIP compiler) on Windows.

---

## Build Instructions

### Linux (Recommended -- full rocBLAS support)

Prerequisites:
- ROCm 6.1+ installed (`/opt/rocm` or custom path)
- hipBLAS, rocBLAS packages
- Find your GPU target: `rocminfo | grep gfx | head -1 | awk '{print $2}'`
  (e.g. `gfx1100` for RX 7900 XTX, `gfx1151` for Strix Halo iGPU)

```bash
# Configure
cmake -S . -B build_hip \
  -DENGINE_ENABLE_HIP=ON \
  -DGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER="$(hipconfig -l)/clang" \
  -DCMAKE_CXX_COMPILER="$(hipconfig -l)/clang++" \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build_hip -j$(nproc)

# Run
./build_hip/bin/audiocpp_server --config server.json
./build_hip/bin/audiocpp_cli --task tts --family index_tts2 --model ./model --backend hip --device 0
```

Multiple GPU targets (for distribution):
```bash
-DGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151"
```

### Windows

> **IMPORTANT:** ROCm on Windows only ships rocBLAS Tensile libraries for the GPU architectures listed below. If your AMD GPU is not in this list (notably: RDNA3 iGPU `gfx1103` / Radeon 780M is NOT supported), you will need to use the workaround build flags.
>
> Supported: `gfx1030`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`, `gfx906`

**Standard build (supported GPU):**

```cmd
# Set ROCm environment
set PATH=C:\Program Files\AMD\ROCm\6.4\bin;%PATH%
set HIP_PATH=C:\Program Files\AMD\ROCm\6.4

# Install ninja if not present:
# Download from https://github.com/ninja-build/ninja/releases

# Configure
cmake -S . -B build_hip -G Ninja ^
  -DCMAKE_MAKE_PROGRAM=<path-to-ninja.exe> ^
  -DENGINE_ENABLE_HIP=ON ^
  -DENGINE_ENABLE_OPENMP=OFF ^
  -DGPU_TARGETS=gfx1100 ^
  -DCMAKE_C_COMPILER="%HIP_PATH%\bin\clang.exe" ^
  -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe" ^
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build_hip
```

**Workaround build (unsupported GPU, e.g. 780M / gfx1103):**

Add `-DGGML_CUDA_FORCE_MMQ=ON -DGGML_HIP_NO_VMM=ON` to the cmake command above. This bypasses hipBLAS/rocBLAS for matrix multiplication and uses GGML's own MMQ kernels instead. Performance will be lower for large matrix multiplications but functionally correct.

Why: rocBLAS requires pre-compiled Tensile library files per GPU architecture. If none exist for your GPU, `hipblasCreate()` fails fatally. `GGML_CUDA_FORCE_MMQ` routes all matmul through GGML's custom dequant+matmul kernels, avoiding rocBLAS entirely.

---

## Architecture Support Matrix

| GPU Target | Type | rocBLAS Tensile (Linux) | rocBLAS Tensile (Windows) | MMQ Workaround |
|---|---|---|---|---|
| gfx1100 | RDNA3 discrete (7900 XTX/XT) | Yes | Yes | Not needed |
| gfx1101 | RDNA3 discrete (7900 GRE) | Yes | Yes | Not needed |
| gfx1102 | RDNA3 discrete (7600 XT) | Yes | Yes | Not needed |
| **gfx1103** | **RDNA3 iGPU (780M)** | Yes 1 | **No** | **Required on Windows** |
| gfx1150 | RDNA3.5 iGPU (Strix Point) | Yes | Yes | Not needed |
| gfx1151 | RDNA3.5 iGPU (Strix Halo) | Yes | Yes | Not needed |
| gfx1200 | RDNA4 discrete | Yes | Yes | Not needed |
| gfx1201 | RDNA4 discrete | Yes | Yes | Not needed |

> 1 gfx1103 on Linux requires `HSA_OVERRIDE_GFX_VERSION=11.0.0`. This environment variable is [not supported on Windows](https://github.com/ROCm/ROCm/issues/2654).

---

## Known Limitations & Future Work

### Model-level GPU optimizations not yet enabled for HIP

The following locations check `BackendType::Cuda` specifically and will not apply their CUDA optimizations for HIP backends. They fall back to generic CPU-equivalent paths, which are functionally correct but may have lower performance.

| File | Line | Optimization |
|---|---|---|
| `src/framework/sampling/torch_random.cpp` | 311 | TorchCUDA random sampling policy probe |
| `src/framework/modules/conv_modules.cpp` | 254 | CUDA depthwise conv1d fast path |
| `src/models/irodori_tts/rf_dit.cpp` | 141, 535 | CUDA-specific attention path |
| `src/models/demucs/pipeline.cpp` | 474, 561 | CUDA-specific tensor pipeline |
| `src/models/demucs/session.cpp` | 154 | CUDA-specific tensor storage |
| `src/framework/modules/optimizations/fast_projection_modules.cpp` | 62 | CUDA projection acceleration |
| `src/models/miocodec/audio_pipeline.cpp` | 294 | CUDA audio pipeline |
| `src/models/index_tts2/gpt.cpp` | 83 | CUDA-required GPT module |
| `src/models/vibevoice/session.cpp` | 164 | CUDA-specific max seconds |
| `src/models/vibevoice_asr/session.cpp` | 543, 545 | CUDA-specific weight storage |
| `src/models/vibevoice_asr/speech_encoder.cpp` | 54 | CUDA-specific speech encoder |

**To enable:** Change `== BackendType::Cuda` to `== BackendType::Cuda || == BackendType::Hip` on a per-file basis after verifying each optimization works correctly on AMD hardware.

### Switch statements missing `Hip` case (compiler warnings)

These produce `-Wswitch` warnings but behave correctly (fall through to `default` or implicit return):

| File | Line | Notes |
|---|---|---|
| `app/server/runtime.cpp` | 61 | `backend_name()` returns `"unknown"` for HIP |
| `src/models/ace_step/planner.cpp` | 153 | Falls through; HIP treated same as CUDA/Cpu/Metal for this path |
| `src/models/moss/moss_tts_local/session.cpp` | 157 | Falls through |

### audio.cpp CUDA-specific `.cu` files

- `src/framework/audio/istft_cuda_runtime.cu` -- uses `cufft`
- `src/framework/sampling/torch_random_cuda_runtime.cu` -- uses CUDA Driver API

These are skipped entirely on HIP builds. To enable GPU acceleration for ISTFT and TorchRandom on AMD GPUs, they would need to be ported (`cufft` -> `hipfft`, `cuInit`/`cuDeviceGet` -> `hipInit`/`hipDeviceGet`).

---

## Verification Checklist

- [ ] Linux: `cmake -DENGINE_ENABLE_HIP=ON ...` configures without errors
- [ ] Linux: `cmake --build build_hip` completes without errors
- [ ] Linux: `--backend hip --device 0` initializes and detects AMD GPU
- [ ] Windows: Same as above (standard GPU only)
- [ ] Windows: `--backend hip` appears in `--help` output
- [ ] `backend_type()` returns `BackendType::Hip` for ROCm-initialized backends
- [ ] `query_backend_memory()` works for HIP
- [ ] `release_backend_graph_resources()` works for HIP
- [ ] Model inference runs on HIP without crashes
- [ ] Model inference produces numerically correct results
- [ ] Server JSON config accepts `"backend": "hip"`

---

## References

- GGML HIP vendor header: `external/ggml/src/ggml-cuda/vendors/hip.h` (~300 lines, ~150 CUDA-to-HIP mappings)
- GGML HIP backend CMake: `external/ggml/src/ggml-hip/CMakeLists.txt`
- GGML backend registry: `external/ggml/src/ggml-backend-reg.cpp` (L116: HIP registers via `GGML_USE_CUDA`)
- llama.cpp HIP build docs: `docs/build.md` (upstream GGML documentation)
