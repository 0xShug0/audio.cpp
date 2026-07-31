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
      -> the ggml CUDA backend registers itself under the name "ROCm" (GGML_CUDA_NAME under HIP)
        -> audio.cpp backend.cpp maps registry name "ROCm" to BackendType::Hip
        -> init_backend() uses the generic dynamic-loading path
           (ggml_backend_load_all + init_device_backend), no #ifdef guards
        -> registry name "ROCm" vs "CUDA"/"MUSA" distinguishes HIP from CUDA
```

Key insight: HIP and CUDA share **the same GGML backend code**. The `ggml-hip/` directory contains only a `CMakeLists.txt` -- zero lines of unique kernel code. The runtime distinction is made via the ggml backend **registry name** (`"ROCm"` vs `"CUDA"`), which is how backends are identified when they are loaded dynamically (`ggml_backend_load_all`).

---

## File-by-File Changes

### 1. `CMakeLists.txt` (root)

**L53** -- New option default:
```cmake
set(ENGINE_DEFAULT_ENABLE_HIP OFF)
```

**L65-L66** -- Forward pre-existing GGML_HIP variable (allows `-DGGML_HIP=ON` to auto-enable):
```cmake
if (DEFINED GGML_HIP)
    set(ENGINE_DEFAULT_ENABLE_HIP ${GGML_HIP})
endif()
```

**L82** -- New CMake option:
```cmake
option(ENGINE_ENABLE_HIP "Build ggml with HIP/ROCm backend support" ${ENGINE_DEFAULT_ENABLE_HIP})
```

**L100-L106** -- Reject configurations that enable both CUDA and HIP. HIP compiles ggml's CUDA backend sources as HIP code, so both options drive the same ggml backend and cannot coexist in one configure:
```cmake
if (ENGINE_ENABLE_CUDA AND ENGINE_ENABLE_HIP)
    message(FATAL_ERROR ...)
endif()
```

**L108** -- Guard CUDA language/Toolkit to CUDA-only builds:
```cmake
if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
```

**L149** -- Forward to vendored GGML:
```cmake
set(GGML_HIP ${ENGINE_ENABLE_HIP} CACHE BOOL "Build ggml with HIP backend support" FORCE)
```

**L156** -- Keep ggml's separate HIP graphs option in sync with the CUDA graphs toggle, so `ENGINE_ENABLE_CUDA_GRAPHS=OFF` also disables graphs on HIP builds:
```cmake
set(GGML_HIP_GRAPHS ${ENGINE_ENABLE_CUDA_GRAPHS} CACHE BOOL "Enable ggml HIP graphs support" FORCE)
```

**L657** -- Guard audio.cpp's own `.cu` files to CUDA-only builds:
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

HIP support is expressed through the dynamic backend-registry architecture shared with the other GPU backends: `ensure_backends_loaded()` calls `ggml_backend_load_all()`, and each backend is identified by the name of the ggml registry that owns it. There are no `#ifdef GGML_USE_CUDA` guards or direct `ggml_backend_cuda_*` calls in this file -- those headers resolve against the backend's own build flags, which are not visible from this translation unit.

**L34-L39** -- Registry name table. HIP builds share the ggml CUDA backend but register as `"ROCm"`, so they get their own entry (upstream maps `"ROCm"` to Cuda; here it is moved to the dedicated Hip type):
```cpp
constexpr BackendRegNames k_backend_reg_names[] = {
    {BackendType::Cuda,   {"CUDA", "MUSA", nullptr}},
    {BackendType::Hip,    {"ROCm", nullptr, nullptr}},
    // ...
};
```

**L71-L73** -- `is_hip_backend_handle()` helper, matching the owning registry's name through the shared `backend_handle_matches()`:
```cpp
bool is_hip_backend_handle(ggml_backend_t backend) {
    return backend_handle_matches(backend, BackendType::Hip);
}
```

**L203-L204** -- `init_backend()` Hip case, using the generic device init path:
```cpp
case BackendType::Hip:
    return init_device_backend(BackendType::Hip, "HIP", config);
```

**L251-L253** -- `backend_type()` HIP/CUDA distinction (HIP is checked first, since both match the ggml CUDA backend family):
```cpp
if (is_hip_backend_handle(backend)) {
    return BackendType::Hip;
}
```

**L290, L294** -- `release_backend_graph_resources()` includes HIP in both overloads:
```cpp
if (is_cuda_backend_handle(backend) || is_hip_backend_handle(backend)) cuda_clear_graph(backend, graph);
// and
if (backend_type == BackendType::Cuda || backend_type == BackendType::Hip) cuda_clear_graph(backend, graph);
```
`cuda_clear_graph()` resolves `ggml_backend_cuda_clear_graph` by proc-address lookup, which works on HIP builds because HIP shares the ggml CUDA backend and is compatible with `GGML_BACKEND_DL`.

**`query_backend_memory()`** -- no Hip-specific case needed: GPU backend types fall through the `switch` to the generic `find_device_by_backend_type()` path, which covers HIP automatically.

---

### 4. `app/cli/args.cpp`

CLI string to BackendType mapping (`rocm` accepted as an alias):
```cpp
if (value == "hip" || value == "rocm") {
    return engine::core::BackendType::Hip;
}
```

### 5. `app/cli/main.cpp`

Help text:
```
--backend cpu|cuda|hip|rocm|vulkan|metal|best  (rocm is an alias for hip)
```

---

### 6. `external/ggml` -- hipBLASLt GEMM path

rocBLAS does not ship Tensile kernels for every AMD GPU arch (e.g. gfx1103 on Windows), while hipBLASLt covers more arches. HIP builds therefore route every cuBLAS-equivalent GEMM through hipBLASLt by default.

**`external/ggml/src/ggml-hip/CMakeLists.txt`** -- New option `GGML_HIP_HIPBLASLT` (default `ON`). Runs `find_package(hipblaslt)`; when found, defines `GGML_HIP_USE_HIPBLASLT` and links `roc::hipblaslt`. When not found (e.g. older ROCm without the hipblaslt dev package), it warns and falls back to the original hipBLAS (rocBLAS) path, so Linux builds are unaffected.

**`external/ggml/src/ggml-cuda/common.cuh`** -- Includes `<hipblaslt/hipblaslt.h>` when `GGML_HIP_USE_HIPBLASLT` is defined; adds a `HIPBLASLT_CHECK` error macro; adds lazily-created per-device `hipblasLtHandle_t` handles and a 32 MiB per-device workspace to `ggml_backend_cuda_context` (freed in the context destructor in `ggml-cuda.cu`).

**`external/ggml/src/ggml-cuda/ggml-cuda.cu`** -- New `ggml_hipblaslt_gemm()` helper mirroring the cublas call semantics (`OP_T`/`OP_N`, column-major, strided-batch support), plus a `hipblasDatatype_t` -> `hipDataType` conversion (the legacy hipBLAS enum values, 150+, differ from hipDataType, 0-based, on ROCm < 6.5). All GEMM call sites are switched to it when `GGML_HIP_USE_HIPBLASLT` is defined, with the original cublas code kept as `#else` fallback:

- `ggml_cuda_op_mul_mat_cublas()`: BF16, FP16->FP32, FP16->FP16, and FP32 GEMM paths
- `ggml_cuda_mul_mat_batched_cublas_impl()`: strided-batched path (native hipBLASLt batched layouts) and pointer-array batched path (emulated with a per-batch-element GEMM loop; hipBLASLt has no pointer-array API)

All Lt GEMMs use `HIPBLAS_COMPUTE_32F` with FP32 scale for accuracy. The algorithm heuristic is queried per call -- caching heuristics per shape is future work.

### 7. `scripts/build_windows_hip.ps1`

Dedicated Windows HIP build script. Auto-detects ROCm (`HIP_PATH` or `C:\Program Files\AMD\ROCm\*`), GPU targets (`amdgpu-arch`), cmake, and ninja (PATH or the VS-bundled copy), then configures and builds with `ENGINE_ENABLE_HIP=ON`. See the Windows build section below for options.

---

## Build Instructions

### Linux

Prerequisites:
- ROCm 6.1+ installed (`/opt/rocm` or custom path)
- hipBLAS, rocBLAS, hipBLASLt packages (hipBLASLt is used for GEMM by default; without it the build falls back to hipBLAS/rocBLAS)
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

**iGPU vs discrete GPU (Linux):**

- **Discrete GPU (gfx1100/1101/1102, gfx1200/1201):** the defaults are fine. `ENGINE_ENABLE_CUDA_GRAPHS` is ON by default and dGPUs have the VRAM headroom for it. Optionally add `-DGGML_HIP_NO_VMM=OFF` — HIP VMM works on Linux dGPUs and improves ggml's memory-pool reuse under varying shapes.
- **iGPU (gfx1103 780M, gfx1150, gfx1151):** add `-DENGINE_ENABLE_CUDA_GRAPHS=OFF` if you see `out of memory` during graph warmup — each cached graph reserves its own buffers out of shared system memory (see Known Limitations). Keep `GGML_HIP_NO_VMM=ON` (the default).
- **gfx1103 note:** with the default hipBLASLt GEMM path, no `HSA_OVERRIDE_GFX_VERSION=11.0.0` is needed. That override is only required if you force the legacy rocBLAS path with `-DGGML_HIP_HIPBLASLT=OFF`.
- **rocWMMA fattn** (`GGML_HIP_ROCWMMA_FATTN`): keep OFF on both — the default `fattn-tile` kernels are faster on RDNA3/RDNA4.

### Windows

> **MSVC 14.51 (Visual Studio 2026) incompatibility:** ROCm's HIP clang headers conflict with the `cmath` shipped in MSVC 14.51 — every `.cu` compile fails with `__device__ function 'isless'/'isgreater' cannot overload __host__ __device__ function` in `__clang_cuda_math_forward_declares.h`. This affects at least ROCm 6.4 and 7.1 and can be intermittent (see [llama.cpp#22570](https://github.com/ggml-org/llama.cpp/issues/22570)). Until AMD ships a fix, configure and build from a prompt that selects an older MSVC toolset (14.44 or earlier):
>
> ```cmd
> "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
> ```
>
> Install an older v143 toolset via the VS Installer if it is missing, then run the build script or the manual cmake commands below from that same prompt — the HIP clang picks up the MSVC headers selected by vcvars. (Verified on ROCm 6.4 / gfx1103: `.cu` compilation succeeds under 14.44, fails under 14.51.)

> **hipBLASLt GEMM (default):** HIP builds use hipBLASLt instead of hipBLAS (rocBLAS) for all cuBLAS-equivalent GEMM calls. hipBLASLt ships Tensile kernels for more GPU architectures than rocBLAS on Windows — notably **gfx1103 (Radeon 780M) works**, even though rocBLAS has no gfx1103 library. Disable with `-DGGML_HIP_HIPBLASLT=OFF` to fall back to hipBLAS.

**Build script (recommended):**

```powershell
# Auto-detects ROCm (HIP_PATH), GPU targets (amdgpu-arch), cmake, and ninja
powershell -ExecutionPolicy Bypass -File scripts\build_windows_hip.ps1

# Useful options:
#   -GpuTargets gfx1103        override target arch
#   -NoHipblasLt               use hipBLAS (rocBLAS) instead of hipBLASLt
#   -ForceMmq                  route quantized matmul through GGML MMQ kernels
#   -WithVmm                   enable HIP virtual memory management
#   -NoNativeCpu               portable CPU kernels (no native ISA); required for distribution builds
#   -ConfigureOnly / -Clean / -Target audiocpp_cli / -Jobs 8
```

**Manual build:**

```cmd
# Set ROCm environment
set PATH=C:\Program Files\AMD\ROCm\6.4\bin;%PATH%
set HIP_PATH=C:\Program Files\AMD\ROCm\6.4
set ROCM_PATH=C:\Program Files\AMD\ROCm\6.4

# Configure (ninja: bundled with Visual Studio CMake tools or standalone)
cmake -S . -B build/hip -G Ninja ^
  -DENGINE_ENABLE_HIP=ON ^
  -DENGINE_ENABLE_OPENMP=OFF ^
  -DGPU_TARGETS=gfx1103 ^
  -DCMAKE_C_COMPILER="%HIP_PATH%\bin\clang.exe" ^
  -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe" ^
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build/hip
```

The resulting binaries need the ROCm `bin` directory on `PATH` at runtime (for `amdhip64_6.dll`, `hipblas.dll`, `hipblaslt.dll`, ...).

**iGPU vs discrete GPU:**

The script defaults are tuned for memory-constrained iGPUs (780M, Strix Point/Halo). On a discrete GPU (gfx1100/1101/1102, gfx1200/1201) adjust the following:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_hip.ps1 `
  -GpuTargets gfx1100 `   # your dGPU arch; see the support matrix below
  -Graphs `               # re-enable CUDA graphs: dGPUs have VRAM headroom, avoids per-request graph rebuild overhead
  -WithVmm                # optional: HIP VMM improves ggml memory-pool reuse; keep OFF on iGPUs
```

- **CUDA graphs** (`-Graphs`): each cached graph reserves its own VRAM buffers. Fine on a dGPU with 8+ GB; on UMA iGPUs it can exhaust shared memory during warmup (see Known Limitations).
- **VMM** (`-WithVmm`): with `GGML_HIP_NO_VMM=ON` (the default) ggml's memory pool reuses less, which costs performance under varying shapes. It is required on Windows iGPUs but generally works on dGPUs.
- **hipBLASLt**: keep the default ON. Unlike gfx1103, rocBLAS does ship Tensile kernels for the dGPU arches on Windows, so `-NoHipblasLt` is viable there — but there is no performance reason to prefer it.
- **rocWMMA fattn**: stays OFF on dGPUs too; the default `fattn-tile` kernels are the faster path on RDNA3/RDNA4.

> **Legacy workaround (pre-hipBLASLt):** `-DGGML_CUDA_FORCE_MMQ=ON` bypasses BLAS for *quantized* matmul only; FP16/FP32 GEMM still required rocBLAS and failed on gfx1103. Renaming rocBLAS `gfx1100` Tensile files to `gfx1103` segfaults with ROCm 6.4 on Windows — do not use. hipBLASLt is the supported path.

---

## Architecture Support Matrix

| GPU Target | Type | rocBLAS Tensile (Linux) | rocBLAS Tensile (Windows) | hipBLASLt (Linux + Windows) |
|---|---|---|---|---|
| gfx1100 | RDNA3 discrete (7900 XTX/XT) | Yes | Yes | Yes |
| gfx1101 | RDNA3 discrete (7900 GRE) | Yes | Yes | Yes |
| gfx1102 | RDNA3 discrete (7600 XT) | Yes | Yes | Yes |
| **gfx1103** | **RDNA3 iGPU (780M)** | Yes 1 | **No** | **Yes** |
| gfx1150 | RDNA3.5 iGPU (Strix Point) | Yes | Yes | Yes |
| gfx1151 | RDNA3.5 iGPU (Strix Halo) | Yes | Yes | Yes |
| gfx1200 | RDNA4 discrete | Yes | Yes | Yes |
| gfx1201 | RDNA4 discrete | Yes | Yes | Yes |

> 1 gfx1103 on Linux requires `HSA_OVERRIDE_GFX_VERSION=11.0.0` for rocBLAS. This environment variable is [not supported on Windows](https://github.com/ROCm/ROCm/issues/2654). With the hipBLASLt GEMM path (default), gfx1103 works on both Linux and Windows without any override.

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
| `src/models/vibevoice_asr/session.cpp` | 543, 545 | CUDA-specific weight storage |
| `src/models/vibevoice_asr/speech_encoder.cpp` | 54 | CUDA-specific speech encoder |

**To enable:** Change `== BackendType::Cuda` to `== BackendType::Cuda || == BackendType::Hip` on a per-file basis after verifying each optimization works correctly on AMD hardware.

**Enabled for HIP:** VibeVoice TTS (`src/models/vibevoice/session.cpp` -- backend whitelist and the 30 s voice-prompt cap) and the conv-transpose1d col2im fast path (`src/framework/modules/conv_modules.cpp`) are enabled for HIP. Validated on gfx1151 (ROCm 7.14, Linux): prompt-side fingerprints (text encoding, acoustic encoder, prompt embeddings, prefill logits/top-5) match CPU/CUDA within ~1e-3 relative float noise, `test-backend-ops` passes 11953/11953 on ROCm0, and the remaining end-to-end divergence vs CPU is the same chaotic argmax amplification that CUDA exhibits vs CPU.

### Switch statements missing `Hip` case (compiler warnings)

- `app/server/runtime.cpp` (`backend_name()`) and `src/models/ace_step/planner.cpp` (`planner_prefill_uses_host_backend()`) now handle `Hip` explicitly.
- `src/models/moss/moss_tts_local/session.cpp` (`resolve_auto_weight_type()`) intentionally keeps HIP on the `Native` dtype default; enabling the CUDA-style BF16 path for HIP needs validation with real models first.

### hipBLASLt GEMM path notes

- Heuristics are queried per GEMM call (shape-keyed algo caching is future work).
- The pointer-array batched GEMM (`cublasGemmBatchedEx` equivalent) is emulated with a per-batch-element loop.
- Verified on gfx1103 / ROCm 6.4 / Windows: F32 and F16 2D GEMM, strided-batched, and broadcast-batched all match CPU reference within 1e-4.

### CUDA graphs on iGPUs / limited VRAM

`ENGINE_ENABLE_CUDA_GRAPHS` defaults to ON (inherited from the CUDA build), but every cached graph reserves its own VRAM buffers. On memory-constrained GPUs (notably UMA iGPUs like the 780M, where `GGML_HIP_NO_VMM` also reduces ggml's memory-pool reuse), a burst of new shapes (e.g. the second inference request) can exhaust VRAM with `cudaMalloc failed: out of memory` during graph warmup. `scripts/build_windows_hip.ps1` therefore builds with `-DENGINE_ENABLE_CUDA_GRAPHS=OFF` by default; pass `-Graphs` to re-enable on discrete GPUs with headroom (recommended there — it removes per-request graph rebuild overhead).

### audio.cpp CUDA-specific `.cu` files

- `src/framework/audio/istft_cuda_runtime.cu` -- uses `cufft`
- `src/framework/sampling/torch_random_cuda_runtime.cu` -- uses CUDA Driver API

These are skipped entirely on HIP builds. To enable GPU acceleration for ISTFT and TorchRandom on AMD GPUs, they would need to be ported (`cufft` -> `hipfft`, `cuInit`/`cuDeviceGet` -> `hipInit`/`hipDeviceGet`).

---

## Verification Checklist

- [ ] Linux: `cmake -DENGINE_ENABLE_HIP=ON ...` configures without errors
- [ ] Linux: `cmake --build build_hip` completes without errors
- [ ] Linux: `--backend hip --device 0` initializes and detects AMD GPU
- [x] Windows: HIP build configures and compiles (ROCm 6.4, gfx1103)
- [x] Windows: `--backend hip` appears in `--help` output
- [x] Windows: HIP backend initializes and detects the GPU (`ROCm0`, gfx1103)
- [x] Windows: F32/F16/BF16 GEMM via hipBLASLt matches CPU reference (2D, strided-batched, broadcast-batched)
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
