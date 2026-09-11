# Windows HIP/ROCm Distribution Guide

This document explains how to build and package audio.cpp HIP/ROCm binaries on Windows so that other AMD GPU users can run them **without installing the ROCm SDK**.

The approach mirrors the llama.cpp Windows HIP release (`.github/workflows/release.yml`, `windows-hip` job), adjusted for audio.cpp and verified against a local ROCm 6.4 / 7.1 installation.

For HIP build instructions (compiler setup, MSVC toolset compatibility, iGPU vs dGPU tuning), see [docs/build/HIP.md](HIP.md).

---

## 1. Build with Multiple GPU Architectures

The default `build_windows_hip.ps1` auto-detects only the **local** GPU via `amdgpu-arch`. For distribution, pass all target architectures explicitly.

### Two Release Tracks

The project publishes two HIP packages (decided after verifying both build tracks locally):

| Track | ROCm | GPU Targets | Notes |
|---|---|---|---|
| `audiocpp-windows-hip-rocm6` | 6.4 | gfx1100/1101/**1102**/1103/1150/1151/1200/1201 (8) | Full coverage, conservative choice; required for RX 7600 (gfx1102) |
| `audiocpp-windows-hip-rocm7` | 7.1 | gfx1100/1101/1103/1150/1151/1200/1201 (7) | Smaller kernel libraries, better-tuned RDNA4 kernels; **no gfx1102** (7.1 hipBLASLt ships no gfx1102 kernels) |

Both tracks cover gfx1103 (780M) via hipBLASLt. The 7.1 package README must state: "RX 7600 series users: use the ROCm 6 package."

Both tracks were verified locally (2026-07-29): 145 `.cu` files compiled for every target arch, dependency chains match §2 (`amdhip64_6` + `hipblas` + `hipblaslt` on 6.4; `amdhip64_7` + `libhipblas` + `libhipblaslt` on 7.1), no `libomp140.x86_64.dll`, embedded model specs resolve.

> **Two build-environment requirements** (both documented in [docs/build/HIP.md](HIP.md)):
>
> 1. Run the build from a prompt that selects the **MSVC 14.44 toolset** (`vcvarsall.bat x64 -vcvars_ver=14.44`). ROCm's HIP clang cannot parse the `cmath` of MSVC 14.51 (VS 2026) — every `.cu` compile fails.
> 2. Pass `-NoNativeCpu` so the CPU backend is not compiled with the build machine's native ISA (see "CPU Portability" below).

**ROCm 6.4 track** (`build\hip`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_hip.ps1 `
  -RocmPath "C:\Program Files\AMD\ROCm\6.4" `
  -GpuTargets "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201" `
  -NoNativeCpu -DeploymentBuild -Target audiocpp_cli -Jobs 16

powershell -ExecutionPolicy Bypass -File scripts\build_windows_hip.ps1 `
  -RocmPath "C:\Program Files\AMD\ROCm\6.4" `
  -GpuTargets "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201" `
  -NoNativeCpu -DeploymentBuild -Target audiocpp_server -Jobs 16
```

`-DeploymentBuild` compiles the `model_specs/*.json` catalog into the binary (`AUDIOCPP_DEPLOYMENT_BUILD`), so end users do not need the `model_specs/` directory. Without it the server/CLI fails with `model spec not found for family ...` unless `--model-spec-override` is given.

**ROCm 7.1 track** (side-by-side in `build\hip71`, gfx1102 excluded):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_hip.ps1 `
  -RocmPath "C:\Program Files\AMD\ROCm\7.1" -BuildDir build\hip71 `
  -GpuTargets "gfx1100;gfx1101;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201" `
  -NoNativeCpu -DeploymentBuild -Target audiocpp_cli -Jobs 16

powershell -ExecutionPolicy Bypass -File scripts\build_windows_hip.ps1 `
  -RocmPath "C:\Program Files\AMD\ROCm\7.1" -BuildDir build\hip71 `
  -GpuTargets "gfx1100;gfx1101;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201" `
  -NoNativeCpu -DeploymentBuild -Target audiocpp_server -Jobs 16
```

### Supported Architecture Matrix

| GPU Target | Type | Example Hardware |
|---|---|---|
| gfx1100 | RDNA3 discrete | RX 7900 XTX / XT |
| gfx1101 | RDNA3 discrete | RX 7900 GRE |
| gfx1102 | RDNA3 discrete | RX 7600 XT |
| gfx1103 | RDNA3 iGPU | Radeon 780M |
| gfx1150 | RDNA3.5 iGPU | Strix Point |
| gfx1151 | RDNA3.5 iGPU | Strix Halo |
| gfx1200 | RDNA4 discrete | RX 9070 XT |
| gfx1201 | RDNA4 discrete | RX 9070 |

Omit architectures you do not need to reduce compile time and package size.

> **RDNA2 (gfx1030/1031/1032):** llama.cpp ships these targets, but the hipBLASLt libraries in ROCm 6.4 and 7.1 contain **no gfx1030 kernels** (verified locally). Since audio.cpp routes all GEMM through hipBLASLt at compile time, RDNA2 would fail at runtime. Do not add gfx103x targets unless you have verified the GEMM path on real RDNA2 hardware.

> **gfx1102 on ROCm 7.1:** the 7.1 hipBLASLt library has no gfx1102 kernels (6.4 has 16 MB). If you build against ROCm 7.1, verify RX 7600-class cards before shipping; ROCm 6.4 covers gfx1102 in hipBLASLt.

### Recommended Build Flags for Distribution

The script defaults are already tuned for broad compatibility:

| Flag | Default | Reason |
|---|---|---|
| hipBLASLt GEMM | ON | gfx1103 has no rocBLAS Tensile kernels on Windows (verified in both ROCm 6.4 and 7.1); hipBLASLt covers all listed arches |
| CUDA graphs | OFF | Each cached graph reserves its own VRAM; exhausts shared memory on UMA iGPUs |
| HIP VMM | OFF (`GGML_HIP_NO_VMM=ON`) | Required on Windows iGPUs |

Do **not** change these defaults unless the package targets only discrete GPUs with 8+ GB VRAM.

### CPU Portability (Important)

`ENGINE_ENABLE_NATIVE_CPU` defaults to **ON** (`CMakeLists.txt:87`), which compiles the ggml CPU backend with the build machine's native ISA (e.g. AVX512). A distribution binary built this way can crash with `illegal instruction` on older CPUs.

Pass `-NoNativeCpu` to `scripts/build_windows_hip.ps1` (it forwards `-DENGINE_ENABLE_NATIVE_CPU=OFF`) when producing release binaries. llama.cpp does the same with `-DGGML_NATIVE=OFF`.

### CI Automation Reference

The llama.cpp `windows-hip` release job is a proven template for CI:

1. Download the AMD HIP SDK installer (`AMD-Software-PRO-Edition-<version>-Win11-For-HIP.exe`) and run it with `-install` (silent).
2. Cache `C:\Program Files\AMD\ROCm` between runs; use ccache for the build.
3. Build, copy runtime DLLs and both kernel libraries next to the binaries (see below), zip with 7z.

---

## 2. Collect Runtime DLLs

The executables dynamically link against ROCm math libraries and the MSVC CRT. Direct dependencies of `audiocpp_cli.exe` (verified with `dumpbin /DEPENDENTS` on a ROCm 6.4 build):

```
audiocpp_cli.exe / audiocpp_server.exe
├── amdhip64_6.dll          (usually provided by the AMD driver — see below)
├── hipblas.dll
│   └── rocblas.dll
├── hipblaslt.dll
└── MSVC CRT (see below)
```

All other dependencies (`KERNEL32.dll`, `ADVAPI32.dll`, `api-ms-win-crt-*`) are Windows system DLLs and do not need to be bundled.

### HIP Runtime: Bundle It

The HIP runtime (`amdhip64_*.dll`) and compiler runtime (`amd_comgr_*.dll`) are installed into `C:\Windows\System32` by the Adrenalin graphics driver — but **which major version a user has depends on their driver generation**:

- Drivers from 2024 through early 2026 ship the ROCm 6 runtime (`amdhip64_6.dll`, `amd_comgr_2.dll`).
- Adrenalin 26.5.1 (May 2026) and later **dropped the ROCm 6 runtime** and ship only ROCm 7 (`amdhip64_7.dll`, `amd_comgr_3.dll`). This broke applications compiled against ROCm 6, e.g. [Blender Cycles](https://videocardz.com/newz/blender-cycles-has-issues-with-amd-adrenalin-26-5-1-after-rocm-runtime-change).

Since the executables link against exactly one major version, relying on the driver-provided runtime fails for a large share of users either way. **Copy both DLLs from the ROCm `bin` directory used for the build into the package** (~130 MB) so the driver version stops mattering — the driver only needs to support the GPU itself:

| DLL (ROCm 6.4) | DLL (ROCm 7.x) | Approx. Size | Role |
|---|---|---|---|
| `amdhip64_6.dll` | `amdhip64_7.dll` | 17 MB | HIP runtime |
| `amd_comgr_2.dll` | `amd_comgr_3.dll` | 116 MB | AMD compiler runtime (dependency of `amdhip64_*.dll`) |

(llama.cpp gets away without bundling only because it builds against ROCm 7 and accepts the new-driver requirement.)

### Math Library DLLs (Must Bundle)

Copy from the ROCm `bin` directory (e.g. `C:\Program Files\AMD\ROCm\6.4\bin`) into the same directory as the `.exe` files:

| DLL (ROCm 6.4) | DLL (ROCm 7.1) | Approx. Size | Role |
|---|---|---|---|
| `hipblas.dll` | `libhipblas.dll` | 1 MB | BLAS interface layer |
| `rocblas.dll` | `rocblas.dll` | 42 MB | rocBLAS backend (dependency of `hipblas.dll`) |
| `hipblaslt.dll` | `libhipblaslt.dll` | 5 MB | hipBLASLt GEMM (default path) |

The DLL names must match what the executables were linked against, i.e. they must come from the **same ROCm installation used for the build**. Do not mix versions.

### MSVC CRT DLLs

Copy from the Visual Studio Build Tools redistributable directory (e.g. `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC\<version>\x64\Microsoft.VC143.CRT`):

- `MSVCP140.dll`
- `VCRUNTIME140.dll`
- `VCRUNTIME140_1.dll`

> OpenMP is disabled in HIP builds (`ENGINE_ENABLE_OPENMP=OFF` plus `-DGGML_OPENMP=OFF`), so `VCOMP140.DLL` is **not** needed. The second flag matters: ggml probes for OpenMP on its own (`GGML_OPENMP` defaults ON), and in a vcvars environment it will find LLVM libomp and silently add a **`libomp140.x86_64.dll`** runtime dependency. After any distribution build, run `dumpbin /DEPENDENTS` on the executables — `libomp140.x86_64.dll` must **not** appear. `scripts/build_windows_hip.ps1` passes both flags.

### HSA Runtime

There is no `hsa-runtime64_1.dll` in the Windows ROCm SDK. On Windows the HSA runtime is provided by the **AMD GPU driver** (Adrenalin Edition). Users only need a normally installed AMD driver.

---

## 3. Copy the Kernel Libraries (rocBLAS **and** hipBLASLt)

Both `rocblas.dll` and `hipblaslt.dll` load GPU kernel binaries from a directory next to the DLL at runtime. **Both must be shipped.** llama.cpp copies both in its release packaging (`release.yml`: `cp ...\bin\rocblas\library\*` and `cp ...\bin\hipblaslt\library\*`).

Copy from the ROCm `bin` directory:

```
C:\Program Files\AMD\ROCm\<version>\bin\rocblas\library\
C:\Program Files\AMD\ROCm\<version>\bin\hipblaslt\library\
```

Place them next to the executables as:

```
<package>\rocblas\library\
<package>\hipblaslt\library\
```

> The hipBLASLt library is easy to overlook and fatal to omit: hipBLASLt is the **default GEMM path**, and it is the **only** GEMM path that covers gfx1103. Without `hipblaslt\library\`, GEMM initialization fails on every GPU.

### Library Sizes by ROCm Version (measured locally)

| Library | ROCm 6.4 | ROCm 7.1 |
|---|---|---|
| `rocblas\library` (full) | ~1060 MB | ~191 MB |
| `rocblas\library` (8 target arches + fallback) | ~783 MB | ~143 MB |
| `hipblaslt\library` (8 target arches) | ~147 MB | ~407 MB |
| **Total, 8 arches** | **~930 MB** | **~550 MB** |

ROCm 7.1 shrinks rocBLAS dramatically (lazy-loading index only) but inflates hipBLASLt for RDNA4 (gfx1200: 189 MB, gfx1201: 131 MB). With the HIP runtime bundled (see §2), driver generation is no longer a differentiator. The remaining trade-offs: ROCm 6.4 covers gfx1102 in hipBLASLt (7.1 does not) and is the conservative choice for the first release; ROCm 7.x yields a ~380 MB smaller package and is the maintained line with better RDNA4/Strix support, but must be verified on gfx1102 hardware first.

### Filtering by Architecture

Delete files whose names contain architectures you do not target. **Do not remove `fallback` files** — they are architecture-independent kernels used when no arch-specific kernel is available.

rocBLAS per-architecture breakdown (ROCm 6.4):

| Architecture | Size | Keep? |
|---|---|---|
| gfx906 (Vega) | 130 MB | No |
| gfx1030 (RDNA2) | 147 MB | No |
| gfx1100 | 280 MB | Yes |
| gfx1101 | 157 MB | Yes |
| gfx1102 | 159 MB | Yes |
| gfx1150 | 159 MB | Yes |
| gfx1151 | 9 MB | Yes |
| gfx1200 | 9 MB | Yes |
| gfx1201 | 9 MB | Yes |
| fallback | 76 MB | Yes (always) |

hipBLASLt per-architecture breakdown:

| Architecture | ROCm 6.4 | ROCm 7.1 |
|---|---|---|
| gfx1100 | 16 MB | 17 MB |
| gfx1101 | 17 MB | 19 MB |
| gfx1102 | 16 MB | **0 MB (missing!)** |
| gfx1103 | 16 MB | 17 MB |
| gfx1150 | 16 MB | 17 MB |
| gfx1151 | 16 MB | 17 MB |
| gfx1200 | 23 MB | 189 MB |
| gfx1201 | 27 MB | 131 MB |

To filter with PowerShell:

```powershell
$rocmBin = "C:\Program Files\AMD\ROCm\6.4\bin"
foreach ($lib in @("rocblas", "hipblaslt")) {
    $src = Join-Path $rocmBin "$lib\library"
    $dst = ".\package\$lib\library"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Get-ChildItem $src -File | Where-Object {
        $_.Name -notmatch "gfx906|gfx1030"
    } | Copy-Item -Destination $dst
}
```

> **Optional size optimization (verify before using):** with hipBLASLt enabled, every GEMM in ggml is routed to hipBLASLt at compile time (`external/ggml/src/ggml-cuda/ggml-cuda.cu`), so the rocBLAS Tensile kernels in `rocblas\library` are never exercised. `rocblas.dll` itself must still ship (it is statically linked by `hipblas.dll`), but the ~780 MB (6.4) of Tensile kernels may be dead weight. llama.cpp ships the full library; if you drop it, test every model family on several architectures first.

---

## 4. Final Package Layout

```
audiocpp-windows-hip/
├── audiocpp_cli.exe
├── audiocpp_server.exe
├── amdhip64_6.dll             (amdhip64_7.dll on ROCm 7.x)
├── amd_comgr_2.dll            (amd_comgr_3.dll on ROCm 7.x)
├── hipblas.dll                (libhipblas.dll on ROCm 7.x)
├── hipblaslt.dll              (libhipblaslt.dll on ROCm 7.x)
├── rocblas.dll
├── MSVCP140.dll
├── VCRUNTIME140.dll
├── VCRUNTIME140_1.dll
├── rocblas/
│   └── library/               (filtered to target architectures + fallback)
├── hipblaslt/
│   └── library/               (filtered to target architectures)
└── README.md
```

### Estimated Size (8 architectures, ROCm 6.4)

| Component | Size |
|---|---|
| Executables | ~160 MB |
| ROCm DLLs (incl. HIP runtime) | ~181 MB |
| rocBLAS library (filtered) | ~783 MB |
| hipBLASLt library | ~147 MB |
| MSVC CRT | ~2 MB |
| **Total (uncompressed)** | **~1.3 GB** |
| **Total (zip)** | **~700–900 MB** |

With ROCm 7.1 the kernel libraries total ~550 MB instead of ~930 MB.

---

## 5. End-User Requirements

- 64-bit Windows
- AMD GPU matching one of the compiled target architectures
- AMD GPU driver (Adrenalin Edition) normally installed — the HIP runtime itself is bundled, so the driver only needs to support the GPU
- Model files downloaded separately

The following are **not** required:

- ROCm SDK / AMD HIP SDK
- Visual Studio or MSVC Build Tools
- CUDA Toolkit

### Quick Start for Users

```powershell
.\audiocpp_cli.exe --backend hip --task tts --family <family> --model C:\path\to\model [options]
.\audiocpp_cli.exe --backend cpu --task tts --family <family> --model C:\path\to\model [options]
```

`rocm` is accepted as an alias for `hip`:

```powershell
.\audiocpp_cli.exe --backend rocm --task tts --family <family> --model C:\path\to\model
```

Server:

```powershell
.\audiocpp_server.exe --config C:\path\to\server.json
```

---

## 6. Notes

- The HIP build also includes the CPU backend, so users can fall back to `--backend cpu`.
- Keep all DLL files and the `rocblas/` and `hipblaslt/` directories next to the `.exe` files. The libraries locate their kernel files relative to the DLL path.
- If GPU initialization fails, the user should update their AMD driver first.
- gfx1103 (Radeon 780M) works via the hipBLASLt GEMM path without any `HSA_OVERRIDE_GFX_VERSION` workaround. That override is [not supported on Windows](https://github.com/ROCm/ROCm/issues/2654) and is not needed when hipBLASLt is enabled (the default).
- On memory-constrained iGPUs (780M, Strix Point/Halo), CUDA graphs are disabled by default to avoid `out of memory` during graph warmup. This is the correct default for a distribution package.
- Redistribution: rocBLAS/hipBLASLt are MIT-licensed open source, but the Windows HIP SDK binaries are governed by the AMD EULA. Verify that the EULA permits redistributing the runtime DLLs with an application before publishing.
- Packaging automation: `scripts/package_windows_prebuilt.ps1` currently supports only `cpu` and `cuda` packages; extending it with a `hip` package kind would automate the DLL/library collection steps above.
