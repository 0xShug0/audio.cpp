# Windows Build and Prebuilt Packages

This document covers native Windows builds and release zip packaging.

## Requirements for Building

- Visual Studio Build Tools 2022 or newer with the C++ desktop workload
- MSVC x64 compiler, Windows SDK, CMake, Ninja, and MSVC OpenMP components
- Official NVIDIA CUDA Toolkit for CUDA builds

The Visual Studio IDE is not required.

## Native Builds

CPU:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli -Jobs 16
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_server -Jobs 16
```

CUDA:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_cli -Jobs 16
.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_server -Jobs 16
```

The CUDA build also includes the CPU backend, so the same binary can run with `--backend cpu` or `--backend cuda`.

## CPU Architecture Profiles

The Windows build script supports explicit CPU architecture selection:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli -CpuArch native
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli -CpuArch avx2
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli -CpuArch baseline
```

| Profile | Build flag | Use when | Performance |
| --- | --- | --- | --- |
| Fast | `-CpuArch native` | You build for your own machine or machines with very similar CPUs. | Fastest, but may use AVX512 or other host-specific instructions. |
| Balance | `-CpuArch avx2` | You want a good default for most modern Windows PCs. | Usually close to native on many systems, more compatible than native. |
| Portable | `-CpuArch baseline` | You want the broadest compatibility. | Slowest, avoids AVX/AVX2/AVX512 selection. |

`-NativeCpu ON/OFF` is still available for compatibility, but release packaging should prefer `-CpuArch`.

## Prebuilt Release Zips

Use the package script to build binaries, copy runtime DLLs, generate a package README, and create zips:

```powershell
.\scripts\package_windows_prebuilt.ps1 -Package all -Profile balance -Jobs 16
```

For CUDA packages, the package script uses a multi-architecture CUDA default derived from the installed CUDA Toolkit instead of the local GPU auto-detect path. This makes release zips more suitable for unknown Windows machines. Pass `-CudaArchitectures auto` only for a local machine-specific package.

Package choices:

```powershell
.\scripts\package_windows_prebuilt.ps1 -Package cpu -Profile balance -Jobs 16
.\scripts\package_windows_prebuilt.ps1 -Package cuda -Profile balance -Jobs 16
```

Release profiles:

```powershell
.\scripts\package_windows_prebuilt.ps1 -Package all -Profile fast -Jobs 16
.\scripts\package_windows_prebuilt.ps1 -Package all -Profile balance -Jobs 16
.\scripts\package_windows_prebuilt.ps1 -Package all -Profile portable -Jobs 16
```

Generated zips are written under `build/prebuilt`:

```text
build/prebuilt/audiocpp-windows-cpu-balance.zip
build/prebuilt/audiocpp-windows-cuda-balance.zip
```

## Choosing a Release Profile

For public releases, `balance` is the recommended default. It avoids native CPU selection while still using AVX2-class kernels for reasonable performance on modern Windows machines.

Use `fast` only when you are comfortable with a machine-specific package. If the build machine has AVX512, the resulting binary may require AVX512.

Use `portable` when compatibility matters more than speed. It disables llamafile SGEMM in addition to using the baseline CPU arch, because that is the safest path for unknown user machines.

## Runtime Requirements for Users

CPU package:

- 64-bit Windows
- Model files downloaded separately

CUDA package:

- 64-bit Windows
- NVIDIA GPU with compute capability 7.5 or newer
- NVIDIA driver 580 or newer
- Model files downloaded separately

The package script copies MSVC/OpenMP runtime DLLs into both packages. The CUDA package also copies the CUDA DLLs used by this build, so users should not need to install the CUDA Toolkit or Visual Studio Build Tools.

The CUDA package is intended for RTX 20/30/40/50 series GPUs and similar NVIDIA datacenter GPUs. Older GPUs such as GTX 10-series Pascal cards or V100-class Volta cards are not covered by the CUDA 13 package; use the CPU package or build a separate package with an older CUDA Toolkit if those GPUs must be supported.
