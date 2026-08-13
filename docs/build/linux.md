# Linux Build

This document covers direct CMake builds on Linux. For the quick script-based path, see the build section in the main README.

## Requirements

- GCC 13 or newer
- CMake
- A supported backend toolchain when enabling CUDA or Vulkan
- For CUDA: CUDA Toolkit 12.0 or newer — any 12.x or 13.x is fine (the Docker image
  pins 12.9, and a CUDA 13 package is documented for Windows). The real upper bound
  is the host GCC the chosen toolkit accepts, not the CUDA version: e.g. CUDA 12.9
  supports up to GCC 14

Native ggml CPU optimization is enabled by default for local performance. If your compiler or assembler rejects a generated CPU instruction such as `vpdpbusd`, reconfigure with `-DENGINE_ENABLE_NATIVE_CPU=OFF` to build portable CPU kernels.

If you use an environment manager or custom toolchain, activate it before running the commands below.

## Configure

CPU-only:

```bash
cmake -S . -B build
```

CUDA:

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON
```

CMake picks the first `nvcc` on `PATH`, which is often **not** the toolkit you want:
distro packages install an old one to `/usr/bin/nvcc` (Ubuntu 22.04's
`nvidia-cuda-toolkit` is CUDA 11.5) while the toolkit from NVIDIA lands in
`/usr/local/cuda-<version>`. Point at it explicitly, and set the architecture of the
GPU you are building for, rather than relying on `PATH`:

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86      # 86 = RTX 3000, 89 = RTX 4000, 120 = Blackwell
```

Set **both**. `CMAKE_CUDA_COMPILER` only chooses the compiler, and `CUDAToolkit_ROOT`
only chooses the libraries — with a distro CUDA also installed, setting just the
compiler produces a build that compiles with the new `nvcc` but silently links the
old `libcudart`/`libcublas` from `/usr/lib/x86_64-linux-gnu` (owned by
`nvidia-cuda-dev`). That mismatch links without warning. Check the result:

```bash
readelf -d build/bin/audiocpp_server | grep NEEDED | grep cuda
# want libcudart.so.12 / libcublas.so.12 — libcudart.so.11.0 means a mixed build
```

Leave `CMAKE_CUDA_ARCHITECTURES` unset to build for the GPUs present at build time
(`native`). Note that CMake caches the CUDA compiler: switching toolkits in an
existing build directory requires deleting `CMakeCache.txt` and `CMakeFiles/`.

### Building an old GPU with the helper script

The `scripts/build_linux.sh --backend cuda` script leaves `CMAKE_CUDA_ARCHITECTURES` unset, so it
builds for the GPU attached to the *build* machine (`native`). If you build on one machine and run
on another — or target an old GPU such as a Pascal compute-capability 6.1 device — pass
`--cuda-arch`, which forwards `-DCMAKE_CUDA_ARCHITECTURES` (and un-defines the sticky cache value so
a previous configure does not win):

```bash
scripts/build_linux.sh --backend cuda --cuda-arch 61 --target audiocpp_cli --target audiocpp_server
```

Pinning an architecture below 8.9 also turns SageAttention2 off at build time
(`external/ggml/src/ggml-cuda/CMakeLists.txt` gates it to `>= 89`), so no SM89+ kernels are packaged
and a 6.1 GPU will not hit an illegal-instruction core dump:
`scripts/build_linux.sh --backend cuda --cuda-arch 61` is equivalent to the working direct CMake
configure `cmake -S . -B build -DENGINE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61`.

You do **not** need `-DENGINE_ENABLE_NATIVE_CPU=OFF` for this: `GGML_NATIVE` only influences the CUDA
architecture as a *fallback* (`if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` in the ggml CUDA CMake
file), so once `--cuda-arch`/`CMAKE_CUDA_ARCHITECTURES` is set, GPU architecture is fully determined
by it.

On WSL2, install the toolkit only — `cuda-toolkit-<version>` from the `wsl-ubuntu`
repo. The `cuda` and `cuda-drivers` metapackages pull a Linux display driver that
breaks the GPU passthrough provided by the Windows host driver.

Vulkan:

```bash
cmake -S . -B build -DENGINE_ENABLE_VULKAN=ON
```

Portable CPU-kernel fallback:

```bash
cmake -S . -B build -DENGINE_ENABLE_NATIVE_CPU=OFF
```

## Build

Build the CLI and server from the configured tree:

```bash
cmake --build build -j$(nproc) --target audiocpp_cli --target audiocpp_server
```

If your machine is memory-constrained, use a smaller `-j` value, for example `-j4`.

## Build Type Notes

- For single-config generators, the recommended config is `RelWithDebInfo`
- For multi-config generators, choose the configuration at build time
- Backend and feature options are independent from build type
