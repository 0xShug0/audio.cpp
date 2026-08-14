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


Old CUDA GPUs (cm<89):

If you build for old CUDA GPUs with CUDA architecture < 8.9, e.g. Pascal compute-capability
6.1, use build_linux.sh with  `--cuda-arch`. Or cmake with `-DCMAKE_CUDA_ARCHITECTURES=<archi-id>` 
(and un-defines the sticky cache value so a previous configure does not win):

```bash
./scripts/build_linux.sh --backend cuda --cuda-arch 61 --target audiocpp_cli --target audiocpp_server
```
or
```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=61      # 61 = GTX 1080Ti
```

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
