# Linux Build

This document covers direct CMake builds on Linux. For the quick script-based path, see the build section in the main README.

## Requirements

- GCC 13 or newer
- CMake
- A supported backend toolchain when enabling CUDA or Vulkan

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
