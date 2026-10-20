# Running audio.cpp in Docker

## Table of Contents

- [Prerequisites](#prerequisites)
- [Image Variants](#image-variants)
- [Published Images](#published-images)
- [Build Images locally](#build-images-locally)
- [Usage](#usage)
- [Examples](#examples)

## Prerequisites

- Docker must be installed and running on your system.
- For CUDA:
  - The [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) must be installed.
- For Vulkan:
  - The host must expose a working Vulkan device to Docker, typically through `/dev/dri` on Linux.
  - The container user needs access to the host render/video device groups.
- For Rockchip NPU (RKNN):
  - The host must expose the NPU device nodes: `/dev/dri` on a mainline
    (rocket DRM) kernel, `/dev/rknpu` on a vendor BSP kernel, plus `/dev/dma_heap`
    for the shared-buffer allocator.
  - The image is arm64-only (the Rockchip runtime library is), and `.rknn` engines are
    compiled per SoC, so engines must match the board (see `docs/build/rockchip-npu.md`).

## Image Variants

The following image variants are available:

- **full**: Provides the main tools **cli** and **server** and test binaries in one image. When running the container, the first argument selects the tool to execute.

The following backends are supported:
- **cuda12**
- **cuda13**
- **vulkan**
- **cpu**
- **rockchip** (Rockchip NPU; arm64 only, local build)

The following architectures are supported:
- **amd64**
- **arm64**

## Published Images

Docker images are published daily when new commits are available. The images are provided
as multiarch images (amd64/arm64).

Pull the latest images using these tags:
- **cuda12**: `ghcr.io/0xshug0/audio.cpp:full-cuda12`
- **cuda13**: `ghcr.io/0xshug0/audio.cpp:full-cuda13`
- **vulkan**: `ghcr.io/0xshug0/audio.cpp:full-vulkan`
- **cpu**: `ghcr.io/0xshug0/audio.cpp:full-cpu`

Images for a specific day/commit can be found in the
[versions](https://github.com/0xShug0/audio.cpp/pkgs/container/audio.cpp/versions?filters%5Bversion_type%5D=tagged)
history.
The format is: `full-<backend>-<date>-<shortsha>`, e.g. `full-cuda12-20260725-db7d2c4`


## Build Images locally

If you would like to build the images locally, you can use the available
Dockerfiles in `.devops`.

### CUDA

Build with the default CUDA 12.x version. See `.devops/cuda.Dockerfile`.

```bash
docker build -f .devops/cuda.Dockerfile -t local/audio.cpp:full-cuda12 .
```

Build with a specific CUDA version, for example 13.3.0:

```bash
docker build -f .devops/cuda.Dockerfile -t local/audio.cpp:full-cuda13 --build-arg CUDA_VERSION=13.3.0 .
```

Build for a specific set of GPU architectures (e.g. for faster, less portable builds):

```bash
docker build -f .devops/cuda.Dockerfile -t local/audio.cpp:full-cuda12 --build-arg CUDA_DOCKER_ARCH="86;89" .
```

### Vulkan

```bash
docker build -f .devops/vulkan.Dockerfile -t local/audio.cpp:full-vulkan .
```

### CPU

```bash
docker build -f .devops/cpu.Dockerfile -t local/audio.cpp:full-cpu .
```

### Rockchip NPU (RKNN)

Arm64 only (the Rockchip runtime library is aarch64): build on the board, on an
arm64 runner, or under binfmt emulation. The build stage downloads `librknnrt.so`
and its headers from the pinned `airockchip/rknn-toolkit2` tag (an ARG, with a
sha256 pin), so the runtime version is explicit and the engines you install must
be converted with the matching toolkit version.

```bash
docker build --platform linux/arm64 -f .devops/rockchip.Dockerfile -t local/audio.cpp:full-rockchip .
```

## Usage

For CLI use, mount the model directory `<models-dir>` into the container.
An additional `<output-dir>` should be mounted for tasks that write files.

### CUDA

```bash
docker run --rm --gpus all -v "<models-dir>:/models:ro" ghcr.io/0xshug0/audio.cpp:full-cuda12 <cli|server> --model /models/<model> <...>
```

### Vulkan

```bash
docker run --rm --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  -v "<models-dir>:/models:ro" \
  ghcr.io/0xshug0/audio.cpp:full-vulkan \
  <cli|server> --backend vulkan --model /models/<model> <...>
```

### Native WebUI

For the native WebUI with model downloads and dynamic model management, mount a
writable model directory and expose the server port:

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v "<models-dir>:/app/models" \
  ghcr.io/0xshug0/audio.cpp:full-cuda12 \
  server --ui --ui-management --host 0.0.0.0 --port 8080 --backend cuda
```

For Vulkan, expose the host render device and use the Vulkan backend:

```bash
docker run --rm --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  -p 8080:8080 \
  -v "<models-dir>:/app/models" \
  ghcr.io/0xshug0/audio.cpp:full-vulkan \
  server --ui --ui-management --host 0.0.0.0 --port 8080 --backend vulkan
```

Open `http://127.0.0.1:8080` on the host. Use a writable mount when the UI
should download or prepare models. For a read-only model directory, omit
`--ui-management` or mount the directory as read-only and load only models that
already exist in the configured path.

### CPU

```bash
docker run --rm -v "<models-dir>:/models:ro" ghcr.io/0xshug0/audio.cpp:full-cpu <cli|server> --model /models/<model> <...>
```

### Rockchip NPU (RKNN)

Pass the NPU device nodes (straced on an RK3588 board: the driver opens the DRM
cards on a mainline kernel; a vendor BSP kernel exposes `/dev/rknpu` — add that
`--device` there) and name the per-SoC engines in the family's `exec_plan`.
Prebuilt RK3588 engines install next to the family weights through the model
manager, so a bare file name resolves:

```bash
docker run --rm --device /dev/dri --device /dev/dma_heap \
  -v "<models-dir>:/app/models" \
  local/audio.cpp:full-rockchip \
  cli --task tts --family piper_tts --model /app/models/<model>.gguf \
  --session-option 'piper_tts.exec_plan={"decoder":{"impl":"rknn","artifacts":{"main":"<decoder>.rknn"}}}' \
  --text "<text>" --out /app/models/out.wav
```

Without the device nodes the run fails loudly — `rknn_init failed with code -1`
on the first plan step — rather than falling back silently.

See the fully working [examples](#examples) below.

## Examples

Examples for Docker, including CUDA and CPU, are available in `examples/docker`.

### CLI

The **[examples](../examples/docker/cli/EXAMPLE.md)** in `examples/docker/cli`
demonstrate how to run the audio.cpp CLI with `docker run`. The examples include:

- **PocketTTS:** Text-to-Speech
- **Qwen3-TTS:** Text-to-Speech with Voice Cloning

### Server

The **[examples](../examples/docker/server/EXAMPLE.md)** in `examples/docker/server`
demonstrate how to run the audio.cpp server with `docker compose`. The examples include:

- **PocketTTS:** Text-to-Speech
- **Qwen3-TTS:** Text-to-Speech with Voice Cloning
