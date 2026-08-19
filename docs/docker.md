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

## Image Variants

The following image variants are available:

- **full**: Provides the main tools **cli** and **server** and test binaries in one image. When running the container, the first argument selects the tool to execute.

The following backends are supported:
- **cuda12**
- **cuda13**
- **cpu**

The following architectures are supported:
- **amd64**
- **arm64**

## Published Images

Docker images are published daily when new commits are available. The images are provided
as multiarch images (amd64/arm64).

Pull the latest images using these tags:
- **cuda12**: `ghcr.io/0xshug0/audio.cpp:full-cuda12`
- **cuda13**: `ghcr.io/0xshug0/audio.cpp:full-cuda13`
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

### CPU

```bash
docker build -f .devops/cpu.Dockerfile -t local/audio.cpp:full-cpu .
```

## Usage

For CLI use, mount the model directory `<models-dir>` into the container.
An additional `<output-dir>` should be mounted for tasks that write files.

### CUDA

```bash
docker run --rm --gpus all -v "<models-dir>:/models:ro" ghcr.io/0xshug0/audio.cpp:full-cuda12 <cli|server> --model /models/<model> <...>
```

### WebUI

For the native WebUI with model downloads and dynamic model management, mount a
writable model directory and expose the server port:

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v "<models-dir>:/app/models" \
  ghcr.io/0xshug0/audio.cpp:full-cuda12 \
  server --ui --ui-management --host 0.0.0.0 --port 8080 --backend cuda
```

Open `http://127.0.0.1:8080` on the host. Use a writable mount when the UI
should download or prepare models. For a read-only model directory, omit
`--ui-management` or mount the directory as read-only and load only models that
already exist in the configured path.

### CPU

```bash
docker run --rm -v "<models-dir>:/models:ro" ghcr.io/0xshug0/audio.cpp:full-cpu <cli|server> --model /models/<model> <...>
```

### Native WebUI

Use `--ui-management` when you want the browser UI to browse, download, remove,
or switch models. Mount a writable models directory to keep downloads across
container runs:

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v "<models-dir>:/app/models" \
  ghcr.io/0xshug0/audio.cpp:full-cuda12 \
  server --ui --ui-management --host 0.0.0.0 --port 8080 --backend cuda
```

Then open `http://127.0.0.1:8080`. Omit `--ui-management` for a read-only UI
serving only the models declared by your server configuration.

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
