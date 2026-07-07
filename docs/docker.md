# Running audio.cpp in Docker

## Prerequisites

- Docker must be installed and running on your system
- For CUDA:
  - The [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) must be installed

## Images

The following image flavors are available for this project:
- **full**: This image provides the main tools ***cli*** and ***server***
  and test binaries in one image. When running the container the first argument
  selects the tool to execute.

## Build the image

### CUDA

Build with default CUDA 12.x version (see *.devops/cuda.Dockerfile*):

```bash
docker build -f .devops/cuda.Dockerfile -t local/audiocpp:full-cuda .
```

Build with a specific CUDA version, e.g. 13.2.0:
```bash
docker build -f .devops/cuda.Dockerfile -t local/audiocpp:full-cuda --build-arg CUDA_VERSION=13.2.0 .
```

### CPU

```bash
docker build -f .devops/cpu.Dockerfile -t local/audiocpp:full-cpu .
```

## Usage

The model directory `<models-dir>` must be mounted into the container.
An additional `<output-dir>` should be mounted for TTS tasks. 

### CUDA
```bash
docker run --rm --gpus all -v "<models-dir>:/models:ro" local/audiocpp:full-cuda <cli|server> <...>
```

### CPU
```bash
docker run --rm -v "<models-dir>:/models:ro" local/audiocpp:full-cpu <cli|server> <...>
```

See the fully working [examples](#examples) below. 

## Examples

Examples for Docker (CUDA and CPU) are available in `examples/docker`.

### Pocket TTS via docker run

Generate speech via audio.cpp cli running in Docker.

See [EXAMPLE.md](../examples/docker/cli/EXAMPLE.md) in `examples/docker/cli`.

### Pocket TTS via Docker Compose

Generate speech via audio.cpp server running in Docker (Compose).

See [EXAMPLE.md](../examples/docker/server/EXAMPLE.md) in `examples/docker/server`.
