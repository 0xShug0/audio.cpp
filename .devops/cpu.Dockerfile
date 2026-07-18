# audio.cpp — CPU Dockerfile
#
# Usage:
#   docker build -f .devops/cpu.Dockerfile -t local/audiocpp:full-cpu .

# ── BUILD: Compile all release binaries ───────────────────────────────────────
ARG UBUNTU_VERSION=24.04
ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG GCC_VERSION=14

FROM docker.io/ubuntu:$UBUNTU_VERSION AS build

ARG GCC_VERSION=14
ARG ENGINE_ENABLE_NATIVE_CPU=ON

# Install build toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} g++-${GCC_VERSION} make cmake libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION}

WORKDIR /app
COPY . .

# Configure and build
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DENGINE_ENABLE_CUDA=OFF \
        -DENGINE_ENABLE_VULKAN=OFF \
        -DENGINE_ENABLE_OPENMP=ON \
        -DENGINE_ENABLE_NATIVE_CPU=$ENGINE_ENABLE_NATIVE_CPU \
        -DENGINE_BUILD_EXAMPLES=OFF \
        -DENGINE_BUILD_TESTS=OFF \
        -DENGINE_BUILD_WARMBENCH=OFF && \
    cmake --build build --parallel $(nproc) \
        --target audiocpp_cli \
        --target audiocpp_server \
        --target model_perf \
        --target miocodec_wavlm_parity

# Collect all binaries + multiplexer into /app/full
RUN mkdir -p /app/full && \
    cp build/bin/audiocpp_cli build/bin/audiocpp_server \
       build/bin/model_perf build/bin/miocodec_wavlm_parity /app/full/ && \
    cp .devops/entrypoint.sh /app/full/entrypoint.sh && \
    chmod +x /app/full/entrypoint.sh

# ── BASE: Shared runtime (OS + common libs) ───────────────────────────────────
FROM docker.io/ubuntu:$UBUNTU_VERSION AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=N/A
ARG IMAGE_SOURCE=N/A

LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="audio.cpp" \
      org.opencontainers.image.description="An all-in-one, pure C++ inference engine for audio models, powered by ggml" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

# Runtime deps: OpenMP threading, curl (healthcheck), ffmpeg (audio I/O)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 curl ffmpeg && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

WORKDIR /app

# ── FULL: All binaries + entrypoint.sh multiplexer ────────────────────────────
FROM base AS full

COPY --from=build /app/full /app

USER ubuntu

ENTRYPOINT ["/app/entrypoint.sh"]
