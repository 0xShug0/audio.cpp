# audio.cpp — Rockchip NPU (RKNN) Dockerfile, linux/arm64
#
# The CPU image (cpu.Dockerfile) runs on a Rockchip board, but its binaries
# carry no librknnrt: ENGINE_ENABLE_RKNN links the board runtime at build time,
# and that runtime is aarch64-only — so this is a separate Dockerfile, built on
# (or emulating) arm64.
#
# Usage (on the board, with the NPU reached through its device nodes):
#   docker build --platform linux/arm64 -f .devops/rockchip.Dockerfile -t local/audio.cpp:full-rockchip .
#   docker run --rm --device /dev/dri --device /dev/dma_heap \
#       -v ./models:/app/models local/audio.cpp:full-rockchip server --help
#
# The --device flags are what the driver itself opens (straced on an RK3588
# board): with a mainline (rocket DRM) kernel the NPU is reached through the
# /dev/dri render nodes; a vendor BSP kernel exposes /dev/rknpu instead, so add
# `--device /dev/rknpu` there. Without the nodes the refusal is loud, not
# silent: `rknn_init failed with code -1` on the first plan step.

# ── RKNN SDK: the board runtime + its headers, pinned ─────────────────────────
# librknnrt.so ships in the airockchip/rknn-toolkit2 release tree; the image
# must carry the same version the engines were converted with (an engine is
# per-SoC and the runtime checks its version), so the tag is an ARG and the
# download is hash-checked.
ARG UBUNTU_VERSION=24.04

FROM docker.io/ubuntu:$UBUNTU_VERSION AS rknn-sdk

ARG RKNN_SDK_TAG=v2.3.2
ARG RKNN_SDK_BASE=https://raw.githubusercontent.com/airockchip/rknn-toolkit2
ARG RKNN_LIB_SHA256=d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /rknn/include && \
    curl -fsSL -o /rknn/librknnrt.so \
        "${RKNN_SDK_BASE}/${RKNN_SDK_TAG}/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" && \
    curl -fsSL -o /rknn/include/rknn_api.h \
        "${RKNN_SDK_BASE}/${RKNN_SDK_TAG}/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h" && \
    curl -fsSL -o /rknn/include/rknn_matmul_api.h \
        "${RKNN_SDK_BASE}/${RKNN_SDK_TAG}/rknpu2/runtime/Linux/librknn_api/include/rknn_matmul_api.h" && \
    echo "${RKNN_LIB_SHA256}  /rknn/librknnrt.so" | sha256sum -c -

# ── BUILD: compile the release binaries with the NPU path on ──────────────────
FROM docker.io/ubuntu:$UBUNTU_VERSION AS build

ARG UBUNTU_VERSION=24.04
ARG GCC_VERSION=14
ARG AUDIOCPP_VERSION=dev

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} g++-${GCC_VERSION} make cmake libgomp1 ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION}

WORKDIR /app
COPY . .

# librknnrt is aarch64 (armhf would be the other board arch); a native arm64
# runner builds this in the time the amd64 image takes.
RUN test "$(uname -m)" = "aarch64"

COPY --from=rknn-sdk /rknn/ /opt/rknn/

# Same shape as cpu.Dockerfile, plus the RKNN trio: the flag, the headers, the
# library. The RKNN sidecar sources join the build per linked model, so the
# full model set keeps working and every family with a board path gets one.
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DAUDIOCPP_MODEL_SET=full \
        -DENGINE_ENABLE_CPU_ALL_VARIANTS=ON \
        -DENGINE_ENABLE_CUDA=OFF \
        -DENGINE_ENABLE_VULKAN=OFF \
        -DENGINE_ENABLE_RKNN=ON \
        -DENGINE_RKNN_INCLUDE_DIR=/opt/rknn/include \
        -DENGINE_RKNN_LIBRARY=/opt/rknn/librknnrt.so \
        -DENGINE_ENABLE_OPENMP=ON \
        -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON \
        -DAUDIOCPP_VERSION="${AUDIOCPP_VERSION}" \
        -DENGINE_BUILD_EXAMPLES=OFF \
        -DENGINE_BUILD_TESTS=OFF \
        -DENGINE_BUILD_WARMBENCH=OFF && \
    cmake --build build --parallel $(nproc) \
        --target audiocpp_cli \
        --target audiocpp_server \
        --target audiocpp_model_manager \
        --target model_perf

# Collect shared libraries (the driver runtime rides along in the $ORIGIN lib
# dir: librknnrt.so carries that SONAME, so the binaries' rpath finds it like
# any other collected library).
RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    cp -P /opt/rknn/librknnrt.so /app/lib/

# Collect binaries + multiplexer into /app/full
RUN mkdir -p /app/full && \
    cp build/bin/audiocpp_cli build/bin/audiocpp_server build/bin/audiocpp_model_manager \
       build/bin/model_perf /app/full/ && \
    cp .devops/entrypoint.sh /app/full/entrypoint.sh && \
    chmod +x /app/full/entrypoint.sh

# ── BASE: shared runtime (OS + common libs + NPU driver runtime) ──────────────
FROM docker.io/ubuntu:$UBUNTU_VERSION AS base

ARG UBUNTU_VERSION=24.04
ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=N/A
ARG IMAGE_SOURCE=N/A

LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="audio.cpp (Rockchip NPU)" \
      org.opencontainers.image.description="An all-in-one, pure C++ inference engine for audio models, powered by ggml, with the RKNN decoder path for Rockchip boards" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

# Runtime deps: OpenMP threading, curl (healthcheck), ffmpeg (audio I/O),
# python3 for the native WebUI's spec-backed model installer, and eSpeak-ng
# (the Piper/Inflect phonemizers dlopen libespeak-ng and need its data dir).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 curl ffmpeg python3 ca-certificates libespeak-ng1 espeak-ng-data && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

WORKDIR /app

# ── FULL: all binaries + entrypoint.sh multiplexer ────────────────────────────
FROM base AS full

COPY --from=build /app/full /app
COPY model_specs/ /app/model_specs/
COPY tools/model_manager_v2.py /app/tools/model_manager_v2.py

RUN mkdir -p /app/models && chown ubuntu:ubuntu /app/models

USER ubuntu

ENTRYPOINT ["/app/entrypoint.sh"]
