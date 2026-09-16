#!/usr/bin/env bash

set -euo pipefail

CONDA_ENV=""
BUILD_DIR=""
BUILD_TYPE="RelWithDebInfo"
CUDA_MODE="auto"
CUDA_ARCH=""
VULKAN_MODE="off"
HIP_MODE="off"
GPU_TARGETS=""
WITH_TESTS="OFF"
WITH_EXAMPLES="OFF"
WITH_WARMBENCH="OFF"
AUDIOCPP_DEPLOYMENT_BUILD="OFF"
DEPLOYMENT_BUILD_SET="OFF"
AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER="OFF"
AUDIOCPP_USE_SYSTEM_OPENSSL="OFF"
AUDIOCPP_BORINGSSL_ARCHIVE=""
AUDIOCPP_MODEL_SET="full"
AUDIOCPP_MODELS=""
NATIVE_CPU="ON"
LLAMAFILE="ON"
TARGETS=()
JOBS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --backend)
            if [[ "$2" == "cuda" ]]; then
                CUDA_MODE="on"
                VULKAN_MODE="off"
                HIP_MODE="off"
            elif [[ "$2" == "cpu" ]]; then
                CUDA_MODE="off"
                VULKAN_MODE="off"
                HIP_MODE="off"
            elif [[ "$2" == "vulkan" ]]; then
                CUDA_MODE="off"
                VULKAN_MODE="on"
                HIP_MODE="off"
            elif [[ "$2" == "hip" || "$2" == "rocm" ]]; then
                CUDA_MODE="off"
                VULKAN_MODE="off"
                HIP_MODE="on"
            else
                echo "Unsupported legacy --backend value: $2" >&2
                exit 1
            fi
            shift 2
            ;;
        --cuda)
            CUDA_MODE="$2"
            shift 2
            ;;
        --vulkan)
            VULKAN_MODE="$2"
            shift 2
            ;;
        --hip)
            HIP_MODE="$2"
            shift 2
            ;;
        --gpu-targets)
            GPU_TARGETS="$2"
            shift 2
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --with-tests)
            WITH_TESTS="ON"
            shift
            ;;
        --with-examples)
            WITH_EXAMPLES="ON"
            shift
            ;;
        --with-warmbench)
            WITH_WARMBENCH="ON"
            shift
            ;;
        --deployment-build)
            AUDIOCPP_DEPLOYMENT_BUILD="ON"
            DEPLOYMENT_BUILD_SET="ON"
            shift
            ;;
        --no-deployment-build)
            AUDIOCPP_DEPLOYMENT_BUILD="OFF"
            DEPLOYMENT_BUILD_SET="ON"
            shift
            ;;
        --native-model-manager)
            AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER="ON"
            shift
            ;;
        --system-openssl)
            AUDIOCPP_USE_SYSTEM_OPENSSL="ON"
            shift
            ;;
        --boringssl-archive)
            AUDIOCPP_BORINGSSL_ARCHIVE="$2"
            shift 2
            ;;
        --model-set)
            case "$2" in
                full|core|custom)
                    AUDIOCPP_MODEL_SET="$2"
                    ;;
                *)
                    echo "--model-set must be full, core, or custom" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --models)
            AUDIOCPP_MODELS="$2"
            shift 2
            ;;
        --native-cpu)
            case "$2" in
                ON|on|On|1|true|TRUE|yes|YES)
                    NATIVE_CPU="ON"
                    ;;
                OFF|off|Off|0|false|FALSE|no|NO)
                    NATIVE_CPU="OFF"
                    ;;
                *)
                    echo "--native-cpu must be ON or OFF" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --llamafile)
            case "$2" in
                ON|on|On|1|true|TRUE|yes|YES)
                    LLAMAFILE="ON"
                    ;;
                OFF|off|Off|0|false|FALSE|no|NO)
                    LLAMAFILE="OFF"
                    ;;
                *)
                    echo "--llamafile must be ON or OFF" >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --target)
            TARGETS+=("$2")
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER" != "ON" ]]; then
    if [[ "$AUDIOCPP_USE_SYSTEM_OPENSSL" == "ON" ]]; then
        echo "--system-openssl requires --native-model-manager" >&2
        exit 1
    fi
    if [[ -n "$AUDIOCPP_BORINGSSL_ARCHIVE" ]]; then
        echo "--boringssl-archive requires --native-model-manager" >&2
        exit 1
    fi
fi

GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
fi

# ccache when it is installed. ggml probes for it upstream; audio.cpp's
# CMakeLists forces that probe off, so nothing was caching anything. Measured on
# this tree: ~2% slower on a cold build, ~14x faster on a rebuild.
CCACHE_ARGS=()
if command -v ccache >/dev/null 2>&1; then
    CCACHE_ARGS=(
        -DCMAKE_C_COMPILER_LAUNCHER=ccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
    )
fi

ENGINE_ENABLE_CUDA="OFF"
ENGINE_ENABLE_VULKAN="OFF"
case "$CUDA_MODE" in
    on)
        ENGINE_ENABLE_CUDA="ON"
        ;;
    off)
        ENGINE_ENABLE_CUDA="OFF"
        ;;
    auto)
        if command -v nvcc >/dev/null 2>&1; then
            ENGINE_ENABLE_CUDA="ON"
        fi
        ;;
    *)
        echo "Unknown --cuda mode: $CUDA_MODE (expected on, off, or auto)" >&2
        exit 1
        ;;
esac

case "$VULKAN_MODE" in
    on)
        ENGINE_ENABLE_VULKAN="ON"
        ;;
    off)
        ENGINE_ENABLE_VULKAN="OFF"
        ;;
    *)
        echo "Unknown --vulkan mode: $VULKAN_MODE (expected on or off)" >&2
        exit 1
        ;;
esac

ENGINE_ENABLE_HIP="OFF"
HIP_CLANG_C=""
HIP_CLANG_CXX=""
case "$HIP_MODE" in
    on)
        ENGINE_ENABLE_HIP="ON"
        # ROCm ships its own clang; hipconfig -l prints the ROCm lib dir's
        # parent on some installs, so prefer ROCM_PATH/HIP_PATH when set.
        ROCM_HOME="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
        HIP_CLANG_C="$ROCM_HOME/bin/clang"
        HIP_CLANG_CXX="$ROCM_HOME/bin/clang++"
        if [[ ! -x "$HIP_CLANG_CXX" ]] && command -v hipconfig >/dev/null 2>&1; then
            HIP_CLANG_C="$(hipconfig -l)/clang"
            HIP_CLANG_CXX="$(hipconfig -l)/clang++"
        fi
        if [[ ! -x "$HIP_CLANG_C" || ! -x "$HIP_CLANG_CXX" ]]; then
            echo "ROCm clang not found (looked in $ROCM_HOME/bin and via hipconfig). Install ROCm or set ROCM_PATH." >&2
            exit 1
        fi
        if [[ -z "$GPU_TARGETS" ]]; then
            AMDGPU_ARCH=""
            if [[ -x "$ROCM_HOME/bin/amdgpu-arch" ]]; then
                AMDGPU_ARCH="$ROCM_HOME/bin/amdgpu-arch"
            elif command -v amdgpu-arch >/dev/null 2>&1; then
                AMDGPU_ARCH="$(command -v amdgpu-arch)"
            fi
            if [[ -n "$AMDGPU_ARCH" ]]; then
                GPU_TARGETS="$("$AMDGPU_ARCH" 2>/dev/null | grep -xE 'gfx[0-9a-f]{3,}' | sort -u | paste -sd ';' -)"
            fi
            # Fall back to rocminfo (e.g. amdgpu-arch missing from minimal
            # ROCm packages, or sandboxed environments where it prints nothing).
            # Only accept full agent "Name: gfxXXX" lines — a loose grep would
            # also catch fragments like "gfx11" from ISA description text.
            if [[ -z "$GPU_TARGETS" ]]; then
                ROCMINFO=""
                if [[ -x "$ROCM_HOME/bin/rocminfo" ]]; then
                    ROCMINFO="$ROCM_HOME/bin/rocminfo"
                elif command -v rocminfo >/dev/null 2>&1; then
                    ROCMINFO="$(command -v rocminfo)"
                fi
                if [[ -n "$ROCMINFO" ]]; then
                    GPU_TARGETS="$("$ROCMINFO" 2>/dev/null | grep -E '^[[:space:]]+Name:[[:space:]]+gfx[0-9a-f]{3,}[[:space:]]*$' | grep -oE 'gfx[0-9a-f]{3,}' | sort -u | paste -sd ';' -)"
                fi
            fi
            if [[ -z "$GPU_TARGETS" ]]; then
                echo "Could not detect GPU targets (tried amdgpu-arch and rocminfo)." >&2
                echo "If this machine has no visible AMD GPU (VM/container), pass --gpu-targets gfxXXXX explicitly (semicolon-separated for several)." >&2
                exit 1
            fi
        fi
        GPU_TARGETS="${GPU_TARGETS//,/;}"
        ;;
    off)
        ENGINE_ENABLE_HIP="OFF"
        ;;
    *)
        echo "Unknown --hip mode: $HIP_MODE (expected on or off)" >&2
        exit 1
        ;;
esac

# HIP builds default to deployment builds so the produced binaries embed
# package specs and run standalone outside the repo checkout. Other backends
# keep the historical OFF default; --no-deployment-build opts out.
if [[ "$ENGINE_ENABLE_HIP" == "ON" && "$DEPLOYMENT_BUILD_SET" == "OFF" ]]; then
    AUDIOCPP_DEPLOYMENT_BUILD="ON"
fi

if [[ "$ENGINE_ENABLE_CUDA" == "ON" && "$ENGINE_ENABLE_VULKAN" == "ON" ]]; then
    echo "CUDA and Vulkan backends cannot both be enabled by this script" >&2
    exit 1
fi

ENABLED_GPU_BACKENDS=0
[[ "$ENGINE_ENABLE_CUDA" == "ON" ]] && ENABLED_GPU_BACKENDS=$((ENABLED_GPU_BACKENDS + 1))
[[ "$ENGINE_ENABLE_VULKAN" == "ON" ]] && ENABLED_GPU_BACKENDS=$((ENABLED_GPU_BACKENDS + 1))
[[ "$ENGINE_ENABLE_HIP" == "ON" ]] && ENABLED_GPU_BACKENDS=$((ENABLED_GPU_BACKENDS + 1))
if [[ "$ENABLED_GPU_BACKENDS" -gt 1 ]]; then
    echo "CUDA, Vulkan, and HIP backends are mutually exclusive in this script" >&2
    exit 1
fi

BACKEND_NAME="cpu"
if [[ "$ENGINE_ENABLE_CUDA" == "ON" ]]; then
    BACKEND_NAME="cuda"
elif [[ "$ENGINE_ENABLE_VULKAN" == "ON" ]]; then
    BACKEND_NAME="vulkan"
elif [[ "$ENGINE_ENABLE_HIP" == "ON" ]]; then
    BACKEND_NAME="hip"
fi

BUILD_TYPE_NAME="$(printf '%s' "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
if [[ "$BUILD_TYPE_NAME" == "relwithdebinfo" ]]; then
    BUILD_TYPE_NAME="release"
fi

if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="build/linux-${BACKEND_NAME}-${BUILD_TYPE_NAME}"
fi

if [[ -z "$JOBS" ]]; then
    JOBS="$(nproc 2>/dev/null || echo 8)"
fi

RUNNER=()
if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required on PATH when --conda-env is used" >&2
        exit 1
    fi
    if ! conda info --envs | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
        echo "Conda environment '$CONDA_ENV' was not found" >&2
        exit 1
    fi
    RUNNER=(conda run -n "$CONDA_ENV")
fi

if [[ -n "$CONDA_ENV" ]]; then
    echo "Using conda env: $CONDA_ENV"
fi
echo "Using generator: $GENERATOR"
echo "Using build dir: $BUILD_DIR"
echo "Including CUDA backend: $ENGINE_ENABLE_CUDA"
if [[ "$ENGINE_ENABLE_CUDA" == "ON" ]]; then
    echo "CUDA architectures: ${CUDA_ARCH:-<portable default list>}"
fi
echo "Including Vulkan backend: $ENGINE_ENABLE_VULKAN"
echo "Including HIP backend: $ENGINE_ENABLE_HIP"
if [[ "$ENGINE_ENABLE_HIP" == "ON" ]]; then
    echo "ROCm clang: $HIP_CLANG_CXX"
    echo "GPU targets: $GPU_TARGETS"
fi
echo "Native CPU optimization: $NATIVE_CPU"
echo "llamafile SGEMM: $LLAMAFILE"
echo "Building examples: $WITH_EXAMPLES"
echo "Building tests: $WITH_TESTS"
echo "Building warmbench: $WITH_WARMBENCH"
echo "Deployment build: $AUDIOCPP_DEPLOYMENT_BUILD"
echo "Native model manager: $AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER"
if [[ "$AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER" == "ON" ]]; then
    echo "System OpenSSL: $AUDIOCPP_USE_SYSTEM_OPENSSL"
    echo "BoringSSL archive: ${AUDIOCPP_BORINGSSL_ARCHIVE:-<download at configure time>}"
fi
echo "Model composite: $AUDIOCPP_MODEL_SET"
if [[ -n "$AUDIOCPP_MODELS" ]]; then
    echo "Selected models: $AUDIOCPP_MODELS"
fi

CMAKE_ARGS=(
    -S .
    -B "$BUILD_DIR"
    -G "$GENERATOR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DENGINE_ENABLE_CUDA="$ENGINE_ENABLE_CUDA"
    -DENGINE_ENABLE_VULKAN="$ENGINE_ENABLE_VULKAN"
    -DENGINE_ENABLE_HIP="$ENGINE_ENABLE_HIP"
    -DENGINE_ENABLE_NATIVE_CPU="$NATIVE_CPU"
    -DENGINE_ENABLE_LLAMAFILE="$LLAMAFILE"
    -DENGINE_BUILD_EXAMPLES="$WITH_EXAMPLES"
    -DENGINE_BUILD_TESTS="$WITH_TESTS"
    -DENGINE_BUILD_WARMBENCH="$WITH_WARMBENCH"
    -DAUDIOCPP_DEPLOYMENT_BUILD="$AUDIOCPP_DEPLOYMENT_BUILD"
    -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER="$AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER"
    -DAUDIOCPP_USE_SYSTEM_OPENSSL="$AUDIOCPP_USE_SYSTEM_OPENSSL"
    -UAUDIOCPP_BORINGSSL_ARCHIVE
    -DAUDIOCPP_MODEL_SET="$AUDIOCPP_MODEL_SET"
    -DAUDIOCPP_MODELS="$AUDIOCPP_MODELS"
)
if [[ -n "$AUDIOCPP_BORINGSSL_ARCHIVE" ]]; then
    CMAKE_ARGS+=(-DAUDIOCPP_BORINGSSL_ARCHIVE="$AUDIOCPP_BORINGSSL_ARCHIVE")
fi

# Empty unless ccache was found, and appended rather than set so an explicit
# CMAKE_*_COMPILER_LAUNCHER passed by the caller still wins (last -D wins).
if [[ ${#CCACHE_ARGS[@]} -gt 0 ]]; then
    CMAKE_ARGS+=("${CCACHE_ARGS[@]}")
fi

if [[ "$ENGINE_ENABLE_HIP" == "ON" ]]; then
    CMAKE_ARGS+=(
        -DCMAKE_C_COMPILER="$HIP_CLANG_C"
        -DCMAKE_CXX_COMPILER="$HIP_CLANG_CXX"
        # GPU_BUILD_TARGETS/AMDGPU_TARGETS are sticky cache entries set by
        # hip-config from GPU_TARGETS; without -U a previous configure's arch
        # list would win over a new --gpu-targets value.
        -UGPU_BUILD_TARGETS
        -UAMDGPU_TARGETS
        -DGPU_TARGETS="$GPU_TARGETS"
    )
fi

# Mirror the HIP path above: when no arch was asked for, build for the GPU that
# is actually here. HIP has done this for --gpu-targets all along, and CUDA
# silently compiled nine architectures instead -- roughly 3x the wall time, for
# hardware the developer running this script does not have.
#
# The two differ in what happens when detection fails, and CUDA can afford to be
# gentler: HIP has no portable default and exits 1, while leaving
# CMAKE_CUDA_ARCHITECTURES unset here falls through to the portable list. So a
# GPU-less machine (VM, container, CI) still gets a working build, and is told
# which way it went rather than left to infer it.
#
# ⚠ This is the DEVELOPER entry point. No release or CI path calls this script
# -- they invoke cmake directly and pin the arch explicitly -- so the fast
# default cannot reach a published binary.
if [[ "$ENGINE_ENABLE_CUDA" == "ON" && -z "$CUDA_ARCH" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # compute_cap reads e.g. "8.6"; CMake wants "86".
        DETECTED_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                       | tr -d ' .' | grep -xE '[0-9]{2,3}' | sort -u | paste -sd ';' -)"
        if [[ -n "$DETECTED_CC" ]]; then
            CUDA_ARCH="$DETECTED_CC"
            echo "Detected CUDA arch $CUDA_ARCH; building for this GPU only."
            echo "  Pass --cuda-arch \"<list>\" for a portable build, e.g. --cuda-arch \"75;80;86;89\"."
        fi
    fi
    if [[ -z "$CUDA_ARCH" ]]; then
        echo "No local CUDA GPU detected; using the portable default arch list (slower to build)."
        echo "  Pass --cuda-arch native on a machine with a GPU for a faster local build."
    fi
fi

if [[ "$ENGINE_ENABLE_CUDA" == "ON" && -n "$CUDA_ARCH" ]]; then
    CMAKE_ARGS+=(
        # CMAKE_CUDA_ARCHITECTURES is a sticky cache entry; -U it so a previous
        # configure's arch list does not win over a new --cuda-arch value
        # (mirrors the HIP --gpu-targets handling above).
        -UCMAKE_CUDA_ARCHITECTURES
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    )
fi

"${RUNNER[@]}" cmake "${CMAKE_ARGS[@]}"

BUILD_CMD=("${RUNNER[@]}" cmake --build "$BUILD_DIR" --parallel "$JOBS")
for target in "${TARGETS[@]}"; do
    BUILD_CMD+=(--target "$target")
done

"${BUILD_CMD[@]}"
