#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR=""
BUILD_TYPE="RelWithDebInfo"
WITH_TESTS="OFF"
WITH_EXAMPLES="OFF"
WITH_WARMBENCH="OFF"
AUDIOCPP_DEPLOYMENT_BUILD="OFF"
OPENMP_MODE="off"
LLAMAFILE="ON"
NATIVE_CPU="ON"
METAL_EMBED_LIBRARY="ON"
DEPLOYMENT_TARGET=""
ARCHS=""
TARGETS=()
JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)"

usage() {
    cat <<'EOF'
Usage: scripts/build_metal.sh [options]

Build audio.cpp on macOS with the ggml Metal backend enabled.

Options:
  --build-dir <path>       CMake build directory.
                           Default: build/macos-metal-<build-type>
  --build-type <type>      CMake build type.
                           Default: RelWithDebInfo
  --deployment-target <v>  Minimum macOS deployment target.
  --archs "<list>"         Space or comma separated macOS architectures.
                           Default: CMake/native default
  --openmp ON|OFF|auto     Enable OpenMP host parallelism.
                           Default: OFF
  --llamafile ON|OFF       Enable ggml llamafile SGEMM.
                           Default: ON
  --native-cpu ON|OFF      Build ggml CPU kernels with native host ISA flags.
                           Default: ON
  --embed-metal ON|OFF     Embed ggml Metal shader library.
                           Default: ON
  --with-tests             Build framework unit tests.
  --with-examples          Build example binaries.
  --with-warmbench         Build warmbench helper binaries.
  --deployment-build       Embed package specs for standalone GGUF/model loading.
  --target <name>          Build a specific CMake target. May be repeated.
  -j, --jobs <n>           Parallel build jobs.
  -h, --help               Show this help.

Examples:
  scripts/build_metal.sh
  scripts/build_metal.sh --target audiocpp_cli
  scripts/build_metal.sh --build-type Release --archs arm64
EOF
}

normalize_on_off() {
    local name="$1"
    local value="$2"
    case "$value" in
        ON|on|On|1|true|TRUE|yes|YES)
            printf 'ON'
            ;;
        OFF|off|Off|0|false|FALSE|no|NO)
            printf 'OFF'
            ;;
        *)
            echo "$name must be ON or OFF" >&2
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --deployment-target)
            DEPLOYMENT_TARGET="$2"
            shift 2
            ;;
        --archs)
            ARCHS="$2"
            shift 2
            ;;
        --openmp)
            OPENMP_MODE="$2"
            shift 2
            ;;
        --llamafile)
            LLAMAFILE="$(normalize_on_off --llamafile "$2")"
            shift 2
            ;;
        --native-cpu)
            NATIVE_CPU="$(normalize_on_off --native-cpu "$2")"
            shift 2
            ;;
        --embed-metal)
            METAL_EMBED_LIBRARY="$(normalize_on_off --embed-metal "$2")"
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
            shift
            ;;
        --target)
            TARGETS+=("$2")
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "audio.cpp Metal builds require macOS." >&2
    exit 1
fi

for tool in cmake xcrun; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool not found on PATH: $tool" >&2
        exit 1
    fi
done

if ! xcrun --sdk macosx --find metal >/dev/null 2>&1; then
    echo "Xcode Metal toolchain was not found. Install Xcode or the Command Line Tools." >&2
    exit 1
fi

case "$OPENMP_MODE" in
    ON|on|On)
        ENGINE_ENABLE_OPENMP="ON"
        ;;
    OFF|off|Off)
        ENGINE_ENABLE_OPENMP="OFF"
        ;;
    auto)
        ENGINE_ENABLE_OPENMP="OFF"
        if command -v brew >/dev/null 2>&1; then
            LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
            if [[ -n "$LIBOMP_PREFIX" && -d "$LIBOMP_PREFIX" ]]; then
                ENGINE_ENABLE_OPENMP="ON"
            fi
        fi
        ;;
    *)
        echo "--openmp must be ON, OFF, or auto" >&2
        exit 1
        ;;
esac

BUILD_TYPE_NAME="$(printf '%s' "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
if [[ "$BUILD_TYPE_NAME" == "relwithdebinfo" ]]; then
    BUILD_TYPE_NAME="release"
fi

if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="$REPO_ROOT/build/macos-metal-${BUILD_TYPE_NAME}"
fi

GENERATOR=""
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
fi

CMAKE_CMD=(
    cmake
    -S "$REPO_ROOT"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DENGINE_ENABLE_CUDA=OFF
    -DENGINE_ENABLE_VULKAN=OFF
    -DENGINE_ENABLE_METAL=ON
    -DENGINE_ENABLE_OPENMP="$ENGINE_ENABLE_OPENMP"
    -DENGINE_ENABLE_LLAMAFILE="$LLAMAFILE"
    -DENGINE_ENABLE_NATIVE_CPU="$NATIVE_CPU"
    -DGGML_OPENMP="$ENGINE_ENABLE_OPENMP"
    -DGGML_METAL_EMBED_LIBRARY="$METAL_EMBED_LIBRARY"
    -DENGINE_BUILD_TESTS="$WITH_TESTS"
    -DENGINE_BUILD_EXAMPLES="$WITH_EXAMPLES"
    -DENGINE_BUILD_WARMBENCH="$WITH_WARMBENCH"
    -DAUDIOCPP_DEPLOYMENT_BUILD="$AUDIOCPP_DEPLOYMENT_BUILD"
)

if [[ -n "$GENERATOR" ]]; then
    CMAKE_CMD+=(-G "$GENERATOR")
fi
if [[ -n "$DEPLOYMENT_TARGET" ]]; then
    CMAKE_CMD+=(-DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET")
fi
if [[ -n "$ARCHS" ]]; then
    CMAKE_CMD+=(-DCMAKE_OSX_ARCHITECTURES="${ARCHS//,/;}")
fi
if [[ "$OPENMP_MODE" == "auto" && "${ENGINE_ENABLE_OPENMP}" == "ON" ]]; then
    CMAKE_CMD+=(-DOpenMP_ROOT="$LIBOMP_PREFIX")
fi

echo "Using source dir: $REPO_ROOT"
echo "Using build dir: $BUILD_DIR"
if [[ -n "$GENERATOR" ]]; then
    echo "Using generator: $GENERATOR"
else
    echo "Using generator: CMake default"
fi
echo "Including Metal backend: ON"
echo "Embedding Metal library: $METAL_EMBED_LIBRARY"
echo "Including OpenMP: $ENGINE_ENABLE_OPENMP"
echo "Including llamafile: $LLAMAFILE"
echo "Native CPU optimization: $NATIVE_CPU"
echo "Building examples: $WITH_EXAMPLES"
echo "Building tests: $WITH_TESTS"
echo "Building warmbench: $WITH_WARMBENCH"
echo "Deployment build: $AUDIOCPP_DEPLOYMENT_BUILD"

"${CMAKE_CMD[@]}"

BUILD_CMD=(cmake --build "$BUILD_DIR" --parallel "$JOBS")
for target in "${TARGETS[@]}"; do
    BUILD_CMD+=(--target "$target")
done

"${BUILD_CMD[@]}"
