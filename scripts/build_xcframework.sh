#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_ROOT="$REPO_ROOT/build/xcframework"
OUTPUT="$BUILD_ROOT/AudioCpp.xcframework"
BUILD_TYPE="Release"
ARCHS="$(uname -m)"
PLATFORM="macos"
DEPLOYMENT_TARGET="13.3"
JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)"
CLEAN="OFF"
LLAMAFILE="ON"
NATIVE_CPU="ON"
OPENMP="OFF"
AUDIOCPP_DEPLOYMENT_BUILD="OFF"
AUDIOCPP_MODEL_SET="full"
AUDIOCPP_MODELS=""

usage() {
    cat <<'EOF'
Usage: scripts/build_xcframework.sh [options]

Build audio.cpp as a static XCFramework.

Options:
  --output <path>       Output XCFramework path.
                        Default: build/xcframework/AudioCpp.xcframework
  --build-root <path>   Intermediate build root.
                        Default: build/xcframework
  --build-type <type>   CMake build type.
                        Default: Release
  --platform macos|ios  Target Apple platform.
                        Default: macos
  --deployment-target <version>
                        Minimum deployment target.
                        Default: 13.3
  --archs "<list>"      Space or comma separated target architectures.
                        Default: current machine architecture
  --universal           Build arm64 and x86_64, then lipo a universal macOS lib.
  --clean               Remove intermediate/output directories first.
  --llamafile ON|OFF    Enable ggml llamafile SGEMM.
                        Default: ON
  --native-cpu ON|OFF   Build ggml CPU kernels with native host ISA flags.
                        Default: ON
  --openmp ON|OFF       Enable OpenMP host parallelism.
                        Default: OFF
  --deployment-build    Embed package specs for standalone GGUF/model loading.
  --model-set full|core|custom
                        Model composite to build.
                        Default: full
  --models "<list>"     Comma or semicolon separated model targets when
                        --model-set custom is used.
  -j, --jobs <n>        Parallel build jobs.
  -h, --help            Show this help.

Examples:
  scripts/build_xcframework.sh
  scripts/build_xcframework.sh --universal
  scripts/build_xcframework.sh --platform ios --archs arm64 --model-set custom --models vibevoice
  scripts/build_xcframework.sh --output /tmp/AudioCpp.xcframework
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --build-root)
            BUILD_ROOT="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --platform)
            case "$2" in
                macos|ios)
                    PLATFORM="$2"
                    ;;
                *)
                    echo "--platform must be macos or ios" >&2
                    exit 1
                    ;;
            esac
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
        --universal)
            ARCHS="arm64 x86_64"
            shift
            ;;
        --clean)
            CLEAN="ON"
            shift
            ;;
        --llamafile)
            LLAMAFILE="$2"
            shift 2
            ;;
        --native-cpu)
            NATIVE_CPU="$2"
            shift 2
            ;;
        --openmp)
            OPENMP="$2"
            shift 2
            ;;
        --deployment-build)
            AUDIOCPP_DEPLOYMENT_BUILD="ON"
            shift
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
    echo "audio.cpp XCFramework builds require macOS." >&2
    exit 1
fi

for tool in cmake xcodebuild libtool lipo xcrun; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool not found on PATH: $tool" >&2
        exit 1
    fi
done

case "$LLAMAFILE" in
    ON|OFF) ;;
    on) LLAMAFILE="ON" ;;
    off) LLAMAFILE="OFF" ;;
    *)
        echo "--llamafile must be ON or OFF" >&2
        exit 1
        ;;
esac

case "$NATIVE_CPU" in
    ON|OFF) ;;
    on) NATIVE_CPU="ON" ;;
    off) NATIVE_CPU="OFF" ;;
    *)
        echo "--native-cpu must be ON or OFF" >&2
        exit 1
        ;;
esac

case "$OPENMP" in
    ON|OFF) ;;
    on) OPENMP="ON" ;;
    off) OPENMP="OFF" ;;
    *)
        echo "--openmp must be ON or OFF" >&2
        exit 1
        ;;
esac

ARCHS="${ARCHS//,/ }"
read -r -a ARCH_LIST <<< "$ARCHS"
if [[ ${#ARCH_LIST[@]} -eq 0 ]]; then
    echo "No architectures requested." >&2
    exit 1
fi

GENERATOR=""
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
fi

SDK_NAME="macosx"
PLATFORM_LABEL="macos"
if [[ "$PLATFORM" == "ios" ]]; then
    SDK_NAME="iphoneos"
    PLATFORM_LABEL="ios"
    GENERATOR="Xcode"
fi

ARTIFACT_ROOT="$BUILD_ROOT/artifacts"
HEADER_DIR="$ARTIFACT_ROOT/Headers"
UNIVERSAL_LIB="$ARTIFACT_ROOT/libMiniTTS.a"

if [[ "$CLEAN" == "ON" ]]; then
    rm -rf "$BUILD_ROOT" "$OUTPUT"
fi

mkdir -p "$ARTIFACT_ROOT"
rm -rf "$HEADER_DIR"
mkdir -p "$HEADER_DIR"

ditto "$REPO_ROOT/include" "$HEADER_DIR"
ditto "$REPO_ROOT/external/ggml/include" "$HEADER_DIR"
find "$HEADER_DIR" -name '._*' -delete

archive_for_arch() {
    local arch="$1"
    local build_dir="$BUILD_ROOT/$PLATFORM_LABEL-$arch"
    local out_dir="$ARTIFACT_ROOT/$PLATFORM_LABEL-$arch"
    local out_lib="$out_dir/libMiniTTS.a"

    mkdir -p "$out_dir"

    echo "==> Configuring audio.cpp for $PLATFORM_LABEL $arch"
    local cmake_cmd=(
        cmake
        -S "$REPO_ROOT" \
        -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_SYSROOT="$SDK_NAME" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
        -DENGINE_ENABLE_CUDA=OFF \
        -DENGINE_ENABLE_VULKAN=OFF \
        -DENGINE_ENABLE_METAL=ON \
        -DENGINE_ENABLE_OPENMP="$OPENMP" \
        -DENGINE_ENABLE_LLAMAFILE="$LLAMAFILE" \
        -DENGINE_ENABLE_NATIVE_CPU="$NATIVE_CPU" \
        -DGGML_OPENMP="$OPENMP" \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DENGINE_BUILD_TESTS=OFF \
        -DENGINE_BUILD_EXAMPLES=OFF \
        -DAUDIOCPP_DEPLOYMENT_BUILD="$AUDIOCPP_DEPLOYMENT_BUILD" \
        -DAUDIOCPP_MODEL_SET="$AUDIOCPP_MODEL_SET" \
        -DAUDIOCPP_MODELS="$AUDIOCPP_MODELS"
    )
    if [[ "$PLATFORM" == "ios" ]]; then
        cmake_cmd+=(
            -DCMAKE_SYSTEM_NAME=iOS
            -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY
        )
    fi
    if [[ -n "$GENERATOR" ]]; then
        cmake_cmd+=(-G "$GENERATOR")
    fi
    "${cmake_cmd[@]}"

    echo "==> Building engine_runtime for $PLATFORM_LABEL $arch"
    cmake --build "$build_dir" --target engine_runtime --config "$BUILD_TYPE" -j "$JOBS"

    local libs=()
    local optional_libs=()
    if [[ "$GENERATOR" == "Xcode" ]]; then
        local lib_root="$build_dir/$BUILD_TYPE-$SDK_NAME"
        libs=(
            "$lib_root/libengine_runtime.a"
            "$build_dir/build/engine_core.build/$BUILD_TYPE-$SDK_NAME/libengine_core.a"
            "$lib_root/libcjson_vendor.a"
            "$lib_root/libyaml_vendor.a"
            "$build_dir/external/sentencepiece/src/$BUILD_TYPE-$SDK_NAME/libsentencepiece.a"
            "$build_dir/ggml/src/$BUILD_TYPE-$SDK_NAME/libggml.a"
            "$build_dir/ggml/src/$BUILD_TYPE-$SDK_NAME/libggml-base.a"
            "$build_dir/ggml/src/$BUILD_TYPE-$SDK_NAME/libggml-cpu.a"
            "$build_dir/ggml/src/ggml-metal/$BUILD_TYPE-$SDK_NAME/libggml-metal.a"
        )
        optional_libs=(
            "$build_dir/ggml/src/ggml-blas/$BUILD_TYPE-$SDK_NAME/libggml-blas.a"
        )
    else
        libs=(
            "$build_dir/libengine_runtime.a"
            "$build_dir/libcjson_vendor.a"
            "$build_dir/libyaml_vendor.a"
            "$build_dir/external/sentencepiece/src/libsentencepiece.a"
            "$build_dir/ggml/src/libggml.a"
            "$build_dir/ggml/src/libggml-base.a"
            "$build_dir/ggml/src/libggml-cpu.a"
            "$build_dir/ggml/src/ggml-metal/libggml-metal.a"
        )
        optional_libs=(
            "$build_dir/ggml/src/ggml-blas/libggml-blas.a"
        )
    fi
    for lib in "${optional_libs[@]}"; do
        if [[ -f "$lib" ]]; then
            libs+=("$lib")
        fi
    done

    for lib in "${libs[@]}"; do
        if [[ ! -f "$lib" ]]; then
            echo "Expected static library missing: $lib" >&2
            exit 1
        fi
    done

    echo "==> Merging static libraries for $PLATFORM_LABEL $arch"
    rm -f "$out_lib"
    libtool -static -o "$out_lib" "${libs[@]}"
    lipo -info "$out_lib"
}

for arch in "${ARCH_LIST[@]}"; do
    archive_for_arch "$arch"
done

rm -f "$UNIVERSAL_LIB"
if [[ ${#ARCH_LIST[@]} -eq 1 ]]; then
    cp "$ARTIFACT_ROOT/$PLATFORM_LABEL-${ARCH_LIST[0]}/libMiniTTS.a" "$UNIVERSAL_LIB"
else
    lipo_inputs=()
    for arch in "${ARCH_LIST[@]}"; do
        lipo_inputs+=("$ARTIFACT_ROOT/$PLATFORM_LABEL-$arch/libMiniTTS.a")
    done
    lipo -create "${lipo_inputs[@]}" -output "$UNIVERSAL_LIB"
fi

lipo -info "$UNIVERSAL_LIB"

echo "==> Creating XCFramework"
rm -rf "$OUTPUT"
xcodebuild -create-xcframework \
    -library "$UNIVERSAL_LIB" \
    -headers "$HEADER_DIR" \
    -output "$OUTPUT"
find "$OUTPUT" -name '._*' -delete

echo "Created: $OUTPUT"
