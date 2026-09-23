#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_ROOT="$REPO_ROOT/build/xcframework-ios"
OUTPUT="$BUILD_ROOT/AudioCpp.xcframework"
BUILD_TYPE="Release"
DEPLOYMENT_TARGET="16.3"
JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 8)"
CLEAN="OFF"
AUDIOCPP_MODEL_SET="custom"
AUDIOCPP_MODELS="yue2"

usage() {
    cat <<'EOF'
Usage: scripts/build_ios_xcframework.sh [options]

Build audio.cpp as an iOS device static XCFramework.

Options:
  --output <path>       Output XCFramework path.
                        Default: build/xcframework-ios/AudioCpp.xcframework
  --build-root <path>   Intermediate build root.
                        Default: build/xcframework-ios
  --build-type <type>   CMake build type.
                        Default: Release
  --deployment-target <version>
                        Minimum iOS deployment target.
                        Default: 16.3
  --clean               Remove intermediate/output directories first.
  --model-set <name>    Model composite to build.
                        Default: custom
  --models "<list>"     Comma or semicolon separated model targets.
                        Default: yue2
  -j, --jobs <n>        Parallel build jobs.
  -h, --help            Show this help.

Example:
  scripts/build_ios_xcframework.sh --clean --models yue2
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
        --deployment-target)
            DEPLOYMENT_TARGET="$2"
            shift 2
            ;;
        --clean)
            CLEAN="ON"
            shift
            ;;
        --model-set)
            AUDIOCPP_MODEL_SET="$2"
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
    echo "audio.cpp iOS XCFramework builds require macOS." >&2
    exit 1
fi

if [[ "$BUILD_ROOT" != /* ]]; then
    BUILD_ROOT="$REPO_ROOT/$BUILD_ROOT"
fi
if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$REPO_ROOT/$OUTPUT"
fi

for tool in cmake xcodebuild libtool xcrun; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool not found on PATH: $tool" >&2
        exit 1
    fi
done

ARTIFACT_ROOT="$BUILD_ROOT/artifacts"
HEADER_DIR="$ARTIFACT_ROOT/Headers"
BUILD_DIR="$BUILD_ROOT/ios-arm64"
OUT_DIR="$ARTIFACT_ROOT/ios-arm64"
OUT_LIB="$OUT_DIR/libMiniTTS.a"

if [[ "$CLEAN" == "ON" ]]; then
    rm -rf "$BUILD_ROOT" "$OUTPUT"
fi

mkdir -p "$ARTIFACT_ROOT" "$OUT_DIR"
rm -rf "$HEADER_DIR"
mkdir -p "$HEADER_DIR"
ditto "$REPO_ROOT/include" "$HEADER_DIR"
ditto "$REPO_ROOT/external/ggml/include" "$HEADER_DIR"

SENTENCEPIECE_XCODE_HELPER="$BUILD_ROOT/sentencepiece_xcode_property.cmake"
cat > "$SENTENCEPIECE_XCODE_HELPER" <<'EOF'
if(NOT COMMAND set_xcode_property)
  macro(set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE XCODE_RELVERSION)
    set(XCODE_RELVERSION_I "${XCODE_RELVERSION}")
    if(XCODE_RELVERSION_I STREQUAL "All")
      set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY} "${XCODE_VALUE}")
    else()
      set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY}[variant=${XCODE_RELVERSION_I}] "${XCODE_VALUE}")
    endif()
  endmacro()
endif()
EOF

echo "==> Configuring audio.cpp for iOS arm64"
cmake \
    -S "$REPO_ROOT" \
    -B "$BUILD_DIR" \
    -G Xcode \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
    -DENGINE_ENABLE_CUDA=OFF \
    -DENGINE_ENABLE_VULKAN=OFF \
    -DENGINE_ENABLE_HIP=OFF \
    -DENGINE_ENABLE_METAL=ON \
    -DENGINE_ENABLE_OPENMP=OFF \
    -DENGINE_ENABLE_NATIVE_CPU=OFF \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DENGINE_BUILD_TESTS=OFF \
    -DENGINE_BUILD_EXAMPLES=OFF \
    -DCMAKE_PROJECT_sentencepiece_INCLUDE="$SENTENCEPIECE_XCODE_HELPER" \
    -DAUDIOCPP_MODEL_SET="$AUDIOCPP_MODEL_SET" \
    -DAUDIOCPP_MODELS="$AUDIOCPP_MODELS"

echo "==> Building engine_runtime for iOS arm64"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --target engine_runtime -j "$JOBS"

libs=(
    "$BUILD_DIR/$BUILD_TYPE-iphoneos/libengine_runtime.a"
    "$BUILD_DIR/$BUILD_TYPE-iphoneos/libcjson_vendor.a"
    "$BUILD_DIR/$BUILD_TYPE-iphoneos/libyaml_vendor.a"
    "$BUILD_DIR/build/engine_core.build/$BUILD_TYPE-iphoneos/libengine_core.a"
    "$BUILD_DIR/external/sentencepiece/src/$BUILD_TYPE-iphoneos/libsentencepiece.a"
    "$BUILD_DIR/ggml/src/$BUILD_TYPE-iphoneos/libggml.a"
    "$BUILD_DIR/ggml/src/$BUILD_TYPE-iphoneos/libggml-base.a"
    "$BUILD_DIR/ggml/src/$BUILD_TYPE-iphoneos/libggml-cpu.a"
    "$BUILD_DIR/ggml/src/ggml-metal/$BUILD_TYPE-iphoneos/libggml-metal.a"
)

while IFS= read -r lib; do
    libs+=("$lib")
done < <(find "$BUILD_DIR/build" -path "*/$BUILD_TYPE-iphoneos/libengine_model_*.a" -type f | sort)

optional_libs=(
    "$BUILD_DIR/ggml/src/ggml-blas/$BUILD_TYPE-iphoneos/libggml-blas.a"
)
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

echo "==> Merging static libraries"
rm -f "$OUT_LIB" "$ARTIFACT_ROOT/libMiniTTS.a"
libtool -static -o "$OUT_LIB" "${libs[@]}"
cp "$OUT_LIB" "$ARTIFACT_ROOT/libMiniTTS.a"
xcrun lipo -info "$OUT_LIB"

echo "==> Creating XCFramework"
rm -rf "$OUTPUT"
xcodebuild -create-xcframework \
    -library "$OUT_LIB" \
    -headers "$HEADER_DIR" \
    -output "$OUTPUT"

echo "Created: $OUTPUT"
