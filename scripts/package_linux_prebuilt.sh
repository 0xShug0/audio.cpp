#!/usr/bin/env bash

set -euo pipefail

BACKEND="cpu"
ARCH="$(uname -m)"
OUTPUT_DIR=""
BUILD_DIR=""
JOBS="$(nproc)"

usage() {
    cat <<'EOF'
Usage: scripts/package_linux_prebuilt.sh [options]

Build and archive a portable Linux deployment package.

Options:
  --backend cpu|vulkan  Backend to include (default: cpu)
  --arch NAME           Architecture label used in the archive name
  --output-dir DIR      Package output directory (default: build/prebuilt)
  --build-dir DIR       CMake build directory
  -j, --jobs N          Parallel build jobs
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
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

if [[ "$BACKEND" != "cpu" && "$BACKEND" != "vulkan" ]]; then
    echo "--backend must be cpu or vulkan" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/build/prebuilt"
fi
if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="$REPO_ROOT/build/linux-${BACKEND}-prebuilt"
fi

PACKAGE_NAME="audiocpp-linux-${ARCH}-${BACKEND}"
STAGE_DIR="$OUTPUT_DIR/$PACKAGE_NAME"
ARCHIVE_PATH="$OUTPUT_DIR/$PACKAGE_NAME.tar.gz"

mkdir -p "$OUTPUT_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

BUILD_ARGS=(
    --backend "$BACKEND"
    --build-dir "$BUILD_DIR"
    --build-type RelWithDebInfo
    --deployment-build
    --cpu-all-variants
    --jobs "$JOBS"
    --target audiocpp_cli
    --target audiocpp_server
    --target audiocpp_gguf
)

(
    cd "$REPO_ROOT"
    bash scripts/build_linux.sh "${BUILD_ARGS[@]}"
)

for binary in audiocpp_cli audiocpp_server audiocpp_gguf; do
    if [[ ! -x "$BUILD_DIR/bin/$binary" ]]; then
        echo "Expected binary was not built: $BUILD_DIR/bin/$binary" >&2
        exit 1
    fi
done

cp -a "$BUILD_DIR/bin/." "$STAGE_DIR/"
cp "$REPO_ROOT/LICENSE" "$STAGE_DIR/"

cat > "$STAGE_DIR/README.md" <<EOF
# audio.cpp Linux ${BACKEND} prebuilt

This package contains \`audiocpp_cli\`, \`audiocpp_server\`, and
\`audiocpp_gguf\` for Linux ${ARCH}. Models are downloaded separately.

The build is a self-contained audio.cpp deployment build with portable,
runtime-selected CPU kernels. Keep all included shared libraries next to the
executables.

Quick check:

\`\`\`bash
./audiocpp_cli --help
\`\`\`
EOF

tar -C "$OUTPUT_DIR" -czf "$ARCHIVE_PATH" "$PACKAGE_NAME"
(
    cd "$OUTPUT_DIR"
    sha256sum "$(basename "$ARCHIVE_PATH")" > "$(basename "$ARCHIVE_PATH").sha256"
)

echo "Created $ARCHIVE_PATH"
echo "Created $ARCHIVE_PATH.sha256"
