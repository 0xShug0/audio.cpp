#!/usr/bin/env bash
# Created by ZeroDot1.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORTED_AMD_TARGETS="gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_PATH
BUILD_ROOT="${AUDIOCPP_BUILD_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/audio.cpp/arch-build}"
INSTALL_ROOT="${AUDIOCPP_INSTALL_ROOT:-$HOME/.local/opt/audio.cpp}"
BIN_ROOT="${AUDIOCPP_BIN_ROOT:-$HOME/.local/bin}"
MODEL_SET="${AUDIOCPP_MODEL_SET:-full}"
MODELS="${AUDIOCPP_MODELS:-}"

info() { printf '[audio.cpp] %s\n' "$*"; }
fail() { printf '[audio.cpp] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Build and install audio.cpp with AMD HIP on Arch Linux and Arch-based systems.

Usage: scripts/install_arch_amd.sh [--help]

The interactive menu offers a build for the detected local GPU, a portable
RDNA3/RDNA3.5/RDNA4 target set, or a dedicated RDNA4 build. Binaries are
installed under ~/.local/opt/audio.cpp and launchers under ~/.local/bin.

For non-interactive use, set AUDIOCPP_ARCH_MODE to auto, portable, or rdna4.
Optional path overrides: AUDIOCPP_BUILD_ROOT, AUDIOCPP_INSTALL_ROOT,
AUDIOCPP_BIN_ROOT, ROCM_PATH. Set AUDIOCPP_MODEL_SET=core or custom to build a
smaller model catalog; AUDIOCPP_MODELS selects models for a custom build.
EOF
}

is_arch_linux() {
    [[ -r /etc/os-release ]] || return 1
    # shellcheck disable=SC1091
    . /etc/os-release
    [[ "${ID:-}" == arch || " ${ID_LIKE:-} " == *" arch "* ]]
}

detect_gpu_target() {
    local arch_tool="${ROCM_PATH}/bin/amdgpu-arch" target=""
    if [[ -x "$arch_tool" ]]; then
        target="$("$arch_tool" 2>/dev/null | grep -E '^gfx[0-9a-f]{3,}$' | sort -u | paste -sd ';' -)"
    elif command -v amdgpu-arch >/dev/null 2>&1; then
        target="$(amdgpu-arch 2>/dev/null | grep -E '^gfx[0-9a-f]{3,}$' | sort -u | paste -sd ';' -)"
    fi

    if [[ -z "$target" ]]; then
        local info_tool="${ROCM_PATH}/bin/rocminfo"
        if [[ ! -x "$info_tool" ]]; then
            info_tool="$(command -v rocminfo || true)"
        fi
        if [[ -n "$info_tool" ]]; then
            target="$("$info_tool" 2>/dev/null \
                | grep -E '^[[:space:]]+Name:[[:space:]]+gfx[0-9a-f]{3,}[[:space:]]*$' \
                | grep -oE 'gfx[0-9a-f]{3,}' | sort -u | paste -sd ';' -)"
        fi
    fi

    [[ -n "$target" ]] || fail "No AMD GPU target detected. Check ROCm with rocminfo or choose the portable/RDNA4 build."
    printf '%s\n' "$target"
}

ensure_dependencies() {
    local missing=() package hip_clang_dir="${ROCM_PATH}/lib/llvm/bin"
    if command -v hipconfig >/dev/null 2>&1; then
        hip_clang_dir="$(hipconfig -l)"
    fi
    for package in git cmake ninja; do
        command -v "$package" >/dev/null 2>&1 || missing+=("$package")
    done
    [[ -x "$hip_clang_dir/clang" && -x "$hip_clang_dir/clang++" ]] \
        || [[ -x "${ROCM_PATH}/bin/clang" && -x "${ROCM_PATH}/bin/clang++" ]] \
        || missing+=(rocm-hip-sdk)
    command -v rocminfo >/dev/null 2>&1 || [[ -x "${ROCM_PATH}/bin/rocminfo" ]] || missing+=(rocminfo)
    for package in hipblas rocblas hipblaslt; do
        [[ -e "${ROCM_PATH}/lib/lib${package}.so" ]] || missing+=("$package")
    done
    command -v pacman >/dev/null 2>&1 || fail "pacman was not found; this installer is for Arch Linux and Arch-based systems."

    ((${#missing[@]} == 0)) && return
    info "Missing build dependencies: ${missing[*]}"
    info "Installing the Arch packages needed for a HIP build."
    sudo pacman -S --needed base-devel git cmake ninja rocminfo rocm-hip-sdk hipblas hipblaslt rocblas
}

select_mode() {
    local selected="${AUDIOCPP_ARCH_MODE:-}"
    if [[ -n "$selected" ]]; then
        printf '%s\n' "$selected"
        return
    fi

    cat <<'EOF' >&2
Choose the AMD HIP build:
  1) Detected GPU       Build only for the GPU(s) visible to ROCm
  2) Portable RDNA set  Build for supported RDNA3, RDNA3.5, and RDNA4 targets
  3) RDNA4 only         Build separately for gfx1200 and gfx1201
  q) Quit
EOF
    local choice
    read -r -p 'Selection [1]: ' choice
    case "${choice:-1}" in
        1) printf 'auto\n' ;;
        2) printf 'portable\n' ;;
        3) printf 'rdna4\n' ;;
        q|Q) printf 'quit\n' ;;
        *) fail "Choose 1, 2, 3, or q." ;;
    esac
}

make_launcher() {
    local executable="$2" launcher="$BIN_ROOT/$1"
    mkdir -p "$BIN_ROOT"
    cat > "$launcher" <<EOF
#!/usr/bin/env bash
export ROCM_PATH="\${ROCM_PATH:-$ROCM_PATH}"
export LD_LIBRARY_PATH="\$ROCM_PATH/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$executable" "\$@"
EOF
    chmod 755 "$launcher"
}

install_build() {
    local mode="$1" targets="$2" build_dir="$BUILD_ROOT/$mode"
    local install_dir="$INSTALL_ROOT/$mode" executable

    info "Building HIP target(s): $targets"
    if [[ -x "${ROCM_PATH}/bin/rocminfo" ]]; then
        export PATH="${ROCM_PATH}/bin:$PATH"
    fi
    cd "$ROOT"
    local build_args=(
        --backend hip
        --gpu-targets "$targets"
        --build-dir "$build_dir"
        --build-type Release
        --deployment-build
        --model-set "$MODEL_SET"
    )
    if [[ "$MODEL_SET" == custom ]]; then
        [[ -n "$MODELS" ]] || fail "Set AUDIOCPP_MODELS when AUDIOCPP_MODEL_SET=custom."
        build_args+=(--models "$MODELS")
    fi
    JOBS="${JOBS:-$(nproc)}" scripts/build_linux.sh "${build_args[@]}"

    [[ -x "$build_dir/bin/audiocpp_cli" ]] || fail "Build completed without audiocpp_cli."
    mkdir -p "$install_dir/bin"
    for executable in audiocpp_cli audiocpp_server audiocpp_gguf; do
        [[ -x "$build_dir/bin/$executable" ]] || continue
        install -m 755 "$build_dir/bin/$executable" "$install_dir/bin/$executable"
        make_launcher "${executable}-amd-${mode}" "$install_dir/bin/$executable"
    done

    info "Installed build: $install_dir"
    info "CLI launcher: $BIN_ROOT/audiocpp_cli-amd-$mode"
}

main() {
    case "${1:-}" in
        --help|-h) usage; return 0 ;;
        "") ;;
        *) usage >&2; fail "Unknown argument: $1" ;;
    esac

    is_arch_linux || fail "This script supports Arch Linux and distributions declaring Arch in ID_LIKE."
    ensure_dependencies

    local mode targets
    mode="$(select_mode)"
    case "$mode" in
        quit) return 0 ;;
        auto)
            targets="$(detect_gpu_target)"
            mode="detected-${targets//;/_}"
            ;;
        portable)
            targets="$SUPPORTED_AMD_TARGETS"
            ;;
        rdna4)
            targets="gfx1200;gfx1201"
            ;;
        *) fail "AUDIOCPP_ARCH_MODE must be auto, portable, or rdna4." ;;
    esac

    install_build "$mode" "$targets"
    info "Build finished. Model files are installed separately; see docs/build/HIP.md."
}

main "$@"
