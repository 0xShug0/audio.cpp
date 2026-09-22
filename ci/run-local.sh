#!/usr/bin/env bash
#
# Run a CI job locally, in the container image the workflow uses.
#
# The GitHub jobs are the only place some build configurations get exercised,
# and a failure there costs a push and a wait. This runs the same steps on the
# same base image so a configuration can be checked before it is submitted.
#
#   ci/run-local.sh                 # list the jobs
#   ci/run-local.sh linux-vulkan    # one job
#   ci/run-local.sh linux           # every job whose name starts with "linux"
#   ci/run-local.sh all
#
# Requires docker. Build trees live in build/ci-local-<job> and are reused
# between runs; pass --clean to start from nothing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_UBUNTU="ubuntu:24.04"
IMAGE_NIX="nixos/nix:latest"

CLEAN=0
JOBS=()
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) JOBS+=("$arg") ;;
    esac
done

ALL_JOBS=(linux-cpu linux-vulkan nix-cpu nix-vulkan)

# Docker state shared by every job.
DEPS_IMAGE="audiocpp-ci-local:ubuntu-24.04"
NIX_STORE_VOLUME="audiocpp-ci-local-nix-store"

usage() {
    cat <<EOF
usage: ci/run-local.sh [--clean] <job>...

jobs:
  linux-cpu       linux-build.yml, ENGINE_ENABLE_VULKAN=OFF
  linux-vulkan    linux-build.yml, ENGINE_ENABLE_VULKAN=ON  (compiles the Vulkan shaders)
  nix-cpu         nix-build.yml, nix build .#cpu
  nix-vulkan      nix-build.yml, nix build .#vulkan

  linux           both linux jobs
  nix             both nix jobs
  all             everything
EOF
}

if [ ${#JOBS[@]} -eq 0 ]; then
    usage
    exit 0
fi

# Expand the group names.
EXPANDED=()
for job in "${JOBS[@]}"; do
    case "$job" in
        all)   EXPANDED+=("${ALL_JOBS[@]}") ;;
        linux) EXPANDED+=(linux-cpu linux-vulkan) ;;
        nix)   EXPANDED+=(nix-cpu nix-vulkan) ;;
        *)
            found=0
            for known in "${ALL_JOBS[@]}"; do
                [ "$job" = "$known" ] && found=1
            done
            if [ $found -eq 0 ]; then
                echo "unknown job: $job" >&2
                usage >&2
                exit 2
            fi
            EXPANDED+=("$job")
            ;;
    esac
done

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2
    exit 1
fi

# Discard the shared docker state once, before any job runs: the dependency
# image is cached by tag and would otherwise survive a change to the package
# list, and the nix store must not be wiped between the two nix jobs.
if [ $CLEAN -eq 1 ]; then
    docker image rm "$DEPS_IMAGE" >/dev/null 2>&1 || true
    docker volume rm "$NIX_STORE_VOLUME" >/dev/null 2>&1 || true
fi

# Run as the invoking user so the build tree is not left root-owned.
docker_run() {
    local image="$1"; shift
    # linux-build.yml sets CC/CXX as job-level env, so they reach every step.
    # The Vulkan shader generator is an ExternalProject that configures itself
    # during the build step, and it needs them there too.
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        --env HOME=/tmp \
        --env CC=gcc-13 \
        --env CXX=g++-13 \
        --volume "$REPO_ROOT:/src" \
        --workdir /src \
        "$image" bash -euo pipefail -c "$*"
}

# The apt install needs root, so the dependencies are baked into a thin image
# once rather than installed on every run.
ubuntu_image_with_deps() {
    local tag="$DEPS_IMAGE"
    if ! docker image inspect "$tag" >/dev/null 2>&1; then
        echo "--- building $tag (one time)" >&2
        # Context is stdin-only: the Dockerfile copies nothing, and the repo
        # root would ship the build trees to the daemon for no reason.
        docker build -q -t "$tag" - >/dev/null <<EOF || return 1
FROM $IMAGE_UBUNTU
RUN apt-get update && apt-get install -y --no-install-recommends \\
        gcc-13 g++-13 cmake make glslc libvulkan-dev spirv-headers \\
        python3 ca-certificates git \\
    && rm -rf /var/lib/apt/lists/*
EOF
    fi
    echo "$tag"
}

run_loader_check() {
    # Both workflows run this before building.
    echo "--- check_loader_catalog_sync"
    docker_run "$1" "python3 tools/check_loader_catalog_sync.py --self-test && \
                     python3 tools/check_loader_catalog_sync.py" || return 1
}

run_linux_job() {
    local backend="$1" vulkan="$2"
    local build_dir="build/ci-local-linux-$backend"
    local image
    image="$(ubuntu_image_with_deps)" || return 1

    if [ $CLEAN -eq 1 ]; then
        rm -rf "$REPO_ROOT/$build_dir"
    fi

    run_loader_check "$image" || return 1

    echo "--- configure ($backend)"
    # Keep these flags in step with the Configure step of linux-build.yml.
    docker_run "$image" "cmake -S . -B '$build_dir' \
        -DCMAKE_BUILD_TYPE=Debug \
        -DAUDIOCPP_VERSION=ci \
        -DENGINE_ENABLE_CUDA=OFF \
        -DENGINE_ENABLE_VULKAN=$vulkan \
        -DENGINE_BUILD_TESTS=ON" || return 1

    echo "--- build ($backend)"
    docker_run "$image" "cmake --build '$build_dir' --parallel \$(nproc) \
        --target audiocpp_cli audiocpp_server audiocpp_gguf" || return 1

    echo "--- build and run unit tests ($backend)"
    docker_run "$image" "cmake --build '$build_dir' --parallel \$(nproc) && \
        ctest --test-dir '$build_dir' --output-on-failure --parallel 4" || return 1
}

run_nix_job() {
    local package="$1"
    local image
    image="$(ubuntu_image_with_deps)" || return 1

    # nix-build.yml runs the loader check too, and nixos/nix has no python3.
    run_loader_check "$image" || return 1

    # The container's /nix is ephemeral under --rm, so without this every run
    # re-downloads the whole closure. A named volume keeps the store between
    # runs; --clean discards it (once, above).
    local store_volume="$NIX_STORE_VOLUME"
    docker volume create "$store_volume" >/dev/null || return 1

    echo "--- nix build .#$package"
    # The container runs as root against a checkout owned by someone else, so
    # libgit2 refuses the flake input until the path is marked safe.
    docker run --rm \
        --volume "$REPO_ROOT:/src" \
        --volume "$store_volume:/nix" \
        --workdir /src \
        "$IMAGE_NIX" \
        bash -euo pipefail -c "
            git config --global --add safe.directory /src
            nix --extra-experimental-features 'nix-command flakes' \
                build '.#$package' --no-link --print-build-logs
        "
}

failed=()
for job in "${EXPANDED[@]}"; do
    echo
    echo "================ $job"
    start=$SECONDS
    ok=1
    case "$job" in
        linux-cpu)    run_linux_job cpu OFF     || ok=0 ;;
        linux-vulkan) run_linux_job vulkan ON   || ok=0 ;;
        nix-cpu)      run_nix_job cpu           || ok=0 ;;
        nix-vulkan)   run_nix_job vulkan        || ok=0 ;;
    esac
    elapsed=$(( SECONDS - start ))
    if [ $ok -eq 1 ]; then
        echo "================ $job PASSED (${elapsed}s)"
    else
        echo "================ $job FAILED (${elapsed}s)"
        failed+=("$job")
    fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "all jobs passed"
else
    echo "failed: ${failed[*]}"
    exit 1
fi
