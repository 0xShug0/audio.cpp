# Releasing audio.cpp prebuilt binaries

This document explains how to publish a release with all prebuilt binaries
(macOS Metal, Linux CPU/Vulkan, Windows CPU/Vulkan/CUDA 12.4 + 13.3).

Automation lives in `.github/workflows/release.yml` (ported from llama.cpp's
release workflow). Publishing is gated to `main`.

## Prerequisites

- The `gh` CLI installed and authenticated: `gh auth login`.
- `scripts/release.sh` on your PATH or run from the repo root.
- You are on the **`main`** branch with a clean working tree.

## Quick: build + publish a release

```bash
./scripts/release.sh --watch
```

This dispatches the Release workflow on `main` with `publish=true`, waits for
it to finish (~1–2 h; the Windows CUDA jobs dominate), and then prints the
GitHub Release URL and the uploaded asset list.

The release tag is `b<N>` (a monotonically increasing build number derived from
commit count, via `.github/actions/get-tag-name`).

## Build without publishing (dry run)

```bash
./scripts/release.sh --dry-run
```

Builds every backend and uploads them as Actions artifacts, but does **not**
create a tag or GitHub Release. Use this to validate before cutting a release.

## Manual alternative (GitHub UI / CLI)

1. Go to **Actions → Release → Run workflow** on `main`.
2. Set **publish** to `true` to create the Release (leave `false` for a dry run).
   - Note: publishing is only effective when the workflow runs on `main`.
3. Run and wait for the `Create release` job.

Or from the CLI (equivalent to the script):

```bash
gh workflow run release.yml --repo drzsdrtfg/audio.cpp --ref main -f publish=true
```

## What gets shipped

| Platform       | Backend            | Artifact |
|----------------|--------------------|----------|
| Windows x64    | CUDA 12.4          | `...-bin-windows-x64-cuda12.4.zip` (+ `cudart-...-cuda12.4.zip`) |
| Windows x64    | CUDA 13.3          | `...-bin-windows-x64-cuda13.3.zip` (+ `cudart-...-cuda13.3.zip`) |
| Windows x64    | Vulkan             | `...-bin-windows-x64-vulkan.zip`    |
| Windows x64    | CPU                | `...-bin-windows-x64-cpu.zip`       |
| Ubuntu x64     | Vulkan / CPU       | `...-bin-ubuntu-x64-vulkan.tar.gz` / `...-cpu.tar.gz` |
| macOS arm64/x64| Metal              | `...-bin-macos-<arch>-metal.tar.gz` |

Notes:

- CUDA builds use `GGML_BACKEND_DL`, so CUDA kernels ship once as
  `ggml-cuda.dll`; the heavy CUDA runtime (`cudart`/`cuBLAS`/`cuBLASLt`/`cuFFT`)
  is bundled in a separate `cudart-...zip`.
- CUDA architectures are pinned (all-real) so binaries are portable across
  supported NVIDIA GPUs instead of being tied to the (GPU-less) CI host.

## Verification

1. Confirm all expected assets exist on the Release page.
2. On a real machine with an NVIDIA GPU, run e.g.
   `audiocpp_cli.exe --backend cuda ...` and check the log reports a CUDA
   device (`ggml_cuda_init` / "found N CUDA devices").
   GitHub CI runners have **no GPU**, so a green build does not prove GPU
   runtime attach — this step is required.

## When releases are triggered

- **Push to `main`** touching source/CMake files → builds **and** publishes.
- **Manual dispatch** → builds artifacts; publishes only with `publish=true`.
- A commit message containing `[no release]` skips the release entirely.