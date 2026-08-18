# Releasing audio.cpp prebuilt binaries

Everything is driven from the **GitHub Actions UI** — no command line needed.
Building + publishing a release is a single click; a dry run is a check-box
away. Publishing is gated to `main`.

## Release via the GitHub UI (recommended)

1. Open the repository on GitHub.
2. Go to the **Actions** tab.
3. In the left sidebar pick **Release**.
4. Click **Run workflow** (top right).
   - **Branch:** keep `main`.
   - **Publish a GitHub Release:** leave it **checked** for a normal release.
     Uncheck it to build the binaries without publishing (a dry run).
5. Click **Run workflow** → wait for `Create release` to finish (~1–2 h; the
   Windows CUDA jobs dominate).

When it completes, the new GitHub Release appears on the **Releases** page
with all prebuilt binaries attached. The tag is `b<N>` (a monotonic build
number derived from commit count).

## What gets shipped

| Platform       | Backend     | Artifact |
|----------------|-------------|----------|
| Windows x64    | CUDA 12.4   | `...-bin-windows-x64-cuda12.4.zip` (+ `cudart-...-cuda12.4.zip`) |
| Windows x64    | CUDA 13.3   | `...-bin-windows-x64-cuda13.3.zip` (+ `cudart-...-cuda13.3.zip`) |
| Windows x64    | Vulkan      | `...-bin-windows-x64-vulkan.zip`    |
| Windows x64    | CPU         | `...-bin-windows-x64-cpu.zip`       |
| Ubuntu x64     | Vulkan/CPU  | `...-bin-ubuntu-x64-vulkan.tar.gz` / `...-cpu.tar.gz` |
| macOS arm64/x64| Metal       | `...-bin-macos-<arch>-metal.tar.gz` |

Notes:

- CUDA builds use `GGML_BACKEND_DL`, so CUDA kernels ship once as
  `ggml-cuda.dll`; the heavy CUDA runtime (`cudart`/`cuBLAS`/`cuBLASLt`/`cuFFT`)
  is bundled in a separate `cudart-...zip`.
- CUDA architectures are pinned (all-real) so binaries are portable across the
  supported NVIDIA GPUs instead of being tied to the (GPU-less) CI host.

## When the workflow runs

- **Push to `main`** touching source/CMake files → builds **and** publishes.
- **Manual run from the Actions UI** → builds; publishes if **Publish** is
  checked (effective only on `main`).
- A commit message containing `[no release]` skips the release entirely.

## Verification

1. Confirm the expected assets exist on the Release page.
2. On a real NVIDIA GPU machine, run e.g.
   `audiocpp_cli.exe --backend cuda ...` and confirm the log reports a CUDA
   device (`ggml_cuda_init` / "found N CUDA devices"). GitHub CI runners have
   **no GPU**, so a green build does not prove GPU runtime attach — this check
   is required.

## Optional CLI convenience (not required)

For maintainers who prefer the terminal, `scripts/release.sh` wraps the same
flow (`--watch` to monitor, `--dry-run` to build only). The GitHub UI flow
above is equivalent.