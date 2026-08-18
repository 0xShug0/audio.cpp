# Releasing audio.cpp prebuilt binaries

Everything is driven from the **GitHub UI** — no command line needed. Releases
are **semantic-version and tag-driven**: there is no auto-release on pushes to
`main`. You release by tagging a version; the built binaries are then attached
to that version's Release.

## Release a version (recommended)

1. Open the repository → **Releases** → **Draft a new release**.
2. **Choose a tag** → create a new `v*` tag, e.g. `v1.2.0` (or `v1.2.0-rc1`).
3. Add a title/notes; check **"Set as a pre-release"** if it's a candidate.
4. **Publish release.**

Publishing pushes the tag, which starts the build. When it finishes (~1–2 h;
the Windows CUDA jobs dominate), the prebuilt binaries are attached to that
Release.

## Build a version without publishing (dry run)

1. **Actions** tab → **Release** → **Run workflow**.
2. **Version:** enter `1.2.0` (or `v1.2.0`).
3. Leave the **Publish a GitHub Release** box **unchecked**.
4. **Run workflow** → builds all backends and uploads them as workflow
   artifacts, but creates no tag or Release. Use this to validate before
   cutting a release.

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

- **Pushing a `v*` tag** (e.g. by drafting + Publishing a Release) → builds and
  attaches the binaries to that tag's Release.
- **Manual run from the Actions UI** (with a version) → builds; publishes only
  if **Publish** is checked.
- `main` pushes do **not** trigger a release.

## Verification

1. Confirm the expected assets exist on the Release page.
2. On a real NVIDIA GPU machine, run e.g.
   `audiocpp_cli.exe --backend cuda ...` and confirm the log reports a CUDA
   device (`ggml_cuda_init` / "found N CUDA devices"). GitHub CI runners have
   **no GPU**, so a green build does not prove GPU runtime attach — this check
   is required.