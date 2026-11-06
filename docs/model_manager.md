# Model Manager

`tools/model_manager_v2.py` downloads supported model packages into the
framework's expected `models/` layout. It reads package metadata from
`model_specs/*.json`, which is the current source of truth for default download
links.

The same package surface is also available through the reusable native C++
`audiocpp_package_manager` library. Two native frontends use that library:

- `audiocpp_server` exposes asynchronous management endpoints for the embedded
  WebUI.
- `audiocpp_model_manager` provides direct headless/CLI and Docker access
  without starting a server.

The Python v2 manager remains a supported alternative for existing scripted
workflows while the native command surface matures.

## Opt-in Native Network Backend

Native model management is disabled by default so ordinary CLI and server
builds do not configure, fetch, compile, or link an HTTP/TLS dependency. Enable
it explicitly to build the reusable library, standalone manager, and server
download/install endpoints:

```bash
cmake -S . -B build -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON
cmake --build build --target audiocpp_model_manager audiocpp_server
```

With build helper scripts, use the native-manager flag:

```bash
./scripts/build_linux.sh --backend cuda --native-model-manager --target audiocpp_server
```

```powershell
.\scripts\build_windows.ps1 -NativeModelManager -Target audiocpp_server
```

Run the managed WebUI with:

```bash
audiocpp_server --ui --ui-management --backend cuda
```

The enabled native library, server, and standalone manager share one vendored
`cpp-httplib` transport on Windows, Linux, and macOS. HTTPS is enabled with a
pinned BoringSSL release that is fetched at configure time and linked
statically. The resulting executables do not require libcurl, a system OpenSSL
development package, or separate TLS DLLs at runtime.

Offline and sandboxed builds may provide the same verified source archive with
`-DAUDIOCPP_BORINGSSL_ARCHIVE=/path/to/boringssl.tar.gz`. The Nix package does
this through a fixed-output derivation, so its CMake phase never accesses the
network.

With build helper scripts, use `--boringssl-archive <path>` on Linux/macOS or
`-BoringSslArchive <path>` on Windows.

Packagers that prefer their distribution's OpenSSL can select it explicitly:

```bash
cmake -S . -B build \
  -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON \
  -DAUDIOCPP_USE_SYSTEM_OPENSSL=ON
```

With build helper scripts, use `--system-openssl` on Linux/macOS or
`-SystemOpenSsl` on Windows.

When using system OpenSSL on machines with multiple OpenSSL installations, make
sure CMake resolves the distribution OpenSSL that will also be available where
the binary runs. Bundled BoringSSL is the simpler portable choice for normal
local builds and release images.

## Native Standalone Manager

The native executable embeds the active `model_specs/*.json` catalog. An
external `model_specs/` below `--repository-root` is an optional development
override, not a deployment requirement.

```bash
audiocpp_model_manager list
audiocpp_model_manager list --remote
audiocpp_model_manager info qwen3_asr_0_6b_q8_0 --remote
audiocpp_model_manager install qwen3_asr_0_6b_q8_0 --models-dir models
audiocpp_model_manager clean qwen3_asr_0_6b_q8_0 --models-dir models
audiocpp_model_manager remove qwen3_asr_0_6b_q8_0 --models-dir models
```

`list` and `info` return machine-readable JSON. Native installation uses the
same downloader, staging, size validation, atomic publication, shared-sidecar,
and version metadata logic as the server/WebUI.

For a container image, no server process is required:

```dockerfile
RUN audiocpp_model_manager install qwen3_asr_0_6b_q8_0 --models-dir /models
```

```text
                     +-- audiocpp_server -> REST API / native WebUI
model_specs/*.json -> audiocpp_package_manager
                     +-- audiocpp_model_manager -> CLI / Docker / scripts
```

When a family has a ready-to-use GGUF package, the default install should be that
GGUF package. The old safetensors/converter catalog is still available as
`tools/model_manager_deprecated.py`, but it is a legacy path for models that have
not migrated to spec-backed packages.

## GGUF Downloads

Ready-to-use GGUF packages are published here:

- Core released models: [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf)
- Fun-ASR-Nano: [FunAudioLLM/Fun-ASR-Nano-2512-GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-GGUF)
- Community OuteTTS package: [mirek190/audio.cpp](https://huggingface.co/mirek190/audio.cpp/tree/main/Text%20to%20audio%20(TTS))

For support status and tested precision coverage, see the [GGUF guide](gguf.md).
For measured 16-bit vs Q8 speed and peak-VRAM results, see the
[Q8 performance report](reports/gguf_q8_performance.md).

## Download Sources

Both the native package manager and `tools/model_manager_v2.py` download from
Hugging Face (`kind: "huggingface_snapshot"`) or ModelScope
(`kind: "modelscope_snapshot"`); see
[maintainers/model_specs.md](maintainers/model_specs.md) for the spec fields.
The native endpoints can be overridden for mirrors or tests with
`AUDIOCPP_HF_BASE_URL` (default `https://huggingface.co`) and
`AUDIOCPP_MS_BASE_URL` (default `https://www.modelscope.cn`). The Python tool
uses the standard `HF_ENDPOINT` for Hugging Face and the same
`AUDIOCPP_MS_BASE_URL` for ModelScope.

Authentication is provider-scoped: Hugging Face requests may carry
`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, while ModelScope requests only carry
`AUDIOCPP_MS_TOKEN` (optional, for access-restricted ModelScope repos). The
Hugging Face token is never sent to a ModelScope endpoint, and vice versa.

### Hugging Face Xet Transfers

Most large files on Hugging Face are stored with
[Xet](https://huggingface.co/docs/xet/index). For those files the native
manager uses the Xet download protocol directly instead of streaming the
`/resolve` redirect over a single connection:

1. The metadata request that already reads file size and revision (redirects
   not followed) also reads `X-Xet-Hash` and the commit-pinned `xet-auth` link.
2. A read-scoped CAS token is requested from the Hub
   (`/api/models/{repo}/xet-read-token/{revision}`).
3. The file's reconstruction (`/v2/reconstructions/{hash}`) is fetched in
   10 GB batches. It lists the compressed xorb byte ranges that make up the
   file, each with a short-lived signed URL.
4. Those ranges are downloaded in parallel, decoded as they stream in (LZ4
   frames, optionally byte-grouped), and written straight to their offsets in
   the usual staging file. Memory stays bounded; whole ranges are never
   buffered.

The installed result goes through the same staging, size check, atomic
publication, and manifest logic as the HTTP route. When the Hub reports the
file's SHA-256 (`X-Linked-ETag`), the reconstructed file is also verified
against it. The hash runs on a background thread while the transfer is in
progress, so it adds little time.

Files without Xet metadata, ModelScope packages, and any Xet failure before
publication (no token, no v2 endpoint, an unexpected response) fall back to
the regular HTTP download. The progress message then says
`(Xet unavailable: …; using HTTP)`. A complete reconstruction that fails size
or SHA-256 verification is reported as an error instead of being retried.

Credentials stay scoped. The Hugging Face token is sent only to the Hub, the
CAS token only to the CAS endpoint the Hub returned, and the signed xorb URLs
receive no credentials. Redirects are not followed for any Xet request.

Environment controls:

| Variable | Default | Effect |
|---|---|---|
| `AUDIOCPP_HF_XET` | enabled | `0` disables Xet and always uses HTTP |
| `AUDIOCPP_HF_XET_CONCURRENCY` | `16` | Parallel xorb range requests per file (1-256) |
| `AUDIOCPP_HF_XET_VERIFY` | enabled | `0` skips the SHA-256 check (size and per-term length checks remain) |

Xet support is download-only and needs no extra system dependency. LZ4 is
vendored under `external/lz4` and SHA-256 comes from the TLS library already
linked for HTTPS.

### Python Source Override

The Python v2 manager can redirect any package to ModelScope on demand,
without editing `model_specs/*.json`:

```bash
python3 tools/model_manager_v2.py install qwen3_tts --source modelscope --source-repo HereIsMark/audio.cpp-gguf
```

`--source modelscope` is accepted by `install` and `sizes`. `--source-repo`
names the ModelScope repo (`namespace/name`); when omitted, the spec's own
repo name is reused on ModelScope. Passing `--source-repo` without
`--source modelscope` is rejected. Revision translation: a spec revision that
is unset or `main` becomes ModelScope's default branch `master`; any other
explicit revision passes through unchanged.

Note that manifest etags are source-specific (Hugging Face etag vs ModelScope
sha256), so cross-source freshness checks can spuriously report that an
installed package has an update. Query with the same `--source` that was used
to install.

## Dependencies

The native manager needs no Python runtime. The default bundled-TLS build needs
network access at CMake configure time to fetch the pinned BoringSSL source
archive, unless `AUDIOCPP_BORINGSSL_ARCHIVE` points at a local copy. Runtime
model installation needs network access to the package source, usually Hugging
Face.

The Python v2 manager needs:

- Python 3
- Network access to the upstream model source

Legacy converter installs through `tools/model_manager_deprecated.py` may also
need `torch`, `safetensors`, `PyYAML`, or model-specific conversion inputs.

## Native Commands

`audiocpp_model_manager` supports the same package ids as the managed WebUI:

- `list` shows the available package ids
- `list --remote` includes remote availability and package-size metadata where
  available
- `info <package>` shows one package's install status and metadata
- `info <package> --remote` also checks the remote package source
- `install <package>` downloads one package into a models root
- `clean <package>` removes incomplete staging data for one package
- `remove <package>` removes the files managed by one installed package

Use `--models-dir PATH` to select the installation root. Use
`--repository-root PATH` only when developing against an external
`model_specs/` override; normal deployed binaries use the embedded catalog.

## Python v2 Commands

- `list` shows the available package ids
- `list --json` prints a machine-readable package catalog
- `info` shows the target layout, required files, and install source for one package
- `info <package> --json` prints machine-readable package details
- `install` downloads or converts one package into a models root
- `installed --json` reports package-file presence without network access
- `sizes --json` reports remote sizes plus local/remote revision status
- `uninstall` removes only files declared by one package precision
- `clean-partial` removes abandoned staging directories left by an interrupted process

The runtime loader catalog is also available from:

```bash
audiocpp_cli --list-loaders --json
```

## Quick Start

List installable packages:

```bash
python3 tools/model_manager_v2.py list
```

Inspect one package:

```bash
python3 tools/model_manager_v2.py info qwen3_tts
```

Install into the default `models/` directory:

```bash
python3 tools/model_manager_v2.py install qwen3_tts
```

Install into a custom models root:

```bash
python3 tools/model_manager_v2.py install vevo2 --models-root /path/to/models
```

Overwrite an existing install:

```bash
python3 tools/model_manager_v2.py install pocket_tts --overwrite
```

Successful installs write a small `.audiocpp-package-<id>.json` manifest beside
the package files. It records the resolved Hugging Face commit and enables the
native WebUI to report whether the local package is current. Packages installed
before this metadata existed remain usable and show `Version unknown` until they
are reinstalled once.

Clean staging directories left by a process or machine interruption:

```bash
python3 tools/model_manager_v2.py clean-partial pocket_tts --models-root models
```

The native WebUI can stop an active download cooperatively. Its worker checks a
cancellation marker between download chunks and deletes its staging directory
before reporting the job as cancelled.

Package variants may declare an identical sidecar at the same destination. The
manager downloads that file once, reuses it when installing a sibling precision,
and keeps it until the last package that declares it is removed. PocketTTS uses
this for the public `alba` preset shared by its English Q8 and BF16 packages.

Install a converter-style package that needs a source file:

```bash
python3 tools/model_manager_deprecated.py info voxcpm2_audiovae
python3 tools/model_manager_deprecated.py install voxcpm2_audiovae --source-file models/VoxCPM2/audiovae.pth --models-root models --overwrite
```

Kroko Community defaults to the ready-to-use GGUF package:

```bash
python3 tools/model_manager_v2.py install kroko_asr_community_q8_0 --models-root models --overwrite
```

The original Kroko `.data` conversion workflow remains documented in the
community model guide for users who want to build from the upstream source
package themselves.

## Package Notes

For shared audio.cpp GGUF packages, the v2 model manager installs the default GGUF.
That is usually `q8_0`; FP32-only packages such as Inflect Micro v2 use
original dtype instead. Other precision variants can be downloaded directly from
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf).

Use `python3 tools/model_manager_v2.py list --json` for the current package
ids and defaults. The legacy loader/catalog sync notes are maintained only for
the deprecated catalog path.
