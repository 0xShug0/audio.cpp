# Native WebUI Guide

audio.cpp ships one browser interface: a SvelteKit/TypeScript single-page app embedded directly in
`audiocpp_server`. The compiled server needs neither Python, Node.js, nor separate frontend files for
inference and normal UI operation.

## Run

Configure the optional native manager when you want model downloads and dynamic
model management from the browser:

```bash
cmake -S . -B build -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON
cmake --build build --target audiocpp_server
```

```powershell
.\build\windows-cuda-release\bin\audiocpp_server.exe --ui --ui-management --backend <backend>
```

```bash
./build/bin/audiocpp_server --ui --ui-management --backend <backend>
```

Open **http://127.0.0.1:8080**. `--ui-management` enables on-demand model loading,
unloading, package management, and temporary browser uploads. Models default to a `models/` directory
beside the server executable. The Models page can select and remember a different directory.

Without `AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON`, the server can still serve the
embedded UI with `--ui`, but the model-management endpoints are unavailable. Use
that mode for config-driven deployments or direct model paths.

An existing server configuration exposes the embedded UI by default:

```bash
audiocpp_server --config server.json
```

Configured models retain their eager or lazy behavior. Add `--ui-management` when the UI should be
allowed to load, switch, unload, download, or delete model packages. Use `--no-ui` for an API-only
server.

## Features

The native UI supports the shared model catalog and spec-driven controls for TTS, cloning, ASR,
generation, conversion, separation, VAD, diarization, alignment, and voice design. It also provides:

- background model downloads with progress, cancellation, partial-download cleanup, version status,
  precision selection, and package deletion;
- on-demand model loading with automatic unloading when switching model or precision;
- sentence-aware long-text synthesis with browser-side WAV merging;
- microphone capture and near-live input for streaming-capable ASR models;
- embedded demo voices with matching transcripts;
- a saved voice library stored in browser IndexedDB;
- multilingual UI resources under `native/lang/`;
- structured results, generated artifacts, and request timing.

Uploaded request files use a per-process temporary directory and are deleted when the server exits.
Saved voices remain in the current browser profile and are only uploaded when selected for a request.

## Model downloads

Inference and normal embedded UI operation do not require Python. When native
model management is enabled, browser downloads use the same C++ package manager
as `audiocpp_model_manager`.

The Python managers remain available for CLI workflows and legacy conversion
packages:

- `tools/model_manager_v2.py` can install spec-backed packages from the command line.
- `tools/model_manager_deprecated.py` is only for legacy checkpoint conversion layouts.

Loading an existing model directory or standalone GGUF does not invoke Python.

## Frontend development

Node.js is needed only to modify and rebuild the frontend:

```bash
cd webui/native
npm ci
npm run check
npm run build
```

The build creates `webui/native/dist/index.html`. CMake converts that single-file application into an
embedded byte array for `audiocpp_server`; rebuild the server after changing it. For live development,
run `npm run dev`; Vite proxies `/health` and `/v1` to a server on port 8080.

The frontend consumes:

- `webui/configs/models_catalog.json` for model/task entries;
- `webui/configs/model_params.json` for model-specific controls;
- `webui/native/lang/lang_<code>.json` for optional translations;
- `webui/native/demo_voices/` for demo reference voices embedded in the server.

English strings are built into `native/src/lib/i18n.ts` and are the fallback for missing translations.
