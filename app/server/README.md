# audio.cpp Server

`audiocpp_server` exposes audio.cpp models through HTTP and an optional WebUI.
The default runtime reuses each loaded model and its configured task session
across requests. You can register several models in one config; this does not
require the experimental parallel runtime.

## Contents

- [Quick start: full WebUI](#full-webui-no-config-file)
- [Quick start: first API request](#configured-api-first-request)
- [Build and choose a server mode](#build)
- [Configure models](#config)
- [Map CLI options to server config](#cli-options-to-server-configuration)
- [Control model loading and memory](#loading-and-memory)
- [Configure offline and streaming sessions](#offline-and-streaming-sessions)
- [Start and inspect the server](#start)
- [Voice presets and reference recordings](#voices)
- [API endpoints](#endpoints)
- [Troubleshooting](#troubleshooting)
- [Optional frontend modules and listeners](#optional-frontend-modules-and-listeners)
- [Experimental features](#experimental-features)

## Quick start

Choose **one** path below. Use an existing server binary, or [build one](#build).
Commands use `build/debug/bin/audiocpp_server` from a source checkout. For a
downloaded release, substitute its executable path, such as
`./audiocpp_server` on Linux/macOS or `.\audiocpp_server.exe` in PowerShell.
Run source-checkout commands from the repository root.

### Full WebUI (no config file)

To download models and run them in your browser:

```bash
build/debug/bin/audiocpp_server --ui --ui-management --backend cuda --log --log-file server.log
```

Open `http://127.0.0.1:8080`, install a model package from the model library,
then select it in the matching studio tab. Replace `cuda` with a backend your
binary and selected model support, such as `cpu`, `vulkan`, or `metal`.

This requires a binary built with `AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON`.
If the server reports that UI management is unavailable, use the configured API
path below or [rebuild with that option](#choose-a-server-mode). The plain `--ui`
flag does not enable downloads or register models by itself.

### Configured API (first request)

**1. Get a model.** Download the complete
[Kokoro package and its dependencies](../../docs/models/kokoro_tts.md).
This example uses its built-in `af_heart` voice: no reference WAV or `voice_dir`
is needed. It uses CPU inference and works with a CPU-only server build.

**2. Create the config.** Save this as `server.json`, replacing the GGUF path
with your actual file. A config does not download model weights.

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cpu",
  "lazy_load": true,
  "max_loaded_models": 1,
  "models": [
    {
      "id": "kokoro",
      "family": "kokoro_tts",
      "path": "/absolute/path/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
      "task": "tts",
      "mode": "offline",
      "default_voice_preset": { "voice_id": "af_heart" }
    }
  ]
}
```

**3. Start the server.** Keep this terminal open:

```bash
build/debug/bin/audiocpp_server --config server.json --log --log-file server.log
```

**4. Send a request.** In another terminal:

```bash
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8080/v1/models
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Welcome. This speech is generated locally with audio.cpp."}' \
  -o speech.wav
```

Open `http://127.0.0.1:8080` for the configured-model WebUI. `/health` confirms
the server is running, not that a lazy model has loaded. The first generation
loads the model and is slower than later requests. Check curl's exit status and
`server.log` before playing the output; an unsuccessful request may save an
error body instead of a WAV.

The commands use a POSIX shell. On Windows, use `curl.exe` and put JSON in a
request file, for example `--data-binary @request.json`. Use forward slashes
or escaped backslashes in JSON paths, such as `C:/models/model.gguf`.
For other models, start with the [ready-to-edit configs and complete requests](../../examples/server/README.md).
The request's `"model": "kokoro"` matches the config's `id`, not its
`family: "kokoro_tts"` or GGUF filename. To use the GPU later, restart with
`--backend cuda` (or another supported backend); the config otherwise stays the same.

## Build

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DENGINE_ENABLE_CUDA=ON
cmake --build build/debug --parallel 8 --target audiocpp_server
```

Enable the backend you plan to run: `ENGINE_ENABLE_CUDA=ON`,
`ENGINE_ENABLE_VULKAN=ON`, `ENGINE_ENABLE_HIP=ON`, or `ENGINE_ENABLE_METAL=ON`.
CPU support is included by default. Build flags decide which backends exist;
`--backend` selects one when starting the server. See the
[Linux](../../docs/build/linux.md), [Windows](../../docs/build/windows.md),
and [HIP](../../docs/build/HIP.md) build guides for dependencies.

Commands here use `build/debug/bin/`. For a downloaded release, substitute the
path to its `audiocpp_server` executable; it does not need rebuilding.

### Choose a server mode

Pick the mode that matches the behavior you want:

| If you want... | Build with... | Run with... | Behavior |
|---|---|---|---|
| API/config-driven server, no UI | default build | `audiocpp_server --config server.json --no-ui` | Uses models declared in the config; does not serve the UI. |
| UI for configured models | default build | `audiocpp_server --config server.json --ui` | Browser UI can run configured models, without downloads, deletes, or dynamic package management. |
| Full UI with downloads and model switching | `-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` | `audiocpp_server --ui --ui-management --backend <backend>` | UI can browse packages, download models, load/unload models, delete packages, and use temporary uploads. |
| Standalone deployed binary without local `model_specs/` | `-DAUDIOCPP_DEPLOYMENT_BUILD=ON` | `audiocpp_server --config server.json` | Binary carries compiled package specs for fallback model-spec lookup. |

Native model management uses bundled BoringSSL by default. Normal server builds
do not build or link that HTTP/TLS dependency. These build choices can be
combined: deployment mode supplies specs, while native management supplies the
download/install functionality. Neither enables experimental parallel execution.
Optional frontend modules are described [later](#optional-frontend-modules-and-listeners).

For native-manager builds, these TLS alternatives do not change the API:

| Need | Additional CMake setting |
|---|---|
| Use a local BoringSSL archive instead of downloading it during configure | `-DAUDIOCPP_BORINGSSL_ARCHIVE=/path/to/boringssl.tar.gz` |
| Use system OpenSSL instead of bundled BoringSSL | `-DAUDIOCPP_USE_SYSTEM_OPENSSL=ON` |

Both require `-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON`. They do not enable an
HTTPS listener; that is a separate frontend extension.

## Config

For ready-to-edit configs with explained paths, voice-library setup, and complete
requests, see [server examples](../../examples/server/README.md). They cover
Qwen3-TTS, BreezeTTS, Kokoro, PocketTTS, YuE2, MiniMax Music3, and Parakeet ASR.

The following example registers TTS and ASR in one process. Replace both paths;
keep PocketTTS's downloaded `embeddings/` directory beside its GGUF. Use only
entries for packages you installed. The defaults for optional server settings
are listed in [loading and memory](#loading-and-memory),
[logging](#logging-and-request-limits), and [busy timeouts](#busy-requests-and-timeouts).

```bash
cat > server.json <<'JSON'
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 4,
  "lazy_load": true,
  "max_loaded_models": 1,
  "log_request_body": false,
  "max_request_body_bytes": 2147483648,
  "models": [
    {
      "id": "pocket-tts",
      "family": "pocket_tts",
      "path": "/absolute/path/PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf",
      "task": "tts",
      "mode": "offline",
      "load_options": {
        "language": "english"
      },
      "default_voice_preset": {
        "voice_id": "alba"
      },
      "voice_presets": {
        "cosette": {
          "voice_id": "cosette"
        }
      }
    },
    {
      "id": "qwen3-asr",
      "family": "qwen3_asr",
      "path": "/absolute/path/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-f16.gguf",
      "task": "asr",
      "mode": "offline"
    }
  ]
}
JSON
```

### Model entries

| Field | Meaning |
|---|---|
| `id` | Your server-side name. Send this exact string as the request's `model`, not the family or GGUF filename. |
| `family` | The implementation to load, such as `pocket_tts` or `qwen3_asr`. Use the model's documented family. |
| `path` | An existing GGUF file or the complete package directory required by that family. Multi-component models may need additional package files. |
| `task` | The model's documented route, such as `tts`, `clon`, `vdes`, `asr`, or `align`. Capabilities such as voice cloning do not imply an identically named route. |
| `mode` | `offline` or `streaming`, supported by that model. It selects the session created by the server. |
| `load_options` | Options consumed while loading weights/assets. |
| `session_options` | Options consumed when creating the reusable session. |
| `default_request_options` | Defaults for the model's inference requests; request values with the same key override them. |

Always set `family`, `task`, and `mode` explicitly rather than relying on inferred
names or defaults. Several entries can use the same family with different GGUFs,
tasks, or modes, provided each entry has a unique `id`.

### Default session options

Top-level `"default_session_options"` sets session options for every model of a
family, keyed `<family>.<option>`. A model gets the entries of its family that its
own `session_options` do not set. This applies to the configured models and to
models loaded later through `POST /v1/models/load`, which is how the WebUI loads
a model with `--ui-management`. The WebUI sends the catalog's session options,
not the config's, so without defaults such a load drops settings like memory
options:

```json
{
  "ui_management": true,
  "default_session_options": {
    "irodori_tts.mem_saver": true,
    "irodori_tts.codec_decode_chunk_steps": 100
  },
  "models": []
}
```

A key without a family prefix is rejected when the config loads. A value for a
family that never loads is ignored.

### Paths and model specs

Relative model and configured voice-library paths resolve against the config
file's directory. Use absolute paths in your first config. Request-time audio
paths refer to files accessible on the **server's computer**, not the client's;
upload audio when the server cannot access the original file. Option-map path
handling is model-specific; do not assume every option is config-relative.

Package/resource loading prefers the spec embedded in a GGUF. For option validation,
capabilities, and help, contracts instead prefer `model_specs/<family>.json` in the
working directory or its parents, then the compiled catalog
(`AUDIOCPP_DEPLOYMENT_BUILD=ON`), then the embedded spec. External specs must be
compatible with the installed runtime. `model_spec_override` takes priority over
both lookup orders. It accepts either one JSON file or a directory containing `<family>.json`.
Set it at the top level to provide a server-wide override, or inside one model entry;
the per-model value takes precedence. The equivalent command-line option is
`--model-spec-override <json-or-directory|embedded>`. Set
`"model_spec_override": "embedded"` to require the selected GGUF's schema-v1
embedded contract, ignoring external and compiled contracts. Missing or legacy
embedded contracts fail without fallback. With `--log`, `runtime.model.contract_spec`
shows the selected contract. Use `./embedded` for a literal file/directory of that name.

### Backends

Set top-level `"backend"` to `"cuda"`, `"cpu"`, `"hip"` (`"rocm"` is an alias),
`"vulkan"`, or `"metal"`. The backend must be included in the binary and supported
by the selected model. CUDA is the primary optimization target; performance and
model coverage differ on other backends. `--backend` overrides the config for
the process, not for an individual HTTP request. Use separate server processes
on different ports if clients need different backends simultaneously.

### CLI options to server configuration

Inference CLI flags are not arbitrary server-config fields. Put each model
option in its documented scope; request options go in `options` on the speech
endpoint, or `request.options` on generic task endpoints.

| Inference CLI | Server config | Per-request override |
|---|---|---|
| `--family qwen3_tts` | `models[].family` | Select the entry's `id` with `model`; do not send `family`. |
| `--model /path/model.gguf` | `models[].path` | Choose a different configured `model` id. |
| `--task tts --mode offline` | `models[].task` and `models[].mode` | Choose the matching entry; a request does not change its session mode. |
| `--load-option key=value` | `models[].load_options` | Not an inference-request option. |
| `--session-option family.key=value` | `models[].session_options` | Not an inference-request option. |
| `--session-option family.key=value` for every model of that family | top-level [`default_session_options`](#default-session-options) | Not an inference-request option. |
| `--request-option key=value` | `models[].default_request_options` | Speech `options`, or generic `request.options`. |
| `--seed 1234`, `--max-tokens 512` | `default_request_options`: `seed`, `max_tokens` | Speech also accepts these as top-level fields. |
| `--instruct "..."` | `default_request_options`: use the model's documented option, usually `instruct` | `options.instruct` for a model accepting `instruct`; see the speech endpoint for the distinct `instructions` field. |
| `--voice-ref` and `--reference-text` | `default_voice_preset` / `voice_presets` | Speech `voice_ref` and `reference_text`; see [voices](#voices). |

Keep family prefixes on session/load option names where the model's docs require
them. The server does not automatically prefix arbitrary keys or translate every
CLI flag. A key accepted by one family may be rejected or unused by another.
Use [model-specific docs](../../docs/usage.md#model-docs) and
[`GET /v1/models?include_params=true`](api.md#get-v1models) for declared request
controls. That endpoint does not list load/session options.

For example, this is the options portion of a BreezeTTS model entry, not a full
server config:

```json
{
  "session_options": {
    "breeze_tts.reference_cache_slots": 1
  },
  "default_request_options": {
    "seed": 1234,
    "instruct": "A calm narrator speaking clearly.",
    "text_chunk_size": 600
  }
}
```

Send `"options": {"instruct": "A cheerful narrator."}` on a speech request to
override the matching default. Use the same option spelling in defaults and
requests: the server merges by key, not by model-specific alias equivalence.
Load/session options are not changed by that request.

### Loading and memory

These are server-level settings. The examples enable lazy loading and allow one
resident model; the defaults below apply when keys are omitted from a config
file. Starting the full UI without a config enables lazy loading automatically.

| Config key | Default | What it controls | Startup override |
|---|---:|---|---|
| `lazy_load` | `false` | Defer model loading/session creation until the first request. Per-model `lazy` overrides it. | Set in config. |
| `max_loaded_models` | `0` | Maximum resident models; `0` disables the limit, `1` keeps one model resident. | `--max-loaded-models <n>` |
| `idle_unload_ms` | `0` | Unload resident models after the server has had no model load/run activity for this long; `0` disables it. | `--idle-unload-ms <ms>` |
| `min_free_memory_mb` | `0` | Free-memory headroom in MiB required after an estimated model load; `0` disables the guard. | `--min-free-memory-mb <mb>` |

> [!WARNING]
> Lazy loading does not unload a model after each request. Used models stay
> resident until explicitly unloaded, evicted by a residency limit, unloaded
> after server inactivity, or the process exits.

When the residency limit is reached, loading another model first evicts the
least recently used **idle** model. Its next request reloads it. Models running
inference are not evicted; if every resident model is busy, the new load fails
with HTTP 503. If more non-lazy entries are configured than the limit permits,
startup loads the first entries up to the limit and defers the remainder.

Idle unloading frees memory during quiet periods, not while inference is active.
It complements the resident-model limit; it is not a per-request teardown mode.
The next request reloads the model.

The optional free-memory guard checks host memory and GPU memory where available.
Its load estimate covers the selected weight file/package plus session auxiliary
files, with a runtime overhead factor and fixed floor. A directory normally
selects `model.gguf` or its sole GGUF; several GGUFs without `model.gguf` are
ambiguous to the estimator. Unsupported family-specific layouts may skip the
guard with a log message, while an invalid spec-driven package can still fail
in the loader. Safetensors/HF layouts use the full checkpoint tree. A failed
guard returns HTTP 503 with `insufficient_memory`. It does not predict the exact
peak workspace of every inference.

> [!NOTE]
> `max_loaded_models: 1` limits how many models are resident, not the peak memory
> needed by one model. `lazy_load` changes when loading happens; it does not make
> a model smaller. The first request after an unload pays the load cost again.

### Logging and request limits

Start with `--log --log-file server.log` when diagnosing failures. Request-body
logging is separate and off by default to avoid accidentally recording prompts,
paths, and model options.

| Setting | Default | Meaning |
|---|---:|---|
| Config `log_request_body` | `false` | Log JSON request bodies only when logging is also enabled with `--log` or `--log-file`. Audio payload bytes are not printed. |
| Config `max_request_body_bytes` | `2147483648` | Maximum buffered HTTP body in bytes (2 GiB), before routing. It covers JSON and audio uploads, including base64 expansion. |
| Flag `--load-progress 0\|1` | `0` | With logging enabled, emit model-load progress keys. This is not generation progress and can be inaccurate for multi-component packages. |

Multipart logging includes filenames and byte counts; raw/live audio logging
includes only route, format/query, and size/stream metadata. Choose a body limit
that fits your expected uploads and host RAM. Values above `2^53 - 1` are rejected
because the config JSON parser stores numbers as doubles. Live ingestion has
its own [incremental body limits](api.md#post-v1audiotranscriptionslive).

### Busy requests and timeouts

The default runtime runs one request per model at a time. Top-level
`busy_timeout_ms` limits waiting for a busy model; it does **not** cancel an
inference already running on the backend. Requests that cannot acquire the
model within the policy return HTTP 503 (`server_busy`), or an error event if
streaming headers have already been sent. The default is `300000` (5 minutes);
`0` disables the guard. Set it above the slowest legitimate inference for that
model. `--busy-timeout-ms <ms>` overrides the server-level config value.

The bound is resolved in three layers, since model runtimes differ by orders of magnitude (a short TTS clip versus minutes of music generation):

1. **Server:** top-level `"busy_timeout_ms"` (or `--busy-timeout-ms`) sets the default.
2. **Model:** `"busy_timeout_ms"` inside a model entry overrides the server default and is the ceiling for requests to that model.
3. **Request:** `"busy_timeout_ms"` in the request body, multipart form, or live-ingest query bounds that caller's wait.

A request may ask for a **shorter** bound than the model's ceiling but never a longer one — `effective = min(request, ceiling)` — so a client cannot weaken the guard and reintroduce the hang it prevents. Because `0` means "unbounded", it compares as infinity on both sides: a request asking for `0` is still capped by the model ceiling, while under a ceiling of `0` a request's own bound is honored.

```json
{
  "busy_timeout_ms": 300000,
  "models": [
    { "id": "tts",   "busy_timeout_ms": 30000,  "...": "..." },
    { "id": "music", "busy_timeout_ms": 900000, "...": "..." },
    { "id": "asr", "...": "..." }
  ]
}
```

Here `asr` inherits 300000, a request to `music` asking for `60000` waits 60 s, and one asking for `999999` is clamped to 900000.

### Offline and streaming sessions

For streaming endpoints, configure the model with `"mode": "streaming"` and
use that entry's id in the request. Setting `stream=true` on an offline entry
does not create a streaming session. Register separate ids if you need both
modes of one model. Each entry creates its own session and can consume memory.
See [streaming_example.json](streaming_example.json):

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 1,
  "lazy_load": true,
  "models": [
    {
      "id": "voxcpm2-stream",
      "family": "voxcpm2",
      "path": "/path/to/models/VoxCPM2",
      "task": "tts",
      "mode": "streaming"
    },
    {
      "id": "nemotron-stream",
      "family": "nemotron_asr",
      "path": "/path/to/models/nemotron-3.5-asr-streaming-0.6b",
      "task": "asr",
      "mode": "streaming"
    }
  ]
}
```

## Start

```bash
build/debug/bin/audiocpp_server --config server.json --log --log-file server.log
```

Command-line server settings override their configured values:

```bash
build/debug/bin/audiocpp_server --config server.json --backend vulkan
build/debug/bin/audiocpp_server --config server.json --voice-dir /absolute/path/to/voice
```

Use `--no-ui` for API-only operation. Full model-management UI requires the
[native-manager build](#choose-a-server-mode) and `--ui --ui-management`.
Run `audiocpp_server --help` for server flags; inference controls belong in the
model option maps, not as arbitrary server command-line flags.

Keep `host` at `127.0.0.1` for local access. `0.0.0.0` binds all interfaces;
clients connect using the server's actual address, not `0.0.0.0`. Do not expose
model-management and server-file access to an untrusted network.

## Voices

### Configured presets and defaults

For TTS models that need repeated voice-clone context, set a model-level `default_voice_preset` so OpenAI-compatible clients can omit `voice_ref` and `reference_text` on each request:

```json
{
  "id": "omnivoice",
  "family": "omnivoice",
  "path": "/absolute/path/to/models/OmniVoice",
  "task": "tts",
  "mode": "offline",
  "default_voice_preset": {
    "voice_ref": "/absolute/path/to/reference.wav",
    "reference_text": "Reference transcript for the reference audio."
  }
}
```

For multiple server-side presets, use `voice_presets` and optionally point `default_voice_preset` at one of those preset names:

```json
{
  "voice_presets": {
    "assistant": {
      "voice_ref": "/absolute/path/to/assistant.wav",
      "reference_text": "Reference transcript for assistant."
    },
    "narrator": {
      "voice_ref": "/absolute/path/to/narrator.wav",
      "reference_text": "Reference transcript for narrator."
    }
  },
  "default_voice_preset": "assistant"
}
```

Define `voice_presets` once and put every named preset inside that object. Duplicate JSON keys are rejected so a config cannot silently drop presets.

When a request sends `"voice": "assistant"`, the server uses that entry's
configured preset. If the name is not a preset, the server checks the shared
reference library before passing it through as a model-native voice ID.
For built-in speakers, configure `voice_id` using an ID valid for that model,
such as `af_heart` for Kokoro; preset names and native voice IDs are not universal.

A preset can also carry request `options` for the model. Explicit request options
override preset options with the same key. Different aliases are not merged;
supplying conflicting aliases may produce a model validation error. The model
validates preset options like any other request option. For example, an
Irodori-TTS speaker embedding selected by name:

```json
{
  "voice_presets": {
    "narrator": {
      "options": {
        "speaker_embedding_file": "/absolute/path/to/narrator.speaker.safetensors"
      }
    }
  }
}
```

A request `{"input": "Hello.", "voice": "narrator"}` then uses that embedding.
Option values are not paths to the server, so a relative path is not resolved
against the config file the way `voice_ref` is; use absolute paths.

### Voice library (`voice_dir`)

Set server-level `"voice_dir"` to a folder of your reference WAV recordings and
a `prompt_text` mapping file. A request `"voice": "demo_01_man"` then selects
`voice_dir/demo_01_man.wav` for cloning, without sending `voice_ref` and
`reference_text` each time. This is a shared server-side reference library,
not a list of speaker presets embedded in model weights:

```json
{
  "voice_dir": "/absolute/path/to/voice",
  "models": [
    {
      "id": "qwen3-tts",
      "family": "qwen3_tts",
      "path": "/absolute/path/to/models/Qwen3-TTS",
      "task": "tts",
      "mode": "offline"
    }
  ]
}
```

The directory holds one `.wav` per reference voice (`demo_01_man.wav`,
`demo_02_woman.wav`, ...) and a file named exactly `prompt_text` (no `.txt`
extension), with one `<basename-without-extension>|<transcript>` line per voice:

```
demo_01_man|okay,I'm Cemo and what you just heard wasn't a human voice.
demo_02_woman|以前我对这句话一知半解，现在好像有点懂了。因为你我开始留意很多以前不曾关心的事，开始对这个世界有了更多的好奇和善意。
```

Relative `voice_dir` paths resolve against the config file's directory. The command-line option `--voice-dir <directory>` overrides the configured value, which is useful when a process manager launches multiple single-model server configurations against one shared voice library. Relative command-line paths resolve against the process working directory.

When a request sends `"voice"` that is not a configured model preset, the server checks `<voice_dir>/<name>.wav`; if the file exists it is loaded as the cloning reference, and the `<name>` transcript from `prompt_text` is injected as `reference_text` unless the request already provides one.

Server-side resolution of reference audio and voice names:

1. Explicit `voice_ref` replaces any preset reference audio and bypasses WAV-name lookup.
2. `voice` matching a configured model preset — preset wins.
3. `voice` matching a wav basename in `voice_dir` — voice-library clone.
4. Otherwise, `voice` is passed as the model-native cached voice ID.

If neither `voice` nor `voice_ref` is supplied, the configured default preset
applies. Prefer one selection method per request: combining `voice` and
`voice_ref` can pass both a native ID and audio to the model, whose interpretation
is model-specific. Top-level `reference_text` overrides the preset/library text.

[`GET /v1/audio/voices`](api.md#get-v1audiovoicesmodelid) lists library WAV
basenames, deduplicated against configured preset and cached voice names.

With native UI management enabled and no configured `voice_dir`, the server
provides embedded demo recordings in a temporary reference library. Those names
are reference recordings, not the selected model's built-in speaker IDs.

Use distinct library filenames and model preset names. The library is shared
across configured models; naming a reference WAV after a model-native speaker
can cause that name to select the recording instead. Only models that support
reference-audio conditioning can use these recordings for cloning.

### Built-in voices and discovery

Some models, including Kokoro and KittenTTS, store built-in voices in the GGUF.
`/v1/audio/voices` does not enumerate those embedded presets. Select them using
the IDs documented for that model, or configure named `voice_presets` pointing
to those IDs. An empty discovery list does not mean the built-in voices are
unavailable. `voice_dir` is not required to use them.

The WebUI's browser-saved voice library is separate from this server directory;
adding a recording in the browser does not itself create a named WAV in
`voice_dir`.

## Endpoints

Choose the endpoint by the result and transport you need. Not every model
supports every task, mode, or optional output.

| Need | Endpoint | Input / response |
|---|---|---|
| Check readiness / configured IDs | <a id="get-health"></a>[`GET /health`](api.md#get-health), <a id="get-v1models"></a>[`GET /v1/models`](api.md#get-v1models) | JSON metadata; a registered model may still be unloaded. |
| Generate speech | <a id="post-v1audiospeech"></a>[`POST /v1/audio/speech`](api.md#post-v1audiospeech) | JSON text and voice controls; WAV by default, or streaming audio/SSE when supported. |
| Transcribe a recording | <a id="post-v1audiotranscriptions"></a>[`POST /v1/audio/transcriptions`](api.md#post-v1audiotranscriptions) | Server-local WAV path or file upload; text/timing, optionally transcript SSE. |
| Get timestamps or speaker labels | <a id="post-v1audiotranscriptionsdetails"></a>[`POST /v1/audio/transcriptions/details`](api.md#post-v1audiotranscriptionsdetails) | Same inputs; includes detail arrays the model actually produces. |
| Align a known transcript | <a id="post-v1audioalignments"></a>[`POST /v1/audio/alignments`](api.md#post-v1audioalignments) | Multipart WAV, text and model-required language; word alignment. |
| Transcribe live capture | <a id="post-v1audiotranscriptionslive"></a>[`POST /v1/audio/transcriptions/live`](api.md#post-v1audiotranscriptionslive) | Incremental raw PCM input and SSE output; native full-duplex HTTP client required. |
| List discoverable voice names | <a id="get-v1audiovoicesmodelid"></a>[`GET /v1/audio/voices`](api.md#get-v1audiovoicesmodelid) | Configured presets, cached IDs and reference-library names, not every embedded model preset. |
| Run another audio task | <a id="post-v1tasksrun"></a>[`POST /v1/tasks/run`](api.md#post-v1tasksrun) | Generic framework request/result, including separation and enhancement. |
| Stream a generic task | <a id="post-v1tasksstream"></a>[`POST /v1/tasks/stream`](api.md#post-v1tasksstream) | Generic events; use `stream_format: "sse"` for incremental delivery. |
| Native offline batching | <a id="post-v1batchestranscriptions"></a>[`POST /v1/batches/transcriptions`](api.md#post-v1batchestranscriptions), <a id="post-v1tasksbatch"></a>[`POST /v1/tasks/batch`](api.md#post-v1tasksbatch) | Multiple inputs, only for models implementing native batching. |
| Release model memory | <a id="post-v1tasksunload_models"></a>[`POST /v1/tasks/unload_models`](api.md#post-v1tasksunload_models), <a id="post-v1tasksunload_all_models"></a>[`POST /v1/tasks/unload_all_models`](api.md#post-v1tasksunload_all_models) | Unload; the next request reloads the model. |

For remote clients, upload the recording rather than sending the client's file
path. The core audio routes accept WAV; MP3/FLAC support requires the optional
[audio decoding frontend](#optional-frontend-modules-and-listeners). Multipart
fields, query parameters, and JSON fields are not interchangeable between routes.

## Troubleshooting

| Symptom | Check first |
|---|---|
| `/health` succeeds but generation fails | With lazy loading, readiness does not load weights. Check the first request's error and `server.log`, the package path, dependencies, and backend support. |
| Unknown model or wrong output task | Request `model` must match `models[].id`; check that entry's `task` and `mode` and select the matching endpoint. |
| Streaming requested but no incremental output | Use a streaming entry and the endpoint's streaming fields. Generic `/v1/tasks/stream` returns buffered JSON unless `stream_format` is `sse`. File-upload transcription cannot emit while the file is being uploaded. |
| Voice list is empty or shows unexpected names | Embedded GGUF speakers are not enumerated. `voice_dir` is shared; use configured presets or the model's documented native IDs and avoid filename collisions. |
| Unknown option or a default is not overridden | Check the exact model option name and its load/session/request scope. Defaults merge by key. Check the binary and GGUF spec versions rather than assuming every CLI flag is a top-level HTTP/config field. |
| HTTP 503 | Read the error: `server_busy` is a wait/admission issue; `insufficient_memory` is the optional load guard. A busy timeout does not abort the request already running. |
| Out of VRAM after switching models | Use `max_loaded_models: 1` or unload idle models. One model's inference still must fit by itself; a weight-size estimate is not a guarantee about graph workspace. |
| Client audio is not found or a file is rejected | A JSON path is server-local. Use multipart WAV upload for remote transcription; MP3/FLAC need the audio decoding frontend. A browser's WebM recording is not a WAV. |
| Browser reports CORS | Inspect the HTTP status/body as well as headers. Cross-origin access is disabled by default; see [experimental CORS](#experimental-cors). The same-origin embedded UI does not require it. |

### Timing and streaming clients

`rtf` is inference wall time divided by audio duration, not elapsed seconds or
time to first output. Speech uses generated audio duration; transcription uses
input recording duration. For offline requests, model loading and waiting for the
model are outside the timed inference; the first HTTP request can take longer
than the returned `wall_ms`. Use a warm request when comparing inference speed.
Streaming `ttft_ms` and live speech timings have endpoint-specific definitions
documented in their sections; they are not whole-request latency measurements.

An HTTP 200 status at the start of SSE does not prove the run succeeded. Read
through the endpoint's completion event and handle error events. Disable proxy
buffering for incremental responses; live ingestion additionally needs
full-duplex request transport. See [live deployment](api.md#post-v1audiotranscriptionslive)
for nginx and client constraints.

## Optional frontend modules and listeners

These are build-time extensions from the external
[server-frontends repository](https://github.com/0xShug0/audio.cpp-server-frontends),
not model options or part of the default server dependency set.

| Need | Build settings | Runtime settings |
|---|---|---|
| Audio decoding / MP3 encoding | `-DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON -DAUDIOCPP_SERVER_FRONTENDS_DIR=external/audio.cpp-server-frontends -DAUDIOCPP_SERVER_FRONTEND_MODULES="audio_decode;mp3_encode"` | Use the extended audio formats on supported routes. |
| Frontend-owned transport, such as HTTPS or WebSocket | Same frontend build flags, with `-DAUDIOCPP_SERVER_FRONTEND_MODULES=<listener>` | `--frontend-listener <listener>` and repeated `--frontend-option key=value` |

Optional frontend modules are selected at configure time with the semicolon-separated
`AUDIOCPP_SERVER_FRONTEND_MODULES` list and an external
`AUDIOCPP_SERVER_FRONTENDS_DIR` package. If you use the bundled submodule path,
fetch it before configuring:

```bash
git submodule update --init external/audio.cpp-server-frontends
```

For a fresh clone, `git clone --recurse-submodules` also fetches it.

The server runs selected modules as an ordered pipeline: every module gets a
pre-processing pass before the core handler, then every module gets a
post-processing pass after the core handler. A module that does not need one side
leaves that method empty. Each active side declares a simple contract over the
HTTP envelope state (`method`, `path`, `request_in/request_out` for
pre-processing, or `response_in/response_out` for post-processing), and module
registration rejects incompatible adjacent transforms on the same route.
The external package owns modules and private dependencies such as miniaudio
and libmp3lame; the default server includes none of these modules or dependencies.

Listener frontends are selected through the same external package but are not
part of the pre/post pipeline. The server core only knows a listener name plus
string options; the external package owns listener implementations, docs, and
dependency detection.

Set top-level `"frontend_listener"` to select a compiled listener and
`"frontend_options"` to an object of listener-specific **string** values. The
equivalent command-line options are `--frontend-listener <name>` and repeated
`--frontend-option key=value`. Check the frontend repository for listener names
and required settings; adding config alone does not compile a missing module.

## Experimental features

Neither feature below is required for a multi-model config, the normal API, or
the same-origin embedded UI. Keep the default runtime unless you specifically
need to evaluate the experimental one.

### Experimental CORS

CORS is disabled by default. For trusted local browser demos on a different
origin from the server, enable it explicitly:

```bash
build/debug/bin/audiocpp_server --config server.json --cors-origins "*"
```

or add this server-level field to `server.json`:

```json
{
  "cors_origins": "*"
}
```

> [!WARNING]
> CORS is not authentication. This experimental support is intended for trusted
> local web apps only. With `"*"`, any browser origin can make cross-origin API
> requests and consume local CPU/GPU resources. Do not expose the server on an
> untrusted network.

### Experimental parallel model slots

The single authoritative
[Parallel Server Behavior Contract](../../docs/maintainers/parallel_server_behavior_contract.md)
defines operation/state matrices, linked tests, validation by area and known
limitations. Enabling the feature prints a warning with the tested and untested
scope. Controlled session tests do not admit real catalogue models to multi-slot
execution; GPU streaming, full lifecycle coverage and hard backend cancellation
remain unverified. The descriptions below are a usage summary.

The existing implementation remains the default. Opt into the separate parallel
runtime at process startup:

```bash
audiocpp_server --config server.json --parallel-jobs
```

Without `--parallel-jobs`, **any explicit `slots` field** in a model entry or
dynamic registration is rejected, including `"slots": 1` and `"slots": null`.
Existing configurations without the field use the original `ServerState`.

With `--parallel-jobs`, every model uses `ParallelServerState`. Omitted slots
means capacity one; otherwise `"slots"` must be an integer from 1 to 16. Capacity
one also uses the new scheduler and session pool. Slot count never selects the
runtime. The requested count must fit the model's advertised parallel capacity;
the common framework retains capacity one unless the model explicitly supports
more. Increasing `slots` does not automatically enable concurrency for a model.
The default HTTP listener in this mode owns its request workers, interrupts
socket I/O on shutdown, and joins workers before destroying the runtime. Active
GPU work must still finish. Custom frontend listeners own their shutdown/drain
behavior through the existing frontend interface.

`GET /v1/models` on the parallel runtime reports configured `slots`, `active_slots`,
`queued_requests` and loaded `max_parallel_slots` (`null` when unloaded). Legacy
model responses retain their schema and do not gain synthetic slots fields.
Parallel requests and management share FIFO admission; unload, eviction and
reconfiguration drain active leases. Counts can be resized across one under an
exclusive lease, but registration never changes the process's selected runtime.
The inference CLI has no new flag and retains its existing behavior.

Each slot needs private execution contexts, graphs, KV/reference caches,
stream state and sampling state. Immutable assets/weights can be shared.
More slots can improve throughput but increase latency and working memory.
This is concurrent session execution; continuous token batching is not included.
See the [adapter guide](../../docs/maintainers/parallel_sessions.md).

The [validation procedure](../../docs/maintainers/parallel_model_validation.md)
keeps model quality/recovery/memory checks separate from shared framework
ownership tests and performance measurements. Shared evidence is referenced once
per applicable revision/backend; every admitted count still needs repeated
quality testing. Optional separate-server benchmarks do not determine admission.

The [busy-timeout policy](#busy-requests-and-timeouts) also applies here. Requests
queue when every slot is occupied or management blocks admission. When every
occupied slot is overdue, new requests fail fast with HTTP 503, or an error event
if streaming has started. Waiting unload/reconfiguration keeps priority but does
not suppress that check, even when the pool has spare slots. A healthy occupied
slot still permits bounded waiting; timeout `0` disables both checks.

All models (including capacity one) use FIFO admission. Earlier queued requests
drain before a later unload/reconfiguration, and later arrivals cannot overtake
a reserved waiter. A request binds model configuration before preparation and
retains its lease through any deferred stream/native batch response. Bulk unload
releases ready idle pools before waiting on busy ones. Logical residency ends
when teardown starts; unload completes when resources have been destroyed, and
physical-memory guards still apply. Registry lookup/snapshot locks never span
model admission or metadata waits. See the
[behavior contract](../../docs/maintainers/parallel_server_behavior_contract.md).
