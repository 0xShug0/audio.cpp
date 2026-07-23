# audio.cpp Server

`audiocpp_server` is an HTTP adapter over the framework runtime registry. It keeps one loaded model and one offline task session per active model id, so repeated HTTP requests reuse the same framework session and model-owned graph/cache state.

## Build

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON
cmake --build build --parallel --target audiocpp_server
```

Enable the backend you plan to run: `ENGINE_ENABLE_CUDA=ON` for CUDA, `ENGINE_ENABLE_VULKAN=ON` for Vulkan, or `ENGINE_ENABLE_METAL=ON` for Metal. CPU support is always available.

## Config

```bash
cat > server.json <<'JSON'
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 1,
  "lazy_load": true,
  "models": [
    {
      "id": "pocket-tts",
      "family": "pocket_tts",
      "path": "/path/to/models/pocket-tts",
      "task": "tts",
      "mode": "offline",
      "load_options": {
        "language": "english"
      },
      "session_options": {
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
      "path": "/path/to/models/Qwen3-ASR-0.6B",
      "model_spec_override": "/optional/path/to/qwen3_asr.json",
      "task": "asr",
      "mode": "offline"
    }
  ]
}
JSON
```

The server resolves model paths from this JSON exactly as written, so use paths that match your machine. Request-time audio paths are also user-provided paths.

Package specs embedded in a GGUF are used automatically. Builds configured with
`AUDIOCPP_DEPLOYMENT_BUILD=ON` also carry a compiled fallback catalog; normal builds
discover `model_specs/<family>.json` on disk. `model_spec_override` explicitly replaces
that resolution order. It accepts either one JSON file or a directory containing `<family>.json`.
Set it at the top level to provide a server-wide override, or inside one model entry;
the per-model value takes precedence. The equivalent command-line option is
`--model-spec-override <json-or-directory>`.

Set top-level `"backend"` to `"cuda"`, `"cpu"`, `"vulkan"`, or `"metal"`. CUDA is the optimized path for audio.cpp; CPU, Vulkan, and Metal are intended for portability and testing when the binary is built with that backend, but performance and model coverage may be lower. The server prints this expectation-setting message when a non-CUDA backend is selected.

Set top-level `"lazy_load": true` to register all configured model ids at startup but defer each model's framework load and session creation until its first request. A model can override the default with `"lazy": true` or `"lazy": false`.

> [!WARNING]
> Lazy loading does not unload models after a request. Once a model is first used, the server keeps that model and session in memory for reuse until the server exits.

### Experimental CORS

CORS is disabled by default. For trusted local browser demos, enable it explicitly:

```bash
build/bin/audiocpp_server --config server.json --cors-origins "*"
```

or in `server.json`:

```json
{
  "cors_origins": "*"
}
```

> [!WARNING]
> CORS support is experimental and intended for trusted local web apps only. Do not expose a server with CORS enabled on an untrusted network. With CORS enabled, any browser page can send requests to the local server and consume local CPU/GPU resources.

Set top-level `"busy_timeout_ms"` to bound how long a request waits for a model that is already running. Each model serializes its requests on an internal lock, so a second request normally queues behind the first. A GPU call that wedges cannot be cancelled from userspace, so without a bound every subsequent request would park a worker thread forever. When the current inference has held the lock past this timeout, a new request fails fast with HTTP 503 (`server_busy`) instead of queuing; streaming requests that have already sent headers surface the same condition as a `{"type":"error"}` stream event. The value must exceed the slowest legitimate single inference (music generation can take minutes). Defaults to `300000` (5 minutes); set `0` to disable the guard and restore unbounded waiting. The `--busy-timeout-ms <ms>` command-line flag overrides the config value.

The bound is resolved in three layers, since model runtimes differ by orders of magnitude (a short TTS clip versus minutes of music generation):

1. **Server** — top-level `"busy_timeout_ms"` (or `--busy-timeout-ms`) sets the fleet default.
2. **Model** — `"busy_timeout_ms"` on an entry in `"models"` overrides that default for one model, and becomes the ceiling for requests to it.
3. **Request** — `"busy_timeout_ms"` in the request body (or as a `busy_timeout_ms` form field on multipart transcription) lets a caller bound its own wait.

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

For streaming endpoints, configure the model with `"mode": "streaming"` and use that model id in the request. A complete example is available at `app/server/streaming_example.json`:

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
      "voice_id": "alba"
    }
  },
  "default_voice_preset": "assistant"
}
```

When a request sends `"voice": "assistant"`, the server uses that configured preset. When `"voice"` does not match a configured preset, it is passed through as the model-native cached voice id, preserving the previous behavior.

## Start

```bash
build/bin/audiocpp_server --config server.json
```

You can override the configured backend at startup:

```bash
build/bin/audiocpp_server --config server.json --backend vulkan
```

## Endpoints

### `GET /health`

Returns server readiness and the number of configured models.

### `GET /v1/models`

Returns OpenAI-style model entries for the configured audio.cpp model ids.

### `POST /v1/audio/speech`

OpenAI-style text-to-audio. The response is `audio/wav` by default.

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{
    "model": "pocket-tts",
    "input": "audio.cpp is serving this request through the framework runtime.",
    "max_tokens": 96,
    "seed": 1234
  }'
```

If no request voice is provided and the configured model has `default_voice_preset`, the server injects that preset automatically. Request-level `voice`, `voice_ref`, and `reference_text` override the configured default.

Set `"response_format": "json"` to receive base64 WAV in a JSON response.

For streaming-capable TTS models configured with `mode: "streaming"`, `stream_format` follows the OpenAI speech streaming shape:

```bash
curl -N http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "model": "voxcpm2-stream",
    "input": "Stream this sentence as audio events.",
    "response_format": "pcm",
    "stream_format": "sse",
    "options": {
      "retry_badcase": false
    }
  }'
```

The SSE stream emits `speech.audio.delta` events with base64 PCM chunks, then `speech.audio.done`, then `data: [DONE]`. VoxCPM2 streaming requires `retry_badcase=false` because retrying a completed bad case is an offline-only behavior. Set `"stream_format": "audio"` with `"response_format": "pcm"` to receive raw PCM bytes over chunked transfer encoding instead.

### `POST /v1/audio/transcriptions`

JSON transcription request using a server-local audio path.

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-asr",
    "audio": "/path/to/input.wav"
  }'
```

Also accepts a `multipart/form-data` upload, matching the OpenAI Whisper API convention used by real clients (e.g. Open WebUI). The request is routed to the multipart path based on the `Content-Type` header; the JSON path above still works unchanged.

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=qwen3-asr \
  -F language=en \
  -F file=@/path/to/input.wav
```

`file` and `model` are required; `language` is optional. Uploaded WAV bytes are decoded in memory and are not written to a temporary file.

For streaming-capable ASR models configured with `mode: "streaming"`, pass `stream=true` to receive OpenAI-style transcription SSE:

```bash
curl -N http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=nemotron-stream \
  -F language=en-US \
  -F stream=true \
  -F file=@/path/to/input.wav
```

The stream emits `transcript.text.delta` events, one final `transcript.text.done` event containing the full transcript, then `data: [DONE]`.

### `GET /v1/audio/voices?model=<id>`

Lists the cached voice ids and configured server voice preset names available for a TTS model, so a client can populate a voice picker instead of guessing generic names. For families that keep voice presets under `model_root/embeddings/*.safetensors` (`pocket_tts` today), this returns those ids too. If `model` is omitted and the server has exactly one configured model, that model is used; if multiple models are configured, omit `model` only when an empty list is acceptable.

```bash
curl 'http://127.0.0.1:8080/v1/audio/voices?model=pocket-tts'
```

```json
{"voices": ["alba", "cosette", "marius"]}
```

### `POST /v1/tasks/run`

Generic framework request route. The `request` object uses the same JSON fields as the `audiocpp_cli` request sequence format.

```bash
curl http://127.0.0.1:8080/v1/tasks/run \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "pocket-tts",
    "request": {
      "text": "Generic audio.cpp request.",
      "voice_ref": "/path/to/reference.wav",
      "max_tokens": 96,
      "seed": 1234
    }
  }'
```
