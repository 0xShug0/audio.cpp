# audio.cpp Server

`audiocpp_server` is an HTTP adapter over the framework runtime registry. It keeps one loaded model and one offline task session per active model id, so repeated HTTP requests reuse the same framework session and model-owned graph/cache state.

## Build

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON
cmake --build build --parallel --target audiocpp_server
```

Enable the backend you plan to run: `ENGINE_ENABLE_CUDA=ON` for CUDA, `ENGINE_ENABLE_VULKAN=ON` for Vulkan, or `ENGINE_ENABLE_METAL=ON` for Metal. CPU support is always available.

### Choose a server mode

Pick the mode that matches the behavior you want:

| If you want... | Build with... | Run with... | Behavior |
|---|---|---|
| API/config-driven server | default build | `audiocpp_server --config server.json` | Uses models declared in the config. The UI is available unless disabled by config or `--no-ui`. |
| Read-only UI for configured models | default build | `audiocpp_server --config server.json --ui` | Browser UI is available for configured models, without downloads, deletes, or dynamic package management. |
| Full UI with downloads and model switching | `-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` | `audiocpp_server --ui --ui-management --backend <backend>` | UI can browse packages, download models, load/unload models, delete packages, and use temporary uploads. |
| Standalone deployed binary without local `model_specs/` | `-DAUDIOCPP_DEPLOYMENT_BUILD=ON` | `audiocpp_server --config server.json` | Binary carries compiled package specs for fallback model-spec lookup. |
| Offline/reproducible native-manager build | `-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON -DAUDIOCPP_BORINGSSL_ARCHIVE=/path/to/boringssl.tar.gz` | `audiocpp_server --ui --ui-management --backend <backend>` | Configure does not fetch BoringSSL from the network. |
| Distro-packaged TLS instead of bundled BoringSSL | `-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON -DAUDIOCPP_USE_SYSTEM_OPENSSL=ON` | `audiocpp_server --ui --ui-management --backend <backend>` | Uses system OpenSSL; useful for packagers. |

Native model management uses bundled BoringSSL by default. Normal server builds
do not build or link that HTTP/TLS dependency.

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
  "log_request_body": false,
  "max_request_body_bytes": 2147483648,
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
      "default_request_options": {
        "speaking_rate": 1.0
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

Set per-model `"default_request_options"` to apply request-option defaults to every request for that model. Values supplied by the actual request body override these defaults.

Set top-level `"max_request_body_bytes"` to bound the largest HTTP request body buffered in host RAM before routing. This protects endpoints that accept JSON or audio uploads from unbounded `Content-Length` claims. The default is `2147483648` bytes (2 GiB). Raise or lower it to match the largest upload your deployment intends to accept. Values above `2^53 - 1` are rejected because this config parser stores JSON numbers as doubles.

Set top-level `"log_request_body": true` and start the server with `--log` to print full JSON request bodies for debugging. This is off by default, and both switches are required so prompt text, paths, and request options are not logged accidentally. Audio bodies are not printed; multipart uploads log filename and byte count, while raw or live/chunked audio requests log only route, content type, query, and size/stream metadata.

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
3. **Request** — `"busy_timeout_ms"` in the request body (or as a `busy_timeout_ms` form field on multipart transcription, or a `busy_timeout_ms` query parameter on the live-ingest route, whose body is audio and has nowhere to put JSON) lets a caller bound its own wait.

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

Define `voice_presets` once and put every named preset inside that object. Duplicate JSON keys are rejected so a config cannot silently drop presets.

When a request sends `"voice": "assistant"`, the server uses that configured preset. When `"voice"` does not match a configured preset, it is passed through as the model-native cached voice id, preserving the previous behavior.

### Voice library (`voice_dir`)

Set server-level `"voice_dir"` to a directory of built-in voice wav files plus a `prompt_text` mapping file, so a request `"voice": "demo_01_man"` clones `voice_dir/demo_01_man.wav` the same way the WebUI's voice tab does — no `voice_ref` / `reference_text` needed from the caller:

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

The directory holds one `.wav` per built-in voice ('demo_01_man.wav', 'demo_02_woman.wav'...) and a `prompt_text` file with one `<basename-without-extension>|<transcript>` line per voice:

```
demo_01_man|okay,I'm Cemo and what you just heard wasn't a human voice.
demo_02_woman|以前我对这句话一知半解，现在好像有点懂了。因为你我开始留意很多以前不曾关心的事，开始对这个世界有了更多的好奇和善意。
```

Relative `voice_dir` paths resolve against the config file's directory. The command-line option `--voice-dir <directory>` overrides the configured value, which is useful when a process manager launches multiple single-model server configurations against one shared voice library. Relative command-line paths resolve against the process working directory.

When a request sends `"voice"` that is not a configured model preset, the server checks `<voice_dir>/<name>.wav`; if the file exists it is loaded as the cloning reference, and the `<name>` transcript from `prompt_text` is injected as `reference_text` unless the request already provides one.

Resolution precedence for a TTS request's voice fields:

1. `voice_ref` — always wins.
2. `voice` matching a configured model preset — preset wins.
3. `voice` matching a wav basename in `voice_dir` — voice-library clone.
4. Otherwise — `voice` is used as the model-native cached voice id (previous behavior).

`GET /v1/audio/voices` also lists the `voice_dir` wav basenames (deduplicated against preset names). If `voice_dir` is unset, behavior is exactly as before.

## Start

```bash
build/bin/audiocpp_server --config server.json
```

You can override configured server settings at startup, including the backend and shared voice library:

```bash
build/bin/audiocpp_server --config server.json --backend vulkan
build/bin/audiocpp_server --config server.json --voice-dir /absolute/path/to/voice
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

For full `uint64` seed values, pass `seed` as a JSON string. JSON numeric seeds
below `2^53` are accepted, but larger JSON numbers may lose precision before
option parsing.

If no request voice is provided and the configured model has `default_voice_preset`, the server injects that preset automatically. Request-level `voice`, `voice_ref`, and `reference_text` override the configured default.

`voice_ref` accepts either a plain path string (server-side file) or an object with a `type`:

```json
"voice_ref": { "type": "path", "path": "voices/alice.wav" }
```

With `"type": "base64"`, the `data` field carries a base64-encoded WAV payload (a `data:audio/wav;base64,...` URI is also accepted), so cloning clients can inline the reference audio instead of staging a file on the server first. The decoded payload is limited to 5 MiB; larger references must use a path:

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{
    "model": "indextts2",
    "input": "Cloned from an inline reference.",
    "voice_ref": { "type": "base64", "data": "UklGRh..." },
    "reference_text": "Transcript of the reference audio."
  }'
```

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

`POST /v1/audio/speech/live` is the live-ingest variant for speech-to-speech models: the request body is raw PCM sent with `Transfer-Encoding: chunked`, and the response can emit audio while the input stream is still open. Its `speech.audio.done` timing reports `ttft_ms` only when first output audio occurs after the input stream ends. If output audio starts before the input stream closes, `ttft_ms` is `null`, `first_audio_before_input_end=true`, and `overlap_ms` reports how much earlier the first output arrived. `request_start_to_first_audio_ms` is also included for transport diagnostics.

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

Note that `stream=true` streams the *output* of an already-uploaded file: the whole recording is sent first, and the deltas describe decoding it. It shortens time-to-first-token on long audio, but nothing can appear while the speaker is still talking. For that, use the live endpoint below.

### `POST /v1/audio/transcriptions/live`

Streams raw PCM **as it is captured** and returns transcript deltas on the same connection, so partial text can appear while the user is still speaking.

The request body is raw interleaved PCM sent with `Transfer-Encoding: chunked`; the response is the same SSE event shape as `stream=true` above, so a client can share one reader. There is no multipart form and no file — the audio never has to exist on disk, and the transport hands each chunk to the model as it arrives rather than assembling the recording first. Whether the *model* then keeps the whole utterance in memory is its own business: `nemotron_asr`, for instance, accumulates internally regardless of how the audio reaches it.

Because the body carries audio rather than JSON, parameters are query parameters:

| parameter | default | meaning |
| --- | --- | --- |
| `model` | required | id of a model configured with `mode: "streaming"` |
| `sample_rate` | `16000` | samples per second of the PCM being sent |
| `channels` | `1` | interleaved channel count |
| `sample_format` | `s16le` | `s16le` or `f32le` |
| `language` | unset | passed through to the model |
| `busy_timeout_ms` | model policy | how long to wait for the model lock, as elsewhere; clamped by the configured ceiling, so a request can shorten its own wait but never weaken the guard |

```bash
# Microphone straight into transcription (macOS; -f alsa on Linux, -f dshow on Windows)
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - \
  | curl -N -X POST -H 'Expect:' -T - \
      'http://127.0.0.1:8080/v1/audio/transcriptions/live?model=voxtral-realtime&sample_rate=16000&channels=1&sample_format=s16le'
```

`-T -` is what makes this live, and it is not interchangeable with `--data-binary @-`: the latter drains stdin to completion before opening the connection, which turns a live capture back into a file upload and defeats the endpoint. `-H 'Expect:'` suppresses curl's `Expect: 100-continue`, which the server does not answer — without it curl waits out its one-second continue timeout before sending any audio. (`-T .` reads stdin non-blocking; measured against this endpoint it behaves the same, so either works.)

A headerless stream carries no format, so the parameters above are a contract the server cannot verify — sending 48 kHz audio while declaring 16 kHz produces a confident, wrong transcript rather than an error.

Whether partial text actually appears *during* capture is a property of the model, not of this endpoint. A model that decodes incrementally (`voxtral_realtime`) emits deltas throughout the utterance; one whose encoder consumes the whole utterance before decoding (`nemotron_asr`) will stream its deltas only after the audio ends. Both work here; only the first feels live.

The request ends when the client sends the terminating chunk. Closing the connection without one is an error, not an end of speech — a truncated transcript that arrives as a normal `transcript.text.done` would be indistinguishable from the speaker stopping, so the endpoint refuses to produce one. The same applies to a stall past the idle timeout, an oversized chunk, or a malformed frame: each surfaces as an SSE `error` event.

Because the model is held for the length of the request, the body is bounded on several axes:

| key | default | meaning |
| --- | --- | --- |
| `idle_timeout_ms` | 30 s | longest wait for more data once the reader asks for it |
| `total_timeout_ms` | 600 s | checked at every point the body advances, so it caps the whole request |
| `max_body_bytes` | 512 MiB | received body bytes, framing included |
| `max_chunk_bytes` | 8 MiB | largest single declared chunk |
| `send_timeout_ms` | 30 s | `SO_SNDTIMEO` on the connection, so a client that stops reading the SSE response cannot hold the model open |

Those defaults suit one dictation at a time. Continuous captioning needs a longer deadline, and a trusted deployment may want a different trade entirely, so they are configurable — server-wide under `live_ingest`, with any subset overridden per model:

```json
{
  "live_ingest": { "total_timeout_ms": 600000, "max_chunk_bytes": 8388608 },
  "models": [
    {
      "id": "voxtral-realtime",
      "family": "voxtral_realtime",
      "mode": "streaming",
      "live_ingest": { "total_timeout_ms": 1800000 }
    }
  ]
}
```

A model entry sets only the values it needs and inherits the rest, so raising one bound for a long-capture model does not mean restating the whole policy and letting it drift. `0` means *disabled*, the same convention as `busy_timeout_ms`; a negative value is rejected at startup rather than silently treated as "no bound". The exception is `max_chunk_bytes`, which must stay positive and is rejected at `0`: a chunk is materialized in memory before it is served, so an unbounded one is not implementable, and the same value backs the overflow check that rejects a declared chunk size of `SIZE_MAX`. The keys are only consulted for this route, since no other one delivers its body incrementally.

`sample_rate` and `channels` are range-checked too (1000–384000 and 1–16), and are not configurable: they size the model's per-chunk buffer, so an absurd value would be an allocation request made while the model lock is held.

**Deployment.** This endpoint streams the request body and the response on one connection at the same time. That is legal HTTP/1.1, but it needs a direct connection or a proxy that does not buffer requests. Behind nginx both of these are required, because `proxy_request_buffering off` still buffers a chunked body unless the upstream connection is HTTP/1.1:

```nginx
proxy_http_version 1.1;
proxy_request_buffering off;
proxy_buffering off;
```

A browser cannot drive it: `fetch()` request streaming requires HTTP/2 and is half-duplex by specification — the whole request is sent before the response is processed. It is intended for native clients — `curl`, an `ffmpeg` pipe, or a backend service. A WebSocket transport would suit browsers and intermediaries better and could be added alongside this without changing it.

### `GET /v1/audio/voices?model=<id>`

Lists the cached voice ids, configured server voice preset names, and voice-library (`voice_dir`) wav names available for a TTS model, so a client can populate a voice picker instead of guessing generic names. For families that keep voice presets under `model_root/embeddings/*.safetensors` (`pocket_tts` today), this returns those ids too. If `model` is omitted and the server has exactly one configured model, that model is used; if multiple models are configured, omit `model` only when an empty list is acceptable.

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

### `POST /v1/tasks/unload_models`

Unload specific models from memory to free resources (e.g. VRAM on GPU backends). Subsequent requests to an unloaded model will trigger a transparent reload. The server waits for any in-flight inference on each target model to complete before unloading it.

The request body must contain a `model_ids` array of strings. Unknown ids are reported in the response rather than causing an error. Models that are not yet loaded (e.g. lazy-loaded models that have not been requested yet) are skipped silently.

```bash
curl http://127.0.0.1:8080/v1/tasks/unload_models \
  -H 'Content-Type: application/json' \
  -d '{
    "model_ids": ["pocket-tts", "qwen3-asr"]
  }'
```

Response:

```json
{
  "unloaded": ["pocket-tts", "qwen3-asr"],
  "not_found": []
}
```

### `POST /v1/tasks/unload_all_models`

Unload all currently loaded models from memory. No request body is required. As with the selective endpoint, subsequent requests will reload models transparently.

```bash
curl -X POST http://127.0.0.1:8080/v1/tasks/unload_all_models
```

Response:

```json
{
  "unloaded": ["pocket-tts", "qwen3-asr"]
}
```
