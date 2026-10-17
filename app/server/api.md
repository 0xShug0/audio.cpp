# Server API Reference

For installation, model configuration, CLI option mapping, voice setup and
troubleshooting, start with the [server guide](README.md). The
[endpoint index](README.md#endpoints) helps choose a route.

Unless stated otherwise, examples use `http://127.0.0.1:8080`. Model IDs must
refer to entries in your server config, with the task/mode required by the route.

## Contents

- [Readiness](#get-health) and [model discovery / available options](#get-v1models)
- [Speech generation, voice references, and streamed audio](#post-v1audiospeech)
- [Transcription](#post-v1audiotranscriptions) and [timestamps / speaker details](#post-v1audiotranscriptionsdetails)
- [Native transcription batches](#post-v1batchestranscriptions)
- [Forced alignment](#post-v1audioalignments)
- [Live PCM transcription and deployment requirements](#post-v1audiotranscriptionslive)
- [Voice discovery](#get-v1audiovoicesmodelid)
- [Generic tasks and artifacts](#post-v1tasksrun)
- [Generic streaming](#post-v1tasksstream) and [native batches](#post-v1tasksbatch)
- [Unload selected models](#post-v1tasksunload_models) or [all models](#post-v1tasksunload_all_models)

## `GET /health`

Returns server readiness and the number of configured models.

## `GET /v1/models`

Returns OpenAI-style model entries for the configured audio.cpp model ids.

Two opt-in query flags add to each entry:

- `include_session_options=true` echoes the model's configured `session_options`.
- `include_params=true` adds `params`: the request options the model's spec
  declares under `options.request`, as the spec writes them (`name`, `type`,
  `required`, `description`, and `default`/`min`/`max` where given). An enum row
  that names a `preset` also carries the preset's `values`. A model whose spec
  declares no request options returns `"params": []`.

```bash
curl 'http://127.0.0.1:8080/v1/models?include_params=true'
```

The list describes options; it does not validate them. Requests are still checked
per model as before.

## `POST /v1/audio/speech`

OpenAI-style text-to-audio. The response is `audio/wav` by default.

The `model` field is a configured entry's `id`. `input` is the text to speak;
`voice` selects a [server preset, reference-library name, or model-native voice](README.md#voices).
Model-specific controls belong in `options` and must be accepted by that family.
For example, a BreezeTTS entry can receive `"options": {"instruct": "A calm narrator."}`.

The OpenAI-style top-level `instructions` field is distinct: the current server
maps it to the model option `instruction`. It is not a universal replacement for
`options.instruct`, and a model must accept that spelling. Use the model's
documented option names consistently with its configured request defaults.

Top-level `speed` (or `speaking_rate`) is a positive speech-rate multiplier when
the model spec declares support. Explicit values, including `1`, override the
configured speed default. For models without declared speed support, `1` is
accepted as omitted; other values are rejected. For Kokoro, use the
[GGUF published on September 22, 2026 (UTC)](https://huggingface.co/audio-cpp/audio.cpp-gguf/commit/406756ee8e3b16e902ce40112986c1010775f888)
or a later published version. Other model controls are not implied by OpenAI
API compatibility.

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

Set `"response_format": "json"` to receive base64 WAV in a JSON response. In builds configured with `-DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON -DAUDIOCPP_SERVER_FRONTENDS_DIR=external/audio.cpp-server-frontends -DAUDIOCPP_SERVER_FRONTEND_MODULES=mp3_encode`, `"response_format": "mp3"` returns `audio/mpeg` MP3 bytes for non-streaming speech requests.

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

Some versions of the optional `mp3_encode` frontend reject `response_format=pcm`
before the core handler. If this example returns that frontend error, use a
server built without that module for PCM streaming.

`POST /v1/audio/speech/live` is the live-ingest variant for speech-to-speech models: the request body is raw PCM sent with `Transfer-Encoding: chunked`, and the response can emit audio while the input stream is still open. `input` is the text a text-to-speech model speaks and is required for it. A speech-to-speech or voice-conversion entry (task `s2s`, `vc` or `svc`) takes it as an optional prompt, such as PersonaPlex's persona prompt or LFM2.5-Audio's system prompt, and without it the model uses its default. The route's `speech.audio.done` timing reports `ttft_ms` only when first output audio occurs after the input stream ends. If output audio starts before the input stream closes, `ttft_ms` is `null`, `first_audio_before_input_end=true`, and `overlap_ms` reports how much earlier the first output arrived. `request_start_to_first_audio_ms` is also included for transport diagnostics.

With `return_text=true` (or `1`), the stream also carries the text the model writes as it speaks, such as a spoken reply's words. `speech.text.delta` events carry the text written since the previous event, in order with the audio deltas (an event's text comes before its audio), and `speech.text.done` carries the whole text before `speech.audio.done`:

```text
data: {"type":"speech.text.delta","delta":"Sure! How about"}
data: {"type":"speech.audio.delta","audio":"<base64 PCM>"}
...
data: {"type":"speech.text.done","text":"Sure! How about ..."}
data: {"type":"speech.audio.done","timing":{...}}
data: [DONE]
```

Without `return_text` the stream is audio only. `return_text` needs `stream_format=sse`; a model that writes no text ends the stream with an error event after its audio.

## `POST /v1/audio/transcriptions`

JSON transcription request using a server-local WAV audio path.

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

`file` and `model` are required; `language` is optional. Uploaded WAV bytes are decoded in memory and are not written to a temporary file. In builds configured with `-DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON -DAUDIOCPP_SERVER_FRONTENDS_DIR=external/audio.cpp-server-frontends -DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode`, the frontend also accepts MP3 and FLAC input for this route, decodes it to a temporary WAV, and forwards that normalized request to the same core transcription handler.

For streaming-capable ASR models configured with `mode: "streaming"`, pass `stream=true` to receive OpenAI-style transcription SSE:

```bash
# Nemotron requires mono 16 kHz audio; prepare it before uploading.
ffmpeg -i /path/to/input.wav -ar 16000 -ac 1 input-16k.wav
curl -N http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=nemotron-stream \
  -F language=en-US \
  -F stream=true \
  -F file=@input-16k.wav
```

The stream emits `transcript.text.delta` events, one final `transcript.text.done` event containing the full transcript, then `data: [DONE]`.

Models with revisable hypotheses, currently Phonon-2, may additionally emit
`{"type":"transcript.text.partial","text":"..."}`. This is a whole provisional
transcript to replace in the display, never append to the committed deltas.
For these models, deltas contain finalized segments only. Other models retain
their existing event behavior. See [Phonon-2](../../docs/models/parakeet_tdt.md#phonon-2-parakeet-tdt-variant).

`stream=true` streams the *output* of an already-uploaded file: the whole
recording is sent first, and the deltas describe decoding it. It can shorten
time to the first transcript on long audio, but nothing appears during the
upload. To send audio incrementally, use [live transcription](#post-v1audiotranscriptionslive).

## `POST /v1/audio/transcriptions/details`

Same request as `POST /v1/audio/transcriptions` — JSON with a server-local path, or a `multipart/form-data` upload — with a richer response. Use it when the model produces timestamps or speaker labels and the caller wants them.

`/v1/audio/transcriptions` returns `text` and `timing` and nothing else, so a model that aligned every word or separated speakers has that work discarded on the way out. This route returns those fields instead. The response schema of the plain route is unchanged; existing clients see exactly what they see today.

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions/details \
  -F model=parakeet-tdt \
  -F file=@/path/to/input.wav
```

```json
{
  "text": "the task has completed successfully",
  "language": "en",
  "words": [
    {"word": "the", "start_sample": 3200, "end_sample": 6400, "confidence": 0.98}
  ],
  "sample_rate": 16000,
  "timing": { "wall_ms": 412.7, "audio_duration_ms": 2400.0, "rtf": 0.17 }
}
```

`text` and `timing` are always present and match the plain route. The rest appear only when the model produced them:

| Field | Present when | Contents |
|---|---|---|
| `language` | the model reports a detected or configured language | Language code. |
| `segments` | the model produces speech segments | `start_sample`, `end_sample`, `confidence`, and `text` where the segment carries it. |
| `speaker_turns` | the model diarizes | `start_sample`, `end_sample`, `speaker_id`, `confidence`, and `text` where present. |
| `words` | the model aligns words | `word`, `start_sample`, `end_sample`, `confidence`. |
| `sample_rate` | any of the three arrays above is present | Rate the sample offsets are counted in. Divide an offset by it for seconds. |

Spans are sample offsets rather than seconds because that is what the models report; `sample_rate` is what converts them, which is why it only appears alongside them.

`stream=true` is rejected with a 400 on this route: the SSE response carries transcript deltas only, so it has nowhere to put the detail arrays. Use `/v1/audio/transcriptions` for a streamed transcript.

## `POST /v1/batches/transcriptions`

Runs multiple uploaded WAV files through one native offline model batch. This is
an audio.cpp extension, not an OpenAI API endpoint. The selected model must
implement native batching; the server rejects unsupported models instead of
silently running each file separately.

Supply `file` more than once in one multipart request:

```bash
ffmpeg -i /path/to/meeting-a.wav -ar 16000 -ac 1 meeting-a-16k.wav
ffmpeg -i /path/to/meeting-b.wav -ar 16000 -ac 1 meeting-b-16k.wav
curl -N http://127.0.0.1:8080/v1/batches/transcriptions \
  -F model=nemotron-3-diar \
  -F file=@meeting-a-16k.wav \
  -F file=@meeting-b-16k.wav
```

Nemotron 3 Diarization requires mono 16 kHz WAV input. This endpoint does not
automatically resample uploaded WAV files.

`model` and at least one `file` are required. `language`, `prompt`,
`busy_timeout_ms`, and a JSON object in `options` are optional and apply to every
file. The response is an SSE stream. Each file is published as soon as the model
finishes it; `index` maps the result back to its upload position. The final event
contains aggregate batch timing measured against the combined audio duration.

```text
data: {"type":"batch.transcription.result","index":0,"filename":"meeting-a.wav","text":"","speaker_turns":[{"start_sample":0,"end_sample":32000,"speaker_id":"speaker_0","confidence":1.0}],"sample_rate":16000}

data: {"type":"batch.transcription.result","index":1,"filename":"meeting-b.wav","text":"","speaker_turns":[...],"sample_rate":16000}

data: {"type":"batch.transcription.done","result_count":2,"timing":{"wall_ms":145.5,"audio_duration_ms":70000.0,"rtf":0.0021}}

data: [DONE]
```

## `POST /v1/audio/alignments`

Multipart forced-alignment request using uploaded audio bytes and a known transcript. Use this when the server cannot see the client's local audio path, for example when the server is remote or running in Docker.

```bash
curl http://127.0.0.1:8080/v1/audio/alignments \
  -F model=qwen3-align \
  -F language=en \
  -F text='The task has completed successfully.' \
  -F file=@/path/to/input.wav
```

`file`, `model`, and `text` are required. `language` is model-dependent: Qwen3 Forced Aligner requires it (for example, `English` or `Chinese`). The selected model must be configured with `task: "align"` and `mode: "offline"`. Uploaded WAV bytes are decoded in memory and are not written to a temporary file. The response includes word timestamps in seconds plus sample offsets.

## `POST /v1/audio/transcriptions/live`

Streams raw PCM **as it is captured** and returns transcript deltas on the same connection, so partial text can appear while the user is still speaking.

The request body is raw interleaved PCM sent with `Transfer-Encoding: chunked`; the response is the same SSE event shape as `stream=true` above, so a client can share one reader. There is no multipart form and no file — the audio never has to exist on disk, and the transport hands each chunk to the model as it arrives rather than assembling the recording first.

Because the body carries audio rather than JSON, parameters are query parameters:

| parameter | default | meaning |
| --- | --- | --- |
| `model` | required | id of a model configured with `mode: "streaming"` |
| `sample_rate` | `16000` | samples per second of the PCM being sent |
| `channels` | `1` | interleaved channel count |
| `sample_format` | `s16le` | `s16le` or `f32le` |
| `language` | unset | passed through to the model |
| `prompt` | unset | URL-encoded recognition context (hotwords, spellings), same as the multipart `prompt` field |
| `busy_timeout_ms` | model policy | how long to wait for the model lock, as elsewhere; clamped by the configured ceiling, so a request can shorten its own wait but never weaken the guard |

```bash
# macOS microphone example; device selection is platform-specific.
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - \
  | curl -N -X POST -H 'Expect:' -T - \
      'http://127.0.0.1:8080/v1/audio/transcriptions/live?model=voxtral-realtime&sample_rate=16000&channels=1&sample_format=s16le'
```

On Linux, use the input arguments `-f alsa -i default` instead of
`-f avfoundation -i ":0"`; on Windows, select a DirectShow audio device with
`-f dshow -i audio="<device name>"`. For a reproducible file-fed check, replace
the microphone input arguments with `-i /path/to/input.wav` and keep the PCM
output options and curl pipe unchanged.

`-T -` is what makes this live, and it is not interchangeable with `--data-binary @-`: the latter drains stdin to completion before opening the connection, which turns a live capture back into a file upload and defeats the endpoint. `-H 'Expect:'` suppresses curl's `Expect: 100-continue`, which the server does not answer — without it curl waits out its one-second continue timeout before sending any audio. (`-T .` reads stdin non-blocking; measured against this endpoint it behaves the same, so either works.)

A headerless stream carries no format, so the parameters above are a contract the server cannot verify — sending 48 kHz audio while declaring 16 kHz produces a confident, wrong transcript rather than an error.

Whether partial text actually appears *during* capture is a property of the model, not of this endpoint. Cache-aware streaming models such as `voxtral_realtime` and `nemotron_asr` emit deltas throughout the utterance; buffered models may emit only after enough audio has accumulated.

For a model configured with `task: "diar"`, this route and the file-backed
`/v1/audio/transcriptions` route with `stream=true` return speaker turns instead
of text deltas:

```text
data: {"type":"diarization.delta","speaker_turns":[{"start_sample":5760,"end_sample":52480,"speaker_id":"speaker_0","confidence":0.99}],"sample_rate":16000}
data: {"type":"diarization.done","speaker_turns":[...],"sample_rate":16000,"timing":{"ttft_ms":3606}}
data: [DONE]
```

Each delta contains newly emitted turns. The final event contains the complete
result, including any turn still open when input ended; do not append it to the
deltas. TTFT measures the first speaker-turn result, not the first internal
probability prediction. A model that emits only completed turns waits for a turn
to end. If no turns are detected, the final array is empty and `ttft_ms` is `null`.
ASR responses keep their `transcript.text.delta` / `transcript.text.done` format.

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

## `GET /v1/audio/voices?model=<id>`

Lists cached voice IDs, configured model preset names, and shared `voice_dir`
WAV basenames. If the configured model path is a directory containing
`embeddings/*.safetensors` (used by PocketTTS), their filename stems are also
listed. For an Irodori-TTS model, `voice_dir` speaker embeddings
(`<name>.speaker.safetensors`) are listed by `<name>`. It does not enumerate
speaker tensors packed inside a GGUF; use the
[model's documented built-in IDs](README.md#built-in-voices-and-discovery) for those.

Pass `model` explicitly when several entries are configured. Omitting it selects
the only model when exactly one exists; with several models it can return an
empty list instead. Shared library entries are not filtered to the model's
voice-cloning capabilities.

```bash
curl 'http://127.0.0.1:8080/v1/audio/voices?model=pocket-tts'
```

```json
{"voices": ["alba", "cosette", "marius"]}
```

## `POST /v1/tasks/run`

For audio input, `request.audio` is a server-local WAV path. Alternatively,
`request.audio_base64` accepts a Base64-encoded WAV or a
`data:audio/wav;base64,...` URI:

```json
{
  "model": "mel_band_roformer",
  "request": {"audio_base64": "<Base64-encoded WAV>"}
}
```

Provide only one of `audio` and `audio_base64`. Invalid inline audio returns
HTTP 400. The existing `max_request_body_bytes` limit applies to the entire
JSON body, including Base64 expansion. The WAV must meet the model's input
sample-rate and channel requirements. Inline audio is supported on this endpoint;
batch and stream endpoints retain their existing input formats.

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

A result's artifacts (opaque payloads such as speaker embeddings or acoustic
tokens) come back in an `artifacts` array. A request takes artifacts in the
same shape, so a client can send what one result returned in a later request:

```json
{
  "model": "my-model",
  "request": {
    "audio": "/path/to/input.wav",
    "artifacts": [
      {"id": "example.state", "kind": "custom", "payload": "<Base64 bytes>", "meta": {"format": "example/1"}},
      {"id": "example.tokens", "kind": "acoustic_tokens", "path": "/path/to/tokens.bin"}
    ]
  }
}
```

| field | meaning |
| --- | --- |
| `id` | non-empty string; ids may repeat |
| `kind` | `speaker_embedding`, `style_embedding`, `prompt_embedding`, `acoustic_tokens`, `midi`, `transcript_alignment`, `diarization_state`, `vad_state` or `custom` |
| `payload` | the bytes as Base64; a `data:...;base64,` URI is also accepted |
| `path` | instead of `payload`: a server-local file whose bytes are the payload; use an absolute path to avoid request-base ambiguity |
| `meta` | optional object of strings; a number or boolean value becomes text as in `options` (`12`, `true`; `1.0` becomes `1`) |

The model gets the artifacts in array order. Which ids and kinds it reads is up
to its family; see the model's docs. Most families that read none ignore them,
while AuK, HeartMuLa, VibeVoice, VoxCPM1, VoxCPM2 and YuE2 turn away a request
that carries any with HTTP 400. The same field works in `/v1/tasks/stream`
requests and in each `/v1/tasks/batch` entry. A malformed entry returns HTTP 400
naming it, for example `artifacts[1] (example.tokens): unknown kind 'tokens'`.
`max_request_body_bytes` bounds inline payloads, as it does `audio_base64`; a
`path` must name a regular file, and the payloads of one request, or of one
`/v1/tasks/batch` entry, may total at most 2 GiB. A batch reads all its entries
before it runs, so with `path` artifacts it can hold up to 2 GiB per entry.

Relative request paths resolve against the server's request base: normally its
working directory, or the executable directory with native UI management.
This differs from config-relative model paths. Absolute server-local paths work
in either mode.

## `POST /v1/tasks/stream`

Runs one generic request through the model's streaming interface and returns
the events it produced with the result. The body is the one `/v1/tasks/run`
takes, artifacts included, except that `audio_base64` is not read here: it is
ignored, not rejected, so audio input goes in `audio` as a server-local path.

By default the response is one JSON body, sent once the run is over:

```text
{"events":[{"partial_text":{"text":"Sure","language":"en"},"audio":"<Base64 WAV>","is_final":false},
           {"audio":"<Base64 WAV>","is_final":false},...],
 "result":{"text":"Sure! ...","language":"en","audio":"<Base64 WAV>","sample_rate":24000,"channels":1,
           "timing":{"ttft_ms":<ms>}}}
```

An event has the fields it carries: `partial_text`, `audio` (a WAV of that
event's samples), `named_audio_outputs`, `word_timestamps` and, for a
diarization model, `speaker_turns`, and always `is_final`. Events do not carry
voice activity or artifacts, so an event can be just `{"is_final":false}`.
`result` has the fields of the `/v1/tasks/run` result, except that its `timing`
holds only `ttft_ms`: the time from the start of the run, after the model is
free and loaded, to its first output (`null` for a diarization run that found
no speaker).

With `"stream_format": "sse"` at the top level of the body, the response is
server-sent events instead, each written as the model produces it:

```bash
curl -N http://127.0.0.1:8080/v1/tasks/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lfm2-audio-s2s-stream",
    "stream_format": "sse",
    "request": {"audio": "/path/to/question.wav", "options": {"return_codes": true}}
  }'
```

```text
data: {"type":"task.stream.event","event":{"partial_text":{"text":"Sure","language":"en"},"audio":"<Base64 WAV>","is_final":false}}
data: {"type":"task.stream.event","event":{"audio":"<Base64 WAV>","is_final":false}}
...
data: {"type":"task.stream.done","result":{"text":"Sure! ...","language":"en","audio":"<Base64 WAV>",...,"timing":{"ttft_ms":<ms>}}}
data: [DONE]
```

Each `event` is the object the JSON response puts in `events`, in the same
order, and `result` is the object it puts under `result`, artifacts included.
`task.stream.done` carries the whole result on one line, so for a spoken reply
it repeats all of the reply's audio, about 640 KB of Base64 per 10 s of 24 kHz
audio; a client that reads lines with a fixed size limit has to allow for that.
A client that closes the stream before `task.stream.done` gets no result.

Only the string `"sse"` streams. Without the field, or with `null`, the
response is the JSON body, and any other value is answered with HTTP 400. This
differs from `/v1/audio/speech`, which streams whenever `stream_format` is
present. The `Accept` header is not consulted.

A request that fails before the stream starts gets the status it gets without
`stream_format`: an unknown model, a malformed artifact, an unreadable `audio`
path and, with `--parallel-jobs`, a model that stays busy past the busy timeout
(503). Once the stream has started, a failure ends it with
`data: {"type":"error","error":{"message":"..."}}`, without `task.stream.done`
or `[DONE]`; the event has no error type or status. That covers a model that
fails to load, a request the model turns away when it runs (HTTP 400 without
`stream_format`) and, without `--parallel-jobs`, a busy model, as on streamed
`/v1/audio/speech`. A client knows the run succeeded when it gets
`task.stream.done`.

An SSE client that disconnects stops the run within the next event or two; a
JSON client that disconnects leaves its run to finish. An SSE client that stops
reading keeps the model, or with `--parallel-jobs` its slot, busy until it
reads or disconnects, as on streamed `/v1/audio/speech` and
`/v1/audio/transcriptions`; `live_ingest.send_timeout_ms` applies only to the
live routes. The SSE response carries `X-Accel-Buffering: no`; a proxy in front
of the server must not buffer it.

## `POST /v1/tasks/batch`

Runs multiple generic requests through a model's native offline batch path. The
selected model must implement native batching; unsupported models are rejected
instead of being run sequentially. Each entry uses the same fields as the
`request` object accepted by `/v1/tasks/run`.

```bash
curl -N http://127.0.0.1:8080/v1/tasks/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "reuse",
    "requests": [
      {"audio": "/path/to/first.wav"},
      {"audio": "/path/to/second.wav"}
    ]
  }'
```

The SSE response emits `task.batch.result` events with the original request
index as results become available, followed by `task.batch.done` with aggregate
batch timing. Per-result `timing` is `null` because fused execution does not
produce an independent wall time for each request.

## `POST /v1/tasks/unload_models`

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

## `POST /v1/tasks/unload_all_models`

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
