# audio_decode

`audio_decode` is a pre-processing module for ASR requests:

```text
POST /v1/audio/transcriptions
client_encoded_audio_request -> core_wav_audio_request
```

It accepts MP3 and FLAC input through JSON paths or multipart uploads, decodes
the file with miniaudio, writes a temporary WAV, rewrites the request to a
core-compatible JSON request, then lets the core ASR handler run normally.

## Build

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  -DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode
```

`audio_decode` uses the vendored miniaudio header and does not add a system
library dependency.

## Request

Multipart upload:

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=qwen3-asr \
  -F file=@speech.mp3
```

JSON path:

```json
{
  "model": "qwen3-asr",
  "file": "speech.flac"
}
```
