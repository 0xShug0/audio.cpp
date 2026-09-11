# audio.cpp Supertonic Speaker Extension

Minimal Chrome/Edge extension demo for speaking selected page text through an
`audiocpp_server` streaming TTS model.

## Start audio.cpp server

The demo includes separate CUDA and CPU server configs. The checked-in configs
use `models/supertonic-3` relative to the directory where `audiocpp_server` is
started; edit the model `path` if your Supertonic model lives elsewhere.

- `server.cuda.json`: CUDA server on `127.0.0.1:8080`
- `server.cpu.json`: CPU server on `127.0.0.1:8081`

CUDA:

```bash
build/debug/bin/audiocpp_server --config examples/browser_extension/supertonic_speaker/server.cuda.json
```

CPU:

```bash
build/debug/bin/audiocpp_server --config examples/browser_extension/supertonic_speaker/server.cpu.json
```

The extension displays the current backend from `/health`.

The config shape is:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 8,
  "lazy_load": true,
  "models": [
    {
      "id": "supertonic-stream",
      "family": "supertonic",
      "path": "/path/to/models/supertonic-3",
      "task": "tts",
      "mode": "streaming",
      "default_voice_preset": {
        "voice_id": "M1"
      }
    }
  ]
}
```

## Load the extension

1. Open `chrome://extensions` or `edge://extensions`.
2. Enable Developer mode.
3. Click **Load unpacked**.
4. Select `examples/browser_extension/supertonic_speaker`.

## Use

1. Open a web page.
2. Select some text.
3. Click the extension icon.
4. Confirm:
   - Server: `http://127.0.0.1:8080`
   - Model id: `supertonic-stream`
   - Sample rate: `44100`
5. Click **Selection**.

The popup splits text into smaller chunks and sends sequential streaming SSE
requests to `/v1/audio/speech`. Keep the popup open while it speaks.

You can also right-click a page and use:

- **Read selection**
- **Read page**

The context menu opens a small player window and starts playback immediately.
The player shows live chunk progress, client-observed first-audio latency, the
server-reported TTFT from `speech.audio.done`, and the number of audio delta
events received for the current chunk.

For non-Supertonic TTS models, set the sample rate to the model's PCM output
rate before speaking.

## Offline Converter Demo

For long-form conversion, open the extension popup and click **Offline converter**.
The page lets you upload a text file, choose a Supertonic preset voice, and run
sequential offline conversion chunks. It shows progress, elapsed time, ETA,
server wall time, generated audio duration, and RTF. As conversion runs, the
page adds one playback bar and download link per generated WAV part.

CUDA offline server:

```bash
build/debug/bin/audiocpp_server --config examples/browser_extension/supertonic_speaker/server.offline.cuda.json
```

CPU offline server:

```bash
build/debug/bin/audiocpp_server --config examples/browser_extension/supertonic_speaker/server.offline.cpu.json
```

Use these URLs in the offline page:

- CUDA: `http://127.0.0.1:8082`
- CPU: `http://127.0.0.1:8083`
