# Any TTS Voice Clone

`any_tts` combines an existing offline TTS model with Tone Color VC. Download
the [base model](../../README.md) and [Tone Color VC](tone_color_vc.md) separately. Intermediate audio stays in memory;
neither model's weights are repackaged. Create a composition manifest:

```json
{
  "base_family": "kokoro_tts",
  "base_model": "/path/to/Kokoro-82M-GGUF/kokoro-82m-bf16.gguf",
  "converter_model": "/path/to/Tone-Color-VC-GGUF/tone-color-vc-f32.gguf"
}
```

Paths may be absolute or relative to the manifest. Use a different `base_family`
and `base_model` to select another TTS model.

```bash
audiocpp_cli --task clon --family any_tts \
  --model /path/to/any-tts.json \
  --backend cuda --threads 8 --voice-id am_michael \
  --voice-ref reference.wav --text "Welcome to the harbor." \
  --seed 42 --log --out speech.wav
```

The base model must provide offline TTS. Its compiled loader must be available,
as must `tone_color_vc`. Base request options (including voice selection) and
base session options retain their usual names. The reference recording is sent
only to Tone Color VC. Language and speaking style are passed to the base TTS.
Kokoro, Piper, and KittenTTS are examples of suitable base models. The base model
determines supported languages and preset voices; the converter does not add
languages to it.

## Common Options

| Option | Meaning |
| --- | --- |
| `--model` | Composition manifest naming the base TTS and converter weights. |
| `--voice-ref` | Target speaker recording. |
| `--voice-id` | Base TTS preset voice, where supported. |
| `--mode` | `offline` or `streaming`. |

## Request Options (`--request-option`)

| Option | Default | Meaning |
| --- | --- | --- |
| `text_chunk_size` | `240` | Positive text chunk budget in Unicode code points. |
| `text_chunk_mode` | `default` | Framework text chunking mode. |
| `vc_temperature` | `0.3` | Converter posterior noise scale; zero disables noise. |
| `vc_seed` | `1234` | Converter seed; `-1` selects a random seed. Independent of base TTS `seed`. |

Other request options are forwarded to, and validated by, the base model.

There are no wrapper-specific session or load options. Base model session and
load options retain their original names.

## Long-Form and Streaming

Both modes split text using the framework text chunker, synthesize each chunk,
and convert that chunk before proceeding. Streaming emits each converted chunk
as it completes; it is chunk-level streaming, not sample-by-sample conversion.
Output is mono 22.05 kHz. Offline and streaming use the same chunk pipeline.

Choose a chunk size within the base model's limits. Conversion preserves the
base speech's delivery; a reference recording does not transfer its transcript
or speaking rhythm. Separate chunks can have audible prosody transitions.

## Server

Add this entry to the server's `models` array:

```json
{
  "id": "my-clone",
  "family": "any_tts",
  "path": "/path/to/any-tts.json",
  "task": "clon",
  "mode": "streaming"
}
```

The standard speech endpoint accepts both ordinary WAV responses and streamed
audio. Set `stream` to `true` for SSE `speech.audio.delta` events; concatenate
their decoded PCM payloads in order. Audio is mono PCM16 at 22.05 kHz. There is
no model-specific server endpoint or response schema.

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-clone","input":"Welcome to the harbor.","voice":"am_michael","voice_ref":"/path/to/reference.wav","seed":42,"options":{"vc_seed":42}}' \
  --output speech.wav
```

The reference path must be accessible to the server. The endpoint also accepts
the standard base64 `voice_ref` object. Use base-model-specific `session_options`
on the server model entry when needed.
