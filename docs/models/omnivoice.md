# OmniVoice

OmniVoice supports multilingual TTS, voice cloning, voice design, non-verbal tag tokens, long-form chunking, and chunked pseudo-streaming output.

| Route | Task | Mode | Inputs |
|---|---|---|---|
| Auto voice | `tts` | `offline`, `streaming` | `--text` |
| Voice clone | `tts` | `offline`, `streaming` | `--text`, `--voice-ref`, optional `--reference-text` |
| Voice design | `tts` | `offline`, `streaming` | `--text`, `--instruct` |

## Model Layout

Use a local OmniVoice model package:

```text
models/OmniVoice/
```

The package should contain the OmniVoice model weights and the audio-tokenizer files expected by `model_specs/omnivoice.json`. GGUF packages can also be used when they embed the package spec.

## Offline CLI

Voice clone:

```bash
audiocpp_cli --task tts --family omnivoice --model /path/to/OmniVoice --backend cuda --text "Hello from OmniVoice." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

Voice design:

```bash
audiocpp_cli --task tts --family omnivoice --model /path/to/OmniVoice --backend cuda --text "Hello from OmniVoice." --instruct "female, young adult, moderate pitch" --out out.wav
```

Auto voice:

```bash
audiocpp_cli --task tts --family omnivoice --model /path/to/OmniVoice --backend cuda --text "Hello from OmniVoice." --out out.wav
```

## Streaming CLI

OmniVoice upstream Python does not expose model-native streaming. audio.cpp exposes chunked pseudo streaming: the session emits one audio event per generated text chunk, then returns a merged final WAV.

```bash
audiocpp_cli --task tts --mode streaming --family omnivoice --model /path/to/OmniVoice --backend cuda --text "Hello from OmniVoice. This text can be split into chunk events." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --text-chunk-size 160 --out stream.wav --out-dir stream_chunks
```

`--out` writes the final merged WAV. `--out-dir` writes emitted chunk WAVs such as `chunk_0.wav`, `chunk_1.wav`, and so on.

## Server Streaming

Configure OmniVoice with `mode: "streaming"`:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 8,
  "models": [
    {
      "id": "omnivoice-stream",
      "family": "omnivoice",
      "path": "/path/to/OmniVoice",
      "task": "tts",
      "mode": "streaming"
    }
  ]
}
```

SSE request:

```bash
curl -N http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "omnivoice-stream",
    "input": "Hello from OmniVoice streaming.",
    "response_format": "pcm",
    "stream_format": "sse",
    "voice": "assets/resources/b.wav",
    "reference_text": "Some call me nature. Others call me Mother Nature. I have been here for over four billion years.",
    "options": {
      "text_chunk_size": 160,
      "text_chunk_mode": "tag_aware"
    }
  }'
```

The SSE stream emits `speech.audio.delta` events followed by `speech.audio.done`.

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for voice cloning. |
| `--reference-text` | text | empty string | Transcript for the reference audio. |
| `--instruct` | text | empty string | Voice-design instruction. |
| `--language` | language hint | auto | Optional language hint. |
| `--text-chunk-size` | integer chars | disabled | Enables framework text chunking and controls pseudo-streaming chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Framework text chunking mode when `--text-chunk-size` is set. |
| `--num-inference-steps` | integer | `32` | Decoder diffusion steps. |
| `--guidance-scale` | float | `2.0` | Decoder CFG strength. |
| `--request-option speed=<float>` | float | `1.0` | Speech speed multiplier. |
| `--request-option audio_chunk_duration=<float>` | seconds | `15.0` | Model-side automatic chunk duration when framework chunking is not explicitly enabled. |
| `--request-option audio_chunk_threshold=<float>` | seconds | `30.0` | Estimated audio length threshold before model-side chunking is used. |
| `--session-option omnivoice.mem_saver=true\|false` | bool | `false` | Release staged generator and audio-tokenizer runtime graphs after request phases to reduce resident VRAM. Later requests may rebuild released graphs. |
| `--session-option omnivoice.perf_mode=off\|flash_attention` | enum | `off` | Opt-in generator attention mode. `off` keeps the exact-safe path; `flash_attention` can improve CUDA throughput with small output drift. |
| `--session-option omnivoice.generator_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Runtime storage type for generator weights. Also accepted as `omnivoice.weight_type`. |
| `--session-option omnivoice.audio_tokenizer_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Runtime storage type for audio tokenizer weights. |

`omnivoice.perf_mode=flash_attention` is only available on the normal graph path and cannot be combined with `omnivoice.mem_saver=true`.

The two `weight_type` options quantize SafeTensors weights at load time, which is not the same as loading a prebuilt Q8 GGUF package. On CUDA, `omnivoice.generator_weight_type=f16` measured 1.26x faster than `native`, while runtime `omnivoice.audio_tokenizer_weight_type=q8_0` produced unusable audio without being faster. See the [OmniVoice weight-type benchmark](../reports/omnivoice_weight_type_benchmark.md).

## Tags

Non-verbal tags are written directly in `--text`. Supported spellings include:

```text
[laughter] [sigh] [confirmation-en] [question-en] [question-ah] [question-oh]
[question-ei] [question-yi] [surprise-ah] [surprise-oh] [surprise-wa]
[surprise-yo] [dissatisfaction-hnn]
```
