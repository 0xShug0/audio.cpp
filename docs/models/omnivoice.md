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

## Common Options (use directly)

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
| `--seed` | unsigned 32-bit integer | random session RNG | Generation seed; omitted requests continue the session RNG state. |

The corresponding request-option names are `reference_text`, `instruction`,
`language`, `text_chunk_size`, `text_chunk_mode`, `num_inference_steps`,
`guidance_scale`, and `seed`.

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `speed` | float | `1.0` | Speech speed multiplier. |
| `duration_sec` | seconds | estimated | Optional target output duration. |
| `shift` | float | `0.1` | Diffusion timestep schedule shift. |
| `denoise` | bool | `true` | Add the denoise control token when reference audio is present. |
| `preprocess_prompt` | bool | `true` | Preprocess reference audio before tokenization. |
| `postprocess_output` | bool | `true` | Apply output postprocessing and trimming. |
| `layer_penalty_factor` | float | `5.0` | Layer-order penalty for masked audio-codebook position selection. |
| `position_temperature` | float | `5.0` | Gumbel temperature for masked-position selection. |
| `class_temperature` | nonnegative float | `0.0` | Audio-token class sampling temperature; `0` selects classes greedily. |
| `audio_chunk_duration_sec` | seconds | `15.0` | Model-side automatic chunk duration when framework chunking is not explicitly enabled. |
| `audio_chunk_threshold_sec` | seconds | `30.0` | Estimated audio length threshold before model-side chunking is used. |

Legacy request names `duration`, `t_shift`, `audio_chunk_duration`, and
`audio_chunk_threshold` remain accepted, respectively. Use one spelling per
option. Both spellings work with existing GGUFs; no reconversion is required.

## Session Options (use with `--session-option`)

Arena and weight-context sizes are in MiB.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `omnivoice.mem_saver` | bool | `false` | Release staged generator and audio-tokenizer runtime graphs after request phases to reduce resident VRAM. Later requests may rebuild released graphs. |
| `omnivoice.perf_mode` | `off`, `flash_attention` | `off` | Opt-in generator attention mode. `off` keeps the exact-safe path; `flash_attention` can improve CUDA throughput with small output drift. |
| `omnivoice.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Generator weight storage fallback. |
| `omnivoice.generator_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | inherits `weight_type` | Explicit generator weight storage override. |
| `omnivoice.audio_tokenizer_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Runtime storage type for audio tokenizer weights. |
| `omnivoice.audio_tokenizer_graph_arena_mb` | positive integer | `128` | Audio tokenizer graph arena. |
| `omnivoice.generator_prefill_graph_arena_mb` | positive integer | `256` | Generator prefill graph arena. |
| `omnivoice.generator_decode_graph_arena_mb` | positive integer | `256` | Generator decode graph arena. |
| `omnivoice.audio_tokenizer_weight_context_mb` | positive integer | `128` | Audio tokenizer weight tensor context. |
| `omnivoice.generator_weight_context_mb` | positive integer | `256` | Generator weight tensor context. |

`omnivoice.perf_mode=flash_attention` is only available on the normal graph path and cannot be combined with `omnivoice.mem_saver=true`.

The two `weight_type` options quantize SafeTensors weights at load time, which is not the same as loading a prebuilt Q8 GGUF package. On CUDA, `omnivoice.generator_weight_type=f16` measured 1.26x faster than `native`, while runtime `omnivoice.audio_tokenizer_weight_type=q8_0` produced unusable audio without being faster. See the [OmniVoice weight-type benchmark](../reports/omnivoice_weight_type_benchmark.md).

## VoiceTut-TTS (Egyptian Arabic)

[VoiceTut-TTS](https://huggingface.co/mohammedaly22/VoiceTut-TTS) is an
OmniVoice fine-tune for Egyptian Arabic and Arabic-English code-switching. It
has the same architecture and tensor layout as OmniVoice and reuses the
OmniVoice audio tokenizer unchanged, so it runs through the `omnivoice` family
with every route and option above. It was trained with the `arz` (Egyptian
Arabic) language id, so pass `--language arz`. Model and integration by
[@Mohammedaly22](https://github.com/Mohammedaly22).

| Package | Format | Notes |
|---|---|---|
| `voicetut_tts_q8_0` | GGUF Q8_0 | Standalone GGUF with the audio tokenizer embedded. |
| `voicetut_tts_f16` | GGUF F16 | Closest to the original weights. |

Voice clone with Egyptian Arabic and English code-switching:

```bash
audiocpp_model_manager install voicetut_tts_q8_0 --models-dir models
audiocpp_cli --task tts --family omnivoice --model models/VoiceTut-TTS-GGUF --backend cuda --language arz --text "عندي meeting بكرة الصبح، فهحاول أخلص الشغل بدري النهارده." --voice-ref speaker.wav --reference-text "<transcript of speaker.wav>" --out out.wav
```

The upstream repository ships 17 reference speakers in `reference_speakers/`
with transcripts in `references.json`. Copy them into a server `voice_dir`
(one `<Name>.wav` per speaker plus a `prompt_text` file with
`<Name>|<transcript>` lines) to use them as named voices, for example
`"voice": "Mohamed"`. See the server [voice library](../../app/server/README.md#voice-library-voice_dir).

To build the GGUF packages, place the VoiceTut weights and the OmniVoice
`audio_tokenizer/` directory in one model root, then convert:

```bash
audiocpp_gguf --input weights=models/VoiceTut-TTS/model.safetensors   --input audio_tokenizer_weights=models/VoiceTut-TTS/audio_tokenizer/model.safetensors   --root models/VoiceTut-TTS --output models/VoiceTut-TTS-GGUF/voicetut-tts-q8_0.gguf   --family omnivoice --type q8_0 --overwrite
```

## Tags

Non-verbal tags are written directly in `--text`. Supported spellings include:

```text
[laughter] [sigh] [confirmation-en] [question-en] [question-ah] [question-oh]
[question-ei] [question-yi] [surprise-ah] [surprise-oh] [surprise-wa]
[surprise-yo] [dissatisfaction-hnn]
```
