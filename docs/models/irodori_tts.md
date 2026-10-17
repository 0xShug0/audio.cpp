# Irodori-TTS

Irodori-TTS is Japanese TTS under `--family irodori_tts`. It supports no-reference speech, optional reference-audio voice cloning, and instruction-based voice design.

The default downloadable package is the GGUF v4 Small Q8_0 checkpoint. v4 Small is the preferred path for new use because one checkpoint covers no-reference TTS, voice cloning, and voice design. The older v3 packages remain supported for existing users and local validation.

## Variants

| Variant | Package | Tasks | Notes |
|---|---|---|---|
| 500M v3 | `Irodori-TTS-500M-v3-GGUF` | `tts`, `clon` | Uses the shared llm-jp tokenizer layout in the original safetensors package. |
| 600M v3 VoiceDesign | `Irodori-TTS-600M-v3-VoiceDesign-GGUF` | `tts`, `clon`, `vdes` | Adds instruction/caption conditioning for voice design. |
| v4 Small | `Irodori-TTS-v4-Small-GGUF` | `tts`, `clon`, `vdes` | Bundles its v4 tokenizer and supports instruction/caption conditioning in the same checkpoint. |

v4 GGUF packages are published in both `q8_0` and `f16`. v3 GGUF packages are also available in `q8_0` and `f16`.

**Checkpoints without a duration predictor** (`use_duration_predictor` absent or `false` in the model config, such as the original 500M v2 safetensors) also load. With no `duration_sec`, they generate 30 seconds and `trim_tail` cuts the trailing silence, as the Python runtime does; `duration_scale` has no effect. The fallback always samples 30 seconds, so passing `duration_sec` is faster when the length is known.

> **v4 reference-conditioning note:** Fresh v4 voice-clone or reference+instruction generations may occasionally add a short extra phrase near the end of the clip. This behavior is also reproducible in the upstream Python path with the same reference/text/seed, so it is treated as a current v4 model/runtime limitation rather than a GGUF-only issue. No-reference and instruction-only paths are usually cleaner; for reference-conditioned use, try a different seed, instruction, or explicit `duration_sec` if the tail matters.

## Quick Start

No-reference v4 speech:

```bash
audiocpp_cli --task tts --family irodori_tts \
  --model models/Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-q8_0.gguf \
  --backend cuda --language ja \
  --text "今日は短い確認です。やさしく、聞き取りやすい声でお願いします。" \
  --request-option no_ref=true \
  --out out.wav
```

v4 voice cloning:

```bash
audiocpp_cli --task clon --family irodori_tts \
  --model models/Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-q8_0.gguf \
  --backend cuda --language ja \
  --text "どうしてもっと早く教えてくれなかったの？私、ずっと待ってたのに。" \
  --voice-ref models/Irodori-TTS-v4-Small/samples/clone_ref1.wav \
  --out out.wav
```

v4 voice design:

```bash
audiocpp_cli --task vdes --family irodori_tts \
  --model models/Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-q8_0.gguf \
  --backend cuda --language ja \
  --text "本日はお越しいただき、誠にありがとうございます。" \
  --request-option instruct="落ち着いた大人の男性。深く響く声で丁寧に話している。" \
  --request-option no_ref=true \
  --request-option guidance_scale=3 \
  --out out.wav
```

v3 voice cloning:

```bash
audiocpp_cli --task clon --family irodori_tts \
  --model models/Irodori-TTS-500M-v3-GGUF/irodori-tts-500m-v3-q8_0.gguf \
  --backend cuda --language ja \
  --text "同じ声で短く話します。" \
  --voice-ref models/Irodori-TTS-500M-v3/samples/clone_ref1.wav \
  --request-option no_ref=false \
  --out out.wav
```

v4 with a Speaker Inversion embedding:

```bash
audiocpp_cli --task tts --family irodori_tts \
  --model models/Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-q8_0.gguf \
  --backend cuda --language ja \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --request-option speaker_embedding_file=path/to/name.speaker.safetensors \
  --out out.wav
```

## Speaker Inversion Embeddings

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)'s Speaker Inversion training learns a few speaker tokens for one voice while the base model stays frozen, and saves them as `*.speaker.safetensors`. audio.cpp uses such a file as the speaker condition in place of reference audio, like `--ref-embed` in the Python `infer.py`: the tokens go to the model as they are, without the speaker encoder.

- **By file:** request option `speaker_embedding_file` (CLI `--request-option`, or `"options"` in a server request). Legacy `speaker_embedding_path` remains accepted.
- **By name:** put the file in an `embeddings` directory next to the model weights as `<name>.safetensors` and pass `<name>` as the voice id (CLI `--voice-id`, server `"voice"`). With the model path set to that directory, the server's `GET /v1/audio/voices` lists the names. A voice id with no such file is ignored, as before.

The file must hold a `speaker_embedding` tensor of shape `[tokens, speaker_dim]` (F32, F16 or BF16). An embedding cannot be combined with reference audio or `no_ref=true`. Use it with the base checkpoint it was trained on: an embedding from another checkpoint with the same `speaker_dim` loads, but the speaker match is not guaranteed.

## Reference Length

Like Python Irodori-TTS, reference audio longer than the checkpoint's `ref_max_seconds` is cut to its first `ref_max_seconds` before it is encoded: 120 seconds for v4, and 30 seconds for checkpoints that do not state it (v3). Request option `reference_duration_sec` (legacy `max_ref_sec`, Python's `max_ref_seconds`) sets another length in seconds; `0` keeps the whole reference. Speaker embeddings are not affected.

## Request Options (use with `--request-option`)

v4 uses the normalized schema-v1 option names directly. New requests should use these names:

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `language` | `ja` | `ja` | Text language code; Irodori-TTS is Japanese-only. |
| `instruct` | text | empty | Voice-design instruction; only useful on caption-conditioned checkpoints. Legacy `instruction` and `caption` are accepted as aliases. |
| `no_ref` | bool | `true` unless a reference is provided | Use no-reference generation. Set `false` with `--voice-ref` for reference conditioning. |
| `speaker_embedding_file` | path | unset | Speaker Inversion embedding (`*.speaker.safetensors`) used instead of reference audio. Legacy `speaker_embedding_path` remains accepted. See [Speaker Inversion Embeddings](#speaker-inversion-embeddings). |
| `reference_duration_sec` | seconds | checkpoint | Trim reference audio to this length before encoding it; `0` keeps the whole reference. Legacy `max_ref_sec` remains accepted. See [Reference Length](#reference-length). |
| `num_inference_steps` | positive integer | `40` | RF diffusion steps. |
| `t_schedule_mode` | `linear`, `sway` | `linear` | Euler timestep schedule. |
| `sway_coeff` | finite float | `-1.0` | Sway coefficient; grid must strictly decrease. |
| `duration_sec` | seconds | unset | Target duration in seconds; omitted or nonpositive values select predicted duration. |
| `duration_scale` | float > 0 | `1.0` | Multiplier for predicted duration. |
| `min_duration_sec` | seconds > 0 | `0.5` | Minimum generated duration; must not exceed `max_duration_sec`. |
| `max_duration_sec` | seconds > 0 | `30` | Maximum generated duration; must be at least `min_duration_sec`. |
| `text_chunk_mode` | `word_budget`, `tag_aware`, `japanese`, `endline` | `endline` | Text chunk policy; word_budget also accepts the legacy spelling default. |
| `text_chunk_size` | integer | model text window | Maximum characters per outer chunk. |
| `text_guidance_scale` | float | `3.0` | Text classifier-free guidance scale; default 3.0. Nonpositive values disable the corresponding CFG branch. |
| `speaker_guidance_scale` | float | `5.0` | Speaker classifier-free guidance scale; default 5.0. Nonpositive values disable the corresponding CFG branch. |
| `caption_guidance_scale` | float | `3.0` | Caption classifier-free guidance scale; default 3.0. Nonpositive values disable the corresponding CFG branch. |
| `guidance_scale` | float | unset | Override all classifier-free guidance scales when set. Nonpositive values disable the corresponding CFG branch. |
| `guidance_mode` | `independent`, `joint`, `alternating` | `independent` | CFG combination mode. |
| `guidance_min_t` | float | `0.5` | Minimum diffusion timestep value where guidance is active; default 0.5. The parser does not enforce a nonnegative bound. |
| `guidance_max_t` | float | `1.0` | Maximum diffusion timestep value where guidance is active; default 1.0. The parser does not enforce a nonnegative bound. |
| `seed` | integer | random | Generation seed. |
| `trim_tail` | bool | `true` | Trim trailing silence-like samples. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `irodori_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Model weight storage type. |
| `irodori_tts.codec_weight_type` | `native`, `f32`, `f16`, `q8_0` | `native` | DACVAE codec weight storage type. |
| `irodori_tts.mem_saver` | bool | `true` | Release staged runtime graphs after request phases. |
| `irodori_tts.reference_cache_slots` | integer | `1` | Prepared reference-speaker cache slots; use `0` to disable reuse. |
| `irodori_tts.condition_graph_arena_mb` | integer MiB >= 1 | `256` | Condition encoder graph arena size. |
| `irodori_tts.rf_graph_arena_mb` | integer MiB >= 1 | `768` | RF sampler graph arena size. |
| `irodori_tts.codec_graph_arena_mb` | integer MiB >= 1 | `512` | DACVAE codec graph arena size. |
| `irodori_tts.codec_decode_chunk_steps` | integer | `0` | Decode the latent in windows of this many latent frames (25 per second); `0` decodes the whole sequence at once. See [Bounding codec memory](#bounding-codec-memory). |
| `irodori_tts.codec_decode_overlap_steps` | integer | `16` | Latent frames of context decoded on each side of a decode window and discarded. |
| `irodori_tts.codec_encode_chunk_steps` | integer | `0` | Encode reference audio in windows of this many latent frames; `0` encodes the whole reference at once. |
| `irodori_tts.codec_encode_overlap_steps` | integer | `16` | Latent frames of reference audio encoded as context on each side of an encode window and discarded. |
| `irodori_tts.condition_weight_context_mb` | integer MiB >= 1 | `32` | Condition encoder weight metadata context size. |
| `irodori_tts.rf_weight_context_mb` | integer MiB >= 1 | `32` | RF sampler weight metadata context size. |
| `irodori_tts.codec_weight_context_mb` | integer MiB >= 1 | `32` | DACVAE codec weight metadata context size. |

### Bounding codec memory

The DACVAE codec decodes the whole generated latent, and encodes the whole reference clip, as one graph, so its activation memory grows with the audio length: about 176 MiB per second of output and 107 MiB per second of reference audio (v4 Small). For anything longer than a few seconds this sets the peak of the request, well above the resident model.

`codec_decode_chunk_steps` and `codec_encode_chunk_steps` process the codec in fixed-size windows instead. Each window carries `overlap_steps` frames of real context on both sides, which are discarded, and one graph of the window size is reused, so the peak no longer depends on the length. Both are off by default. `100` (4 seconds) with the default overlap of `16` is a good setting; shorter windows save little more and run slower. Audio shorter than one window (`chunk + 2 x overlap` frames) is processed in one piece as before.

```bash
--session-option irodori_tts.codec_decode_chunk_steps=100 \
--session-option irodori_tts.codec_encode_chunk_steps=100
```

Peak GPU memory, v4 Small Q8_0 on CUDA, 48 steps:

| request | default | decode chunks | decode + encode chunks |
|---|---:|---:|---:|
| short text (6.6 s output) | 2,645 MiB | 2,404 MiB | 2,405 MiB |
| long text (25.8 s output) | 6,208 MiB | 2,416 MiB | 2,423 MiB |
| 34.7 s reference clip, first request (includes the encode) | 5,154 MiB | 5,176 MiB | 2,427 MiB |

Requests take about 5-10% longer. The output is not bit-identical: on CPU the waveform differs by at most one 16-bit step; on CUDA the chunked decode is about 67 dB SNR from the whole-sequence decode, and a chunked reference encode changes the speaker condition about as much as `codec_weight_type=f16` does. Keep the overlap at `16`: with `0` the window edges click.

## Compatibility

The runtime accepts the old option names below for existing local scripts and older standalone GGUF packages. Prefer the v1 names for new requests.

| Legacy option | v1 option |
|---|---|
| `caption` | `instruct` |
| `instruction` | `instruct` |
| `duration_seconds` | `duration_sec` |
| `min_seconds` | `min_duration_sec` |
| `max_seconds` | `max_duration_sec` |
| `mem_saver` | `irodori_tts.mem_saver` |
| `reference_cache_slots` | `irodori_tts.reference_cache_slots` |

## RF Sampling: Linear and Sway

The Euler solver supports the same timestep schedules as upstream Irodori-TTS:
`linear` (default) and `sway`. Sway changes the time grid, not the ODE solver.
`num_inference_steps` defaults to 40; `sway_coeff` defaults to -1.0 and only
changes the grid when `t_schedule_mode=sway`. Negative coefficients place more
steps near the initial noise. Zero gives the linear grid. Coefficients must be
finite and the resulting timesteps must strictly decrease; invalid combinations
are rejected instead of silently producing an invalid trajectory.

CLI example (the model path must point to your downloaded weights):

```bash
audiocpp_cli --task tts --family irodori_tts \
  --model models/Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-q8_0.gguf \
  --text "こんにちは。音声合成のテストです。" \
  --request-option no_ref=true \
  --request-option t_schedule_mode=sway \
  --request-option sway_coeff=-1.0 \
  --request-option num_inference_steps=24 \
  --out sway.wav
```

The model-specific request options are `t_schedule_mode=sway`,
`sway_coeff=-1.0`, and `num_inference_steps=24`. These also work with existing
GGUF packages whose embedded option definitions predate Sway support.

For a running `audiocpp_server`, use the registered model id in `/v1/audio/speech`:

```bash
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori","input":"こんにちは。", "options":{"no_ref":true,
       "t_schedule_mode":"sway","sway_coeff":-1.0,"num_inference_steps":24}}' \
  --output sway.wav
```

Place these settings inside `options` in server requests and CLI batch JSON.
Both single-slot and parallel server runtimes use the existing generic options
path.

Reference: [upstream RF sampler](https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/rf.py).

## Streaming through audiocpp_server

Configure the model with `"mode": "streaming"`. Existing GGUF files may embed
an older offline-only model spec; use `model_spec_override` to select this
updated spec without reconverting the weights. Example model entry (paths are
relative to the server config file):

```json
{
  "id": "irodori-tts",
  "family": "irodori_tts",
  "task": "tts",
  "mode": "streaming",
  "path": "./models/irodori-tts-v4-small-q8_0.gguf",
  "model_spec_override": "./model_specs/irodori_tts.json"
}
```

Request incremental PCM (signed 16-bit little-endian, mono, at the codec's
sample rate; standard Irodori-TTS is 48000 Hz):

```bash
curl -N http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori-tts","input":"こんにちは。音声合成のテストです。",
       "stream":true,"stream_format":"audio","response_format":"pcm",
       "options":{"no_ref":true,"t_schedule_mode":"sway","sway_coeff":-1.0,
                  "num_inference_steps":24,"stream_frames_per_event":12}}' \
  --output speech.pcm
```

Use `"stream_format":"sse"` for base64 `speech.audio.delta` events followed by
`speech.audio.done`. Model-specific settings belong inside `options`.
`stream_frames_per_event` is a positive integer (default 12, about 0.48 seconds
at 25 latent frames/second). Streaming uses this window size and the existing
`irodori_tts.codec_decode_overlap_steps` session setting (default 16); the
context samples are discarded before each chunk is sent. Duration and tail
trimming are applied before transmission, and text chunks are emitted in order.

The DiT sampling for each text chunk must finish before its codec decoding
begins. Each decoded window is then sent immediately, before the following
windows or text chunks finish. This follows the windowed codec approach
[described in speech.cpp](https://nyosegawa.com/posts/speech-cpp/); window sizes
here are fixed rather than dynamically selected. Windowed decoding can differ
slightly from whole-sequence decoding. It does not emit unfinished DiT latents.
