# Irodori-TTS

Irodori-TTS is Japanese TTS under `--family irodori_tts`. It supports no-reference speech, optional reference-audio voice cloning, and instruction-based voice design.

The default downloadable package is the GGUF v4 Small Q8_0 checkpoint. v4 Small is the preferred path for new use because one checkpoint covers no-reference TTS, voice cloning, and voice design. The older v3 packages remain supported for existing users and local validation.

## Variants

| Variant | Package | Tasks | Notes |
|---|---|---|---|
| v4 Small | `Irodori-TTS-v4-Small-GGUF` | `tts`, `clon`, `vdes` | Bundles its v4 tokenizer and supports instruction/caption conditioning in the same checkpoint. |
| 500M v3 | `Irodori-TTS-500M-v3-GGUF` | `tts`, `clon` | Uses the shared llm-jp tokenizer layout in the original safetensors package. |
| 600M v3 VoiceDesign | `Irodori-TTS-600M-v3-VoiceDesign-GGUF` | `tts`, `clon`, `vdes` | Adds instruction/caption conditioning for voice design. |

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
  --request-option instruction="落ち着いた大人の男性。深く響く声で丁寧に話している。" \
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
  --request-option speaker_embedding_path=path/to/name.speaker.safetensors \
  --out out.wav
```

## Speaker Inversion Embeddings

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)'s Speaker Inversion training learns a few speaker tokens for one voice while the base model stays frozen, and saves them as `*.speaker.safetensors`. audio.cpp uses such a file as the speaker condition in place of reference audio, like `--ref-embed` in the Python `infer.py`: the tokens go to the model as they are, without the speaker encoder.

- **By path:** request option `speaker_embedding_path` (CLI `--request-option`, or `"options"` in a server request).
- **By name:** put the file in an `embeddings` directory next to the model weights as `<name>.safetensors` and pass `<name>` as the voice id (CLI `--voice-id`, server `"voice"`). With the model path set to that directory, the server's `GET /v1/audio/voices` lists the names. A voice id with no such file is ignored, as before.

The file must hold a `speaker_embedding` tensor of shape `[tokens, speaker_dim]` (F32, F16 or BF16). An embedding cannot be combined with reference audio or `no_ref=true`. Use it with the base checkpoint it was trained on: an embedding from another checkpoint with the same `speaker_dim` loads, but the speaker match is not guaranteed.

## Request Options (use with `--request-option`)

v4 uses the normalized schema-v1 option names directly. New requests should use these names:

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `language` | `ja` | `ja` | Text language code; Irodori-TTS is Japanese-only. |
| `instruction` | text | empty | Voice-design instruction; only useful on caption-conditioned checkpoints. Legacy `caption` is accepted as an alias. |
| `no_ref` | bool | `true` unless a reference is provided | Use no-reference generation. Set `false` with `--voice-ref` for reference conditioning. |
| `speaker_embedding_path` | path | unset | Speaker Inversion embedding (`*.speaker.safetensors`) used instead of reference audio. See [Speaker Inversion Embeddings](#speaker-inversion-embeddings). |
| `num_inference_steps` | integer | `40` | RF diffusion steps. |
| `duration_sec` | seconds | unset | Explicit output duration; omitted uses predicted duration. |
| `duration_scale` | float | `1.0` | Multiplier for predicted duration. |
| `min_duration_sec` | seconds | `0.5` | Minimum generated duration. |
| `max_duration_sec` | seconds | `30` | Maximum generated duration. |
| `text_chunk_mode` | `japanese`, `endline` | `endline` | Long-form chunking mode. |
| `text_chunk_size` | integer | model text window | Maximum characters per outer chunk. |
| `text_guidance_scale` | float | `3.0` | Text CFG strength. |
| `speaker_guidance_scale` | float | `5.0` | Speaker CFG strength. |
| `caption_guidance_scale` | float | `3.0` | Caption CFG strength. |
| `guidance_scale` | float | unset | Override all CFG strengths when set. |
| `guidance_mode` | `independent`, `joint`, `alternating` | `independent` | CFG combination mode. |
| `guidance_min_t` | float | `0.5` | Minimum diffusion timestep where guidance is active. |
| `guidance_max_t` | float | `1.0` | Maximum diffusion timestep where guidance is active. |
| `seed` | integer | random | Generation seed. |
| `trim_tail` | bool | `true` | Trim trailing silence-like samples. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `irodori_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Model weight storage type. |
| `irodori_tts.codec_weight_type` | `native`, `f32`, `f16`, `q8_0` | `native` | DACVAE codec weight storage type. |
| `irodori_tts.mem_saver` | bool | `true` | Release staged runtime graphs after request phases. |
| `irodori_tts.reference_cache_slots` | integer | `1` | Prepared reference-speaker cache slots; use `0` to disable reuse. |
| `irodori_tts.condition_graph_arena_mb` | MiB | `256` | Condition encoder graph arena size. |
| `irodori_tts.rf_graph_arena_mb` | MiB | `768` | RF sampler graph arena size. |
| `irodori_tts.codec_graph_arena_mb` | MiB | `512` | DACVAE codec graph arena size. |
| `irodori_tts.condition_weight_context_mb` | MiB | `32` | Condition encoder weight metadata context size. |
| `irodori_tts.rf_weight_context_mb` | MiB | `32` | RF sampler weight metadata context size. |
| `irodori_tts.codec_weight_context_mb` | MiB | `32` | DACVAE codec weight metadata context size. |

## Compatibility

The runtime accepts the old option names below for existing local scripts and older standalone GGUF packages. Prefer the v1 names for new requests.

| Legacy option | v1 option |
|---|---|
| `caption` | `instruction` |
| `duration_seconds` | `duration_sec` |
| `min_seconds` | `min_duration_sec` |
| `max_seconds` | `max_duration_sec` |
| `mem_saver` | `irodori_tts.mem_saver` |
| `reference_cache_slots` | `irodori_tts.reference_cache_slots` |
