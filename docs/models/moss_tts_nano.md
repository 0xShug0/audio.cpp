# MOSS-TTS-Nano

MOSS-TTS-Nano is the smaller MOSS TTS path. It supports text-only continuation generation and voice cloning through the framework speaker-reference interface.

| Field | Value |
|---|---|
| Family | `moss_tts_nano` |
| Model directory | `models/MOSS-TTS-Nano-100M` |
| Required codec layout | `audio_tokenizer/` directory inside the model root |
| Task | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | Model auto-handles supported languages |
| Voice input | Optional reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

Nano supports `--mode streaming` for both `tts` and `clon`. It decodes each
generated codec frame using cached causal attention state and publishes only
new audio. State resets for each text chunk and request, while the graph is
retained for reuse. Offline decoding remains unchanged. In the server, use
`"mode": "streaming"` and `"stream_format": "sse"` on `/v1/audio/speech`.
This streams output audio from a complete text request, not live text input.

## Quantized weights on CUDA/HIP

With quantized weights (a `q8_0` GGUF, or `moss_tts_nano.weight_type=q8_0`),
incremental decoding may drift numerically from the batched prompt prefill on
CUDA/HIP backends: ggml selects different quantized kernels for single-row and
batched matmuls, and the small per-step difference compounds over the
autoregressive frame loop. Whether the drift becomes audible is
hardware-dependent (it varies with the GPU compute capability); on affected
cards long-form or cloned speech can degrade or fall into repetition loops.
CPU backends are unaffected (both paths are numerically exact there).

If you run into this, use an f16 GGUF, or load the q8_0 values into f16
runtime storage with the session option:

```bash
audiocpp_cli --task clon --family moss_tts_nano --model /path/to/moss-tts-nano-100m-q8_0.gguf \
  --backend cuda --session-option moss_tts_nano.weight_type=f16 --text "..." --out out.wav
```

For the 100M model this costs roughly 100 MB of extra VRAM and removes the
drift; measured parity between the incremental and one-shot paths drops from
several percent to well below the bf16 configuration.

Text-only continuation:

```bash
audiocpp_cli --task tts --family moss_tts_nano --model /path/to/MOSS-TTS-Nano-100M --backend cuda --text "Hello from MOSS-TTS-Nano." --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task clon --family moss_tts_nano --model /path/to/MOSS-TTS-Nano-100M --backend cuda --text "Hello from MOSS-TTS-Nano." --voice-ref /path/to/reference.wav --reference-text "Reference transcript when available." --out out.wav
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for cloning. When omitted, Nano uses text-only continuation mode. |
| `--reference-text` | text | empty string | Transcript for reference audio; valid only with `--voice-ref`. |
| `--max-tokens` | integer | `300` | Maximum generated audio frames per chunk. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic audio-token sampling. |
| `--temperature` | float | `1.7` | Audio-token sampling temperature. |
| `--top-p` | float | `0.8` | Audio-token nucleus sampling limit. |
| `--top-k` | integer | `25` | Audio-token top-k sampling limit. |
| `--repetition-penalty` | float | `1.0` | Audio-token repetition penalty. |
| `--text-chunk-size` | characters | `256` | Framework long-form text chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `reference_text` | text | empty string | Transcript for reference audio; a nonempty value requires `--voice-ref`. |
| `max_tokens` | integer >= 1 | `300` | Maximum generated audio frames. |
| `active_codebooks` | integer >= 1 | model default | Active codebooks; defaults to the checkpoint n_vq. |
| `do_sample` | bool | `true` | Enable sampling. |
| `temperature` | float | `1.7` | Audio sampling temperature. |
| `top_p` | float | `0.8` | Audio nucleus sampling probability. |
| `top_k` | integer | `25` | Audio top-k sampling limit. |
| `repetition_penalty` | float | `1` | Audio repetition penalty. |
| `text_temperature` | float | `1.5` | Text-gate sampling temperature. |
| `text_top_p` | float | `1` | Text-gate nucleus sampling limit. |
| `text_top_k` | integer | `50` | Text-gate top-k sampling limit. |
| `seed` | integer >= 0 | random | Sampling seed; omitted chooses a random seed. |
| `text_chunk_size` | integer | `256` | Framework long-form text chunk size. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunking mode. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `moss_tts_nano.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Global and local-frame weight storage type. See [Quantized weights on CUDA/HIP](#quantized-weights-on-cudahip) before running q8_0 on GPU. |
| `moss_tts_nano.global_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | shared setting | Global transformer weight storage type. |
| `moss_tts_nano.local_frame_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | shared setting | Local frame decoder weight storage type. |
| `moss_tts_nano.global_prefill_graph_arena_mb` | MB | `256` | Global prefill graph arena size. |
| `moss_tts_nano.global_decode_graph_arena_mb` | MB | `128` | Global decode graph arena size. |
| `moss_tts_nano.global_weight_context_mb` | MB | `512` | Global transformer weight context size. |
| `moss_tts_nano.local_frame_graph_arena_mb` | MB | `64` | Local frame decoder graph arena size. |
| `moss_tts_nano.local_frame_weight_context_mb` | MB | `128` | Local frame decoder weight context size. |
| `moss_tts_nano.audio_tokenizer_encoder_graph_arena_mb` | MB | `64` | Audio tokenizer encoder graph arena size. |
| `moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb` | MB | `64` | Audio tokenizer decoder graph arena size. |
| `moss_tts_nano.audio_tokenizer_weight_context_mb` | MB | `128` | Audio tokenizer weight context size. |
