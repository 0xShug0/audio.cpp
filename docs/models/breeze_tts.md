# BreezeTTS 2

BreezeTTS 2 is a GGUF TTS family for instruction-conditioned speech and
prompt-audio voice cloning. The default package is Q8_0.

## Quick Start

Download the default Q8_0 package:

```bash
python3 tools/model_manager_v2.py install breeze_tts_2_q8_0 --models-root models
```

Voice cloning:

```bash
audiocpp_cli \
  --task clon \
  --family breeze_tts \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --backend cuda \
  --text "Please read this line in a clear and natural voice." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years." \
  --request-option instruction="Speak clearly and naturally." \
  --out breeze_tts_clone.wav
```

Voice design:

```bash
audiocpp_cli \
  --task tts \
  --family breeze_tts \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --backend cuda \
  --text "Welcome to the local voice demo." \
  --request-option instruction="A warm female narrator with calm pacing and studio clarity." \
  --out breeze_tts_design.wav
```

Streaming:

```bash
audiocpp_cli \
  --task tts \
  --mode streaming \
  --family breeze_tts \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --backend cuda \
  --text "Welcome to the BreezeTTS 2 streaming demo." \
  --request-option instruction="A confident product demo narrator with steady pacing." \
  --request-option stream_frames_per_event=16 \
  --out breeze_tts_stream.wav \
  --out-dir breeze_tts_stream_chunks
```

## Model

| Field | Value |
|---|---|
| Family | `breeze_tts` |
| Default GGUF | `models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | `zh`, `en` |
| Voice input | Optional for `tts`; required for `clon` |

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required for `clon` | Prompt/reference speaker audio. |
| `--reference-text` | text | empty | Transcript for prompt audio when cloning. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `reference_text` | text | empty | Request-option alias for the prompt-audio transcript. |
| `instruction` | text | `Speak clearly and naturally.` | Voice or style instruction. |
| `text_chunk_size` | integer > 0 | `600` | Long-form text chunk size. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunk mode. |
| `max_tokens` | integer > 0 | `1500` | Maximum generated acoustic frames. |
| `guidance_scale` | float >= 0 | `1.0` | Classifier-free guidance scale. |
| `temperature` | float >= 0 | `0.9` | Backbone sampling temperature. |
| `depth_temperature` | float >= 0 | `0.9` | Depth decoder sampling temperature. |
| `top_k` | integer >= 0 | `50` | Top-k sampling limit; `0` disables top-k filtering. |
| `top_p` | `0..1` | `1.0` | Top-p sampling limit. |
| `seed` | integer >= 0 | `0` | Generation seed. |
| `stream_frames_per_event` | integer > 0 | `16` | Streaming codec frames per emitted audio event. Smaller values can reduce TTFT but increase event/decoder overhead. |
| `stream_lookahead_margin` | integer >= 0 | `12` | Trailing codec frames held before emission to reduce streaming boundary artifacts. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `breeze_tts.reference_cache_slots` | integer >= 0 | `1` | Prepared reference-audio cache slots. |
| `breeze_tts.attention` | `auto`, `flash`, `eager` | `auto` | Attention kernel. `auto` uses flash except on Volta/Turing GPUs (e.g. V100), where it falls back to eager to avoid missing MMA kernels. |
| `breeze_tts.bf16_activations` | `auto`, `on`, `off` | `auto` | Reference bf16 activation rounding (and, on Metal, the bf16 KV cache). `auto` is on for CUDA/HIP/Vulkan and off on Metal; see [Metal and the reference bf16 path](#metal-and-the-reference-bf16-path). |
| `weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Weight storage type; quantized types convert at load time from the BF16 package. |

BreezeTTS streaming is incremental by default. It emits audio events from the
generated codec-frame stream instead of waiting for a whole text chunk. For the
OpenAI-compatible speech endpoint, pass streaming options inside the request
`options` object:

```json
{
  "model": "breeze-stream",
  "input": "Welcome to the BreezeTTS 2 streaming demo.",
  "stream": true,
  "stream_format": "sse",
  "response_format": "pcm",
  "options": {
    "instruction": "A confident product demo narrator with steady pacing.",
    "stream_frames_per_event": "16",
    "stream_lookahead_margin": "12"
  }
}
```

Quantized weight storage is the largest measured speedup and applies to CUDA
and HIP alike: `q8_0` cut the fixed 100-token regression case from RTF ~1.5 to
~0.95 on gfx1151 and from ~0.77 to ~0.56 on an RTX 2080 Ti, and `q4_k` reached
~0.84 / ~0.49 respectively, with no audible quality regression in the Chinese
voice-design regression cases. Counter to intuition, fp32 is the one
configuration known to be *worse* for this model (mispronunciations and
runaway repetition), because the model is trained and tuned in bf16.

## Metal and the reference bf16 path

The official BreezeTTS 2 inference rounds activations to bf16 at every decoder
stage and keeps a bf16 KV cache. CUDA, HIP and Vulkan match that by default; the
casts are cheap enough there. On Metal the same rounding costs a visible share
of the AR loop — even after adding the fused round-to-bf16 unary op for Metal
(which removes the f32 -> bf16 -> f32 cast pair at every rounding point) the
reference path measured roughly **20% slower on the AR component** on a Mac
mini M4, and about the same on an M4 MacBook Air — so on Metal it is **opt-in**:

```bash
# Reference bf16 parity on Metal (slower, matches the official implementation)
audiocpp_cli --task tts --family breeze_tts \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --backend metal \
  --text "Welcome to the local voice demo." \
  --session-option breeze_tts.bf16_activations=on \
  --out breeze_tts_bf16.wav
```

| Value | Behavior |
|---|---|
| `auto` (default) | on for CUDA/HIP/Vulkan, off on Metal |
| `on` | bf16 activation rounding on every GPU backend; on Metal it also switches the KV cache to bf16 |
| `off` | f32 activations; Metal/CUDA/Vulkan keep an f16 KV cache (HIP keeps bf16) |

The trade-off is real in both directions: the Metal default (`auto` -> off) is
the faster path but is the same f32 configuration this page warns about above,
so if a prompt mispronounces or collapses into repetition, re-run it with
`bf16_activations=on` before changing anything else. On non-Metal GPUs the
default already matches the reference, and `off` is only useful for A/B
measurements.
