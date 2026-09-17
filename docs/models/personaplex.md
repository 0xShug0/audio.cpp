# PersonaPlex

PersonaPlex is wired as `--family personaplex` for speech-to-speech
conversation. It consumes user speech, applies a packaged or user-provided voice
prompt, and generates an assistant speech response.

## Install

```bash
python3 tools/model_manager_v2.py install personaplex_7b_v1_q4_k
```

The Q4_K package is the default. A Q8_0 package is also available:

```bash
python3 tools/model_manager_v2.py install personaplex_7b_v1_q8_0
```

Both packages install into:

```text
models/PersonaPlex-GGUF
```

## Quick Start

Packaged voice prompt:

```bash
audiocpp_cli --task s2s --family personaplex \
  --model models/PersonaPlex-GGUF \
  --backend cuda \
  --audio user.wav \
  --text "You are a concise assistant. Answer naturally and briefly." \
  --request-option voice_id=NATF2 \
  --out reply.wav
```

Raw reference voice:

```bash
audiocpp_cli --task s2s --family personaplex \
  --model models/PersonaPlex-GGUF \
  --backend cuda \
  --audio user.wav \
  --voice-ref assets/resources/a.wav \
  --text "You are a calm support specialist. Keep the caller reassured." \
  --out reply_ref.wav
```

Streaming mode:

```bash
audiocpp_cli --task s2s --mode streaming --family personaplex \
  --model models/PersonaPlex-GGUF \
  --backend cuda \
  --audio user.wav \
  --request-option voice_id=NATM1 \
  --text "You are a helpful assistant. Answer the user directly." \
  --out reply_stream.wav
```

## Model

| Field | Value |
|---|---|
| Family | `personaplex` |
| Task | `s2s` |
| Modes | `offline`, `streaming` |
| Model directory | `models/PersonaPlex-GGUF` |
| Input | User speech WAV through `--audio` |
| Prompt | System/persona prompt through `--text` or `system_prompt` |
| Voice input | Packaged `voice_id` or user WAV through `--voice-ref` |
| Language | English |

If `system_prompt` is not provided, the runtime uses `--text` as the system
prompt and wraps plain text with the tags expected by the model.

## Packaged Voices

The package includes these voice prompt ids:

```text
NATF0 NATF1 NATF2 NATF3
NATM0 NATM1 NATM2 NATM3
VARF0 VARF1 VARF2 VARF3 VARF4
VARM0 VARM1 VARM2 VARM3 VARM4
```

Use `--request-option voice_id=<id>` to select one. If omitted, `NATF2` is used.
If `--voice-ref` is supplied, the runtime uses that reference audio instead of a
packaged prompt.

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | User speech input. |
| `--text` | text | empty | Assistant persona/system prompt. |
| `--voice-ref` | WAV path | not set | User reference voice prompt. Overrides packaged `voice_id` when present. |
| `--temperature` | float >= 0 | `0.8` | Audio-token sampling temperature. |
| `--top-k` | integer >= 0 | `250` | Audio-token top-k sampling limit. |
| `--seed` | integer >= 0 | `42424242` | Seed for text and audio token sampling. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `system_prompt` | text | empty | Assistant persona/system prompt. |
| `voice_id` | packaged id | `NATF2` | Packaged PersonaPlex voice prompt id. |
| `temperature` | float >= 0 | `0.8` | Audio-token sampling temperature. |
| `text_temperature` | float >= 0 | follows `temperature` | Text-token sampling temperature. |
| `top_k` | integer >= 0 | `250` | Audio-token top-k sampling limit. |
| `text_top_k` | integer >= 0 | follows `top_k` | Text-token top-k sampling limit. |
| `do_sample` | bool | `true` | Enable stochastic sampling. Set false for greedy decoding. |
| `seed` | integer >= 0 | `42424242` | Seed for text and audio token sampling. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `personaplex.graph_arena_mb` | integer MiB | `1024` | Reusable graph arena size for LM, depformer, and Mimi graphs. |
| `personaplex.lm_weight_context_mb` | integer MiB | `64` | Main LM weight metadata arena size. |
| `personaplex.depformer_weight_context_mb` | integer MiB | `64` | Depth transformer weight metadata arena size. |
| `personaplex.mimi_weight_context_mb` | integer MiB | `64` | Mimi codec weight metadata arena size. |
| `personaplex.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | LM and Mimi matmul weight storage type when supported. |
