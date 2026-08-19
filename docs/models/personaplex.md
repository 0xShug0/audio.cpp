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

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | User speech input. |
| `--text` / `--request-option system_prompt=<text>` | text | empty | Assistant persona/system prompt. |
| `--request-option voice_id=<id>` | packaged id | `NATF2` | Packaged PersonaPlex voice prompt id. |
| `--voice-ref` | WAV path | not set | User reference voice prompt. Overrides packaged `voice_id` when present. |
| `--temperature` / `--request-option temperature=<f>` | float >= 0 | `0.8` | Audio-token sampling temperature. |
| `--request-option text_temperature=<f>` | float >= 0 | follows `temperature` | Text-token sampling temperature. |
| `--top-k` / `--request-option top_k=<n>` | integer >= 0 | `250` | Audio-token top-k sampling limit. |
| `--request-option text_top_k=<n>` | integer >= 0 | follows `top_k` | Text-token top-k sampling limit. |
| `--request-option do_sample=true\|false` | bool | `true` | Enable stochastic sampling. Set false for greedy decoding. |
| `--seed` / `--request-option seed=<n>` | integer >= 0 | `42424242` | Seed for text and audio token sampling. |
| `--session-option personaplex.graph_arena_mb=<mb>` | integer MiB | `1024` | Reusable graph arena size for LM, depformer, and Mimi graphs. |
| `--session-option personaplex.lm_weight_context_mb=<mb>` | integer MiB | `64` | Main LM weight metadata arena size. |
| `--session-option personaplex.depformer_weight_context_mb=<mb>` | integer MiB | `64` | Depth transformer weight metadata arena size. |
| `--session-option personaplex.mimi_weight_context_mb=<mb>` | integer MiB | `64` | Mimi codec weight metadata arena size. |
| `--session-option personaplex.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | LM and Mimi matmul weight storage type when supported. |

For GGUF packages, leave weight options at `native` unless you are deliberately
testing conversion or storage behavior.
