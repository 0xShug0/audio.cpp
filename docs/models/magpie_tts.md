# MagpieTTS

MagpieTTS Multilingual 357M is wired as `--family magpie_tts` for offline text
to speech. It uses baked speaker context prompts, a decoder transformer, an
autoregressive local transformer, and NanoCodec waveform decoding.

## Install

```bash
python3 tools/model_manager_v2.py install magpie_tts_orig
```

The default package installs a standalone GGUF directory:

```text
models/MagpieTTS-Multilingual-357M-GGUF
```

## Quick Start

```bash
audiocpp_cli --task tts --family magpie_tts \
  --model models/MagpieTTS-Multilingual-357M-GGUF \
  --backend cuda \
  --language en \
  --text "The production coordinator reviewed the overnight audio report and sent one clear update." \
  --request-option voice_id=Sofia \
  --out out.wav
```

German example:

```bash
audiocpp_cli --task tts --family magpie_tts \
  --model models/MagpieTTS-Multilingual-357M-GGUF \
  --backend cuda \
  --language de \
  --text "Das Team überprüfte am Morgen die Aufzeichnungen und bereitete einen kurzen Bericht vor." \
  --request-option voice_id=Jason \
  --out out_de.wav
```

## Model

| Field | Value |
|---|---|
| Family | `magpie_tts` |
| Task | `tts` |
| Modes | `offline` |
| Model directory | `models/MagpieTTS-Multilingual-357M-GGUF` |
| Languages | `ar-AE`, `ar-MSA`, `ar-SA`, `de`, `en`, `es`, `fr`, `hi`, `it`, `ko`, `pt-BR`, `vi`, `zh` |
| Voice input | Baked voice prompt through `voice_id` |
| Reference cloning | Not exposed by this integration |

Language note: the upstream NVIDIA checkpoint advertises 12 languages,
including Japanese (`ja`). The current audio.cpp integration exposes only the
language frontends listed above. The standalone GGUF embeds the model config and
sidecar resources, but Japanese requires NeMo's `JapanesePhonemeTokenizer` /
`JapaneseKatakanaAccentG2p` path backed by OpenJTalk-style text processing.
That tokenizer path has not been ported to the native C++ frontend yet, so
`ja` is intentionally not listed as supported here.

`voice_id` accepts either a baked speaker name from the package speaker map or a
numeric speaker index.

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--language <code>` / `--request-option language=<code>` | language code | `en` | Target synthesis language used by the Magpie text frontend. |
| `--request-option voice_id=<name-or-index>` | packaged speaker name or index | `0` | Baked speaker context prompt. |
| `--temperature` / `--request-option temperature=<f>` | float >= 0 | `0.6` | Audio-token sampling temperature. |
| `--top-k` / `--request-option top_k=<n>` | integer > 0 | `80` | Audio-token top-k sampling limit. |
| `--guidance-scale` / `--request-option guidance_scale=<f>` | float >= 0 | `2.5` | Classifier-free guidance scale for decoder logits. |
| `--max-tokens` / `--request-option max_tokens=<n>` | integer > 0 | `500` | Maximum decoder frames per text segment. |
| `--text-chunk-size` / `--request-option text_chunk_size=<n>` | integer > 0 | `300` | Long-form text chunk budget. |
| `--text-chunk-mode` / `--request-option text_chunk_mode=<mode>` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunking mode. |
| `--seed` / `--request-option seed=<n>` | integer >= 0 | `0` | Audio-token sampling seed. |
| `--session-option magpie_tts.graph_arena_mb=<mb>` | integer MiB | `1024` | Reusable graph arena size for Magpie stages. |
| `--session-option magpie_tts.weight_context_mb=<mb>` | integer MiB | `2048` | Weight metadata arena size. |
| `--session-option magpie_tts.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Matmul weight storage type when supported. |
| `--session-option magpie_tts.conv_weight_type=<type>` | `native`, `f32`, `f16` | `native` | Convolution weight storage type when supported. |

For GGUF packages, leave weight options at `native` unless you are deliberately
testing a conversion or storage policy.
