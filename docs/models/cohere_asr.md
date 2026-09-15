# Cohere Transcribe

[Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
is a multilingual speech recognition model released under Apache-2.0.

| Field | Value |
|---|---|
| Family | `cohere_asr` |
| Task | `asr` |
| Mode | `offline` |
| Output | Transcript text |
| Long-form audio | Quiet-energy chunks, up to 35 seconds each |
| Languages | English, French, German, Spanish, Italian, Portuguese, Dutch, Polish, Greek, Arabic, Japanese, Chinese, Vietnamese, Korean |

## Convert

Accept the upstream model's access conditions and authenticate with `hf auth login`.
Download the original weights, then create a standalone GGUF:

```bash
hf download CohereLabs/cohere-transcribe-03-2026 \
  --local-dir models/cohere-transcribe-03-2026
build/debug/bin/audiocpp_gguf \
  --input models/cohere-transcribe-03-2026/model.safetensors \
  --root models/cohere-transcribe-03-2026 \
  --family cohere_asr --model-spec model_specs/cohere_asr.json \
  --type orig --output models/cohere-transcribe-03-2026-bf16.gguf
```

The GGUF embeds the configuration and tokenizer. The original safetensors directory
can also be passed directly to `--model`.

## CLI

```bash
audiocpp_cli --task asr --family cohere_asr \
  --model models/cohere-transcribe-03-2026-bf16.gguf \
  --backend cuda --audio speech.wav \
  --request-option language=en --text-out transcript.txt --log
```

Long recordings are split automatically at quiet points. The five-second boundary
search does not duplicate audio between chunks. Up to eight chunks run together.
For shorter chunks, the search window is capped at half the chunk duration.
Use `--backend cpu --threads 8` for CPU inference.

To transcribe French without punctuation and capitalization:

```bash
audiocpp_cli --task asr --family cohere_asr \
  --model models/cohere-transcribe-03-2026-bf16.gguf \
  --backend cuda --audio french.wav \
  --request-option language=fr --request-option pnc=false \
  --text-out transcript.txt --log
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `language` | `en` | `en`, `fr`, `de`, `es`, `it`, `pt`, `nl`, `pl`, `el`, `ar`, `ja`, `zh`, `vi`, or `ko`. |
| `pnc` | `true` | Request punctuation and capitalization. |
| `audio_chunk_mode` | `auto` | `auto` and `quiet_energy` use quiet boundaries; `fixed` uses equal-length chunks; `none` requires at most 35 seconds. |
| `audio_chunk_duration_sec` | `35` | Maximum audio chunk length, from 0.04 to 35 seconds. |
| `max_tokens` | `256` | Maximum generated tokens per chunk, up to 1014. Increase if generation reaches the limit before EOS. |
| `cohere_asr.weight_type` (session) | `native` | Matmul storage: `native`, `f32`, `f16`, `bf16`, or `q8_0`. |

Request options use `--request-option name=value`; session options use
`--session-option name=value`. Audio chunks must contain at least 20 ms.
The decoder uses FP16 KV storage. Translation, diarization, timestamps, and live
audio streaming are not exposed.
