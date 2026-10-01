# KugelAudio-0-Open

KugelAudio generates 24 kHz mono speech from text and a bundled preset voice.
The open model does not support cloning a voice from reference audio. This port
does not apply a watermark.

Upstream: [KugelAudio-0-Open](https://huggingface.co/kugelaudio/kugelaudio-0-open).
The model and upstream implementation use the MIT license.

## Quick Start

```bash
audiocpp_cli --task tts --family kugelaudio \
  --model /path/to/kugelaudio-0-open-q8_0.gguf --backend cuda \
  --text "The next train leaves in ten minutes." \
  --request-option voice_id=english_female --seed 1234 \
  --out speech.wav --log
```

The GGUF includes the tokenizer and all four preset voices. No separate Python
environment or tokenizer download is required for inference.

Packages are available in BF16, Q8_0 (default), and Q4_K. Each package contains
the same four voices; select its GGUF file with `--model`.

## Common Options

Use these as ordinary CLI flags.

| Option | Description |
| --- | --- |
| `--text` | Text to speak. The model infers the language from the text. |
| `--seed` | Random seed; `-1` chooses a random seed. |
| `--mode` | `offline` or `streaming`. |
| `--out` | Output WAV path. |

## Request Options

Use with `--request-option name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `voice_id` | `default` | `default`, `clear`, `english_female`, or `english_male`. |
| `num_inference_steps` | `20` | Diffusion steps per acoustic frame. |
| `guidance_scale` | `3` | Diffusion guidance; `1` disables the unconditional pass. |
| `max_tokens` | `2048` | Maximum generated tokens per text chunk. |
| `do_sample` | `false` | Sample AR control tokens rather than choose the highest logit. |
| `temperature` | `1` | AR temperature when sampling is enabled; `0` uses greedy decoding. |
| `seed` | `1234` | Random seed; `-1` chooses a random seed. |
| `text_chunk_size` | `300` | Maximum Unicode codepoints per long-form chunk. |
| `text_chunk_mode` | `default` | `default`, `tag_aware`, `japanese`, or `endline`. |

The `default` and `clear` presets are German female voices. `english_female`
and `english_male` are British English voices. Presets can speak other supported
languages, but upstream recommends their native languages for best reliability.

## Session Options

Use with `--session-option kugelaudio.name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `weight_type` | `native` | Weight storage type. |

## Long-Form and Streaming

Long text uses the framework text chunker. Each text chunk starts a new
generation with the same preset and an incremented seed.

Streaming sends decoded audio frames as they become available. Offline output
is peak-normalized if it exceeds full scale. Streaming cannot retrospectively
normalize already-delivered audio, so it sends the decoder output directly.

## Conversion

Build `audiocpp_gguf` first, then run:

```bash
python tests/kugelaudio/convert_gguf.py \
  --model-dir /path/to/original-checkpoint \
  --tokenizer-dir /path/to/Qwen2.5-tokenizer \
  --output-dir /path/to/KugelAudio-0-Open-GGUF
```

The original checkpoint must include its safetensors index and `voices/*.pt`.
Add `--type q8_0` or `--type q4_k` to create a quantized package.
The tokenizer directory must contain `tokenizer.json` and `tokenizer_config.json`
from `Qwen/Qwen2.5-1.5B`. Conversion excludes the unused audio and semantic
encoders and embeds the preset acoustic features in the GGUF.
