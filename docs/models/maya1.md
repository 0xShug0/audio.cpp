# Maya1

`maya1` provides native offline inference for
[Maya Research Maya1](https://huggingface.co/maya-research/maya1), an English
text-to-speech model with natural-language voice design and inline emotion
controls. Audio is decoded at 24 kHz with the SNAC speech codec.

## Install

```bash
python tools/model_manager_v2.py install maya1_orig --models-root models
# Or install the smaller Q8_0 package:
python tools/model_manager_v2.py install maya1_q8_0 --models-root models
```

## Run

```bash
audiocpp_cli --task tts --family maya1 \
  --model models/Maya1-GGUF/maya1-orig.gguf \
  --backend cuda \
  --text "Welcome back. <laugh> Today is going to be interesting." \
  --request-option 'instruct=Warm female voice in her 30s with an American accent and conversational pacing.' \
  --out maya1.wav
```

Emotion tags such as `<laugh>`, `<whisper>`, `<sigh>`, and `<gasp>` may be
placed directly in the input text. See the upstream Maya1 model card for its
full prompt and emotion-tag guidance.

### Request options (use with `--request-option`)

| Option | Values | Default | Description |
|---|---|---:|---|
| `instruct` | text | required | Natural-language voice, accent, timbre, age, and delivery description. |
| `max_tokens` | positive integer | `2048` | Maximum generated audio-token count per text chunk. |
| `min_tokens` | non-negative integer | `28` | Minimum token count before EOS may stop generation. |
| `temperature` | positive float | `0.4` | Autoregressive sampling temperature. |
| `top_p` | `0.0`-`1.0` | `0.9` | Nucleus sampling probability. |
| `repetition_penalty` | positive float | `1.1` | Autoregressive repetition penalty. |
| `seed` | non-negative integer | random | Sampling and codec-noise seed. |
| `text_chunk_size` | positive integer | `300` | Maximum Unicode codepoints per long-form chunk. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Framework long-form text chunking mode. |
### Session options (use with `--session-option`)

| Option | Values | Default | Description |
|---|---|---:|---|
| `weight_type` | framework-supported storage type | `native` | Shared AR and SNAC decoder storage type. |
| `ar_weight_type` | same as above | `weight_type` | AR backbone storage override. |
| `codec_weight_type` | same as above | `weight_type` | SNAC decoder storage override. |

## Convert

Convert the original Maya1 and SNAC checkpoints into a self-contained GGUF:

```bash
conda run --no-capture-output -n qwen3-tts \
  python tests/maya1/convert_gguf.py \
  --model-dir /path/to/maya1 \
  --codec-dir /path/to/snac_24khz \
  --output-dir /path/to/Maya1-GGUF
```

Maya1 is Apache-2.0 licensed. The bundled SNAC decoder weights are MIT
licensed; retain the notices for both upstream projects when redistributing the
combined GGUF.
