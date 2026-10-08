# Hviske v6

`hviske_asr_v6` is a separate family from the Conformer-based `hviske_asr`.
It uses a Whisper encoder with partial rotary positions, four-frame stacking,
an RMSNorm/linear projector, and a Qwen3 causal decoder.

Upstream: [syvai/hviske-v6](https://huggingface.co/syvai/hviske-v6).
The checkpoint is Danish-only and licensed under CC BY-NC 4.0; see the
upstream model card and notices for the underlying models' licenses.

## Usage

```bash
audiocpp_cli --task asr --family hviske_asr_v6 \
  --model /path/to/hviske-v6-bf16.gguf --backend cuda \
  --audio danish.wav --log
```

The model supports offline transcription, without timestamps, diarization, or
native streaming. Input is mixed to mono and resampled to 16 kHz. Recordings
longer than 30 seconds are split at silence boundaries into chunks of at most
28 seconds, following the upstream transcription wrapper.

## Request Options

| Option | Default | Meaning |
|---|---|---|
| `audio_chunk_mode` | `auto` | `auto`: silence splitting above 30 seconds; `silence`: silence splitting above the configured duration; `fixed`: fixed-duration cuts; `none`: no splitting, input must be at most 30 seconds. |
| `audio_chunk_duration_sec` | `28` | Maximum chunk duration (0.001-30 seconds). Does not change the 30-second trigger in `auto` mode. |
| `max_tokens` | `440` | Per-chunk generation limit, further capped to eight tokens per second plus twelve. |
| `num_beams` | `2` | Beam-search width; `1` selects greedy decoding. |
| `length_penalty` | `1.0` | Beam-score length-normalization exponent. |
| `cased` | Unset | Request cased or lowercase output. |
| `punctuated` | Unset | Request or suppress punctuation. |

Set both `cased` and `punctuated` to override automatic formatting. With either
option omitted, the model selects formatting from the audio, matching upstream.
These are model conditioning tokens, not post-processing guarantees.

```bash
--request-option cased=true --request-option punctuated=true
```

## Session Options

| Option | Default | Meaning |
|---|---|---|
| `hviske_asr_v6.weight_type` | `native` | Preserve checkpoint tensor types, or explicitly override weight storage. |

## Conversion

No model-specific checkpoint rewrite is required. Use the standard GGUF tool:

```bash
audiocpp_gguf --family hviske_asr_v6 \
  --model-spec model_specs/hviske_asr_v6.json \
  --input weights=/path/to/hviske-v6/model.safetensors \
  --root /path/to/hviske-v6 --type orig \
  --output /path/to/hviske-v6-bf16.gguf
```

The original checkpoint tensors are BF16. The converter embeds the config,
tokenizer, and model spec for standalone loading.
