# Voxtral TTS

Voxtral 4B TTS generates 24 kHz mono speech using a Ministral AR backbone,
a per-frame acoustic flow-matching transformer, and Voxtral Codec. The
self-contained GGUF includes the tokenizer, codec decoder, and all 20 presets.

Upstream: [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).
The released model and reference voices use **CC BY-NC 4.0**.

The open checkpoint does not include the codec encoder required for arbitrary
reference-audio voice cloning. Select an embedded preset instead.

## Quick Start

```bash
audiocpp_cli --task tts --family voxtral_tts \
  --model /path/to/voxtral-4b-tts-2603-bf16.gguf --backend cuda \
  --text "The next train leaves in ten minutes." \
  --request-option voice_id=casual_male --seed 42 \
  --out speech.wav --log
```

No separate codec, tokenizer, or voice files are needed at inference time.
The BF16 package preserves upstream precision; Q8_0 reduces weight memory and
was also validated with the embedded presets. Both keep preset embeddings in
their original BF16 precision.

## Request Options

Use with `--request-option name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `voice_id` | `casual_male` | Embedded preset, listed below. The CLI/server voice field can also select it. |
| `max_tokens` | `2048` | Maximum generated audio frames per text chunk, at 12.5 frames/s. |
| `num_inference_steps` | `7` | Euler integration steps per acoustic frame, matching the released reference implementation. |
| `guidance_scale` | `1.2` | Acoustic CFG scale; no CFG is applied to the AR backbone. |
| `seed` | Random | Acoustic noise seed, also settable through `--seed`. |
| `text_chunk_size` | `300` | Maximum Unicode codepoints per framework text chunk. |
| `text_chunk_mode` | `default` | `default`, `tag_aware`, `japanese`, or `endline`. |

Semantic tokens use greedy selection, matching the reference implementation.

## Embedded Voices

| Group | Voice IDs |
| --- | --- |
| General presets | `casual_female`, `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male` |
| Portuguese | `pt_male`, `pt_female` |
| Dutch | `nl_male`, `nl_female` |
| Italian | `it_male`, `it_female` |
| French | `fr_male`, `fr_female` |
| Spanish | `es_male`, `es_female` |
| German | `de_male`, `de_female` |
| Arabic | `ar_male` |
| Hindi | `hi_male`, `hi_female` |

## Session Options

Use with `--session-option voxtral_tts.name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `weight_type` | `native` | Weight storage type; preserves the GGUF tensor types by default. |
| `kv_cache_type` | `f16` | AR cache storage: `f32`, `f16`, or `bf16`. |

## Long-Form and Streaming

Long text uses the framework text chunker. Each text chunk uses the selected
preset and increments the acoustic seed. `--mode streaming` emits new audio
chunks as generation proceeds, not repeated accumulated waveforms.

Codec graphs have fixed initial and continuation capacities. Continuation
decoding retains bounded left context instead of building a graph for the
complete recording. Graphs and weights remain owned by the session.
The AR decode graph grows only when a request needs more cache capacity. The
runtime retains one prefill graph, replacing it when the prompt length changes.

The server uses the existing `/v1/audio/speech` and `/v1/audio/speech/live`
routes. Select a preset through `voice`, or `request_options.voice_id`.
Configure the model with `"mode": "streaming"` to use streaming speech:

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"voxtral-tts","input":"Paris is a beautiful city!","voice":"casual_male","stream":true,"seed":42}'
```

The SSE `speech.audio.delta` payloads contain new PCM16 mono samples, at 24 kHz.
Concatenate their decoded audio bytes in order; do not replace preceding chunks.

## Conversion

```bash
python tests/voxtral_tts/convert_gguf.py \
  --model-dir /path/to/Voxtral-4B-TTS-2603 \
  --output-dir /path/to/Voxtral-4B-TTS-2603-GGUF
```

The converter retains original-precision transformer weights, folds codec
weight normalization, and embeds the preset voice embeddings and text tokenizer.
