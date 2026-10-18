# CrisperWhisper 2.0

[CrisperWhisper 2.0 Large](https://huggingface.co/nyralabs/CrisperWhisper2.0_large)
is a Whisper encoder-decoder model for verbatim and intended transcription.
The model weights and generated outputs use the upstream non-commercial research
license. See the checkpoint's license before using or redistributing them.

| Field | Value |
|---|---|
| Family | `crisperwhisper` |
| Tasks | `asr`, `align` |
| Modes | ASR: `offline`, `streaming`; alignment: `offline` |
| Output | Transcript and optional word timestamps |
| Long-form audio | Overlapping continuation windows, up to 30 seconds each |
| Languages | Explicit Whisper language codes; English by default |

The BF16 and Q8_0 GGUF packages contain the tokenizer and configuration. Use
`crisperwhisper-2-large-q8_0.gguf` for the smaller package; no external tokenizer
files are required.

## CLI

```bash
audiocpp_cli --task asr --family crisperwhisper \
  --model models/CrisperWhisper2.0-GGUF/crisperwhisper-2-large-bf16.gguf \
  --backend cuda --audio speech.wav \
  --request-option language=en \
  --text-out transcript.txt --words-out words.json --log
```

Use `transcription_mode=intended` for cleaned transcription, including normalized
numbers. The default `verbatim` mode preserves disfluencies and vocal events.
Language is supplied explicitly; it is not automatically detected.

To restore disfluencies in a known clean transcript, use
`--request-option transcription_mode=verbatimize --text "Your clean transcript"`.
This mode requires audio no longer than 30 seconds.

To align an existing transcript, use `--task align --text "The words spoken"`
with `--audio` and `--words-out`. Alignment first transcribes the recording,
then maps the supplied words onto those timestamps. Unmatched words are
interpolated between neighboring anchors; they are not independently detected
in the audio. Alignment supports long recordings in offline mode.

Long recordings are processed in overlapping windows. Each window receives the
last confirmed words as context. `--mode streaming` emits newly committed text
after each window. It requires the complete input recording; it is not live PCM
or causal streaming. Concatenate text deltas to obtain the final transcript.

## Server

Register separate model IDs for transcription and alignment in the server config:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "lazy_load": true,
  "models": [
    {
      "id": "crisperwhisper",
      "family": "crisperwhisper",
      "path": "/path/to/crisperwhisper-2-large-q8_0.gguf",
      "task": "asr",
      "mode": "streaming"
    },
    {
      "id": "crisperwhisper-align",
      "family": "crisperwhisper",
      "path": "/path/to/crisperwhisper-2-large-q8_0.gguf",
      "task": "align",
      "mode": "offline"
    }
  ]
}
```

```bash
audiocpp_server --config server.json --log

curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=crisperwhisper -F file=@speech.wav -F language=en -F stream=true

curl http://127.0.0.1:8080/v1/audio/alignments \
  -F model=crisperwhisper-align -F file=@speech.wav \
  -F language=en -F 'text=The words spoken in the recording.'
```

Omit `stream=true` for a final transcription response. Streaming emits ordinary
`transcript.text.delta` events after each continuation window; it does not accept
live microphone chunks. See [usage](../usage.md) for server options.

## Common Options

| Option | Default | Meaning |
|---|---|---|
| `--audio` | Required | Input recording. |
| `--text-out` | None | Save the transcript. |
| `--words-out` | None | Save word timestamps as JSON; also enables timestamp output. |
| `--mode` | `offline` | `offline` or window-based `streaming`. |

## Request Options (use with `--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `language` | `en` | Whisper language code, such as `de`, `fr`, `es`, or `ja`. |
| `transcription_mode` | `verbatim` | `verbatim`, `intended`, or `verbatimize` with a clean transcript supplied through `--text`. |
| `return_timestamps` | `false` | Return word timestamps. |
| `max_tokens` | `256` | Maximum generated tokens per window, up to 439 and limited by remaining decoder context. |
| `audio_chunk_mode` | `auto` | `auto` and `fixed` use continuation windows; `none` requires at most 30 seconds. |
| `audio_chunk_duration_sec` | `30` | Encoder window length, at most 30 seconds. |
| `audio_chunk_overlap_sec` | `4` | Overlap between windows; must be less than window length. |

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `crisperwhisper.weight_type` | `native` | Matmul weight storage override. |

## Conversion

Convert the original checkpoint with the standard GGUF tool. Configuration,
tokenizer, license, and model spec are embedded in the resulting file.

```bash
audiocpp_gguf --input /path/to/checkpoint/model.safetensors \
  --root /path/to/checkpoint --family crisperwhisper \
  --model-spec model_specs/crisperwhisper.json --type orig \
  --output crisperwhisper-2-large-bf16.gguf
```
