# MOSS-Transcribe-Diarize

MOSS-Transcribe-Diarize transcribes audio with speaker labels and segment timestamps.

## Quick Start

```bash
audiocpp_cli \
  --task asr \
  --family moss_transcribe_diarize \
  --model models/MOSS-Transcribe-Diarize-GGUF/moss-transcribe-diarize-bf16.gguf \
  --backend cuda \
  --audio recording.wav \
  --text-out transcript.txt \
  --segments-out segments.json \
  --turns-out turns.json \
  --log
```

## Model

| Field | Value |
|---|---|
| Family | `moss_transcribe_diarize` |
| Task | `asr` |
| Modes | `offline`, `streaming` (text output only) |
| Model directory | `models/MOSS-Transcribe-Diarize-GGUF` |
| Default weights | `moss-transcribe-diarize-bf16.gguf` |
| Other weights | `moss-transcribe-diarize-q8_0.gguf`, `moss-transcribe-diarize-q4_k.gguf` |
| Input | Recording through `--audio` |
| Output | Transcript, speech segments, and speaker turns |

Each GGUF includes the tokenizer and configuration; no separate encoder or
diarization model is required. Q4_K can change segment boundaries and timestamps.
Use BF16 or Q8_0 when accurate timing matters.

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Recording to transcribe. |
| `--mode` | `offline`, `streaming` | `offline` | Emit the final result or text deltas during generation. |
| `--instruct` | text | upstream instruction | Replace the transcription instruction; also accepted as `--request-option instruct=<text>`. |
| `--request-option max_tokens=<n>` | integer > 0 | `5120` | Maximum generated transcript tokens. |
| `--text-out` | TXT path | not set | Save the timestamped, speaker-labelled transcript. |
| `--segments-out` | JSON path | not set | Save speech segments. |
| `--turns-out` | JSON path | not set | Save speaker turns. |

Increase `max_tokens` if a long recording reaches the output-token limit.

There are no model-specific session options. Select weight precision through
the GGUF filename passed to `--model`.

## Text Streaming

Add `--mode streaming` to the quick-start command to print text as it is
generated. The complete recording is processed before text generation starts;
live microphone input is not supported. Output files are written when the
transcription finishes.
